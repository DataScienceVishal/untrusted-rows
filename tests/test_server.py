"""The tool as a caller meets it: a real subprocess, a real client, real JSON-RPC.

Calling the tool function in-process would test the same Python and prove less.
What these check that the unit tests cannot is that a refusal survives the
protocol with its rule id attached, which it did not in the first draft: the
SDK sorts exceptions into deliberate and crashed, and everything that is not a
`ToolError` reaches the caller as "Error executing tool duckdb_query" with the
reason stripped.

No network. `stdio_client` spawns a child process and talks to its pipes, which
is why the suite still passes under `--disable-socket`.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from transcript import text_of


async def _call(database: Path, sql: str, timeout: str = "5", row_cap: str = "500"):
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m", "untrusted_rows.cli", "serve", str(database),
            "--timeout", timeout, "--row-cap", row_cap,
        ],
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return await session.call_tool("duckdb_query", {"sql": sql})


def call(database: Path, sql: str, timeout: str = "5", row_cap: str = "500"):
    return asyncio.run(_call(database, sql, timeout, row_cap))


def error_text(result) -> str:
    """The script and the suite read a result the same way, from one place."""
    return text_of(result)


def test_the_server_exposes_exactly_one_tool(small_db):
    async def go():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "untrusted_rows.cli", "serve", str(small_db)],
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            return await session.list_tools()

    tools = asyncio.run(go())
    assert [tool.name for tool in tools.tools] == ["duckdb_query"]


def test_a_read_comes_back_as_structured_content(small_db):
    result = call(small_db, "SELECT id, body FROM posts WHERE id < 2 ORDER BY id")
    assert result.is_error is False
    assert result.structured_content["columns"] == ["id", "body"]
    assert result.structured_content["rows"] == [[0, "body 0"], [1, "body 1"]]


def test_an_uncapped_result_says_so(small_db):
    truncation = call(small_db, "SELECT id FROM posts").structured_content["truncation"]
    assert truncation["rows_returned"] == 6
    assert truncation["hit_row_cap"] is False


def test_a_capped_result_says_so(small_db):
    """A capped result that does not say it was capped is a wrong answer, not a short one."""
    result = call(small_db, "SELECT id FROM posts ORDER BY id", row_cap="2")
    truncation = result.structured_content["truncation"]
    assert result.structured_content["rows"] == [[0], [1]]
    assert truncation["hit_row_cap"] is True
    assert truncation["row_cap"] == 2


@pytest.mark.parametrize(
    ("sql", "rule"),
    [
        ("ATTACH '/tmp/other.duckdb'", "statement-type"),
        ("COPY posts TO '/tmp/leak.csv'", "statement-type"),
        ("SELECT 1; DROP TABLE posts", "multiple-statements"),
        ("SELECT * FROM read_csv('/etc/passwd')", "table-source"),
        ("SELECT * FROM '/etc/passwd'", "unknown-table"),
    ],
)
def test_a_refusal_keeps_its_rule_id_across_the_protocol(small_db, sql, rule):
    result = call(small_db, sql)
    assert result.is_error is True
    assert f"[{rule}]" in error_text(result)


def test_a_timeout_reaches_the_caller_as_a_timeout(slow_db):
    result = call(slow_db, "SELECT count(*) FROM posts a, posts b WHERE a.id + b.id < 0", "0.05")
    assert result.is_error is True
    assert "[timeout]" in error_text(result)


def test_a_database_error_reaches_the_caller_with_its_reason(small_db):
    """The blank-error class, found after it had been fixed for two exception types.

    Only `Refused` and `TimedOut` were wrapped as deliberate, so every other
    `duckdb.Error` arrived as "Error executing tool duckdb_query" and nothing
    else, on a server whose stated design is that a failure names its cause.
    """
    result = call(small_db, "SELECT no_such_column FROM posts")
    assert result.is_error is True
    message = error_text(result)
    assert "failed [" in message
    assert "no_such_column" in message


def test_a_table_in_another_schema_needs_its_schema(tmp_path):
    """The catalog used to drop the schema, so the gate passed this and DuckDB did not."""
    source = tmp_path / "schemas.duckdb"
    build = duckdb.connect(str(source))
    build.execute("CREATE SCHEMA sales")
    build.execute("CREATE TABLE sales.orders AS SELECT 1 AS id, 9.99 AS total")
    build.close()

    qualified = call(source, "SELECT * FROM sales.orders")
    assert qualified.is_error is False
    # The total is '9.99' rather than 9.99 because DuckDB types the literal as
    # DECIMAL and pydantic serialises a Decimal to a string to keep the digits
    # it would lose as a float.
    assert qualified.structured_content["rows"] == [[1, "9.99"]]

    bare = call(source, "SELECT * FROM orders")
    assert bare.is_error is True
    assert "[unknown-table]" in error_text(bare)


def test_a_cap_that_cannot_mean_anything_stops_the_server_starting(small_db):
    """At startup rather than once per caller, and with an exit code rather than a traceback.

    A negative row cap used to reach `fetchmany` and raise a pybind11
    TypeError, which is not a ToolError, so the caller got a blank error. It
    then briefly became an unhandled BadCap, which killed the server with a
    traceback and exit 1.
    """
    finished = subprocess.run(
        [sys.executable, "-m", "untrusted_rows.cli", "serve", str(small_db), "--row-cap", "-5"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert finished.returncode == 2
    assert "row cap must be at least 1" in finished.stderr
    assert "Traceback" not in finished.stderr
