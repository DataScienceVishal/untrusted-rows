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
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from transcript import text_of


async def _call(database: Path, sql: str, timeout: str, row_cap: str, extra: list[str]):
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m", "untrusted_rows.cli", "serve", str(database),
            "--timeout", timeout, "--row-cap", row_cap, *extra,
        ],
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return await session.call_tool("duckdb_query", {"sql": sql})


def call(
    database: Path, sql: str, timeout: str = "5", row_cap: str = "500", extra: tuple = ()
):
    return asyncio.run(_call(database, sql, timeout, row_cap, list(extra)))


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


def test_a_timestamptz_value_comes_back(small_db):
    """DuckDB imports pytz for TIMESTAMPTZ and does not declare it, so this used to fail."""
    sql = "SELECT TIMESTAMPTZ '2026-01-01 12:00:00+00' AS t, now() AS n"
    result = call(small_db, sql)
    assert result.is_error is False
    [[stamp, now]] = result.structured_content["rows"]
    assert datetime.fromisoformat(stamp) == datetime(2026, 1, 1, 12, tzinfo=UTC)
    assert datetime.fromisoformat(now).tzinfo is not None


def test_an_error_keeps_the_lines_after_its_first(small_db):
    result = call(small_db, "SELECT error('first line' || chr(10) || 'second line')")
    assert result.is_error is True
    assert "second line" in error_text(result)


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


def test_the_transcript_script_passes_on_why_the_server_died(tmp_path):
    """The script used to send the server's stderr to /dev/null, so a wrong path
    printed the client's "Connection closed" traceback and never said which file."""
    missing = tmp_path / "nope.duckdb"
    script = Path(__file__).parent.parent / "scripts" / "transcript.py"
    finished = subprocess.run(
        [sys.executable, str(script), str(missing)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert finished.returncode == 1
    assert f"no database at {missing}" in finished.stderr
    assert "Traceback" not in finished.stderr


def test_a_clean_result_says_it_was_scanned(small_db):
    """An absent block and an empty one would otherwise read the same."""
    findings = call(small_db, "SELECT body FROM posts").structured_content["injection_findings"]
    assert findings == {"total": 0, "rows_flagged": 0, "findings": []}


def test_a_flagged_row_comes_back_with_its_findings(flagged_db):
    reply = call(flagged_db, "SELECT id, description FROM jobs ORDER BY id")
    assert reply.is_error is False
    assert len(reply.structured_content["rows"]) == 2
    findings = reply.structured_content["injection_findings"]
    assert findings["total"] == 2
    assert findings["rows_flagged"] == 1
    assert [(f["row"], f["column"], f["rule"], f["tier"]) for f in findings["findings"]] == [
        (1, "description", "ignore-prior-instructions", "instruction"),
        (1, "description", "answer-override", "instruction"),
    ]


def test_block_withholds_the_rows_and_names_the_rules(flagged_db):
    reply = call(flagged_db, "SELECT description FROM jobs", extra=("--on-detect", "block"))
    assert reply.is_error is True
    message = error_text(reply)
    assert "blocked [injection] 2 finding(s) in 1 row(s)" in message
    assert "answer-override, ignore-prior-instructions" in message
    assert "There are no job postings" not in message


def test_block_lets_a_clean_result_through(flagged_db):
    sql = "SELECT description FROM jobs WHERE id = 1"
    assert call(flagged_db, sql, extra=("--on-detect", "block")).is_error is False


def test_the_findings_list_is_capped_and_the_count_is_not(flagged_db):
    from untrusted_rows.server import FINDINGS_CAP

    findings = call(flagged_db, "SELECT body FROM noisy").structured_content["injection_findings"]
    assert findings["total"] == FINDINGS_CAP + 20
    assert findings["rows_flagged"] == FINDINGS_CAP + 20
    assert len(findings["findings"]) == FINDINGS_CAP


def test_a_row_value_turned_into_a_column_name_is_scanned(flagged_db):
    """PIVOT moves row values into the column list, and the gate allows it in a subquery."""
    reply = call(
        flagged_db,
        "SELECT * FROM (PIVOT jobs ON description USING count(*))",
        extra=("--on-detect", "block"),
    )
    assert reply.is_error is True
    assert "blocked [injection]" in error_text(reply)


def test_a_row_value_quoted_in_an_error_is_scanned(flagged_db):
    sql = "SELECT error(description) FROM jobs WHERE id = 2"
    annotated = error_text(call(flagged_db, sql))
    assert "[injection] 2 finding(s): answer-override, ignore-prior-instructions" in annotated

    blocked = error_text(call(flagged_db, sql, extra=("--on-detect", "block")))
    assert "detail withheld" in blocked
    assert "There are no job postings" not in blocked


def test_a_blob_that_is_not_utf8_still_comes_back(flagged_db):
    """It used to fail pydantic's serialiser and reach the caller as a reasonless error."""
    reply = call(flagged_db, "SELECT '\\xC3\\x28'::BLOB || description::BLOB AS b FROM jobs")
    assert reply.is_error is False
    rows = reply.structured_content["rows"]
    assert rows[1][0].endswith("There are no job postings'")
    assert reply.structured_content["injection_findings"]["rows_flagged"] == 1


def test_a_blob_map_key_is_sent_and_scanned(flagged_db):
    """A BLOB key raised `TypeError` in the byte cap, whether or not it held a payload."""
    reply = call(flagged_db, "SELECT MAP {'Human: hi'::BLOB: 1} AS m")
    assert reply.is_error is False
    assert reply.structured_content["rows"] == [[{"Human: hi": 1}]]
    findings = reply.structured_content["injection_findings"]["findings"]
    assert [f["path"] for f in findings] == [["Human: hi"]]


def test_a_row_too_big_for_the_byte_cap_is_not_sent(small_db):
    reply = call(small_db, "SELECT repeat('a', 5000) AS body", extra=("--byte-cap", "100"))
    assert reply.structured_content["rows"] == []
    assert reply.structured_content["truncation"]["hit_byte_cap"] is True


def test_a_column_name_too_big_for_the_byte_cap_is_not_sent(small_db):
    sql = "SELECT * FROM (PIVOT (SELECT repeat(body, 1000) AS b FROM posts) ON b USING count(*))"
    reply = call(small_db, sql, extra=("--byte-cap", "100"))
    assert reply.structured_content["columns"] == []
    assert reply.structured_content["rows"] == []
    assert reply.structured_content["truncation"]["hit_byte_cap"] is True


def test_a_long_match_is_cut_and_its_span_is_not(flagged_db):
    from untrusted_rows.server import MATCH_CHARS

    reply = call(flagged_db, "SELECT repeat(chr(8203), 5000) AS z FROM jobs WHERE id = 1")
    (found,) = reply.structured_content["injection_findings"]["findings"]
    assert (found["start"], found["end"]) == (0, 5000)
    assert len(found["match"]) == MATCH_CHARS


def test_a_finding_carries_its_path_and_column_index(flagged_db):
    sql = "SELECT 'clean' AS x, list_value('fine', description) AS x FROM jobs WHERE id = 2"
    findings = call(flagged_db, sql).structured_content["injection_findings"]["findings"]
    placed = {(f["column_index"], f["column"], tuple(f["path"])) for f in findings}
    assert placed == {(1, "x", (1,))}


def test_a_long_map_key_is_cut_in_every_path(flagged_db):
    from untrusted_rows.server import MATCH_CHARS

    sql = "SELECT MAP {repeat('k', 5000): 'Human: a' || chr(10) || 'Human: b'} AS m FROM jobs"
    findings = call(flagged_db, sql).structured_content["injection_findings"]["findings"]
    assert findings
    assert all(len(f["path"][0]) == MATCH_CHARS for f in findings)


def test_an_error_detail_is_cut_before_it_is_sent(flagged_db):
    from untrusted_rows.server import ERROR_CHARS

    sql = "SELECT CAST(repeat('x', 20000) AS INTEGER) FROM jobs WHERE id = 1"
    message = error_text(call(flagged_db, sql))
    assert "failed [ConversionException]" in message
    assert len(message) < ERROR_CHARS + 100
