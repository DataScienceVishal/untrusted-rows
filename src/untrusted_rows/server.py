"""The MCP server, exposing one tool over stdio.

One tool, `duckdb_query`, taking one string. The narrow surface is the point:
every extra parameter is another thing the gate has to reason about, and a
`table` or `limit` argument would move part of the query out of the statement
the parser sees.

A refused statement is an error and says so. `sql_gate.Refused` carries a rule
id and that id reaches the caller, because "refused: table-source" is something
an agent can act on and a stack trace is not. The rule ids are stable; the
detail strings are not, and `sql_gate` says so.

Nothing here writes to stdout. On the stdio transport stdout is the protocol
channel, so a stray print would not be noise, it would be a malformed message.
sqlglot's fallback warning for syntax it does not model goes through logging,
which lands on stderr.

The scanner that gives this project its name is not here yet. Slice 2 adds
`scan_cell` and an `injection_findings` block alongside the rows. The shape of
the result was chosen with that in mind: findings are metadata about rows that
are returned anyway, not a reason to withhold them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from untrusted_rows import query
from untrusted_rows.connection import catalog_names, open_read_only
from untrusted_rows.sql_gate import Refused, check


class QueryOutcome(BaseModel):
    """What `duckdb_query` returns on the success path."""

    sql: str
    columns: list[str]
    rows: list[list[Any]]
    truncation: dict[str, Any] = Field(
        description="Caps and whether either was hit. Rows are dropped whole, never split."
    )
    elapsed_ms: float


class QueryRefused(ToolError):
    """A gate refusal or a timeout, as the caller sees it.

    Subclassing the SDK's `ToolError` rather than `Exception` is what gets the
    rule id to the other end. `tools/base.py` sorts exceptions into deliberate
    and crashed: a `ToolError` reaches the caller with its message intact,
    while anything else becomes `UnexpectedToolError` with the text
    "Error executing tool duckdb_query" and nothing else. A refusal is the
    server working, so it belongs in the first group. This cost one round of
    the transcript printing eight identical error lines to find.

    `QueryFailed` below exists because fixing that for two exception types and
    leaving it open for the rest was the same bug with a smaller blast radius.
    Any other `duckdb.Error` was still arriving as the bare string.
    """


class QueryFailed(ToolError):
    """The statement passed the gate and DuckDB refused it anyway.

    The message carries DuckDB's own text, which can name a column the caller
    guessed at. That is a deliberate trade against the reason `unknown-table`
    exists at the gate: table names are worth withholding because they map
    what a database holds, while a column error on a statement whose table the
    caller may already `SELECT *` from tells them little they could not get by
    asking. A blank error tells them nothing at all and is the worse failure.
    """


def build_server(
    database: Path | str,
    *,
    row_cap: int = query.DEFAULT_ROW_CAP,
    byte_cap: int = query.DEFAULT_BYTE_CAP,
    timeout_s: float = query.DEFAULT_TIMEOUT_S,
) -> MCPServer:
    # Before the first request rather than on it. A cap that cannot mean
    # anything is a startup error, not something every caller finds out about.
    query.check_limits(row_cap, byte_cap, timeout_s)
    conn = open_read_only(database)
    # Read once at startup. The database is opened read-only and no statement
    # that could add a table survives the gate, so re-reading the catalog per
    # query would be a query per query to learn something that cannot change.
    catalog = catalog_names(conn)

    server = MCPServer(
        name="untrusted-rows",
        version="0.1.0",
        instructions=(
            "Read-only DuckDB access. One statement per call, SELECT or a set "
            "operation only. Rows come back capped and the caps are reported."
        ),
    )

    @server.tool(
        name="duckdb_query",
        description=(
            "Run one read-only SQL statement against the local DuckDB database "
            "and return the rows. Refuses anything that is not a single SELECT, "
            "WITH or set operation over a table in the catalog."
        ),
    )
    def duckdb_query(sql: str) -> QueryOutcome:
        try:
            check(sql, catalog)
        except Refused as refusal:
            raise QueryRefused(f"refused [{refusal.rule}] {refusal.detail}") from refusal

        try:
            outcome = query.execute(
                conn, sql, row_cap=row_cap, byte_cap=byte_cap, timeout_s=timeout_s
            )
        except query.TimedOut as expiry:
            raise QueryRefused(f"refused [timeout] {expiry}") from expiry
        except duckdb.Error as failure:
            detail = str(failure).splitlines()[0]
            raise QueryFailed(f"failed [{type(failure).__name__}] {detail}") from failure

        return QueryOutcome(
            sql=sql,
            columns=outcome.columns,
            rows=outcome.rows,
            truncation=outcome.truncation(),
            elapsed_ms=round(outcome.elapsed_ms, 3),
        )

    return server
