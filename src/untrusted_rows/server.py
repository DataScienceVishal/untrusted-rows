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

Every string in a result goes through `scanner.scan_rows` after the caps have
applied, column names included. BLOB values and MAP keys are converted once,
by `scanner.sendable` called from `query._apply_byte_cap`, and that one
converted value is measured, scanned and sent.

Findings come back in an `injection_findings` block next to the rows and do
not set `isError`. The rows come back too, because the server cannot know the
agent's policy and a query that silently loses rows is worse than one that
returns them with a warning. `--on-detect block` withholds the whole result
instead, clean rows included, and is not the default.

A DuckDB error message is stored text too: `SELECT error(description)` or a
failed cast quotes a row value back to the caller. So the error detail is
cut to `ERROR_CHARS`, scanned, annotated with the rules it hit, and withheld
under `block`. The cut is by length only. An earlier version kept the first
line, which cut multi-line errors in half: DuckDB's failed-import error puts
the Python exception on its second line.

The block lists at most `FINDINGS_CAP` findings and always carries the full
count. Every string a finding repeats from the result is cut to `MATCH_CHARS`:
the match, the column name and each MAP key in the path. Without that a
result that fits the byte cap could send its payload again inside its own
findings, once per finding, since a run of zero-width characters is one
finding whose match is the entire run and a long MAP key sits in the path of
every finding beneath it. The span still covers the whole match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import duckdb
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from untrusted_rows import query
from untrusted_rows.connection import catalog_names, open_read_only
from untrusted_rows.scanner import CellFinding, scan_rows
from untrusted_rows.sql_gate import Refused, check

FINDINGS_CAP = 100
MATCH_CHARS = 200
ERROR_CHARS = 500

OnDetect = Literal["annotate", "block"]


class InjectionFindings(BaseModel):
    total: int = Field(description="Every finding in the result, including any not listed.")
    rows_flagged: int
    findings: list[dict[str, Any]] = Field(
        description="The first findings in row order, capped. Spans index the string at `path`."
    )



class QueryOutcome(BaseModel):
    """What `duckdb_query` returns on the success path."""

    sql: str
    columns: list[str]
    rows: list[list[Any]]
    truncation: dict[str, Any] = Field(
        description="Caps and whether either was hit. Rows are dropped whole, never split."
    )
    elapsed_ms: float
    injection_findings: InjectionFindings


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
    asking. A failure with no reason in it tells them nothing and is the worse
    failure.
    Under `--on-detect block` the text is withheld when the scanner flags it,
    because a failed cast or `error()` can quote a row value.
    """


class QueryBlocked(ToolError):
    """The result carried findings and the server was started with `--on-detect block`.

    The message names the rules and the counts and none of the matched text,
    since the matched text is the thing being withheld.
    """


def build_server(
    database: Path | str,
    *,
    row_cap: int = query.DEFAULT_ROW_CAP,
    byte_cap: int = query.DEFAULT_BYTE_CAP,
    timeout_s: float = query.DEFAULT_TIMEOUT_S,
    on_detect: OnDetect = "annotate",
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
            raise QueryFailed(_failure_message(failure, on_detect)) from failure

        placed = scan_rows(outcome.columns, outcome.rows)
        if placed and on_detect == "block":
            raise QueryBlocked(_blocked_message(placed))

        return QueryOutcome(
            sql=sql,
            columns=outcome.columns,
            rows=outcome.rows,
            truncation=outcome.truncation(),
            elapsed_ms=round(outcome.elapsed_ms, 3),
            injection_findings=InjectionFindings(
                total=len(placed),
                rows_flagged=len({p.row for p in placed if p.row is not None}),
                findings=[_as_dict(p) for p in placed[:FINDINGS_CAP]],
            ),
        )

    return server


def _as_dict(placed: CellFinding) -> dict[str, Any]:
    found = placed.finding
    return {
        "row": placed.row,
        "column_index": placed.column_index,
        "column": placed.column[:MATCH_CHARS],
        "path": [step[:MATCH_CHARS] if isinstance(step, str) else step for step in placed.path],
        "rule": found.rule,
        "tier": found.tier,
        "start": found.start,
        "end": found.end,
        "match": found.match[:MATCH_CHARS],
    }


def _blocked_message(placed: list[CellFinding]) -> str:
    rules = ", ".join(sorted({p.finding.rule for p in placed}))
    rows = len({p.row for p in placed if p.row is not None})
    names = len({p.column_index for p in placed if p.row is None})
    where = f"{rows} row(s)" + (f" and {names} column name(s)" if names else "")
    return f"blocked [injection] {len(placed)} finding(s) in {where}: {rules}"


def _failure_message(failure: duckdb.Error, on_detect: OnDetect) -> str:
    detail = str(failure)[:ERROR_CHARS]
    head = f"failed [{type(failure).__name__}]"
    placed = scan_rows(["error"], [[detail]])
    if not placed:
        return f"{head} {detail}"
    rules = ", ".join(sorted({p.finding.rule for p in placed}))
    if on_detect == "block":
        return f"{head} detail withheld [injection] {len(placed)} finding(s): {rules}"
    return f"{head} {detail} [injection] {len(placed)} finding(s): {rules}"
