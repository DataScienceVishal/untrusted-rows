#!/usr/bin/env python3
"""Drive the real server over stdio and print the transcript the README carries.

A real subprocess and a real MCP client, not a function call dressed up as one.
The point of the transcript is that the refusals happen over the protocol, so
calling `duckdb_query` in-process would demonstrate something weaker than what
the README claims.

    uv run python scripts/make_demo_db.py demo.duckdb
    uv run python scripts/transcript.py demo.duckdb

The timeout line reports a real elapsed time and will differ between machines
and between runs. Every other line is deterministic.

Six of the nine calls are meant to fail, and the server logs each one at INFO
level, so the child's stderr goes to a temporary file below. Without that the
transcript arrives with six lines of "Tool 'duckdb_query' failed" interleaved
into it. Here the failures are the result rather than a problem with it.
Setting the level inside the server was the first attempt and did nothing,
because the logger doing the writing is in the child process and not in this
one. The file is printed only if the server dies: it then holds the reason,
such as "no database at nope.duckdb", which the client never sees.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError

# Each pair is one call and the one-line reason it is in the transcript. The
# order walks outward: ordinary read, the row cap, a row the scanner flags, then
# the five refusals condition C7 names by shape rather than by statement.
CALLS = [
    ("SELECT id, author_id FROM posts WHERE id < 3 ORDER BY id", "an ordinary read"),
    ("SELECT id, body FROM posts ORDER BY id", "more rows than the cap"),
    ("SELECT id, description FROM jobs ORDER BY id", "RI.1 from the ICSE paper, stored in a row"),
    ("ATTACH '/tmp/other.duckdb'", "a second database"),
    ("COPY posts TO '/tmp/leak.csv'", "a write to a path of the caller's choosing"),
    ("SELECT 1; DROP TABLE posts", "a second statement after a semicolon"),
    ("SELECT * FROM read_csv('/etc/passwd')", "a table function that reads the filesystem"),
    ("SELECT * FROM '/etc/passwd'", "the same read with no function call in it"),
    ("SELECT count(*) FROM posts a, posts b WHERE a.id + b.id < 0", "a runaway query"),
]


async def run(database: Path) -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "untrusted_rows.cli", "serve", str(database), "--timeout", "1"],
    )
    # A real file, because stdio_client hands this to Popen and an in-memory
    # buffer has no fileno.
    with tempfile.TemporaryFile("w+") as server_stderr:
        closed = False
        try:
            await _drive(params, database, server_stderr)
        except* MCPError:
            # A server that exits early reaches the client as "Connection
            # closed" inside two nested exception groups. `return` is not
            # allowed in an except* block, hence the flag.
            closed = True
        if closed:
            server_stderr.seek(0)
            reason = server_stderr.read().strip() or "the server closed the connection"
            print(reason, file=sys.stderr)
            return 1
    return 0


async def _drive(params, database: Path, errlog) -> None:
    async with (
        stdio_client(params, errlog=errlog) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        print(f"$ untrusted-rows serve {database} --timeout 1")
        print(f"tools: {', '.join(tool.name for tool in tools.tools)}")
        print()

        for sql, reason in CALLS:
            print(f"> {sql}")
            result = await session.call_tool("duckdb_query", {"sql": sql})
            print(f"  {_summarise(result)}")
            print(f"  ({reason})")
            print()


def _summarise(result) -> str:
    """One line per call: the rows, or the refusal with its rule id."""
    if result.is_error:
        return text_of(result).splitlines()[0]
    structured = result.structured_content or {}
    truncation = structured.get("truncation", {})
    rows = structured.get("rows", [])
    if len(rows) > 2:
        shown = f"{len(rows)} row(s), first two {rows[:2]}"
    else:
        shown = f"{len(rows)} row(s) {rows}"
    if truncation.get("hit_row_cap"):
        shown += f", capped at {truncation['row_cap']}"
    if truncation.get("hit_byte_cap"):
        shown += f", capped at {truncation['byte_cap']} bytes"
    for found in structured.get("injection_findings", {}).get("findings", []):
        shown += (
            f"\n  finding: rows[{found['row']}] {found['column']} [{found['tier']}/{found['rule']}]"
            f" {found['start']}-{found['end']} {found['match']!r}"
        )
    return shown


def text_of(result) -> str:
    """Concatenate the text blocks of a `CallToolResult`. Shared with the tests."""
    return "\n".join(block.text for block in result.content if getattr(block, "text", None))


if __name__ == "__main__":
    sys.exit(asyncio.run(run(Path(sys.argv[1] if len(sys.argv) > 1 else "demo.duckdb"))))
