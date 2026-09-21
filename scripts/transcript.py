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
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Each pair is one call and the one-line reason it is in the transcript. The
# order walks outward: ordinary read, then the caps, then the five refusals
# condition C7 names by shape rather than by statement.
CALLS = [
    ("SELECT id, author_id FROM posts WHERE id < 3 ORDER BY id", "an ordinary read"),
    ("SELECT id, body FROM posts ORDER BY id", "more rows than the cap"),
    ("ATTACH '/tmp/other.duckdb'", "a second database"),
    ("COPY posts TO '/tmp/leak.csv'", "a write to a path of the caller's choosing"),
    ("SELECT 1; DROP TABLE posts", "a second statement after a semicolon"),
    ("SELECT * FROM read_csv('/etc/passwd')", "a table function that reads the filesystem"),
    ("SELECT * FROM '/etc/passwd'", "the same read with no function call in it"),
    ("SELECT count(*) FROM posts a, posts b WHERE a.id + b.id < 0", "a runaway query"),
]


async def run(database: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "untrusted_rows.cli", "serve", str(database), "--timeout", "1"],
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
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
    if result.is_error:
        return _text_of(result).splitlines()[0]
    structured = result.structured_content or {}
    truncation = structured.get("truncation", {})
    rows = structured.get("rows", [])
    shown = f"{len(rows)} row(s) {rows[:2]}"
    if truncation.get("hit_row_cap"):
        shown += f", capped at {truncation['row_cap']}"
    if truncation.get("hit_byte_cap"):
        shown += f", capped at {truncation['byte_cap']} bytes"
    return shown


def _text_of(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "text", None))


if __name__ == "__main__":
    asyncio.run(run(Path(sys.argv[1] if len(sys.argv) > 1 else "demo.duckdb")))
