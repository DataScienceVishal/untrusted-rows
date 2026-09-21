"""Command line entry point.

`serve` is the whole of it in slice 1. `fetch`, `eval` and `report` arrive with
the corpora in slices 3 and 4 and are not stubbed here, because a subcommand
that parses and then says it is not implemented is a placeholder with a help
string.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from untrusted_rows import query
from untrusted_rows.server import build_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="untrusted-rows", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the MCP server on stdio")
    serve.add_argument("database", type=Path, help="path to an existing DuckDB file")
    serve.add_argument("--row-cap", type=int, default=query.DEFAULT_ROW_CAP)
    serve.add_argument("--byte-cap", type=int, default=query.DEFAULT_BYTE_CAP)
    serve.add_argument("--timeout", type=float, default=query.DEFAULT_TIMEOUT_S)

    args = parser.parse_args(argv)

    try:
        server = build_server(
            args.database,
            row_cap=args.row_cap,
            byte_cap=args.byte_cap,
            timeout_s=args.timeout,
        )
    except FileNotFoundError as missing:
        print(missing, file=sys.stderr)
        return 2

    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
