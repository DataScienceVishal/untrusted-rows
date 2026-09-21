#!/usr/bin/env python3
"""Build the synthetic database the slice 1 transcript runs against.

Synthetic, and it is not evidence of anything. Its only jobs are to give the
gate a catalog to resolve names against and to be large enough that a cross
join takes longer than the timeout. The real corpora arrive in slice 3:
BIPIA's payloads and 20,000 Stack Exchange post bodies, both downloaded and
checksummed rather than generated here.

200,000 posts because that is the size at which `posts a, posts b` with a
predicate matching nothing overruns a 1 second deadline on this laptop, which
is the only property of the row count that matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROWS = 200_000
USERS = 50


def build(target: Path) -> None:
    target.unlink(missing_ok=True)
    conn = duckdb.connect(str(target))
    conn.execute(f"""
        CREATE TABLE users AS
        SELECT i AS id, 'user_' || i AS name
        FROM range({USERS}) t(i)
    """)
    conn.execute(f"""
        CREATE TABLE posts AS
        SELECT
            i AS id,
            i % {USERS} AS author_id,
            'Post body number ' || i || '. Synthetic filler text.' AS body
        FROM range({ROWS}) t(i)
    """)
    conn.execute("CREATE VIEW recent_posts AS SELECT * FROM posts WHERE id > 199000")
    conn.close()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo.duckdb")
    build(out)
    print(f"wrote {out} with {ROWS} posts, {USERS} users and one view", file=sys.stderr)
