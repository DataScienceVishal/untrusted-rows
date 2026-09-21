"""Databases the tests run against.

Everything here is built in a temp directory and thrown away. No fixture reads
the demo database at the repository root, because that one is a convenience for
the transcript and a test that depends on a file a reader may not have built is
a test that fails in a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def refusal_cases() -> dict:
    return json.loads((FIXTURES / "refusals.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def catalog(small_db) -> frozenset[str]:
    """Read out of the real database rather than written down here.

    A hand-written set drifts from what `catalog_names` produces, and it
    drifted the moment the catalog learned about schemas: the literal held
    bare names only, so every gate test agreed with a catalog the server would
    never build.
    """
    from untrusted_rows.connection import catalog_names, open_read_only

    return catalog_names(open_read_only(small_db))


@pytest.fixture(scope="session")
def small_db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("small") / "small.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE users AS SELECT i AS id, 'user_' || i AS name FROM range(3) t(i)")
    conn.execute("""
        CREATE TABLE posts AS
        SELECT i AS id, i % 3 AS author_id, 'body ' || i AS body
        FROM range(6) t(i)
    """)
    conn.execute("CREATE VIEW recent_posts AS SELECT * FROM posts WHERE id > 3")
    conn.close()
    return path


@pytest.fixture(scope="session")
def slow_db(tmp_path_factory) -> Path:
    """Wide enough that `posts a, posts b` outlives a short deadline.

    20,000 rows is 400 million pairs. The timeout test asks for 50 ms, so the
    margin is large enough that a loaded machine does not turn the test into a
    coin flip in the direction of a false pass.
    """
    path = tmp_path_factory.mktemp("slow") / "slow.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE posts AS SELECT i AS id FROM range(20000) t(i)")
    conn.close()
    return path
