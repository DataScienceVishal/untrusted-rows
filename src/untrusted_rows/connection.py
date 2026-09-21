"""Open the database in the state the gate assumes it is in.

`sql_gate.check` is the first layer and this is the second, and the second one
is the one that holds if the first has a hole. Every statement in
`tests/fixtures/refusals.json` is refused here too, by DuckDB itself, which
`tests/test_connection.py` checks rather than asserts in a comment.

The five settings are not interchangeable. `enable_external_access=false` is
the one that stops a file read, and `lock_configuration=true` is the one that
stops a caller turning it back on. Without the lock the first setting is a
default rather than a restriction. The two extension settings are there because
an extension is new code with its own filesystem and network access, and
autoloading would fetch it on first use without anyone typing INSTALL.

`read_only=True` is separate from all of that: it refuses writes and it has
nothing to say about reads of other files, which is the distinction the two
layers exist to keep straight.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

LOCKED_CONFIG = {
    "enable_external_access": "false",
    "allow_unsigned_extensions": "false",
    "autoinstall_known_extensions": "false",
    "autoload_known_extensions": "false",
    # Last, and it locks everything above it. Verified by the refusal fixture's
    # `SET enable_external_access=true`, which DuckDB rejects at execution.
    "lock_configuration": "true",
}


def open_read_only(database: Path | str) -> duckdb.DuckDBPyConnection:
    """Open an existing database file. Missing files are an error, not a create."""
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(f"no database at {path}")
    return duckdb.connect(str(path), read_only=True, config=dict(LOCKED_CONFIG))


def catalog_names(conn: duckdb.DuckDBPyConnection) -> frozenset[str]:
    """Lower-cased tables and views, which is what the gate resolves FROM names against.

    Views are in here with the tables. A view is a name a caller may read and
    the gate has no reason to care which kind it is. `information_schema.tables`
    reports both and DuckDB's own catalog functions need two calls to say the
    same thing.
    """
    rows = conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
    return frozenset(str(row[0]).lower() for row in rows)
