"""The second layer, which is the one that holds if the gate has a hole.

The gate is a parser and parsers have blind spots. These tests do not go
through it: they hand each refused statement straight to DuckDB on a connection
opened the way `open_read_only` opens it, and check the engine refuses it too.
That is the difference between claiming defence in depth and having it.
"""

from __future__ import annotations

import duckdb
import pytest

from untrusted_rows.connection import LOCKED_CONFIG, catalog_names, open_read_only

# The C7 statements the locked read-only connection runs anyway. Kept as a
# named set rather than a skip, so the test below fails if the list grows and
# also if DuckDB ever closes one.
ENGINE_ALLOWS = {"PRAGMA database_list", "SELECT 1; SELECT 2"}

# The six pragmas the lock does not cover. One list, used twice below: when it
# was written out twice, adding a pragma to one copy and not the other was the
# only way the test could fail for the wrong reason.
UNLOCKED_PRAGMAS = [
    "PRAGMA database_list",
    "PRAGMA show_tables",
    "PRAGMA version",
    "PRAGMA table_info('posts')",
    "PRAGMA enable_profiling",
    "PRAGMA disable_optimizer",
]


def test_the_engine_refuses_the_same_list_the_gate_does(small_db, refusal_cases):
    """Every C7 statement, with the gate taken out of the path.

    Only the two empty statements are skipped, because there is nothing there
    for DuckDB to refuse. An earlier version also skipped the unparseable and
    multi-statement cases. Both skips were wrong. DuckDB raises a Parser Error
    for the unparseable ones, which is a `duckdb.Error` and counts as a
    refusal, so skipping them understated the engine. The multi-statement skip
    was worse: it was justified by a claim that `execute` refuses a batch, and
    duckdb 1.5.5 runs one, so the skip was hiding a second gate-only class.

    35 statements reach the engine and it refuses 33. The two it runs are
    `PRAGMA database_list` and `SELECT 1; SELECT 2`. The other two-statement
    cases are refused, but for the wrong reason: read-only stops the DROP and
    the DELETE in their second halves, not the fact that there are two.
    """
    conn = open_read_only(small_db)
    allowed_through = []
    for case in refusal_cases["must_refuse"]:
        sql = case["sql"]
        if not sql.strip():
            continue
        try:
            conn.execute(sql).fetchall()
        except duckdb.Error:
            continue
        allowed_through.append(sql)
    assert set(allowed_through) == ENGINE_ALLOWS
    assert len(allowed_through) == 2


def test_pragma_runs_on_the_locked_read_only_connection(small_db):
    """Found by the test above failing, and worth more written down than patched.

    `lock_configuration` stops the SET-shaped pragmas, `memory_limit` and
    `threads`, and does nothing about these six. Two of them, `enable_profiling`
    and `disable_optimizer`, change session state on a connection opened
    read-only. `database_list` returns the absolute path of the database file,
    which is the kind of thing a caller that only gets rows should not learn.

    So for PRAGMA the usual order runs backwards. Everywhere else the parser is
    the first layer and the engine holds if the parser has a hole. Here the
    parser is the only layer, and `sql_gate` refuses every PRAGMA by node type
    before any of this is reachable.
    """
    conn = open_read_only(small_db)
    runs = []
    for sql in UNLOCKED_PRAGMAS:
        try:
            conn.execute(sql).fetchall()
            runs.append(sql)
        except duckdb.Error:
            pass
    assert runs == UNLOCKED_PRAGMAS


def test_two_statements_in_one_call_are_the_other_gate_only_class(small_db):
    """duckdb 1.5.5 runs both and returns the last result, so there is no second layer here.

    This is why `test_pragma_runs_on_the_locked_read_only_connection` is no
    longer named as the only one. The gate refuses these by counting parsed
    statements, and nothing behind it would.
    """
    conn = open_read_only(small_db)
    assert conn.execute("SELECT 1; SELECT 2").fetchall() == [(2,)]


def test_the_configuration_lock_still_stops_the_set_shaped_pragmas(small_db):
    """The half of PRAGMA the lock does cover, so the finding above is not overstated."""
    conn = open_read_only(small_db)
    for sql in ["PRAGMA memory_limit='1GB'", "PRAGMA threads=1"]:
        with pytest.raises(duckdb.Error):
            conn.execute(sql)


def test_external_access_cannot_be_switched_back_on(small_db):
    """The lock is the setting that makes the other four restrictions rather than defaults."""
    conn = open_read_only(small_db)
    with pytest.raises(duckdb.Error):
        conn.execute("SET enable_external_access=true")


def test_the_lock_applies_whatever_order_the_settings_are_in(tmp_path):
    """Replaces a test that asserted a mechanism which does not exist.

    It used to check `list(LOCKED_CONFIG)[-1] == "lock_configuration"`, on the
    claim that moving the lock earlier would apply the other four to a locked
    configuration and fail the open. That is a dict key position, which is the
    implementation rather than the behaviour, and the claim is false. Opening
    with the lock first succeeds and the restriction still holds, so this opens
    it that way and checks the restriction rather than the ordering.
    """
    source = tmp_path / "ordered.duckdb"
    duckdb.connect(str(source)).execute("CREATE TABLE posts AS SELECT 1 AS id")

    reversed_config = dict(reversed(list(LOCKED_CONFIG.items())))
    assert list(reversed_config)[0] == "lock_configuration"

    conn = duckdb.connect(str(source), read_only=True, config=reversed_config)
    assert conn.execute("SELECT current_setting('enable_external_access')").fetchone() == (False,)
    with pytest.raises(duckdb.Error):
        conn.execute("SET enable_external_access=true")


def test_a_missing_database_is_not_quietly_created(tmp_path):
    """DuckDB's default is to create. A server that invents an empty database
    on a typo answers every query with nothing and looks like it is working."""
    target = tmp_path / "absent.duckdb"
    with pytest.raises(FileNotFoundError):
        open_read_only(target)
    assert not target.exists()


def test_the_catalog_carries_views_as_well_as_tables(small_db):
    names = catalog_names(open_read_only(small_db))
    assert {"posts", "users", "recent_posts"} <= names


def test_the_catalog_qualifies_every_name_and_keeps_the_bare_one_for_the_current_schema(
    tmp_path,
):
    """The bug: a table living only in another schema put its bare name in the set.

    The gate then passed `SELECT * FROM orders`, DuckDB refused it as a catalog
    error, and the caller received a failure with no reason in it.
    """
    source = tmp_path / "schemas.duckdb"
    build = duckdb.connect(str(source))
    build.execute("CREATE TABLE posts AS SELECT 1 AS id")
    build.execute("CREATE SCHEMA sales")
    build.execute("CREATE TABLE sales.orders AS SELECT 1 AS id")
    build.close()

    names = catalog_names(open_read_only(source))
    assert "posts" in names
    assert "main.posts" in names
    assert "sales.orders" in names
    assert "orders" not in names


def test_the_catalog_is_lower_cased(small_db):
    assert all(name == name.lower() for name in catalog_names(open_read_only(small_db)))
