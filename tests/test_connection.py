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

# The one C7 statement the locked read-only connection runs anyway. Kept as a
# named set rather than a skip, so the test below fails if the list grows and
# also if DuckDB ever closes it.
ENGINE_ALLOWS = {"PRAGMA database_list"}


def test_the_engine_refuses_the_same_list_the_gate_does(small_db, refusal_cases):
    """Every C7 statement, with the gate taken out of the path.

    Three kinds of statement are skipped and each skip is a real limit on what
    this test proves. The empty and unparseable cases have nothing for DuckDB
    to refuse. The two-statement cases are refused by `execute` as a batch API
    question rather than as a permission one, so counting them here would
    flatter the engine.
    """
    conn = open_read_only(small_db)
    allowed_through = []
    for case in refusal_cases["must_refuse"]:
        sql = case["sql"]
        if not sql.strip() or ";" in sql or sql.startswith("SELEKT") or sql == "SELECT FROM WHERE":
            continue
        try:
            conn.execute(sql).fetchall()
        except duckdb.Error:
            continue
        allowed_through.append(sql)
    assert set(allowed_through) == ENGINE_ALLOWS


def test_pragma_is_the_one_class_the_gate_holds_alone(small_db):
    """Found by the test above failing, and worth more written down than patched.

    `lock_configuration` stops the SET-shaped pragmas, `memory_limit` and
    `threads`, and does nothing about the rest. Three of these change session
    state on a connection opened read-only, and `database_list` returns the
    absolute path of the database file, which is the kind of thing a caller
    that only gets rows should not learn.

    So for PRAGMA the usual story runs backwards. Everywhere else the parser is
    the first layer and the engine is the one that holds if the parser has a
    hole. Here the parser is the only layer, and `sql_gate` refuses every
    PRAGMA by node type before any of this is reachable.
    """
    conn = open_read_only(small_db)
    runs = []
    for sql in [
        "PRAGMA database_list",
        "PRAGMA show_tables",
        "PRAGMA version",
        "PRAGMA table_info('posts')",
        "PRAGMA enable_profiling",
        "PRAGMA disable_optimizer",
    ]:
        try:
            conn.execute(sql).fetchall()
            runs.append(sql)
        except duckdb.Error:
            pass
    assert runs == [
        "PRAGMA database_list",
        "PRAGMA show_tables",
        "PRAGMA version",
        "PRAGMA table_info('posts')",
        "PRAGMA enable_profiling",
        "PRAGMA disable_optimizer",
    ]


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


def test_the_lock_is_the_last_setting_applied(small_db):
    """Ordering is load-bearing and a dict literal is the only thing recording it.

    `lock_configuration` locks whatever is set when it runs. If an edit moved it
    above the other four they would be applied to a locked configuration and
    the open would fail, so this asserts on position rather than on behaviour.
    """
    assert list(LOCKED_CONFIG)[-1] == "lock_configuration"
    assert LOCKED_CONFIG["enable_external_access"] == "false"


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


def test_the_catalog_is_lower_cased(small_db):
    assert all(name == name.lower() for name in catalog_names(open_read_only(small_db)))
