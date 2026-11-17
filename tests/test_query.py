"""Caps and the timeout.

The caps are easy to get subtly wrong in a direction nobody notices: a result
that is quietly short looks exactly like a result that was quietly short. Every
test here checks the truncation metadata as well as the rows, because the
metadata is the only thing that tells those two apart.
"""

from __future__ import annotations

import threading
import time
import tracemalloc

import duckdb
import pytest

from untrusted_rows.connection import open_read_only
from untrusted_rows.query import BadCap, TimedOut, check_limits, execute


def test_rows_come_back_with_their_column_names(small_db):
    outcome = execute(open_read_only(small_db), "SELECT id, body FROM posts ORDER BY id")
    assert outcome.columns == ["id", "body"]
    assert outcome.rows[0] == [0, "body 0"]


def test_a_result_under_the_cap_is_not_marked_truncated(small_db):
    outcome = execute(open_read_only(small_db), "SELECT id FROM posts", row_cap=100)
    assert len(outcome.rows) == 6
    assert outcome.truncation()["hit_row_cap"] is False


def test_the_row_cap_cuts_and_says_it_cut(small_db):
    outcome = execute(open_read_only(small_db), "SELECT id FROM posts ORDER BY id", row_cap=2)
    assert outcome.rows == [[0], [1]]
    assert outcome.truncation()["hit_row_cap"] is True


def test_a_result_exactly_at_the_cap_is_not_a_truncation(small_db):
    """The off-by-one that makes every full page look like a cut page."""
    outcome = execute(open_read_only(small_db), "SELECT id FROM posts", row_cap=6)
    assert len(outcome.rows) == 6
    assert outcome.truncation()["hit_row_cap"] is False


def test_the_byte_cap_drops_whole_rows(small_db):
    outcome = execute(
        open_read_only(small_db), "SELECT id, body FROM posts ORDER BY id", byte_cap=40
    )
    assert outcome.truncation()["hit_byte_cap"] is True
    assert all(len(row) == 2 for row in outcome.rows)
    assert outcome.bytes_returned <= 40


def test_a_first_row_over_the_cap_is_dropped_and_reported(small_db):
    """A row bigger than the whole cap used to go out with `hit_byte_cap` false."""
    outcome = execute(
        open_read_only(small_db), "SELECT body FROM posts WHERE id = 0", byte_cap=10
    )
    assert outcome.columns == ["body"]
    assert outcome.rows == []
    assert outcome.truncation()["hit_byte_cap"] is True
    assert outcome.bytes_returned == len('["body"]') + len("[]")


def test_rows_past_the_byte_cap_are_not_held_in_memory(small_db):
    """Python memory stays near the kept rows, however long the result is.

    The kept rows plus the one being measured peak at about 1.1 MB under
    tracemalloc. The 5 MB limit is margin, somewhere between that and the
    roughly 50 MB that holding all 501 fetched rows reads. tracemalloc does not
    see DuckDB's own buffers, which still hold the whole result, so this is not
    a bound on process memory.
    """
    tracemalloc.start()
    try:
        outcome = execute(
            open_read_only(small_db), "SELECT repeat('x', 100000) AS c FROM range(600)"
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(outcome.rows) == 9
    assert outcome.truncation()["hit_byte_cap"] is True
    assert outcome.truncation()["hit_row_cap"] is True
    assert peak < 5 * outcome.byte_cap


def test_the_byte_cap_firing_first_does_not_fake_a_row_cap_hit(small_db):
    """Six rows under a row cap of 500, so the row cap never fired.

    The byte cap stops keeping rows after the second. This guards the drain in
    `execute`: without it, the fetchone after the byte cap stops would see the
    fourth row, since the byte cap already read the third to find it too big,
    and report a row cap hit.
    """
    outcome = execute(
        open_read_only(small_db), "SELECT id, body FROM posts ORDER BY id", byte_cap=40
    )
    assert len(outcome.rows) == 2
    assert outcome.truncation()["hit_byte_cap"] is True
    assert outcome.truncation()["hit_row_cap"] is False


def test_column_names_exactly_at_the_byte_cap_are_kept(small_db):
    """`["id"]` with an empty `[]` is 8 bytes and with `[[0]]` is 11."""
    sql = "SELECT id FROM posts WHERE id = 0"
    nothing = execute(open_read_only(small_db), sql, byte_cap=7)
    assert nothing.columns == []

    names_only = execute(open_read_only(small_db), sql, byte_cap=8)
    assert names_only.columns == ["id"]
    assert names_only.rows == []
    assert names_only.bytes_returned == 8

    short_by_one = execute(open_read_only(small_db), sql, byte_cap=10)
    assert short_by_one.rows == []

    whole = execute(open_read_only(small_db), sql, byte_cap=11)
    assert whole.rows == [[0]]
    assert whole.truncation()["hit_byte_cap"] is False


@pytest.mark.parametrize(
    ("sql", "sent"),
    [
        ("SELECT '\\xFF\\xFE'::BLOB AS x", '["\ufffd\ufffd"]'),
        ("SELECT 'nan'::DOUBLE AS x", "[null]"),
        ("SELECT INTERVAL 1 DAY AS x", '["P1D"]'),
        ("SELECT 'é' AS x", '["é"]'),
    ],
)
def test_a_row_is_measured_as_the_bytes_that_are_sent(small_db, sql, sent):
    outcome = execute(open_read_only(small_db), sql)
    assert outcome.bytes_returned == len('["x"]') + len(f"[{sent}]".encode())


def test_the_commas_between_rows_count_against_the_byte_cap(small_db):
    """`["id"]` is 6 bytes and `[[0],[1],[2]]` is 13: 9 for the rows, 2 brackets, 2 commas."""
    sql = "SELECT id FROM posts WHERE id < 3 ORDER BY id"
    whole = execute(open_read_only(small_db), sql, byte_cap=19)
    assert whole.rows == [[0], [1], [2]]
    assert whole.bytes_returned == 19
    assert whole.truncation()["hit_byte_cap"] is False

    cut = execute(open_read_only(small_db), sql, byte_cap=18)
    assert cut.rows == [[0], [1]]
    assert cut.bytes_returned == len('["id"]') + len("[[0],[1]]")


def test_a_long_column_name_counts_against_the_byte_cap(small_db):
    """PIVOT makes a row value a column name, so the name counts against the cap."""
    outcome = execute(
        open_read_only(small_db),
        "PIVOT (SELECT repeat('a', 5000) AS body) ON body USING count(*)",
        byte_cap=100,
    )
    assert outcome.columns == []
    assert outcome.rows == []
    assert outcome.truncation()["hit_byte_cap"] is True
    assert outcome.bytes_returned == 0


def test_column_names_that_fit_leave_less_room_for_rows(small_db):
    short = execute(open_read_only(small_db), "SELECT id FROM posts ORDER BY id", byte_cap=20)
    long = execute(
        open_read_only(small_db), f"SELECT id AS {'i' * 12} FROM posts ORDER BY id", byte_cap=20
    )
    assert len(short.rows) > len(long.rows)
    assert long.bytes_returned <= 20


@pytest.mark.parametrize(
    ("key_sql", "key"),
    [
        ("DATE '2024-01-01'", "2024-01-01"),
        ("'k'::BLOB", "k"),
        ("1.5::DECIMAL(4, 2)", "1.50"),
        ("'00000000-0000-0000-0000-000000000001'::UUID", "00000000-0000-0000-0000-000000000001"),
    ],
)
def test_date_blob_decimal_and_uuid_map_keys_come_back_as_strings(small_db, key_sql, key):
    outcome = execute(open_read_only(small_db), f"SELECT MAP {{{key_sql}: 1}} AS m")
    assert outcome.rows == [[{key: 1}]]


def test_both_caps_are_reported_even_when_neither_fired(small_db):
    """The caller cannot tell a short result from a capped one without the numbers."""
    truncation = execute(open_read_only(small_db), "SELECT id FROM posts").truncation()
    assert truncation["row_cap"] > 0
    assert truncation["byte_cap"] > 0
    assert truncation["rows_returned"] == 6


def test_a_runaway_query_is_interrupted(slow_db):
    """The bound is on the interrupt, not on the test finishing eventually.

    This asserted `< 10` against a 50 ms deadline, which passes if cancelling
    takes nine seconds. 2 seconds is still 40 times the deadline, which leaves
    room for a loaded machine without the assertion becoming decorative.
    """
    conn = open_read_only(slow_db)
    started = time.perf_counter()
    with pytest.raises(TimedOut):
        execute(conn, "SELECT count(*) FROM posts a, posts b WHERE a.id + b.id < 0", timeout_s=0.05)
    assert time.perf_counter() - started < 2


def test_the_connection_survives_the_interrupt(slow_db):
    """A timeout that poisons the session turns one bad query into a dead server."""
    conn = open_read_only(slow_db)
    with pytest.raises(TimedOut):
        execute(conn, "SELECT count(*) FROM posts a, posts b WHERE a.id + b.id < 0", timeout_s=0.05)
    assert execute(conn, "SELECT count(*) FROM posts").rows == [[20000]]


def test_a_database_error_reaches_the_caller_rather_than_the_worker_thread(small_db):
    """The blanket catch in `execute` exists for this and would otherwise hide it.

    The query runs on a worker thread, where an uncaught exception is printed
    and lost, and the caller would see an empty result instead of a failure.
    """
    with pytest.raises(duckdb.Error):
        execute(open_read_only(small_db), "SELECT nope FROM posts")


def test_two_callers_at_once_get_their_own_rows(small_db):
    """The bug this file did not catch until an audit went looking for it.

    DuckDB's `execute` returns the connection, so the result set is connection
    state, and the MCP SDK runs a synchronous tool body in a thread pool. On
    one shared connection a caller that asked for `posts` came back with five
    rows of `users`, under its own `sql` field, with no error and no truncation
    flag. Ten rounds because it did not reproduce on every attempt.
    """
    conn = open_read_only(small_db)
    seen = {}

    def ask(tag: str, sql: str) -> None:
        seen[tag] = execute(conn, sql)

    for _ in range(10):
        threads = [
            threading.Thread(target=ask, args=("posts", "SELECT id, body FROM posts ORDER BY id")),
            threading.Thread(target=ask, args=("users", "SELECT id, name FROM users ORDER BY id")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert seen["posts"].columns == ["id", "body"]
        assert len(seen["posts"].rows) == 6
        assert seen["users"].columns == ["id", "name"]
        assert len(seen["users"].rows) == 3


def test_one_callers_timeout_does_not_cancel_another_query(slow_db):
    """`conn.interrupt()` is connection-wide and killed every query in flight.

    The victim used to die with `InvalidInputException: No open result set`
    rather than returning its rows.
    """
    conn = open_read_only(slow_db)
    victim = {}

    def slow_but_within_budget() -> None:
        try:
            victim["rows"] = execute(
                conn, "SELECT count(*) FROM posts", timeout_s=30
            ).rows
        except Exception as failure:  # noqa: BLE001
            victim["error"] = failure

    thread = threading.Thread(target=slow_but_within_budget)
    thread.start()
    with pytest.raises(TimedOut):
        execute(conn, "SELECT count(*) FROM posts a, posts b WHERE a.id + b.id < 0", timeout_s=0.05)
    thread.join()

    assert victim.get("error") is None
    assert victim["rows"] == [[20000]]


@pytest.mark.parametrize(
    ("row_cap", "byte_cap", "timeout_s"),
    [(0, 1000, 1.0), (-5, 1000, 1.0), (10, 0, 1.0), (10, 1000, 0), (10, 1000, -1.0)],
)
def test_a_cap_that_cannot_mean_anything_is_refused_before_the_query(row_cap, byte_cap, timeout_s):
    """`--row-cap -5` reached `fetchmany(-4)` and raised a pybind11 TypeError,
    which is not a ToolError, so the caller got a blank error. A timeout of 0
    was worse: every query raised TimedOut before it started."""
    with pytest.raises(BadCap):
        check_limits(row_cap, byte_cap, timeout_s)
