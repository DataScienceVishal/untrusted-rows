"""Caps and the timeout.

The caps are easy to get subtly wrong in a direction nobody notices: a result
that is quietly short looks exactly like a result that was quietly short. Every
test here checks the truncation metadata as well as the rows, because the
metadata is the only thing that tells those two apart.
"""

from __future__ import annotations

import threading
import time

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


def test_a_single_row_over_the_cap_is_still_returned(small_db):
    """Returning nothing tells the caller less than returning the row that was too big."""
    outcome = execute(
        open_read_only(small_db), "SELECT body FROM posts WHERE id = 0", byte_cap=1
    )
    assert outcome.rows == [["body 0"]]
    assert outcome.truncation()["hit_byte_cap"] is False
    assert outcome.bytes_returned > 1


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
