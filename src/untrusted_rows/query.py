"""Run one checked statement under a row cap, a byte cap and a timeout.

DuckDB has no statement timeout setting. The mechanism here is the one
`Shimiyahya/mcp-data-warehouse-server` uses and names the caveat for: the query
runs on a worker thread, the calling thread waits, and on expiry it calls
`conn.interrupt()`. It is best-effort in a specific way worth stating. The
interrupt takes effect where DuckDB checks for it, so a statement can overrun
the deadline by however long the current operator runs without checking, and
the number in `elapsed_ms` is the real one rather than the deadline.

Measured on this machine while writing it: a cross join over 200,000 rows with
a predicate that matches nothing was cancelled 1.01s after a 1.0s deadline, and
the connection answered an ordinary count afterwards. That the connection
survives is the part worth testing, because a timeout that poisons the session
turns one bad query into a dead server.

Both caps apply after execution, not as a rewritten LIMIT. Rewriting the
caller's SQL means the rows returned are not the rows the statement asked for,
and an aggregate over a silently limited scan is a wrong answer rather than a
short one.

Rows are fetched one at a time and measured as they arrive, so the Python side
holds the kept rows plus the one being measured. The row cap's truncation is
detected by fetching row `row_cap + 1`. Rows the byte cap left unread are still
fetched and dropped one at a time, so `hit_row_cap` means the same thing
whichever cap fired first. This bounds Python memory only. DuckDB builds the
whole result in its own memory inside `cursor.execute`, before the first fetch,
so process memory still grows with the full result. On 600 cells of 100,000
characters under the default caps, peak Python memory measured with
`tracemalloc` is about 1.1 MB to return 0.9 MB. The measuring runs on the
worker thread, so it counts toward the deadline and toward `elapsed_ms`.

Every call gets its own `conn.cursor()`, and that is a correctness requirement
rather than tidiness. DuckDB's `execute` returns the connection, so the result
set is connection state. The MCP SDK dispatches a synchronous tool body into a
thread pool, so two calls in flight on one connection overwrite each other's
results: measured before the fix, a caller that asked for `posts` received five
rows of `users` under its own `sql` field, with no error and no truncation flag.
`interrupt()` is scoped the same way, so one caller's timeout aborted every
other caller's query. Both are fixed by the cursor and neither is fixed by a
lock around `execute`.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import duckdb
from pydantic import TypeAdapter

from untrusted_rows.scanner import sendable

DEFAULT_ROW_CAP = 500
DEFAULT_BYTE_CAP = 1_000_000
DEFAULT_TIMEOUT_S = 10.0

_ROW = TypeAdapter(list[Any])


class TimedOut(Exception):
    """The deadline passed and the statement was interrupted."""


class BadCap(ValueError):
    """A cap or deadline that cannot mean anything, caught before a query runs."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    elapsed_ms: float
    bytes_returned: int
    row_cap: int
    byte_cap: int
    hit_row_cap: bool = False
    hit_byte_cap: bool = False

    def truncation(self) -> dict[str, Any]:
        """What the caller needs to know that the row list cannot tell it."""
        return {
            "hit_row_cap": self.hit_row_cap,
            "hit_byte_cap": self.hit_byte_cap,
            "row_cap": self.row_cap,
            "byte_cap": self.byte_cap,
            "rows_returned": len(self.rows),
            "bytes_returned": self.bytes_returned,
        }


@dataclass
class _Outcome:
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    bytes_returned: int = 0
    hit_row_cap: bool = False
    hit_byte_cap: bool = False
    error: BaseException | None = None


def execute(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    row_cap: int = DEFAULT_ROW_CAP,
    byte_cap: int = DEFAULT_BYTE_CAP,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> QueryResult:
    check_limits(row_cap, byte_cap, timeout_s)
    outcome = _Outcome()
    # One cursor per call. See the module docstring: sharing the connection
    # crosses results between concurrent callers.
    cursor = conn.cursor()

    def run() -> None:
        try:
            cursor.execute(sql)
            names = [d[0] for d in (cursor.description or [])]
            within_row_cap = itertools.islice(iter(cursor.fetchone, None), row_cap)
            (
                outcome.columns,
                outcome.rows,
                outcome.bytes_returned,
                outcome.hit_byte_cap,
            ) = _apply_byte_cap(names, within_row_cap, byte_cap)
            # Fetch and drop whatever the byte cap left unread. Without this the
            # fetchone below sees the row after the one the byte cap stopped at
            # and reports a row cap hit.
            for _ in within_row_cap:
                pass
            outcome.hit_row_cap = cursor.fetchone() is not None
        except BaseException as exc:  # noqa: BLE001
            # Deliberately blanket. This runs on a worker thread, where an
            # uncaught exception is printed to stderr and then lost, and the
            # caller would see an empty result rather than a failure. The
            # exception is re-raised on the calling thread below.
            outcome.error = exc

    started = time.perf_counter()
    worker = threading.Thread(target=run, name="untrusted-rows-query", daemon=True)
    worker.start()
    worker.join(timeout_s)

    if worker.is_alive():
        # The cursor's interrupt, not the connection's, so a slow query does
        # not cancel every other caller's work.
        cursor.interrupt()
        worker.join()
        elapsed = (time.perf_counter() - started) * 1000
        raise TimedOut(
            f"interrupted after {elapsed:.0f} ms, deadline was {timeout_s * 1000:.0f} ms"
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    if outcome.error is not None:
        raise outcome.error

    return QueryResult(
        columns=outcome.columns,
        rows=outcome.rows,
        elapsed_ms=elapsed_ms,
        bytes_returned=outcome.bytes_returned,
        row_cap=row_cap,
        byte_cap=byte_cap,
        hit_row_cap=outcome.hit_row_cap,
        hit_byte_cap=outcome.hit_byte_cap,
    )


def check_limits(row_cap: int, byte_cap: int, timeout_s: float) -> None:
    """Reject a cap that cannot mean anything, before it reaches DuckDB.

    `--row-cap -5` used to reach `fetchmany(-4)` and raise a pybind11
    `TypeError`, which is not a `ToolError`, so the caller received
    "Error executing tool duckdb_query" with the reason stripped. A
    non-positive timeout was worse: every query raised `TimedOut` before it
    started, and the server looked like it was working.
    """
    if row_cap < 1:
        raise BadCap(f"row cap must be at least 1, got {row_cap}")
    if byte_cap < 1:
        raise BadCap(f"byte cap must be at least 1, got {byte_cap}")
    if timeout_s <= 0:
        raise BadCap(f"timeout must be above zero, got {timeout_s}")


def _apply_byte_cap(
    columns: list[str], rows: Iterable[tuple[Any, ...]], byte_cap: int
) -> tuple[list[str], list[list[Any]], int, bool]:
    """Drop whole rows, or the column names too, until the serialised size fits.

    A half-row is not a row. The column names are counted first, as their JSON
    list, because `PIVOT` turns row values into column names.
    `test_a_long_column_name_counts_against_the_byte_cap` pins this. Names that
    alone exceed the cap are dropped along with every row. Cutting a name short
    would send a column the statement never produced.

    The size measured is the column names' JSON plus the rows' JSON list as
    pydantic writes it into `structuredContent`, after `sendable`, and the rows
    returned are the converted ones. The rows list's brackets and the commas
    between rows are counted, so `bytes_returned` is the two lists' combined
    length. The one exception is names over the cap, reported as 0 although
    the two empty lists are 4 bytes.
    The SDK also sends a second, indented copy as a text block, so the
    response is larger than this figure. The previous call,
    `json.dumps(row, default=str)`, measured a BLOB as its `b'...'` repr, a
    non-ASCII character as a six-byte escape, and NaN as three bytes where
    pydantic sends `null`, so a list of NaN could pass the cap and send five
    bytes for every four measured.

    A first row that alone exceeds the cap is dropped too, and the result comes
    back empty with `hit_byte_cap` true. It used to be kept with the flag false.
    A single cell of any size then went out reported as uncapped, and the
    scanner read all of it after the timeout had stopped applying.
    """
    total = len(_ROW.dump_json(columns)) + len(b"[]")
    if total > byte_cap:
        return [], [], 0, True
    kept: list[list[Any]] = []
    for row in rows:
        as_list = [sendable(cell) for cell in row]
        size = len(_ROW.dump_json(as_list)) + (1 if kept else 0)
        if total + size > byte_cap:
            return columns, kept, total, True
        kept.append(as_list)
        total += size
    return columns, kept, total, False
