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
short one. Fetching one row past the cap is how truncation is detected without
pulling the whole result into memory.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import duckdb

DEFAULT_ROW_CAP = 500
DEFAULT_BYTE_CAP = 1_000_000
DEFAULT_TIMEOUT_S = 10.0


class TimedOut(Exception):
    """The deadline passed and the statement was interrupted."""


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
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error: BaseException | None = None


def execute(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    row_cap: int = DEFAULT_ROW_CAP,
    byte_cap: int = DEFAULT_BYTE_CAP,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> QueryResult:
    outcome = _Outcome()

    def run() -> None:
        try:
            cursor = conn.execute(sql)
            outcome.columns = [d[0] for d in (cursor.description or [])]
            outcome.rows = cursor.fetchmany(row_cap + 1)
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
        conn.interrupt()
        worker.join()
        elapsed = (time.perf_counter() - started) * 1000
        raise TimedOut(
            f"interrupted after {elapsed:.0f} ms, deadline was {timeout_s * 1000:.0f} ms"
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    if outcome.error is not None:
        raise outcome.error

    fetched = outcome.rows
    hit_row_cap = len(fetched) > row_cap
    kept, total_bytes, hit_byte_cap = _apply_byte_cap(fetched[:row_cap], byte_cap)

    return QueryResult(
        columns=outcome.columns,
        rows=kept,
        elapsed_ms=elapsed_ms,
        bytes_returned=total_bytes,
        row_cap=row_cap,
        byte_cap=byte_cap,
        hit_row_cap=hit_row_cap,
        hit_byte_cap=hit_byte_cap,
    )


def _apply_byte_cap(
    rows: list[tuple[Any, ...]], byte_cap: int
) -> tuple[list[list[Any]], int, bool]:
    """Drop whole rows until the serialised size fits. A half-row is not a row.

    The size measured is the JSON the server is about to send, not the width of
    the DuckDB values, because the cap exists to bound what crosses the
    transport. One row over the cap on its own is kept: returning nothing at all
    tells the caller less than returning the row that was too big.
    """
    kept: list[list[Any]] = []
    total = 0
    for row in rows:
        as_list = list(row)
        size = len(json.dumps(as_list, default=str).encode("utf-8"))
        if kept and total + size > byte_cap:
            return kept, total, True
        kept.append(as_list)
        total += size
    return kept, total, False
