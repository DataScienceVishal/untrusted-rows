#!/usr/bin/env python3
"""Time `scan_rows` over a 500-row result, for condition C9.

C9 says the scanner adds under 10 ms per 500-row result on a laptop CPU, and
it is meant to be timed over the benign corpus, which does not exist here
until slice 3. Until then the cells are cut from this repository's README as
it stood at the end of slice 1, read out of git rather than off disk, because
the current README quotes this script's output and a number measured over its
own paragraph would move every time the paragraph did. It is technical English
about prompt injection, so it is dense in exactly the words the instruction
tier's needles look for. It is also pure ASCII, which takes the fastest path
through `scanner._candidates`, so the second column appends one RIGHT SINGLE
QUOTATION MARK to each cell. That one character moves the cell off the ASCII
path and makes Python store it at two bytes a character, which every regex
then pays for, and real post bodies carry curly apostrophes routinely. Slice 3
re-times over real post bodies and that run, not this one, decides C9.

Cell length is swept up to 1,900 characters, a round figure under the 2,000
bytes a row that the 1,000,000-byte cap leaves for 500 rows. The JSON quoting
takes some of each row, so the true longest depends on the text. From 500
characters up the cost grows faster than the length, because a longer cell is
more likely to contain a needle and so runs more regexes.

Needs a clone with history, since the input is read out of git.

    uv run python scripts/throughput.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import time
from pathlib import Path

from untrusted_rows.scanner import RULES, scan_rows

ROWS = 500
REPEATS = 20
CELL_CHARS = (200, 500, 1000, 1900)
CURLY = "\u2019"
ROOT = Path(__file__).resolve().parent.parent
SOURCE = "ea0f105:README.md"


def cells(text: str, length: int, tail: str = "") -> list[list[object]]:
    step = max(1, (len(text) - length) // ROWS)
    return [[text[n * step : n * step + length - len(tail)] + tail] for n in range(ROWS)]


def median_ms(rows: list[list[object]]) -> float:
    timings = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        scan_rows(["body"], rows)
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


def main() -> int:
    shown = subprocess.run(["git", "show", SOURCE], cwd=ROOT, capture_output=True, text=True)
    if shown.returncode != 0:
        print(f"git show {SOURCE} failed, which a shallow clone or a ZIP download does:")
        print(shown.stderr.strip())
        return 2
    text = shown.stdout
    print(f"{platform.machine()} {platform.system()}, Python {platform.python_version()}")
    print(f"cells cut from {SOURCE}, {len(text)} characters")
    print(f"{len(RULES)} rules, {ROWS} rows per result, median of {REPEATS} runs")
    print()
    print(f"{'chars per cell':>14}  {'ASCII ms':>9}  {'one curly quote ms':>18}")
    for length in CELL_CHARS:
        plain = median_ms(cells(text, length))
        curly = median_ms(cells(text, length, CURLY))
        print(f"{length:>14}  {plain:>9.1f}  {curly:>18.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
