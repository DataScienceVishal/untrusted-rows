"""Inspect one cell of text for content aimed at the model that will read it.

`scan_cell(text)` returns every finding in `text` and does nothing else: no
model, no network, no state, no configuration. The rule list below is fixed
before either corpus is downloaded, and the commit that adds this file is the
record of it. A rule added after seeing a miss in slice 3 is a rule
fitted to the test set, so a change here after that point has to say so in its
own commit and in the README.

Three tiers, kept apart because they are expected to differ sharply in
precision and averaging them would hide exactly that.

`format` matches strings that exist to move a model between roles: chat
template markers and role tags. Expected to be close to zero false positives,
with one rule predicted in advance to be the tier's worst. Line-initial
`Human:` and `Assistant:` are ordinary English on a forum where people quote
conversations and chatbot output.

`obfuscation` matches characters stored text has little benign reason to hold:
the Unicode tag block, zero-width characters, bidirectional overrides, and a
single token mixing Latin with Cyrillic or Greek. Also predicted in advance:
`mixed-script` will fire on a microsecond figure typed with GREEK SMALL LETTER
MU rather than MICRO SIGN, since the token `10`, mu, `s` then mixes Greek with
Latin. That is a units habit and not an attack. And `zero-width` will
fire on emoji sequences, which join their parts with U+200D, the same
ZERO WIDTH JOINER used to smuggle text. Both stay in, because exempting them
now would be tuning against a corpus this repository has not fetched yet.

`instruction` is the lexical tier and the one expected to be expensive in false
positives.
Second-person imperatives addressed to a reading model, in the four families
the spec names: negating prior context, reassigning a role, dictating the
response, and telling the model what to say or withhold from the user. A family is
split where one phrasing is predicted to carry most of its false positives, so
slice 3 can charge that phrasing to its own rule rather than to a tier average.
`role-reassignment` is not split, and its five phrasings share one count.

Runs of invisible characters are one finding, not one per character, so a
kilobyte of zero-width spaces produces one entry with the span of the run.

Most cells cannot match most rules, and the first draft of this file ran every
regex over every cell. So each format and instruction rule carries needles,
and its regex only runs on a cell that contains one of them, while the
obfuscation tier is gated by the characters it looks for. The skip is exact
rather than heuristic, and `test_scanner.py` checks it against the unfiltered
scan. A `format` needle is a literal the pattern cannot
match without. An `obfuscation` rule needs one of a known set of characters,
all outside ASCII, so a cell with none of them skips the tier. An
`instruction` needle is a word every match must contain, looked up in the
cell folded the way `re.IGNORECASE` folds it: ASCII
lowercase plus the four non-ASCII characters that the flag matches to an ASCII
letter, which are the dotted and dotless I, the long s and the Kelvin sign.
Plain `str.lower()` misses those four, so without `_FOLD` an attacker writing
the long s in "re\u017fpond" would match the regex and be skipped by the filter.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import TypeAdapter

Tier = Literal["format", "obfuscation", "instruction"]
TIERS: tuple[Tier, ...] = ("format", "obfuscation", "instruction")


@dataclass(frozen=True)
class Finding:
    rule: str
    tier: Tier
    start: int
    end: int
    match: str


@dataclass(frozen=True)
class Rule:
    id: str
    tier: Tier
    pattern: re.Pattern[str]
    needles: tuple[str, ...] = ()


# What each obfuscation rule cannot match without, as one character class
# each. A cell with none of the first set skips the tier; a cell with no Greek
# or Cyrillic letter skips `mixed-script`, the tier's slowest rule by far.
_GREEK_CYRILLIC = "\u0370-\u03ff\u0400-\u04ff"
_OBFUSCATION_GATE = re.compile(
    f"[\U000e0000-\U000e007f\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e\u2066-\u2069{_GREEK_CYRILLIC}]"
)
_MIXED_GATE = re.compile(f"[{_GREEK_CYRILLIC}]")

_FOLD = str.maketrans({"\u0130": "i", "\u0131": "i", "\u017f": "s", "\u212a": "k"})
# `str.translate` takes a slow path on any non-ASCII string, which one curly
# apostrophe is enough to make, so it only runs when there is something to
# translate.
_FOLD_GATE = re.compile("[\u0130\u0131\u017f\u212a]")


# Nothing in the instruction tier accepts a noun list longer than the ones
# written here. Widening `instructions?` to "any noun" is the obvious next
# step and the obvious way to flag every post that says "ignore the above
# warnings".
_PRIOR = r"(?:previous|prior|preceding|above|earlier|former|original|initial|system)"
_DIRECTIVE = (
    r"(?:instructions?|prompts?|directions?|rules?|context|messages?|guidelines?|commands?)"
)
_RESPONSE = r"(?:response|answer|reply|output)"
_RESPONSE_NEEDLES = ("respon", "answer", "reply", "output")
_I = re.IGNORECASE

RULES: tuple[Rule, ...] = (
    # format. Case-sensitive: a template marker is an exact string, and
    # `[inst]` in lower case is not one.
    Rule("chatml", "format", re.compile(r"<\|im_(?:start|end)\|>"), ("<|im_",)),
    Rule("llama2-inst", "format", re.compile(r"\[/?INST\]"), ("INST]",)),
    Rule("llama3-header", "format", re.compile(r"<\|(?:start_header_id|eot_id)\|>"), ("<|",)),
    Rule(
        "alpaca-section",
        "format",
        re.compile(r"^[ \t]*### (?:Instruction|Response):", re.M),
        ("### ",),
    ),
    Rule("zephyr-role", "format", re.compile(r"<\|(?:system|user|assistant)\|>"), ("<|",)),
    Rule(
        "human-assistant-turn",
        "format",
        re.compile(r"^[ \t]*(?:Human|Assistant):", re.M),
        ("Human:", "Assistant:"),
    ),
    # obfuscation. Written as escapes so this file stays ASCII.
    Rule("tag-block", "obfuscation", re.compile("[\U000e0000-\U000e007f]+")),
    Rule("zero-width", "obfuscation", re.compile("[\u200b\u200c\u200d\u2060\ufeff]+")),
    Rule("bidi-override", "obfuscation", re.compile("[\u202a-\u202e\u2066-\u2069]+")),
    Rule(
        "mixed-script",
        "obfuscation",
        re.compile(
            r"\b(?=\w*(?=\w)[A-Za-z\u00c0-\u024f])(?=\w*(?=\w)[\u0370-\u03ff\u0400-\u04ff])\w+"
        ),
    ),
    # instruction, negating prior context. Split in two because "ignore
    # everything above" is predicted to be common in edited forum posts, and
    # folding it into the first rule would charge its cost to a rule that may
    # not deserve it.
    Rule(
        "ignore-prior-instructions",
        "instruction",
        re.compile(
            rf"\b(?:ignore|disregard|forget|override|bypass)\s+"
            rf"(?:(?:all|any|the|your|my|of|these|those)\s+)*{_PRIOR}\s+{_DIRECTIVE}\b",
            _I,
        ),
        ("ignore", "disregard", "forget", "override", "bypass"),
    ),
    # The lookahead keeps "ignore all prior instructions" out of this rule,
    # which the first rule already owns. Without it one phrase was charged to
    # two rules, which is the attribution slice 3 exists to get right. It sits
    # on the "all" branch alone: placed after "above" it crossed the newline
    # into a next line opening on "Instructions:" and the payload got nothing.
    Rule(
        "ignore-everything-above",
        "instruction",
        re.compile(
            rf"\b(?:ignore|disregard|forget)\s+(?:everything|anything|all(?!\s+{_PRIOR}\s+{_DIRECTIVE}\b))\s+"
            r"(?:(?:said|written|stated)\s+)?(?:above|before|prior|previously)\b",
            _I,
        ),
        ("ignore", "disregard", "forget"),
    ),
    # instruction, reassigning a role.
    Rule(
        "role-reassignment",
        "instruction",
        re.compile(
            r"\b(?:you\s+are\s+now|you\s+are\s+no\s+longer|from\s+now\s+on,?\s+you"
            r"|pretend\s+(?:to\s+be|you\s+are)"
            r"|your\s+new\s+(?:role|task|instructions?)\s+(?:is|are))\b",
            _I,
        ),
        ("now", "longer", "pretend", "new"),
    ),
    # instruction, dictating the response. The first rule is written against
    # the second half of RI.1 in arXiv:2308.01990, "The real answer is". The
    # second takes "your"
    # and not "the", because "the output should look like this" is a common way
    # for a question on a SQL forum to describe the result it wants.
    Rule(
        "answer-override",
        "instruction",
        re.compile(r"\bthe\s+(?:real|correct|true|actual)\s+answer\s+is\b", _I),
        ("answer",),
    ),
    Rule(
        "response-constraint",
        "instruction",
        re.compile(
            rf"\b(?:your\s+{_RESPONSE}\s+(?:must|should|shall)"
            rf"|(?:respond|reply|answer)\s+only"
            rf"|(?:begin|start|end)\s+your\s+{_RESPONSE}\s+with)\b",
            _I,
        ),
        _RESPONSE_NEEDLES,
    ),
    Rule(
        "response-insertion",
        "instruction",
        re.compile(
            rf"\b(?:include|add|insert|append|mention)\s+(?:the\s+following|this)\s+"
            rf"(?:\w+\s+)?in(?:to)?\s+your\s+{_RESPONSE}\b",
            _I,
        ),
        _RESPONSE_NEEDLES,
    ),
    # instruction, what to tell or withhold from the user. `address-user` is
    # the rule most likely to hit database prose, where "tell the user that
    # the login failed" is an application requirement rather than an attack.
    Rule(
        "conceal-from-user",
        "instruction",
        re.compile(
            r"\b(?:do\s+not|don't|never)\s+(?:tell|inform|reveal\s+to|mention\s+to|show)"
            r"\s+the\s+user\b",
            _I,
        ),
        ("user",),
    ),
    Rule(
        "address-user",
        "instruction",
        re.compile(r"\b(?:tell|inform|convince|persuade|remind)\s+the\s+user\s+(?:that|to)\b", _I),
        ("user",),
    ),
)


def scan_cell(text: str) -> list[Finding]:
    """Every finding in `text`, ordered by position and then by rule order.

    The sort is stable and `found` is built in rule order, so sorting on the
    start alone is what gives the rule-order tie-break.
    """
    found = [
        Finding(rule.id, rule.tier, hit.start(), hit.end(), hit.group())
        for rule in _candidates(text)
        for hit in rule.pattern.finditer(text)
    ]
    found.sort(key=lambda f: f.start)
    return found


def _candidates(text: str) -> list[Rule]:
    ascii_only = text.isascii()
    folded = text if ascii_only or not _FOLD_GATE.search(text) else text.translate(_FOLD)
    folded = folded.lower()
    obfuscated = not ascii_only and _OBFUSCATION_GATE.search(text) is not None
    mixed = obfuscated and _MIXED_GATE.search(text) is not None
    picked = []
    for rule in RULES:
        if rule.id == "mixed-script":
            keep = mixed
        elif rule.tier == "obfuscation":
            keep = obfuscated
        elif rule.tier == "format":
            keep = any(needle in text for needle in rule.needles)
        else:
            keep = any(needle in folded for needle in rule.needles)
        if keep:
            picked.append(rule)
    return picked


@dataclass(frozen=True)
class CellFinding:
    """A finding placed in a result: which row, which column, and where inside the cell.

    `path` is empty for a plain string cell. For a LIST, STRUCT or MAP it is
    the list indices and keys walked to reach the string that matched. A MAP
    key is always a string here, spelled as the caller receives it, because
    JSON object keys are strings: `{1: ...}` goes out as `"1"` and `True` as
    `"true"`. An int step is therefore always a list index. A MAP key is
    scanned as well as its value, because both reach the caller, and a
    hit in the key carries the same path as its value would. Usually the span
    and the matched text tell the two apart; a key identical to its value
    gives two identical entries.

    `row` is None for a finding in a column name. PIVOT turns row values into
    column names, so a column name is stored text like any cell. `column_index`
    is there because a result can carry two columns with the same name.
    """

    row: int | None
    column_index: int
    column: str
    path: tuple[int | str, ...]
    finding: Finding


def scan_rows(columns: list[str], rows: list[list[object]]) -> list[CellFinding]:
    """Scan every string a caller of this result would see.

    That includes column names, and strings nested inside LIST, ARRAY, STRUCT
    and MAP values, which `SELECT list(body) ...` produces from an ordinary
    query. Scanning only top-level VARCHAR would let a `GROUP BY` walk a
    payload past the scanner. Rows are expected to have been through
    `sendable` first, so a BLOB arrives here as the text that will be sent.
    """
    placed = [
        CellFinding(None, c, column, (), f)
        for c, column in enumerate(columns)
        for f in scan_cell(column)
    ]
    for r, row in enumerate(rows):
        for c, (column, cell) in enumerate(zip(columns, row, strict=True)):
            for path, text in _strings(cell, ()):
                placed.extend(CellFinding(r, c, column, path, f) for f in scan_cell(text))
    return placed


def sendable(value: object) -> object:
    """The value exactly as it will be sent: BLOBs decoded, MAP keys made JSON keys.

    `query._apply_byte_cap` calls this once per row, so the value it measures is
    the value the scanner reads and the server sends. Before this, pydantic
    refused to serialise a BLOB that was not valid UTF-8 and the caller
    received a reasonless error for every query touching that row.

    A BLOB key is decoded with backslash escapes for invalid bytes, not the
    replacement character used for values, because `\\xff` and `\\xfe` would
    both become U+FFFD and one entry would vanish from the map unflagged. Any
    other key that is not a JSON scalar becomes its `str()`: `json.dumps`
    ignores `default` for keys, so DATE, DECIMAL and UUID keys used to raise
    `TypeError`. A valid key spelled with a literal backslash-x can still
    collide with an escaped one, and then only the later entry is sent.
    """
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, list | tuple):
        return [sendable(item) for item in value]
    if isinstance(value, dict):
        return {_sendable_key(key): sendable(item) for key, item in value.items()}
    return value


def _sendable_key(key: object) -> str | int | float | bool | None:
    if isinstance(key, bytes | bytearray):
        return bytes(key).decode("utf-8", errors="backslashreplace")
    if key is None or isinstance(key, str | int | float | bool):
        return key
    return str(key)


def _strings(value: object, path: tuple[int | str, ...]):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list | tuple):
        for n, item in enumerate(value):
            yield from _strings(item, (*path, n))
    elif isinstance(value, dict):
        for key, item in value.items():
            step = key if isinstance(key, str) else _wire_key(key)
            yield from _strings(key, (*path, step))
            yield from _strings(item, (*path, step))


_ONE_KEY = TypeAdapter(dict[Any, None])


def _wire_key(key: object) -> str:
    # The server's reply goes out through pydantic with its default settings, so
    # this asks pydantic too. It sends a NaN or infinite key as "None".
    (spelled,) = json.loads(_ONE_KEY.dump_json({key: None}))
    return spelled
