"""Condition C7, plus the rules that the fixture exercises only in passing."""

from __future__ import annotations

import pytest

from untrusted_rows.sql_gate import Refused, check


def test_every_committed_refusal_is_refused(refusal_cases, catalog):
    """C7. The fixture was committed in 2b2901f, one commit before this gate existed.

    Reported as one list rather than one assertion per case, because the first
    question after a failure is whether one statement got through or the gate
    fell over, and a per-case parametrise answers that one statement at a time.
    """
    leaked = []
    for case in refusal_cases["must_refuse"]:
        try:
            check(case["sql"], catalog)
        except Refused:
            continue
        leaked.append(case["sql"])
    assert leaked == []


def test_every_committed_allowance_is_allowed(refusal_cases, catalog):
    """The counterweight. Without this, refusing everything passes C7."""
    wrongly_refused = []
    for case in refusal_cases["must_allow"]:
        try:
            check(case["sql"], catalog)
        except Refused as refusal:
            wrongly_refused.append((case["sql"], refusal.rule))
    assert wrongly_refused == []


def test_every_refusal_found_in_review_is_refused(refusal_cases, catalog):
    """The second list, added after the gate existed and honest about that.

    Each of these got through the first version, apart from the nested ones at
    the end, which crashed it with RecursionError. They are separate from the C7
    list so that the pre-registration claim on it stays exactly true.
    """
    leaked = []
    for case in refusal_cases["must_refuse_found_in_review"]:
        try:
            check(case["sql"], catalog)
        except Refused:
            continue
        leaked.append(case["sql"])
    assert leaked == []


def test_every_allowance_found_in_review_is_allowed(refusal_cases, catalog):
    wrongly_refused = []
    for case in refusal_cases["must_allow_found_in_review"]:
        try:
            check(case["sql"], catalog)
        except Refused as refusal:
            wrongly_refused.append((case["sql"], refusal.rule))
    assert wrongly_refused == []


def test_the_fixture_has_both_halves(refusal_cases):
    """A fixture emptied of its allow list would make the test above vacuous."""
    assert len(refusal_cases["must_refuse"]) >= 30
    assert len(refusal_cases["must_allow"]) >= 10


@pytest.mark.parametrize(
    ("sql", "rule"),
    [
        ("", "empty"),
        ("   \n  ", "empty"),
        ("SELEKT 1", "unparseable"),
        ("SELECT 1; SELECT 2", "multiple-statements"),
        ("DROP TABLE posts", "statement-type"),
        ("SELECT * FROM glob('/x')", "table-source"),
        ("SELECT read_text('/x')", "denied-function"),
        ("SELECT * FROM nope", "unknown-table"),
        ("SELECT " + "abs(" * 100 + "1" + ")" * 100, "too-deep"),
    ],
)
def test_the_rule_id_says_which_rule_fired(sql, rule, catalog):
    """The ids are the part of a refusal a caller can branch on, so they are pinned."""
    with pytest.raises(Refused) as caught:
        check(sql, catalog)
    assert caught.value.rule == rule


def test_a_cte_name_resolves_against_the_statement_not_the_catalog(catalog):
    check("WITH t AS (SELECT 1 AS a) SELECT a FROM t", catalog)


def test_a_cte_does_not_launder_a_table_function(catalog):
    with pytest.raises(Refused) as caught:
        check("WITH t AS (SELECT * FROM read_csv('/x')) SELECT * FROM t", catalog)
    assert caught.value.rule == "table-source"


def test_without_a_catalog_the_name_rule_is_off(catalog):
    """Parser-only mode, which is what leaves the replacement-scan path open.

    Kept as a test rather than a comment because it is the one configuration in
    which `SELECT * FROM '/etc/passwd'` gets past the gate, and the server never
    uses it. If the default ever flips to None, this test is what says so.
    """
    check("SELECT * FROM '/etc/passwd'")
    with pytest.raises(Refused):
        check("SELECT * FROM '/etc/passwd'", catalog)


def test_a_set_operation_is_a_read(catalog):
    check("SELECT id FROM posts UNION SELECT id FROM users", catalog)
    check("SELECT id FROM posts EXCEPT SELECT id FROM users", catalog)
    check("SELECT id FROM posts INTERSECT SELECT id FROM users", catalog)


def test_table_names_are_matched_without_regard_to_case(catalog):
    check("SELECT * FROM POSTS", catalog)
    check('SELECT * FROM "Posts"', catalog)


def test_a_schema_qualifier_is_resolved_as_a_schema_qualifier(catalog):
    """`main.posts` resolves and `sales.posts` does not, on the same table name."""
    check("SELECT * FROM main.posts", catalog)
    with pytest.raises(Refused) as caught:
        check("SELECT * FROM sales.posts", catalog)
    assert caught.value.rule == "unknown-table"


def test_a_cte_name_does_not_resolve_when_it_carries_a_schema(catalog):
    """`main.t` is a table reference even where a CTE named `t` exists."""
    check("WITH t AS (SELECT 1 AS a) SELECT a FROM t", catalog)
    with pytest.raises(Refused):
        check("WITH t AS (SELECT 1 AS a) SELECT a FROM main.t", catalog)


def test_a_refusal_can_be_pickled(catalog):
    """The frozen dataclass version could not, which is a trap for any later
    path that moves a refusal between processes."""
    import copy
    import pickle

    with pytest.raises(Refused) as caught:
        check("DROP TABLE posts", catalog)
    assert copy.copy(caught.value).rule == "statement-type"
    assert pickle.loads(pickle.dumps(caught.value)).rule == "statement-type"


def test_the_message_for_a_non_function_source_does_not_call_it_a_function(catalog):
    """It said "is a table function" for every source that was not an identifier."""
    with pytest.raises(Refused) as caught:
        check("SELECT * FROM a.b.c.d", catalog)
    assert "function" not in caught.value.detail
