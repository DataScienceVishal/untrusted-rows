"""Decide whether one statement is allowed to reach the database.

The rule is a positive allow-list over parsed node types, not a pattern over
the string. `crystaldba/postgres-mcp` states the case for that better than a
paraphrase would: it parses the SQL before execution with `pglast` rather than
matching on text. `Shimiyahya/mcp-data-warehouse-server` applies the same idea
to DuckDB through sqlglot, which is the shape this file copies.

Two things here are not in either of those, and both came out of running the
parser rather than reading about it.

The first is the FROM clause. Refusing `read_csv` by name is a deny-list, and a
deny-list over DuckDB's table functions is a promise to track every release.
Instead every table reference in a FROM clause has to be one of two things: an
identifier naming something in the catalog, or a name bound by a CTE in the
same statement. Any function call there fails by construction, including the
ones nobody has thought of yet, and including harmless ones like `range(10)`.
That cost is real and it buys a rule that does not age.

Two FROM sources are not table references at all and this rule never sees them.
sqlglot models `(SELECT ...)` as a `Subquery` and `UNNEST([1,2,3])` as an
`Unnest`, so neither arrives as a `Table`. Both are allowed and neither reads
anything outside the statement. An earlier version of this docstring claimed
every table function fails by construction, and `SELECT * FROM UNNEST([1,2,3])`
is the counterexample that made the claim false.

The second is `SELECT * FROM '/etc/passwd'`. DuckDB's replacement scan reads a
bare string in a FROM clause as a file path with no function call anywhere in
it, and sqlglot parses it to `Identifier(quoted=True)`, which is the same node
`"posts"` produces. Node type cannot separate those two, so the catalog lookup
is what separates them: `/etc/passwd` is not a table and `posts` is.

A scalar deny-list survives for `read_text` and `read_blob`, which appear in a
projection where there is no FROM source to check. It is named as a deny-list
in `DENIED_SCALARS` rather than dressed up as anything sturdier.

None of this is the only layer. `connection.py` opens the database read-only
with external access off and the configuration locked, and that refuses every
statement in `tests/fixtures/refusals.json` on its own. The gate is here so a
refusal is a sentence naming the rule instead of a DuckDB exception, and
because which statements the engine rejects is a property of a DuckDB version
rather than a promise to the caller.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# WITH is not here because sqlglot hangs a CTE off the SELECT that follows it,
# so `WITH a AS (...) SELECT * FROM a` arrives as a Select and never as a node
# of its own.
ALLOWED_TOP_LEVEL = (exp.Select, exp.SetOperation)

# Functions that read something outside the statement. The FROM-clause rule
# already stops every one of these there, so this set is for the projection,
# where there is no table reference to resolve against the catalog.
#
# It held two names until a review, on the argument that "the binder happens to
# stop it" is not a rule anyone wrote down. That argument covers the whole
# family, and the file was applying it to two members of it: `read_csv`,
# `read_csv_auto`, `read_parquet`, `read_json` and `glob` all returned clean
# from `check` and were stopped only by DuckDB's binder complaining that a
# table function was used as a scalar.
#
# `current_setting` is here for a different reason. It is an ordinary scalar
# function that the engine is happy to run, and `current_setting('temp_directory')`
# returns an absolute path on the server's filesystem. The gate refuses
# `PRAGMA database_list` for exactly that, and allowing the SELECT that yields
# the same string would make the PRAGMA rule decorative.
FILE_READING_FUNCTIONS = frozenset({
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan",
    "read_json", "read_json_auto", "read_json_objects", "read_ndjson",
    "read_ndjson_auto", "read_ndjson_objects",
    "read_text", "read_blob", "glob", "sniff_csv",
    "current_setting",
})


class Refused(Exception):
    """Why one statement is not going to run. `rule` is stable, `detail` is not.

    A plain exception rather than a frozen dataclass. The dataclass version
    could not be copied or pickled, because both go through `__setattr__` on
    a frozen instance, which is a trap for any later path that moves a refusal
    across a process boundary.
    """

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail

    def __reduce__(self):
        # `Exception.__reduce__` hands back `self.args`, which is the single
        # formatted string, and `__init__` needs two. Without this, copy and
        # pickle both raise TypeError on the way back in.
        return (type(self), (self.rule, self.detail))


def check(sql: str, known_tables: frozenset[str] | None = None) -> exp.Expression:
    """Return the parsed statement, or raise `Refused`.

    `known_tables` holds lower-cased catalog names. Passing None skips the
    catalog rule, which is what the parser-only tests do. The server always
    passes the real catalog, because without it the replacement-scan path stays
    open.
    """
    if not sql.strip():
        raise Refused("empty", "no statement")

    try:
        parsed = sqlglot.parse(sql, dialect="duckdb")
    except ParseError as exc:
        raise Refused("unparseable", _first_line(exc)) from exc
    except RecursionError as exc:
        # sqlglot's parser recurses once per level of nesting. Run from a plain
        # script, 40 nested function calls were enough to hit Python's limit,
        # and a caller whose stack is already deep gets less than that. A
        # bracket count would not match that limit, so this catches the error
        # itself.
        raise Refused("too-deep", "nested deeper than the parser can follow") from exc

    statements = [node for node in parsed if node is not None]
    if not statements:
        raise Refused("empty", "no statement")
    if len(statements) > 1:
        kinds = ", ".join(type(node).__name__.upper() for node in statements)
        raise Refused("multiple-statements", f"{len(statements)} statements: {kinds}")

    statement = statements[0]
    if not isinstance(statement, ALLOWED_TOP_LEVEL):
        raise Refused("statement-type", f"{type(statement).__name__.upper()} is not a read")

    _check_sources(statement, known_tables)
    _check_scalars(statement)
    return statement


def _check_sources(statement: exp.Expression, known_tables: frozenset[str] | None) -> None:
    bound = {
        cte.alias_or_name.lower()
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }
    for table in statement.find_all(exp.Table):
        source = table.this
        if isinstance(source, exp.Func):
            raise Refused(
                "table-source", f"{_function_name(source)} is a function, not a table"
            )
        if not isinstance(source, exp.Identifier):
            raise Refused(
                "table-source",
                f"a {type(source).__name__.lower()} is not a table name",
            )
        if table.catalog:
            # ATTACH is refused, so there is exactly one catalog and naming it
            # buys the caller nothing while adding a resolution path the
            # catalog lookup below would have to model.
            raise Refused("table-source", f"remove the catalog qualifier {table.catalog!r}")
        name = source.name.lower()
        if not table.db and name in bound:
            continue
        qualified = f"{table.db.lower()}.{name}" if table.db else name
        if known_tables is not None and qualified not in known_tables:
            raise Refused("unknown-table", f"no table or view named {qualified!r}")


def _check_scalars(statement: exp.Expression) -> None:
    """Functions outside the FROM clause, where there is no table reference to resolve."""
    for call in statement.find_all(exp.Func):
        if _function_name(call) in FILE_READING_FUNCTIONS:
            raise Refused("denied-function", f"{_function_name(call)} reads outside the statement")


def _function_name(call: exp.Expression) -> str:
    """The name the caller typed, which sqlglot puts in two different places.

    A function it does not model arrives as `Anonymous` carrying its name. One
    it does model becomes its own node type, so `read_csv` is a `ReadCSV` whose
    name attribute holds the path argument instead. `sql_name()` gives READ_CSV
    for those, and lower-casing it back to what the caller typed is the
    difference between a message they can act on and one about sqlglot.
    """
    if isinstance(call, exp.Anonymous):
        return call.name.lower()
    sql_name = getattr(call, "sql_name", None)
    if callable(sql_name):
        return str(sql_name()).lower()
    return type(call).__name__.lower()


def _first_line(exc: ParseError) -> str:
    return str(exc).splitlines()[0].strip()
