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
Instead every source in a FROM clause has to be one of three things: a bare
identifier that names something in the catalog, a name bound by a CTE in the
same statement, or a subquery. Every table function fails that by construction,
including the ones nobody has thought of yet, and including harmless ones like
`range(10)`. That cost is real and it buys a rule that does not age.

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

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# WITH is not here because sqlglot hangs a CTE off the SELECT that follows it,
# so `WITH a AS (...) SELECT * FROM a` arrives as a Select and never as a node
# of its own.
ALLOWED_TOP_LEVEL = (exp.Select, exp.SetOperation)

# Read a file into a column, from a projection where there is no FROM source to
# resolve against the catalog. Both are table functions in DuckDB today, so the
# engine already rejects this use, and both are listed because "the binder
# happens to stop it" is not a rule anyone wrote down.
DENIED_SCALARS = frozenset({"read_text", "read_blob"})


@dataclass(frozen=True)
class Refused(Exception):
    """Why one statement is not going to run. `rule` is stable, `detail` is not."""

    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


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
        if not isinstance(source, exp.Identifier):
            raise Refused(
                "table-source",
                f"{_source_name(source)} is a table function, not a table",
            )
        name = source.name.lower()
        if name in bound:
            continue
        if known_tables is not None and name not in known_tables:
            raise Refused("unknown-table", f"no table or view named {source.name!r}")


def _check_scalars(statement: exp.Expression) -> None:
    for call in statement.find_all(exp.Anonymous):
        if call.name.lower() in DENIED_SCALARS:
            raise Refused("denied-function", f"{call.name} reads the filesystem")


def _source_name(source: exp.Expression | None) -> str:
    """The function's name, which sqlglot puts in two different places.

    A function it does not model arrives as `Anonymous` carrying its name.
    One it does model becomes its own node type, so `read_csv` is a `ReadCSV`
    whose name attribute holds the path argument instead. `sql_name()` gives
    READ_CSV for those, and lower-casing it back to what the caller typed is
    the difference between a message they can act on and one about sqlglot.
    """
    if source is None:
        return "an empty source"
    if isinstance(source, exp.Anonymous):
        return source.name
    sql_name = getattr(source, "sql_name", None)
    if callable(sql_name):
        return str(sql_name()).lower()
    return type(source).__name__.lower()


def _first_line(exc: ParseError) -> str:
    return str(exc).splitlines()[0].strip()
