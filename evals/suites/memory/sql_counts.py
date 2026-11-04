"""Recognize canonical count predicates in saved SQL tool traces."""

from datetime import UTC
from typing import Any

from pglast import ast, enums, parse_sql
from pglast.parser import ParseError

from .dataset import instant


def _normalize(value: Any, alias: str | None) -> None:
    if isinstance(value, (tuple, list)):
        for item in value:
            _normalize(item, alias)
    elif isinstance(value, ast.Node):
        for field in value:
            _normalize(getattr(value, field), alias)
        if isinstance(value, ast.ColumnRef):
            fields = value.fields or ()
            if (
                len(fields) == 2
                and isinstance(fields[0], ast.String)
                and fields[0].sval in {alias, "messages"}
            ):
                value.fields = fields[1:]
        elif isinstance(value, ast.FuncCall):
            names = value.funcname or ()
            if (
                len(names) == 2
                and isinstance(names[0], ast.String)
                and names[0].sval == "pg_catalog"
            ):
                value.funcname = names[1:]
        elif (
            isinstance(value, ast.TypeCast)
            and isinstance(value.arg, ast.A_Const)
            and isinstance(value.arg.val, ast.String)
        ):
            names = (value.typeName.names or ()) if value.typeName is not None else ()
            if names and names[-1].sval == "timestamptz":
                value.arg.val.sval = (
                    instant(value.arg.val.sval or "").astimezone(UTC).isoformat()
                )


def _conditions(value: ast.Node | None) -> list[ast.Node]:
    if isinstance(value, ast.BoolExpr) and value.boolop == enums.BoolExprType.AND_EXPR:
        return [item for child in value.args or () for item in _conditions(child)]
    return [value] if value is not None else []


def matches_count_sql(query: str, aggregation: dict[str, Any]) -> bool:
    literal = str(aggregation["text_contains_literal"] or "")
    pattern = (
        "%" + literal.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    )
    pattern = pattern.replace("'", "''")
    author = int(aggregation["author_id"])
    after = instant(aggregation["after_inclusive"]).isoformat()
    before = instant(aggregation["before_exclusive"]).isoformat()
    expected_sql = (
        f"SELECT count(*) FROM messages WHERE author_id = {author} "  # ruff: ignore[hardcoded-sql-expression] -- Parse-only trusted gold SQL.
        f"AND sent_at >= TIMESTAMPTZ '{after}' AND sent_at < TIMESTAMPTZ '{before}' "
        f"AND text ILIKE '{pattern}'"
    )
    try:
        return _same_predicate(query, expected_sql)
    except ParseError, TypeError, ValueError, RecursionError:
        return False


def _same_predicate(query: str, expected_sql: str) -> bool:
    statements = parse_sql(query)
    if len(statements) != 1 or not isinstance(statements[0].stmt, ast.SelectStmt):
        return False
    actual = statements[0].stmt
    expected = parse_sql(expected_sql)[0].stmt
    assert isinstance(expected, ast.SelectStmt)
    relations = actual.fromClause or ()
    if len(relations) != 1 or not isinstance(relations[0], ast.RangeVar):
        return False
    relation = relations[0]
    if relation.schemaname not in {None, "pg_temp"}:
        return False
    alias = relation.alias.aliasname if relation.alias is not None else None
    relation.alias = None
    relation.schemaname = None
    for target in actual.targetList or ():
        target.name = None
    _normalize(actual, alias)
    _normalize(expected, None)
    remaining = _conditions(expected.whereClause)
    for condition in _conditions(actual.whereClause):
        match = next(
            (index for index, wanted in enumerate(remaining) if condition == wanted),
            None,
        )
        if match is None:
            return False
        remaining.pop(match)
    if remaining:
        return False
    actual.whereClause = expected.whereClause
    return bool(actual == expected)
