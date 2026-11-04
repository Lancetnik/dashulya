"""Bounded SQL over a transaction-local, authorized view of canonical messages."""

import asyncio
import json
import logging
from datetime import date, datetime
from decimal import Decimal
from time import perf_counter
from typing import Any
from uuid import uuid4

from pglast import ast, parse_sql
from pglast.parser import ParseError
from psycopg import AsyncConnection, AsyncServerCursor, Error, sql
from psycopg.rows import dict_row

from dashulya.database import Database
from dashulya.database.migrate import HISTORY_ROLE
from dashulya.domain.memory import MemoryAccess, MemoryInputError

logger = logging.getLogger(__name__)

MAX_QUERIES = 5
MAX_ROWS = 50
MAX_CELL_CHARS = 12_000
MAX_RESULT_CHARS = 48_000

_NODES = frozenset([
    "SelectStmt",
    "ResTarget",
    "ColumnRef",
    "A_Star",
    "RangeVar",
    "JoinExpr",
    "Alias",
    "RangeSubselect",
    "WithClause",
    "CommonTableExpr",
    "A_Const",
    "String",
    "Integer",
    "Float",
    "Boolean",
    "BitString",
    "A_Expr",
    "BoolExpr",
    "NullTest",
    "BooleanTest",
    "FuncCall",
    "TypeCast",
    "TypeName",
    "SortBy",
    "SubLink",
    "CaseExpr",
    "CaseWhen",
    "CoalesceExpr",
    "MinMaxExpr",
    "A_ArrayExpr",
    "A_Indirection",
    "A_Indices",
    "RowExpr",
    "WindowDef",
    "GroupingSet",
    "SQLValueFunction",
    "NamedArgExpr",
    "RangeFunction",
])
_FUNCTIONS = frozenset([
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "bool_and",
    "bool_or",
    "every",
    "string_agg",
    "array_agg",
    "json_agg",
    "jsonb_agg",
    "json_build_object",
    "jsonb_build_object",
    "json_object_agg",
    "jsonb_object_agg",
    "jsonb_array_length",
    "jsonb_typeof",
    "jsonb_extract_path_text",
    "lower",
    "upper",
    "length",
    "char_length",
    "octet_length",
    "substr",
    "substring",
    "btrim",
    "ltrim",
    "rtrim",
    "trim",
    "replace",
    "regexp_replace",
    "regexp_match",
    "regexp_matches",
    "split_part",
    "strpos",
    "position",
    "starts_with",
    "left",
    "right",
    "concat",
    "concat_ws",
    "date_trunc",
    "date_part",
    "extract",
    "timezone",
    "to_char",
    "make_interval",
    "make_date",
    "make_timestamp",
    "make_timestamptz",
    "age",
    "now",
    "statement_timestamp",
    "abs",
    "round",
    "ceil",
    "ceiling",
    "floor",
    "sqrt",
    "power",
    "mod",
    "sign",
    "row_number",
    "rank",
    "dense_rank",
    "lag",
    "lead",
    "first_value",
    "last_value",
    "ntile",
    "unnest",
    "generate_series",
])
_TYPES = frozenset([
    "bool",
    "boolean",
    "int2",
    "int4",
    "int8",
    "smallint",
    "integer",
    "bigint",
    "numeric",
    "decimal",
    "float4",
    "float8",
    "real",
    "double",
    "text",
    "varchar",
    "bpchar",
    "char",
    "date",
    "timestamp",
    "timestamptz",
    "time",
    "timetz",
    "interval",
    "json",
    "jsonb",
    "uuid",
    "bytea",
])
_OPERATORS = frozenset([
    "=",
    "<>",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    "+",
    "-",
    "*",
    "/",
    "%",
    "^",
    "||",
    "~~",
    "~~*",
    "!~~",
    "!~~*",
    "~",
    "~*",
    "!~",
    "!~*",
    "&",
    "|",
    "#",
    "<<",
    ">>",
    "@>",
    "<@",
    "&&",
    "->",
    "->>",
    "#>",
    "#>>",
    "?",
    "?|",
    "?&",
])


def _nodes(value: Any) -> list[tuple[str, dict[str, Any]]]:
    found = []
    if isinstance(value, dict):
        if "@" in value:
            found.append((value["@"], value))
        for child in value.values():
            found.extend(_nodes(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            found.extend(_nodes(child))
    return found


def _allowed_relation(node: dict[str, Any], ctes: set[str]) -> bool:
    schema = node.get("schemaname")
    relation = node["relname"]
    if node.get("catalogname") or schema not in {None, "pg_temp"}:
        return False
    return relation == "messages" or (schema is None and relation in ctes)


def _names(value: list[dict[str, Any]]) -> list[str]:
    return [item["sval"] for item in value]


def validate_query(query: str) -> str:
    if not query.strip() or len(query) > 16_000:
        msg = "Supply one SQL query of 1-16000 characters."
        raise MemoryInputError(msg)
    try:
        parsed = parse_sql(query)
    except ParseError, RecursionError, ValueError:
        msg = "Invalid PostgreSQL query syntax."
        raise MemoryInputError(msg) from None
    statement = parsed[0].stmt if len(parsed) == 1 else None
    if not isinstance(statement, ast.SelectStmt):
        msg = "Only one SELECT or WITH ... SELECT query is allowed."
        raise MemoryInputError(msg)
    try:
        nodes = _nodes(statement())
    except RecursionError:
        msg = "The SQL query is nested too deeply."
        raise MemoryInputError(msg) from None
    ctes = {node["ctename"] for kind, node in nodes if kind == "CommonTableExpr"}
    for kind, node in nodes:
        if kind not in _NODES:
            raise MemoryInputError("This SQL construct is not allowed: " + kind + ".")
        if kind == "SelectStmt" and (node.get("intoClause") or node.get("lockingClause")):
            msg = "SELECT INTO and row locks are not allowed."
            raise MemoryInputError(msg)
        if kind == "RangeVar" and not _allowed_relation(node, ctes):
            msg = "Read only messages and query-local CTEs; other relations are unavailable."
            raise MemoryInputError(msg)
        if kind in {"FuncCall", "TypeName"}:
            names = _names(node["funcname" if kind == "FuncCall" else "names"])
            allowed = _FUNCTIONS if kind == "FuncCall" else _TYPES
            if (
                len(names) > 2
                or (len(names) == 2 and names[0] != "pg_catalog")
                or names[-1] not in allowed
            ):
                msg = "This SQL function or type is not allowed. Use standard text, date and aggregate functions."
                raise MemoryInputError(msg)
        if kind == "A_Expr":
            names = _names(node.get("name", []))
            between = node.get("kind", {}).get("name") in {
                "AEXPR_BETWEEN",
                "AEXPR_NOT_BETWEEN",
                "AEXPR_BETWEEN_SYM",
                "AEXPR_NOT_BETWEEN_SYM",
            }
            if not between and (len(names) != 1 or names[0] not in _OPERATORS):
                msg = "Only built-in SQL operators are allowed."
                raise MemoryInputError(msg)
    return query


def _value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {key: _value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    return value


async def _rows(
    cursor: AsyncServerCursor[dict[str, Any]], max_chars: int
) -> dict[str, Any]:
    columns = [column.name for column in cursor.description or ()]
    if len(columns) > 32 or len(columns) != len(set(columns)):
        message = (
            "Return at most 32 columns with unique names; use aliases for joined columns."
        )
        raise MemoryInputError(message)
    rows: list[dict[str, Any]] = []
    truncated = False
    remaining = max_chars
    for _ in range(MAX_ROWS + 1):
        row = await cursor.fetchone()
        if row is None:
            break
        if len(rows) == MAX_ROWS:
            truncated = True
            break
        bounded = {}
        for key, item in row.items():
            value = _value(item)
            encoded = (
                value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            )
            if len(encoded) > MAX_CELL_CHARS:
                value = encoded[:MAX_CELL_CHARS]
                truncated = True
            bounded[key] = value
        size = len(json.dumps(bounded, ensure_ascii=False))
        if size > remaining:
            truncated = True
            break
        rows.append(bounded)
        remaining -= size
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "coverage": {
            "complete": False,
            "notice": "Only authorized saved originals through the request cutoff are available; missing history is not fetched from Telegram.",
        },
    }


async def _select(
    connection: AsyncConnection[dict[str, Any]],
    query: str,
    *,
    deadline: float,
    max_chars: int,
) -> dict[str, Any]:
    query = validate_query(query)
    remaining = deadline - perf_counter()
    if remaining <= 0:
        message = "The batch deadline was reached; this query was not executed. Retry it separately."
        raise MemoryInputError(message)
    if max_chars <= 0:
        message = "The batch result limit was reached; this query was not executed. Retry it with a smaller selection."
        raise MemoryInputError(message)
    async with (
        asyncio.timeout(min(12, remaining)),
        connection.transaction(),
        connection.cursor(name="history_result", row_factory=dict_row) as cursor,
    ):
        await cursor.execute(query)
        return await _rows(cursor, max_chars)


async def _one(
    connection: AsyncConnection[dict[str, Any]],
    query: str,
    *,
    deadline: float,
    max_chars: int,
    fields: dict[str, Any],
) -> dict[str, Any]:
    fields = {**fields, "sql": query[:16_000], "sql_truncated": len(query) > 16_000}
    started = perf_counter()
    logger.info("History SQL query started", extra=fields)
    try:
        result = await _select(connection, query, deadline=deadline, max_chars=max_chars)
    except asyncio.CancelledError:
        logger.info(
            "History SQL query completed",
            extra={
                **fields,
                "outcome": "cancelled",
                "duration_ms": round((perf_counter() - started) * 1000),
            },
        )
        raise
    except (MemoryInputError, TimeoutError, Error) as error:
        reason = _failure_reason(error)
        logger.warning(
            "History SQL query completed",
            extra={
                **fields,
                "outcome": "error",
                "duration_ms": round((perf_counter() - started) * 1000),
                "error_type": type(error).__name__,
                "error_message": reason,
            },
        )
        return {"error": reason}
    logger.info(
        "History SQL query completed",
        extra={
            **fields,
            "outcome": "ok",
            "duration_ms": round((perf_counter() - started) * 1000),
            "row_count": result["row_count"],
            "truncated": result["truncated"],
        },
    )
    return result


class SqlHistory:
    def __init__(self, database: Database, access: MemoryAccess) -> None:
        self._database = database
        self._access = access

    def _condition(self) -> sql.Composable:
        access = self._access
        condition = (
            sql.SQL("TRUE")
            if access.is_private
            else sql.SQL("chat_id = {}").format(sql.Literal(access.chat_id))
        )
        if access.before is not None:
            condition += sql.SQL(
                " AND (edited_at IS NULL OR edited_at < {cutoff}) AND "
                "CASE WHEN {message}::bigint IS NULL OR chat_id IS DISTINCT FROM {chat} "
                "THEN sent_at < {cutoff} ELSE (sent_at,message_id) < ({cutoff},{message}) END"
            ).format(
                cutoff=sql.Literal(access.before),
                chat=sql.Literal(access.chat_id),
                message=sql.Literal(access.before_id),
            )
        return condition

    async def _prepare(self, connection: AsyncConnection[dict[str, Any]]) -> None:
        cursor = await connection.execute(
            "SELECT r.rolsuper OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication OR r.rolbypassrls "
            "OR EXISTS (SELECT 1 FROM pg_auth_members WHERE member=r.oid) "
            "OR has_table_privilege(r.oid,'messages','SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') "
            "OR has_function_privilege(r.oid,'pg_catalog.set_config(text,text,boolean)','EXECUTE') AS unsafe "
            "FROM pg_roles r WHERE r.rolname=%s",
            (HISTORY_ROLE,),
        )
        role = await cursor.fetchone()
        if role is None or role["unsafe"]:
            message = "The history database role is missing or overprivileged; apply the database role migration."
            raise RuntimeError(message)
        await connection.execute("SET LOCAL statement_timeout = '10s'")
        await connection.execute("SET LOCAL lock_timeout = '1s'")
        await connection.execute("SET LOCAL work_mem = '4MB'")
        await connection.execute(
            sql.SQL(
                "CREATE TEMP VIEW messages WITH (security_barrier=true) AS SELECT "
                "'source:' || left(encode(sha256(convert_to(collection,'UTF8')),'hex'),24) || ':' || message_id AS source_id, "
                "collection AS chat, chat_id, message_id, body AS text, sent_at, "
                "author_id, author, author_username AS username, is_bot, chat_title, chat_username, "
                "topic_id, reply_to_id AS reply_to_message_id, edited_at, media_type, url, provenance, archive_key "
                "FROM messages WHERE {}"
            ).format(self._condition())
        )
        await connection.execute(
            sql.SQL("GRANT SELECT ON pg_temp.messages TO {}").format(
                sql.Identifier(HISTORY_ROLE)
            )
        )
        await connection.execute(
            sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(HISTORY_ROLE))
        )
        await connection.execute("SET LOCAL search_path = pg_temp, pg_catalog")
        await connection.execute("SET TRANSACTION READ ONLY")

    async def _execute(self, queries: list[str]) -> list[dict[str, Any]]:
        results = []
        remaining = MAX_RESULT_CHARS
        deadline = perf_counter() + 30
        fields = {
            "chat_id": self._access.chat_id,
            "message_id": self._access.before_id,
            "tool_name": "chat_history",
            "batch_id": uuid4().hex,
            "query_count": len(queries),
        }
        async with (
            self._database.connection() as connection,
            connection.transaction(force_rollback=True),
        ):
            async with asyncio.timeout(12):
                await connection.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"
                )
                await self._prepare(connection)
            for index, query in enumerate(queries, 1):
                result = await _one(
                    connection,
                    query,
                    deadline=deadline,
                    max_chars=remaining,
                    fields={**fields, "query_index": index},
                )
                remaining -= sum(
                    len(json.dumps(row, ensure_ascii=False))
                    for row in result.get("rows", [])
                )
                results.append(result)
        return results

    async def query(self, queries: list[str]) -> dict[str, Any]:
        if not 1 <= len(queries) <= MAX_QUERIES:
            message = "Supply a batch of 1-5 SQL query strings."
            raise MemoryInputError(message)
        try:
            results = await self._execute(queries)
        except (TimeoutError, Error) as error:
            raise MemoryInputError(_failure_reason(error)) from None
        return {
            "query_count": len(queries),
            "results": [
                {
                    "query_index": index,
                    "outcome": "error" if "error" in result else "ok",
                    **result,
                }
                for index, result in enumerate(results, 1)
            ],
        }


def _failure_reason(error: Exception) -> str:
    if isinstance(error, MemoryInputError):
        return str(error)
    code = getattr(error, "sqlstate", "") or ""
    if isinstance(error, TimeoutError):
        code = "57014"
    return {
        "42601": "Invalid PostgreSQL query syntax.",
        "42703": "Unknown column; use the messages schema in the tool description.",
        "42803": "GROUP BY must include every selected nonaggregate column or expression; IDs do not imply other columns. Alternatively aggregate replies in a CTE, then join messages.",
        "42P01": "Unknown relation; read messages and query-local CTEs only.",
        "42501": "The history role cannot access that relation or function.",
        "42883": "Unknown function or incompatible argument types.",
        "22007": "Invalid SQL timestamp; include an explicit timezone offset.",
        "57014": "The history query exceeded its execution deadline; narrow the query.",
    }.get(code, "The history SQL query failed; simplify the SELECT and retry.")
