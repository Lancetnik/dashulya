"""Local indexing and inspection without Telegram or model API calls."""

import argparse
import asyncio
import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from dashulya.database import Database
from dashulya.memory.index import build_memory
from dashulya.memory.migrate import migrate_sqlite
from dashulya.memory.search import MemorySearch


async def run_migration(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    load_dotenv()
    url = os.getenv("DATABASE_URL", "")
    if not url:
        parser.error("Set DATABASE_URL before migration.")
    if not any((args.evidence, args.live, args.groups, args.vectors)):
        parser.error("Migration requires at least one source file.")
    database = Database(url)
    try:
        result = await migrate_sqlite(
            database,
            evidence=args.evidence,
            live=args.live,
            groups=args.groups,
            vectors=args.vectors,
            chat_id=args.chat_id,
            owner_user_id=args.owner_user_id,
        )
    finally:
        await database.close()
    print(json.dumps(result, ensure_ascii=False))


async def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("evals/datasets/fastnewsdev/memory.sqlite3")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Build a new index from a slop-writer snapshot."
    )
    build.add_argument("source", type=Path)
    build.add_argument(
        "--chat", required=True, help="Public Telegram group username, without @."
    )
    search = commands.add_parser(
        "search", help="Search messages and derived records together."
    )
    search.add_argument("query")
    search.add_argument("--author", type=int)
    search.add_argument("--after")
    search.add_argument("--before")
    search.add_argument("--limit", type=int, default=3)
    expand = commands.add_parser(
        "expand", help="Read a returned memory ID and its original evidence."
    )
    expand.add_argument("memory_id")
    remember = commands.add_parser(
        "remember", help="Add a source-backed summary or fact."
    )
    remember.add_argument("text")
    remember.add_argument("--sources", type=int, nargs="+", required=True)
    remember.add_argument("--kind", choices=("summary", "fact"), default="summary")
    migrate = commands.add_parser(
        "migrate",
        help="Import legacy originals, groups and cached vectors into PostgreSQL.",
    )
    migrate.add_argument("--evidence", type=Path)
    migrate.add_argument("--live", type=Path)
    migrate.add_argument("--groups", type=Path)
    migrate.add_argument("--vectors", type=Path)
    migrate.add_argument("--chat-id", type=int)
    migrate.add_argument("--owner-user-id", type=int, required=True)
    args = parser.parse_args()
    if args.command == "migrate":
        await run_migration(args, parser)
        return
    try:
        if args.command == "build":
            result = build_memory(args.source, args.index, chat_username=args.chat)
        else:
            memory = await MemorySearch.open(args.index)
            if args.command == "search":
                result = await memory.search(
                    args.query,
                    author_id=args.author,
                    after=args.after,
                    before=args.before,
                    limit=args.limit,
                )
            elif args.command == "expand":
                result = await memory.search(memory_id=args.memory_id)
            else:
                result = {
                    "memory_id": memory.remember(args.text, args.sources, kind=args.kind)
                }
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
