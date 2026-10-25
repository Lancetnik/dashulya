"""Local indexing and inspection without Telegram or model API calls."""

import argparse
import json
import sqlite3
from pathlib import Path

from dashulya.memory.index import build_memory
from dashulya.memory.search import MemorySearch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/memory.sqlite3"))
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
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build_memory(args.source, args.index, chat_username=args.chat)
        else:
            memory = MemorySearch(args.index)
            if args.command == "search":
                result = memory.search(
                    args.query,
                    author_id=args.author,
                    after=args.after,
                    before=args.before,
                    limit=args.limit,
                )
            elif args.command == "expand":
                result = memory.search(memory_id=args.memory_id)
            else:
                result = {
                    "memory_id": memory.remember(args.text, args.sources, kind=args.kind)
                }
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
