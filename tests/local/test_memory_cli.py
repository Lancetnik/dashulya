import json
import sqlite3
from contextlib import AsyncExitStack, chdir, closing, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import pytest

from dashulya.memory.__main__ import main


@pytest.mark.usefixtures("_setup")
class TestMemoryCommand:
    @pytest.fixture
    def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        directory = TemporaryDirectory()
        self.cleanup.callback(directory.cleanup)
        self.path = Path(directory.name)
        self.source = self.path / "snapshot.sqlite3"
        self.index = self.path / "derived.sqlite3"
        with closing(sqlite3.connect(self.source)) as source:
            source.execute(
                "CREATE TABLE group_messages(id INTEGER,date TEXT,text TEXT,user_id INTEGER,author TEXT,reply_to_msg_id INTEGER,thread_post_id INTEGER)"
            )
            source.executemany(
                "INSERT INTO group_messages VALUES(?,?,?,?,?,?,?)",
                [
                    (
                        mid,
                        f"2026-01-0{mid}T12:00:00+00:00",
                        f"Original needle {mid}",
                        10 if mid % 2 else 20,
                        "Alice" if mid % 2 else "Bob",
                        None,
                        None,
                    )
                    for mid in range(1, 5)
                ],
            )
            source.commit()
        self.original_bytes = self.source.read_bytes()

    def invoke(self, *arguments: str) -> dict[str, Any]:
        output = StringIO()
        with (
            patch(
                "sys.argv", ["dashulya-memory", "--index", str(self.index), *arguments]
            ),
            patch("dotenv.main.find_dotenv", return_value=str(self.path / "cli.env")),
            redirect_stdout(output),
        ):
            main()
        result: dict[str, Any] = json.loads(output.getvalue())
        assert self.source.read_bytes() == self.original_bytes
        return result

    def build(self) -> dict[str, Any]:
        return self.invoke("build", str(self.source), "--chat", "test_chat")

    def test_build_search_filters_and_limits_use_the_supplied_index(
        self, subtests: pytest.Subtests
    ) -> None:
        coverage = self.build()
        assert coverage["message_count"] == 4
        assert coverage["chat_username"] == "test_chat"
        assert coverage["source_path"] == str(self.source.resolve())
        assert len(self.invoke("search", "needle")["hits"]) == 3
        result = self.invoke("search", "needle", "--limit", "1")
        assert len(result["hits"]) == 1
        for options, expected in (
            (("--author", "20"), {2, 4}),
            (("--after", "2026-01-02T00:00:00+00:00"), {2, 3, 4}),
            (("--before", "2026-01-03T00:00:00+00:00"), {1, 2}),
        ):
            with subtests.test(options=options):
                hits = self.invoke("search", "needle", "--limit", "5", *options)["hits"]
                assert len(hits) == len(expected)
                assert {
                    source["message_id"] for hit in hits for source in hit["sources"]
                } == expected
        filtered = self.invoke(
            "search",
            "needle",
            "--author",
            "20",
            "--after",
            "2026-01-02T00:00:00+00:00",
            "--before",
            "2026-01-03T00:00:00+00:00",
            "--limit",
            "5",
        )
        assert len(filtered["hits"]) == 1
        assert filtered["hits"][0]["sources"] == [
            {
                "message_id": 2,
                "sent_at": "2026-01-02T12:00:00+00:00",
                "user_id": 20,
                "author": "Bob",
                "text": "Original needle 2",
                "url": "https://t.me/test_chat/2",
                "reply_to_message_id": None,
                "truncated": False,
                "excerpt_start": 0,
            }
        ]

    def test_remember_and_expand_preserve_kind_text_and_original_source_ids(
        self, subtests: pytest.Subtests
    ) -> None:
        self.build()
        for kind in (None, "summary", "fact"):
            with subtests.test(kind=kind):
                arguments = ["remember", "Решение хранения", "--sources", "1", "3"]
                if kind is not None:
                    arguments.extend(["--kind", kind])
                result = self.invoke(*arguments)
                assert set(result) == {"memory_id"}
                expanded = self.invoke("expand", result["memory_id"])
                assert len(expanded["hits"]) == 1
                hit = expanded["hits"][0]
                assert hit["memory_id"] == result["memory_id"]
                assert hit["kind"] == (kind or "summary")
                assert hit["excerpt"] == "Решение хранения"
                assert [s["message_id"] for s in hit["sources"]] == [1, 3]
                assert [s["text"] for s in hit["sources"]] == [
                    "Original needle 1",
                    "Original needle 3",
                ]

    def test_cli_errors_do_not_overwrite_sources_or_existing_indices(
        self, subtests: pytest.Subtests
    ) -> None:
        self.build()
        index_bytes = self.index.read_bytes()
        for arguments, reason in (
            (("build", str(self.source), "--chat", "test_chat"), "already exists"),
            (("search", "needle", "--limit", "0"), "limit"),
            (("search", "needle", "--author", "invalid"), "invalid int"),
            (("search", "needle", "--limit", "invalid"), "invalid int"),
            (("remember", "Invalid source", "--sources", "invalid"), "invalid int"),
            (("remember", "Unsupported source", "--sources", "999"), "source"),
            (("remember", "No sources"), "--sources"),
            (
                ("remember", "Invalid kind", "--sources", "1", "--kind", "other"),
                "invalid choice",
            ),
            (("build", str(self.source)), "--chat"),
            (("expand",), "memory_id"),
            ((), "command"),
        ):
            with subtests.test(arguments=arguments):
                errors = StringIO()
                with redirect_stderr(errors), pytest.raises(SystemExit) as raised:
                    self.invoke(*arguments)
                assert raised.value.code == 2
                assert reason in errors.getvalue()
                assert self.source.read_bytes() == self.original_bytes
                assert self.index.read_bytes() == index_bytes

    def test_migration_requires_database_and_a_source_before_opening_either(
        self, subtests: pytest.Subtests
    ) -> None:
        for environment, arguments, reason in (
            (
                {"DATABASE_URL": ""},
                ["--live", str(self.path / "absent.sqlite3")],
                "DATABASE_URL",
            ),
            ({}, ["--live", str(self.path / "absent.sqlite3")], "DATABASE_URL"),
            ({"DATABASE_URL": "postgresql://unused/test"}, [], "source file"),
        ):
            with subtests.test(reason=reason):
                errors = StringIO()
                with (
                    patch.dict("os.environ", environment, clear=True),
                    patch("dashulya.memory.__main__.load_dotenv"),
                    redirect_stderr(errors),
                    pytest.raises(SystemExit) as raised,
                ):
                    self.invoke("migrate", "--owner-user-id", "42", *arguments)
                assert raised.value.code == 2
                assert reason in errors.getvalue()
                assert not (self.path / "absent.sqlite3").exists()

    def test_migration_owner_must_be_present_and_numeric(
        self, subtests: pytest.Subtests
    ) -> None:
        for arguments, reason in (
            ([], "--owner-user-id"),
            (["--owner-user-id", "invalid"], "invalid int"),
        ):
            with subtests.test(arguments=arguments):
                errors = StringIO()
                with (
                    patch.dict("os.environ", {"DATABASE_URL": ""}, clear=True),
                    patch("dashulya.memory.__main__.load_dotenv"),
                    redirect_stderr(errors),
                    pytest.raises(SystemExit) as raised,
                ):
                    self.invoke("migrate", "--evidence", str(self.source), *arguments)
                assert raised.value.code == 2
                assert reason in errors.getvalue()

    def test_default_index_path_is_created_beneath_the_current_directory(self) -> None:
        output = StringIO()
        with (
            chdir(self.path),
            patch("dotenv.main.find_dotenv", return_value=str(self.path / "cli.env")),
            patch(
                "sys.argv",
                ["dashulya-memory", "build", str(self.source), "--chat", "test_chat"],
            ),
            redirect_stdout(output),
        ):
            main()
        assert json.loads(output.getvalue())["message_count"] == 4
        assert (self.path / "evals/datasets/fastnewsdev/memory.sqlite3").is_file()
        assert self.source.read_bytes() == self.original_bytes
