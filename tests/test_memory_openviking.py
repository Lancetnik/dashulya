import json
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from dashulya.memory.openviking import OpenVikingMemory
from dashulya.memory.openviking_index import reconcile_resources
from evals.suites.memory.openviking import OpenVikingVariant, _ingest_export
from evals.suites.memory.types import QueryContext, SourceMessage

# ruff: file-ignore[ambiguous-unicode-character-string] -- Preserve multilingual synthetic source text.


class OpenVikingVariantTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)
        self.messages = [
            SourceMessage(
                i,
                "fixture_chat",
                f"2026-01-{i:02}T10:00:00+00:00",
                "" if i == 8 else f"Original storage discussion {i}",
                10 if i % 2 else 20,
                "alice" if i % 2 else "bob",
                "Alice" if i % 2 else "Bob",
                reply_to_msg_id=1 if i == 2 else None,
                is_bot=i == 8,
            )
            for i in range(1, 10)
        ]
        self.messages[0] = replace(
            self.messages[0], text="Хранилище «Альфа» сохраняет 🧠.\nСледующая строка."
        )
        self.context = QueryContext("fixture_chat", "2026-02-01T00:00:00+00:00")
        self.variant = OpenVikingVariant(self.workspace, url="http://fixture")
        self.candidates: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []

        def importer(export: Path, _url: str, _key: str) -> dict[str, Any]:
            manifest = export / "manifest.json"
            records = json.loads(manifest.read_text())["records"]
            return reconcile_resources(
                manifest,
                [{"uri": uri, "isDir": not uri.endswith(".md")} for uri in records],
            )

        self.importer = patch(
            "evals.suites.memory.openviking._ingest_export", side_effect=importer
        ).start()
        self.addCleanup(patch.stopall)
        patch.object(OpenVikingMemory, "check_ready", new_callable=AsyncMock).start()
        original_client = httpx.AsyncClient

        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json={"result": {"resources": self.candidates}})

        def client_factory(**kwargs: Any) -> httpx.AsyncClient:
            return original_client(**kwargs, transport=httpx.MockTransport(handle))

        patch(
            "openviking_sdk.client.httpx.AsyncClient", side_effect=client_factory
        ).start()

    def uri(self, key: str) -> str:
        data = json.loads((self.variant.export_path / "manifest.json").read_text())
        records: dict[str, str] = data["records"]
        return next(
            uri
            for uri, mapped in records.items()
            if mapped == key and uri.endswith(".md")
        )

    async def test_sources_round_trip_including_empty_text_without_numeric_chat_id(
        self,
    ) -> None:
        result = await self.variant.prepare(self.messages)
        self.assertEqual(result["source_count"], 9)
        self.assertIsInstance(self.variant.get_backend(), OpenVikingMemory)
        with closing(sqlite3.connect(self.variant.source_path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT chat_key, text, is_bot FROM group_messages WHERE id=8"
                ).fetchone(),
                ("fixture_chat", "", 1),
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(group_messages)")
            }
            self.assertNotIn("chat_id", columns)
        expanded = await self.variant.get_backend().search(memory_id="message:8")
        source = expanded["hits"][0]["sources"][0]
        self.assertEqual(source["message_id"], 8)
        self.assertEqual(source["text"], "")
        self.assertEqual(source["url"], "https://t.me/fixture_chat/8")
        first = await self.variant.get_backend().search(memory_id="message:1")
        self.assertEqual(first["hits"][0]["sources"][0]["text"], self.messages[0].text)

    async def test_resume_skips_import_and_refuses_changed_source_prefix(self) -> None:
        first = await self.variant.prepare(self.messages)
        resumed = OpenVikingVariant(self.workspace, url="http://fixture")
        self.assertEqual(await resumed.prepare(self.messages), first)
        self.assertEqual(self.importer.call_count, 1)
        with self.assertRaisesRegex(ValueError, "different source prefix"):
            await resumed.prepare(self.messages[:-1])

    async def test_prefixes_have_separate_sources_exports_and_namespaces(self) -> None:
        early = OpenVikingVariant(self.workspace / "early", url="http://fixture")
        early_result = await early.prepare(self.messages[:3])
        full_result = await self.variant.prepare(self.messages)
        self.assertNotEqual(early_result["root_uri"], full_result["root_uri"])
        self.assertEqual(early_result["source_count"], 3)
        self.assertEqual(
            (await early.get_backend().search(memory_id="message:9"))["hits"], []
        )
        exported = "\n".join(p.read_text() for p in early.export_path.rglob("*.md"))
        self.assertNotIn("Original storage discussion 9", exported)

    async def test_native_find_retains_five_hit_cap_and_conversation_diversity(
        self,
    ) -> None:
        await self.variant.prepare(self.messages)
        self.candidates = [{"uri": self.uri(f"message:{i}")} for i in range(1, 10)]
        result = await self.variant.search("storage", self.context, limit=12)
        self.assertEqual([h.source_ids for h in result], [(1,), (3,), (4,), (5,), (6,)])
        self.assertTrue(all(h.chat_key == "fixture_chat" for h in result))
        self.assertEqual(self.requests[0]["limit"], 100)
        self.assertEqual(self.requests[0]["target_uri"], self.variant.get_backend().root)
        self.assertTrue(self.requests[0]["include_provenance"])

    async def test_candidate_seam_precedes_diversity_and_native_finalization_matches_baseline(
        self,
    ) -> None:
        await self.variant.prepare(self.messages)
        self.candidates = [{"uri": self.uri(f"message:{i}")} for i in range(1, 10)]
        baseline = await self.variant.search("storage", self.context)
        pool = await self.variant.candidate_pool("storage", self.context, limit=100)
        self.assertEqual([h.source_ids for h in pool], [(i,) for i in range(1, 10)])
        self.assertEqual(self.variant.finalize_candidates(pool, limit=100), baseline)
        self.assertEqual([h.source_ids for h in baseline], [(1,), (3,), (4,), (5,), (6,)])
        self.assertTrue(all(request["limit"] == 100 for request in self.requests))

    async def test_candidate_seam_rejects_unmapped_uris_and_exact_cutoff_sources(
        self,
    ) -> None:
        messages = [
            replace(m, date="2026-01-01T10:00:00+00:00") for m in self.messages[:2]
        ]
        await self.variant.prepare(messages)
        self.candidates = [
            {"uri": "viking://resources/foreign/unknown.md"},
            {"uri": self.uri("message:2")},
            {"uri": self.uri("message:1")},
        ]
        context = QueryContext("fixture_chat", messages[0].date, before_id=2)
        pool = await self.variant.candidate_pool("storage", context)
        self.assertEqual([h.source_ids for h in pool], [(1,)])
        self.assertEqual(
            await self.variant.candidate_pool(
                "storage", replace(context, chat_key="foreign_chat")
            ),
            [],
        )

    async def test_foreign_chat_never_queries_backend(self) -> None:
        await self.variant.prepare(self.messages)
        self.assertEqual(
            await self.variant.search(
                "storage", replace(self.context, chat_key="foreign_chat")
            ),
            [],
        )
        self.assertEqual(self.requests, [])

    async def test_exact_tuple_cutoff_rejects_native_inclusive_boundary(self) -> None:
        messages = [
            replace(m, date="2026-01-01T10:00:00+00:00") for m in self.messages[:2]
        ]
        await self.variant.prepare(messages)
        self.candidates = [
            {"uri": self.uri("message:2")},
            {"uri": self.uri("message:1")},
        ]
        context = QueryContext("fixture_chat", messages[0].date, before_id=2)
        # Native diversity selects message2 first. The adapter rejects that hit
        # rather than leaking it or substituting its own retrieval algorithm.
        self.assertEqual(await self.variant.search("storage", context), [])

    async def test_author_and_after_filters_use_production_original_sources(self) -> None:
        await self.variant.prepare(self.messages)
        self.candidates = [{"uri": self.uri(f"message:{i}")} for i in range(1, 10)]
        context = replace(self.context, author_id=10, after="2026-01-04T00:00:00+00:00")
        result = await self.variant.search("storage", context)
        self.assertEqual([h.source_ids for h in result], [(5,), (7,), (9,)])

    async def test_only_common_profile_bounds_question_seed_and_records_adaptation(
        self,
    ) -> None:
        await self.variant.prepare(self.messages)
        self.candidates = [{"uri": self.uri("message:1")}]
        query = " ".join(["word"] * 21)
        result = await self.variant.search(query, self.context)
        self.assertEqual(self.requests[0]["query"], " ".join(["word"] * 20))
        assert self.variant.last_query is not None
        self.assertTrue(self.variant.last_query["normalized"])
        self.assertIn("retrieval_query", json.loads(result[0].text))
        with self.assertRaisesRegex(ValueError, "1 to 20 keywords"):
            await self.variant.get_backend().search(query)
        self.assertEqual(len(self.requests), 1)

    async def test_offline_build_never_imports_or_exposes_ready_backend(self) -> None:
        result = self.variant.build_local(self.messages)
        self.assertEqual(result["stage"], "exported")
        self.assertFalse(result["prepared"])
        self.importer.assert_not_called()
        self.assertTrue((self.variant.export_path / "manifest.json").is_file())
        with self.assertRaisesRegex(RuntimeError, "Prepare OpenViking"):
            self.variant.get_backend()

    async def test_tampered_evidence_cannot_be_reused(self) -> None:
        await self.variant.prepare(self.messages)
        with closing(sqlite3.connect(self.variant.index_path)) as connection, connection:
            connection.execute("UPDATE messages SET body='changed' WHERE id=1")
        with self.assertRaisesRegex(ValueError, "exactly the supplied originals"):
            await self.variant.prepare(self.messages)

    async def test_requires_one_chat_and_unique_source_identities(self) -> None:
        with self.assertRaisesRegex(ValueError, "one non-empty chat"):
            await self.variant.prepare([
                self.messages[0],
                replace(self.messages[1], chat_key="foreign_chat"),
            ])
        with self.assertRaisesRegex(ValueError, "unique"):
            await self.variant.prepare([self.messages[0], self.messages[0]])


class OpenVikingImportTests(unittest.TestCase):
    def test_deep_wrapper_enumeration_repairs_default_listing_without_reimport(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            export = Path(directory)
            root = "viking://resources/fixture"
            exported = root + "/discussion/hash.md"
            wrapped = root + "/discussion/hash/Heading/Original_sources/source_1.md"
            manifest = export / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "root_uri": root,
                    "index_sha256": "fixture-digest",
                    "records": {exported: "message:1"},
                })
            )
            truncated = [{"uri": root + "/discussion/hash/Heading", "isDir": True}]
            with self.assertRaisesRegex(ValueError, "no mapped original leaf"):
                reconcile_resources(manifest, truncated)
            (export / "import-task.json").write_text(
                json.dumps({
                    "root_uri": root,
                    "index_sha256": "fixture-digest",
                    "task_id": "finished",
                })
            )
            client = MagicMock()
            client.get_task.return_value = {"status": "completed", "result": {}}
            client.tree_page.return_value = {
                "result": [*truncated, {"uri": wrapped, "isDir": False}],
                "has_more": False,
            }
            with patch(
                "evals.suites.memory.openviking.openviking_sdk.SyncHTTPClient",
                return_value=client,
            ):
                result = _ingest_export(export, "http://fixture", "fixture-key")
            self.assertEqual(result["coverage"]["records_with_leaves"], 1)
            self.assertEqual(
                json.loads(manifest.read_text())["records"][wrapped], "message:1"
            )
            self.assertEqual(client.tree_page.call_args.kwargs["level_limit"], 1000)
            client.add_resource.assert_not_called()

    def test_explicit_tree_depth_still_rejects_a_genuinely_missing_original(self) -> None:
        with TemporaryDirectory() as directory:
            export = Path(directory)
            root = "viking://resources/fixture"
            manifest = export / "manifest.json"
            manifest.write_text(
                json.dumps({
                    "root_uri": root,
                    "index_sha256": "fixture-digest",
                    "records": {root + "/missing.md": "message:1"},
                })
            )
            (export / "import-task.json").write_text(
                json.dumps({
                    "root_uri": root,
                    "index_sha256": "fixture-digest",
                    "task_id": "finished",
                })
            )
            client = MagicMock()
            client.get_task.return_value = {"status": "completed", "result": {}}
            client.tree_page.return_value = {"result": [], "has_more": False}
            with (
                patch(
                    "evals.suites.memory.openviking.openviking_sdk.SyncHTTPClient",
                    return_value=client,
                ),
                self.assertRaisesRegex(ValueError, "no mapped original leaf"),
            ):
                _ingest_export(export, "http://fixture", "fixture-key")
            self.assertNotIn("import_coverage", json.loads(manifest.read_text()))
            client.add_resource.assert_not_called()

    def test_new_native_task_is_checkpointed_before_failed_processing(self) -> None:
        with TemporaryDirectory() as directory:
            export = Path(directory)
            manifest = {
                "root_uri": "viking://resources/fixture",
                "index_sha256": "fixture-digest",
                "records": {},
            }
            (export / "manifest.json").write_text(json.dumps(manifest))
            client = MagicMock()
            client.add_resource.return_value = {"task_id": "new-task"}
            client.get_task.return_value = {
                "status": "failed",
                "error": "Provider body must never be exposed",
            }
            with (
                patch(
                    "evals.suites.memory.openviking.openviking_sdk.SyncHTTPClient",
                    return_value=client,
                ),
                self.assertRaisesRegex(RuntimeError, "import did not complete") as raised,
            ):
                _ingest_export(export, "http://fixture", "fixture-key")
            self.assertNotIn("Provider body", str(raised.exception))
            task = json.loads((export / "import-task.json").read_text())
            self.assertEqual(task["task_id"], "new-task")
            self.assertFalse(client.add_resource.call_args.kwargs["wait"])
            self.assertNotIn(
                "import_coverage", json.loads((export / "manifest.json").read_text())
            )
            client.tree_page.assert_not_called()

    def test_resumes_native_task_and_reconciles_all_paginated_parser_leaves(self) -> None:
        with TemporaryDirectory() as directory:
            export = Path(directory)
            root = "viking://resources/fixture"
            leaf = root + "/record/doc.md"
            manifest = {
                "root_uri": root,
                "index_sha256": "fixture-digest",
                "records": {leaf: "message:1"},
            }
            (export / "manifest.json").write_text(json.dumps(manifest))
            (export / "import-task.json").write_text(
                json.dumps({
                    "root_uri": root,
                    "index_sha256": "fixture-digest",
                    "task_id": "existing-task",
                })
            )
            client = MagicMock()
            client.get_task.side_effect = [
                {"status": "running"},
                {"status": "completed", "result": {"root_uri": root}},
            ]
            client.tree_page.side_effect = [
                {"result": [{"uri": root, "isDir": True}], "has_more": True},
                {
                    "result": [{"uri": leaf, "isDir": False}],
                    "has_more": False,
                },
            ]
            with (
                patch(
                    "evals.suites.memory.openviking.openviking_sdk.SyncHTTPClient",
                    return_value=client,
                ) as constructor,
                patch("evals.suites.memory.openviking.sleep"),
            ):
                result = _ingest_export(export, "http://fixture", "fixture-key")
            client.add_resource.assert_not_called()
            self.assertEqual(client.get_task.call_args.args, ("existing-task",))
            self.assertEqual(client.tree_page.call_args.kwargs["offset"], 1)
            self.assertEqual(client.tree_page.call_args.kwargs["level_limit"], 1000)
            client.ls_page.assert_not_called()
            self.assertEqual(result["coverage"]["records_with_leaves"], 1)
            self.assertTrue(result["enumeration"]["strict_original_leaf_coverage"])
            self.assertEqual(constructor.call_args.kwargs["api_key"], "fixture-key")
            client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
