import json
import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, patch

from ag2.events import ToolCallEvent
from ag2.testing import TestConfig, TrackingConfig

from dashulya.ai.agents.factory import create_reply_agent
from dashulya.ai.prompts import (
    CHARACTER_PROMPT,
    INPUT_FORMAT_PROMPT,
    MEMORY_TOOLS_PROMPT,
    RESPONSE_RULES_PROMPT,
)
from evals.suites.memory.corpus import SourceArchive, aware_time
from evals.suites.memory.production import ProductionAnswerSession
from evals.suites.memory.types import QueryContext, SourceMessage
from tests.support import reply_json


def message(mid: int, *, parent: int | None = None) -> SourceMessage:
    return SourceMessage(
        id=mid,
        chat_key="group",
        date="2026-01-01T10:00:00Z",
        text=f"Original source {mid}",
        user_id=100 + mid,
        author=f"author_{mid}",
        user_name=f"Name {mid}",
        reply_to_msg_id=parent,
    )


def evidence(m: SourceMessage) -> dict[str, Any]:
    return {
        "message_id": m.id,
        "sent_at": m.date,
        "user_id": m.user_id,
        "author": m.author,
        "text": m.text,
        "url": f"https://t.me/{m.chat_key}/{m.id}",
        "reply_to_message_id": m.reply_to_msg_id,
        "truncated": False,
        "excerpt_start": 0,
    }


class ScriptedBackend:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {"coverage": {"complete": False}, "hits": []}
        self.calls: list[dict[str, Any]] = []

    async def search(
        self,
        query: str = "",
        *,
        memory_id: str | None = None,
        author_id: int | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 3,
    ) -> dict[str, Any]:
        self.calls.append({
            "query": query,
            "memory_id": memory_id,
            "author_id": author_id,
            "after": after,
            "before": before,
            "limit": limit,
        })
        return self.result


class ProductionAnswerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.scope = QueryContext("group", "2026-01-02T00:00:00Z")
        self.archive = SourceArchive([message(i) for i in range(1, 31)])
        self.backend = ScriptedBackend()

    async def test_actual_production_prompt_and_only_production_tool(self) -> None:
        config = TrackingConfig(TestConfig(reply_json("Answer")))
        session = ProductionAnswerSession(self.archive, self.backend, self.scope)
        with patch(
            "evals.suites.memory.production.create_reply_agent",
            wraps=create_reply_agent,
        ) as factory:
            result = await session.answer("Question", config)
        self.assertEqual(len(factory.call_args.kwargs["tools"]), 1)
        self.assertIn("search_memory", repr(config.calls[0].tools[0]))
        self.assertEqual(len(config.calls[0].tools), 1)
        self.assertEqual(
            config.calls[0].prompt,
            (
                CHARACTER_PROMPT,
                RESPONSE_RULES_PROMPT,
                INPUT_FORMAT_PROMPT,
                MEMORY_TOOLS_PROMPT,
            ),
        )
        self.assertEqual(self.backend.calls, [])  # No automatic initial search.
        self.assertIsNone(result["answer"]["abstained"])
        self.assertIsNone(result["answer"]["aggregation_count"])
        self.assertEqual(result["answer"]["text"], "Answer")
        self.assertIn("total", result["usage"])
        await session.close()

    async def test_latest_twenty_includes_current_and_preserves_foreign_bot_flag(
        self,
    ) -> None:
        messages = [message(i) for i in range(1, 31)]
        messages[-1] = replace(messages[-1], is_bot=True, user_id=800)
        session = ProductionAnswerSession(
            SourceArchive(messages), self.backend, self.scope
        )
        result = await session.answer("Question", TestConfig(reply_json("Answer")))
        self.assertEqual(
            result["production"]["selected_context_ids"], list(range(12, 32))
        )
        self.assertEqual(result["seen_source_ids"], list(range(12, 31)))
        last = result["trace"][0]["result"]["messages"][-1]
        self.assertEqual(last["text"], "Original source 30")
        self.assertEqual(last["user_id"], 800)
        self.assertTrue(last["is_bot"])
        self.assertTrue(result["production"]["synthetic_request"])
        self.assertGreater(
            aware_time(result["production"]["current_request"]["date"]),
            aware_time(self.scope.before),
        )

    async def test_reply_parents_expand_five_hops_and_never_future_or_other_chat(
        self,
    ) -> None:
        messages = [message(i) for i in range(1, 41)]
        for child, parent in ((40, 6), (6, 5), (5, 4), (4, 3), (3, 2), (2, 1)):
            messages[child - 1] = replace(messages[child - 1], reply_to_msg_id=parent)
        messages[38] = replace(messages[38], reply_to_msg_id=100)
        messages.extend([
            replace(message(100), date="2026-01-03T00:00:00Z"),
            replace(message(6), chat_key="private", text="Private secret"),
        ])
        result = await ProductionAnswerSession(
            SourceArchive(messages), self.backend, self.scope
        ).answer("Question", TestConfig(reply_json("Answer")))
        selected = result["production"]["selected_context_ids"]
        self.assertEqual(selected[:5], [2, 3, 4, 5, 6])
        self.assertNotIn(1, selected)
        self.assertNotIn(100, selected)
        self.assertNotIn("Private secret", json.dumps(result))

    async def test_historical_request_uses_original_sender_and_canonical_history(
        self,
    ) -> None:
        current = replace(message(25), text="Historical question", reply_to_msg_id=2)
        archive = SourceArchive(
            [message(i) for i in range(1, 25)] + [current, message(26)]
        )
        scope = QueryContext("group", current.date, before_id=current.id)
        config = TrackingConfig(TestConfig(reply_json("Answer", 2)))
        result = await ProductionAnswerSession(archive, self.backend, scope).answer(
            current.text,
            config,
            input_context={"preceding_messages": [{"id": 24, "text": message(24).text}]},
        )
        self.assertEqual(result["production"]["current_request"]["user_id"], 125)
        self.assertEqual(config.calls[0].variables, {"user_id": 125, "chat_id": -1})
        self.assertFalse(result["production"]["synthetic_request"])
        self.assertIn(
            6, result["seen_source_ids"]
        )  # Reconstructed latest19, not supplied1.
        self.assertNotIn(26, result["seen_source_ids"])
        self.assertNotIn(26, result["seen_source_ids"])
        self.assertEqual(result["answer"]["reply_to_message_id"], 2)

    async def test_forged_historical_question_and_supplied_context_rejected_pre_model(
        self,
    ) -> None:
        config = TrackingConfig(TestConfig(reply_json("Answer")))
        scope = QueryContext("group", message(25).date, before_id=25)
        session = ProductionAnswerSession(self.archive, self.backend, scope)
        with self.assertRaisesRegex(ValueError, "canonical original"):
            await session.answer("Forged question", config)
        with self.assertRaisesRegex(ValueError, "canonical source"):
            await session.answer(
                message(25).text,
                config,
                input_context={
                    "preceding_messages": [{"id": 26, "text": message(26).text}]
                },
            )
        self.assertEqual(config.calls, [])

    async def test_real_injected_tool_preserves_arguments_excerpts_and_exposure(
        self,
    ) -> None:
        original = message(1)
        source = evidence(original)
        source.update(text=original.text[4:10], truncated=True, excerpt_start=4)
        response = {
            "coverage": {"complete": False},
            "hits": [{"memory_id": "record:1", "sources": [source], "context": []}],
        }
        backend = ScriptedBackend(response)
        config = TestConfig(
            ToolCallEvent(
                "search_memory",
                arguments='{"query":"keywords","author_id":101,"limit":5}',
            ),
            reply_json("Source https://t.me/group/1"),
        )
        result = await ProductionAnswerSession(self.archive, backend, self.scope).answer(
            "Question", config
        )
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["limit"], 5)
        self.assertEqual(backend.calls[0]["author_id"], 101)
        self.assertEqual(result["trace"][1]["result"], response)
        self.assertTrue(
            result["trace"][1]["result"]["hits"][0]["sources"][0]["truncated"]
        )
        self.assertIn(1, result["seen_source_ids"])
        self.assertEqual(result["answer"]["cited_message_ids"], [1])

    async def test_noncanonical_or_future_tool_evidence_fails_closed(self) -> None:
        for source in (
            evidence(replace(message(1), text="Fabricated fact")),
            evidence(replace(message(1), chat_key="private")),
            evidence(message(100)),
        ):
            with self.subTest(source=source):
                backend = ScriptedBackend({"hits": [{"sources": [source]}]})
                config = TestConfig(
                    ToolCallEvent("search_memory", arguments='{"query":"keyword"}'),
                    reply_json("Answer"),
                )
                session = ProductionAnswerSession(self.archive, backend, self.scope)
                with self.assertRaisesRegex(ValueError, "noncanonical or out-of-scope"):
                    await session.answer("Question", config)

    async def test_citations_come_only_from_text_and_reply_target_stays_production(
        self,
    ) -> None:
        text = "See https://t.me/group/12 and [13]; [99](https://example.org) and https://t.me/foreign/14"
        result = await ProductionAnswerSession(
            self.archive, self.backend, self.scope
        ).answer("Question", TestConfig(reply_json(text, 999)))
        self.assertEqual(result["answer"]["cited_message_ids"], [12, 13])
        self.assertIsNone(result["answer"]["reply_to_message_id"])
        self.assertNotEqual(
            result["answer"]["cited_message_ids"], result["seen_source_ids"]
        )

    async def test_original_text_time_identity_reach_actual_dialogue_unchanged(
        self,
    ) -> None:
        calls: list[AsyncMock] = []

        def factory(*args: Any, **kwargs: Any) -> Any:
            agent = create_reply_agent(*args, **kwargs)
            ask = AsyncMock(wraps=agent.ask)
            ask_patch = patch.object(agent, "ask", ask)
            ask_patch.start()
            self.addCleanup(ask_patch.stop)
            calls.append(ask)
            return agent

        current = replace(
            message(25), text="Original\n  whitespace", date="2026-01-01T13:00:00+03:00"
        )
        archive = SourceArchive([message(24), current])
        scope = QueryContext("group", current.date, before_id=25)
        with patch(
            "evals.suites.memory.production.create_reply_agent", side_effect=factory
        ):
            await ProductionAnswerSession(archive, self.backend, scope).answer(
                current.text, TestConfig(reply_json("Answer"))
            )
        actual = calls[0].call_args.args[0]
        self.assertIn("13:00:00\nTimezone: Europe/Moscow (UTC+03:00)", actual)
        self.assertIn("Name 25 (Telegram user ID: 125):\nOriginal\n  whitespace", actual)
        self.assertIn("Telegram message ID: 25", actual)

    async def test_tool_timeout_preserves_exposed_context_and_failed_call_trace(
        self,
    ) -> None:
        backend = ScriptedBackend()
        with patch.object(backend, "search", side_effect=TimeoutError):
            session = ProductionAnswerSession(self.archive, backend, self.scope)
            config = TestConfig(
                ToolCallEvent("search_memory", arguments='{"query":"keyword"}')
            )
            with self.assertRaises(TimeoutError):
                await session.answer("Question", config)
        self.assertEqual(session.trace[0]["tool"], "production_group_context")
        self.assertEqual(session.trace[1]["arguments"]["query"], "keyword")
        self.assertEqual(session.trace[1]["error_type"], "TimeoutError")
        self.assertEqual(session.seen_sources, set(range(12, 31)))
