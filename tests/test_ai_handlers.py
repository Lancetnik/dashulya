import json
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import AsyncMock, patch

from ag2 import Agent
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import SendMessage, SendRichMessage, TelegramMethod
from aiogram.types import Update, User

from dashulya.bot.formatting import text_chunks
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.storage.groups import GroupStore
from tests.support import TEST_API_KEY, BotOverride, DecisionOverride, DialogueOverride

BOT_TOKEN = "123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN"
BOT_USER: dict[str, Any] = {"id": 123456789, "is_bot": True, "first_name": "Dashulya"}
USER: dict[str, Any] = {"id": 42, "is_bot": False, "first_name": "Anna"}
PRIVATE: dict[str, Any] = {"id": 42, "type": "private", "first_name": "Anna"}
GROUP: dict[str, Any] = {"id": -123, "type": "supergroup", "title": "Test"}


class AIHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bot = Bot(BOT_TOKEN)
        self.addAsyncCleanup(self.bot.session.close)
        me_patch = patch.object(
            self.bot,
            "me",
            AsyncMock(return_value=User(**BOT_USER, username="dashulya_bot")),
        )
        me_patch.start()
        self.addCleanup(me_patch.stop)
        self.requests = AsyncMock(return_value=True)
        request_patch = patch.object(self.bot.session, "make_request", self.requests)
        request_patch.start()
        self.addCleanup(request_patch.stop)
        self.dialogue = AsyncMock()
        self.dialogue.reply.return_value = "Hello <Anna> 💋"
        self.dialogue.recent_context.return_value = []
        self.decision_agent = AsyncMock(spec=Agent)
        self.decision_agent.ask.return_value.content = AsyncMock(return_value=False)
        self.container = create_container(
            Settings(
                bot_token=BOT_TOKEN,
                openai_api_key=TEST_API_KEY,
                owner_user_id=USER["id"],
                database_path=Path(self.directory.name) / "groups.sqlite3",
            ),
            BotOverride(self.bot),
            DialogueOverride(self.dialogue),
            DecisionOverride(self.decision_agent),
        )
        self.addAsyncCleanup(self.container.close)
        self.groups = await self.container.get(GroupStore)
        self.dispatcher = await self.container.get(Dispatcher)
        self.update_id = 0

    async def message(
        self,
        text: str | None = None,
        chat: dict[str, Any] | None = None,
        sender: dict[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        self.update_id += 1
        payload = {
            "message_id": self.update_id,
            "date": 1700000000,
            "chat": chat or PRIVATE,
            "from": sender or USER,
            **fields,
        }
        if text is not None:
            payload["text"] = text
            if text.startswith("/"):
                payload["entities"] = [
                    {"type": "bot_command", "offset": 0, "length": len(text.split()[0])}
                ]
        await self.dispatcher.feed_update(
            self.bot,
            Update.model_validate({"update_id": self.update_id, "message": payload}),
        )

    def replies(self) -> list[SendMessage]:
        return [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]

    def rich_replies(self) -> list[SendRichMessage]:
        return [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], SendRichMessage)
        ]

    async def test_private_text_invokes_dialogue_and_replies_as_plain_text(self) -> None:
        await self.message("Hi")
        self.dialogue.reply.assert_awaited_once_with(
            (42, None),
            "Hi",
            user_id=42,
            user_name="Anna",
            sent_at=datetime.fromtimestamp(1700000000, UTC),
            group_context=None,
        )
        self.decision_agent.ask.assert_not_awaited()
        reply = self.replies()[-1]
        self.assertEqual(reply.text, "Hello <Anna> 💋")
        self.assertIsNone(reply.parse_mode)
        assert reply.reply_parameters is not None
        self.assertEqual(reply.reply_parameters.message_id, 1)

    async def test_markdown_bold_and_source_links_use_native_rich_messages(
        self,
    ) -> None:
        link_label = "Обсуждение"
        self.dialogue.reply.return_value = (
            "Вчера обсуждали **Codex и Claude** 😄\n"
            f"[{link_label}](https://t.me/fastnewsdev_chat/19008)"
        )
        await self.message("Что обсуждали?")
        reply = self.rich_replies()[-1]
        self.assertEqual(reply.rich_message.markdown, self.dialogue.reply.return_value)
        self.assertEqual(reply.chat_id, PRIVATE["id"])
        assert reply.reply_parameters is not None
        self.assertEqual(reply.reply_parameters.message_id, 1)
        self.assertEqual(self.replies(), [])

    async def test_native_tables_and_collapsible_details_are_preserved(self) -> None:
        text = "## Models\n\n| Model | Cost |\n| --- | --- |\n| Luna | Low |\n\n<details><summary>Details</summary>\n\n**More information**\n\n</details>"
        self.dialogue.reply.return_value = text
        await self.message("Compare models")
        self.assertEqual(len(self.rich_replies()), 1)
        self.assertEqual(self.rich_replies()[0].rich_message.markdown, text)

    async def test_rejected_rich_message_falls_back_to_formatted_text(self) -> None:
        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, SendRichMessage):
                raise TelegramBadRequest(method, "Bad Request: can't parse rich message")
            return True

        self.requests.side_effect = request
        self.dialogue.reply.return_value = "**Codex** [source](https://example.com)"
        with self.assertLogs("dashulya.bot.replies", level="WARNING"):
            await self.message("Hi")
        self.assertEqual(len(self.rich_replies()), 1)
        reply = self.replies()[-1]
        self.assertEqual(reply.text, "Codex source")
        self.assertEqual(
            [entity.type for entity in reply.entities or []], ["bold", "text_link"]
        )

    async def test_rich_network_failure_is_not_retried_as_another_message(self) -> None:
        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, SendRichMessage):
                raise TelegramNetworkError(method, "Connection lost after sending")
            return True

        self.requests.side_effect = request
        self.dialogue.reply.return_value = "**Codex**"
        with self.assertRaises(TelegramNetworkError):
            await self.message("Hi")
        self.assertEqual(len(self.rich_replies()), 1)
        self.assertEqual(self.replies(), [])

    async def test_oversized_rich_answer_keeps_formatting_in_text_chunks(self) -> None:
        text = "a" * 33000
        self.dialogue.reply.return_value = f"**{text}**"
        await self.message("Long answer")
        self.assertEqual(self.rich_replies(), [])
        self.assertEqual("".join(reply.text for reply in self.replies()), text)
        for reply in self.replies():
            self.assertEqual([entity.type for entity in reply.entities or []], ["bold"])

    async def test_other_users_cannot_use_ai_or_commands_in_any_chat(self) -> None:
        outsider = {**USER, "id": 43}
        for chat in ({**PRIVATE, "id": 43}, GROUP):
            for text in (
                "Hello",
                "Дашуля, ответь",
                "@dashulya_bot hello",
                "/start",
                "/help",
                "/groups",
                "/clear",
            ):
                with self.subTest(chat=chat["type"], text=text):
                    await self.message(text, chat=chat, sender=outsider)
        self.dialogue.reply.assert_not_awaited()
        self.dialogue.clear.assert_not_awaited()
        self.decision_agent.ask.assert_not_awaited()
        self.requests.assert_not_awaited()
        self.assertEqual(self.groups.list_groups()[0].chat_id, GROUP["id"])

    async def test_forwarding_from_owner_does_not_grant_access(self) -> None:
        await self.message(
            "Hello",
            chat={**PRIVATE, "id": 43},
            sender={**USER, "id": 43},
            forward_origin={"type": "user", "date": 1700000000, "sender_user": USER},
        )
        self.dialogue.reply.assert_not_awaited()
        self.requests.assert_not_awaited()

    async def test_messages_without_sender_are_ignored(self) -> None:
        await self.dispatcher.feed_update(
            self.bot,
            Update.model_validate({
                "update_id": 100,
                "message": {
                    "message_id": 100,
                    "date": 1700000000,
                    "chat": PRIVATE,
                    "text": "/clear",
                },
            }),
        )
        self.dialogue.clear.assert_not_awaited()
        self.requests.assert_not_awaited()

    async def test_bot_sender_is_rejected_even_with_owner_id(self) -> None:
        await self.message("/help", sender={**USER, "is_bot": True})
        self.requests.assert_not_awaited()

    async def test_owner_can_still_use_group_commands(self) -> None:
        await self.message("/help@dashulya_bot", chat=GROUP)
        self.assertIn("/groups", self.replies()[-1].text)
        self.assertIn("/clear works only in private chats", self.replies()[-1].text)
        self.dialogue.reply.assert_not_awaited()

    async def test_commands_media_bots_and_unaddressed_group_text_skip_ai(self) -> None:
        for text in ("/start", "/help", "/groups", "/unknown", "/help@other_bot"):
            await self.message(text)
        await self.message(
            sticker={
                "file_id": "x",
                "file_unique_id": "x",
                "type": "regular",
                "width": 1,
                "height": 1,
                "is_animated": False,
                "is_video": False,
            }
        )
        await self.message("Hi", sender=BOT_USER)
        await self.message("Group chatter", chat=GROUP)
        await self.message(
            "@someone_else Hi",
            chat=GROUP,
            entities=[{"type": "mention", "offset": 0, "length": 13}],
        )
        await self.message("News", chat={"id": -789, "type": "channel", "title": "News"})
        self.dialogue.reply.assert_not_awaited()
        self.assertEqual(self.groups.list_groups()[0].chat_id, -123)

    async def test_group_mentions_and_topic_messages_bypass_decisions(self) -> None:
        text = "💋 @DASHULYA_BOT hello"
        for chat_type in ("group", "supergroup"):
            await self.message(
                text,
                chat={**GROUP, "type": chat_type},
                entities=[{"type": "mention", "offset": 3, "length": 13}],
            )
        await self.message(
            text,
            chat=GROUP,
            message_thread_id=7,
            is_topic_message=True,
            entities=[{"type": "mention", "offset": 3, "length": 13}],
        )
        self.assertEqual(self.dialogue.reply.await_count, 3)
        self.assertEqual(self.dialogue.reply.call_args.args[0], (-123, 7))
        self.assertEqual(len(self.replies()), 3)
        self.decision_agent.ask.assert_not_awaited()

    async def test_replies_to_this_bot_bypass_decisions_and_carry_reply_target(
        self,
    ) -> None:
        original = {"message_id": 100, "date": 1700000000, "chat": GROUP, "text": "Hi"}
        await self.message(
            "Hello", chat=GROUP, reply_to_message={**original, "from": BOT_USER}
        )
        self.decision_agent.ask.assert_not_awaited()
        self.dialogue.reply.assert_awaited_once()
        context = self.dialogue.reply.call_args.kwargs["group_context"]
        self.assertEqual(context["reply_to"]["sender_id"], BOT_USER["id"])
        self.assertEqual(context["reply_to"]["text"], "Hi")
        reply_parameters = self.replies()[-1].reply_parameters
        assert reply_parameters is not None
        self.assertEqual(reply_parameters.message_id, 1)

    async def test_text_mentions_by_bot_id_bypass_decisions(self) -> None:
        await self.message(
            "Dashulya hi",
            chat=GROUP,
            entities=[
                {"type": "text_mention", "offset": 0, "length": 8, "user": BOT_USER}
            ],
        )
        self.dialogue.reply.assert_awaited_once()
        self.decision_agent.ask.assert_not_awaited()

    async def test_other_reply_targets_and_partial_usernames_do_not_trigger(self) -> None:
        original = {"message_id": 100, "date": 1700000000, "chat": GROUP, "text": "Hi"}
        for sender in (USER, {**BOT_USER, "id": 987654321}):
            await self.message(
                "Hello", chat=GROUP, reply_to_message={**original, "from": sender}
            )
        await self.message(
            "@dashulya_bot_other hello",
            chat=GROUP,
            entities=[{"type": "mention", "offset": 0, "length": 19}],
        )
        self.dialogue.reply.assert_not_awaited()
        self.decision_agent.ask.assert_not_awaited()
        self.requests.assert_not_awaited()

    async def test_name_in_text_uses_decisions_before_replying(self) -> None:
        self.decision_agent.ask.return_value.content.return_value = True
        for text in ("Дашуля, помоги", "Расскажи про Дашулю", "DASHULYA, help"):
            await self.message(text, chat=GROUP, message_thread_id=7)
        self.assertEqual(self.decision_agent.ask.await_count, 3)
        self.assertEqual(self.dialogue.reply.await_count, 3)
        key = self.dialogue.recent_context.call_args.args[0]
        context = json.loads(self.decision_agent.ask.call_args.args[0])["current_message"]
        self.assertEqual(key, (-123, 7))
        self.assertEqual(context["text"], "DASHULYA, help")
        self.assertEqual(context["sender_id"], 42)

    async def test_negative_decision_is_silent_without_typing_or_reply(self) -> None:
        await self.message("Дашуля вчера сказала...", chat=GROUP)
        self.decision_agent.ask.assert_awaited_once()
        self.dialogue.reply.assert_not_awaited()
        self.requests.assert_not_awaited()

    async def test_classifier_gets_reply_to_another_participant(self) -> None:
        await self.message(
            "Тебе нравится ответ Дашули?",  # ruff: ignore[ambiguous-unicode-character-string] -- Russian addressing example.
            chat=GROUP,
            reply_to_message={
                "message_id": 100,
                "date": 1700000000,
                "chat": GROUP,
                "from": {**USER, "id": 43, "first_name": "Boris"},
                "text": "Let's discuss it",
            },
        )
        context = json.loads(self.decision_agent.ask.call_args.args[0])["current_message"]
        self.assertEqual(context["reply_to"]["sender_id"], 43)
        self.assertEqual(context["reply_to"]["text"], "Let's discuss it")
        self.dialogue.reply.assert_not_awaited()

    async def test_classifier_failure_is_silent_and_direct_ping_still_works(self) -> None:
        self.decision_agent.ask.side_effect = RuntimeError("sk-secret provider data")
        with self.assertLogs("dashulya.bot.handlers.dialogue", level="WARNING") as logs:
            await self.message("Dashulya, help", chat=GROUP)
        self.assertNotIn("sk-secret", str(logs.output))
        self.dialogue.reply.assert_not_awaited()
        self.requests.assert_not_awaited()
        await self.message(
            "@dashulya_bot help",
            chat=GROUP,
            entities=[{"type": "mention", "offset": 0, "length": 13}],
        )
        self.dialogue.reply.assert_awaited_once()
        self.decision_agent.ask.assert_awaited_once()

    async def test_agent_name_requires_word_boundaries(self) -> None:
        for text in ("dashulya_bot", "megadashulya", "Дашуля123"):  # ruff: ignore[ambiguous-unicode-character-string] -- Cyrillic next to digits.
            await self.message(text, chat=GROUP)
        self.decision_agent.ask.assert_not_awaited()
        self.dialogue.reply.assert_not_awaited()

    async def test_clear_targets_private_chat_and_never_calls_ai(self) -> None:
        await self.message("/clear")
        self.dialogue.clear.assert_awaited_once_with((42, None))
        self.dialogue.reply.assert_not_awaited()
        self.assertIn("cleared", self.replies()[-1].text)

    async def test_clear_in_groups_is_ignored(self) -> None:
        for chat_type in ("group", "supergroup"):
            await self.message("/clear@dashulya_bot", chat={**GROUP, "type": chat_type})
        await self.message(
            "/clear", chat=GROUP, message_thread_id=7, is_topic_message=True
        )
        self.dialogue.clear.assert_not_awaited()
        self.dialogue.reply.assert_not_awaited()
        self.requests.assert_not_awaited()

    async def test_errors_are_redacted_and_next_message_still_works(self) -> None:
        self.dialogue.reply.side_effect = RuntimeError("sk-secret provider response")
        with self.assertLogs("dashulya.bot.replies", level="WARNING") as logs:
            await self.message("Hi")
        self.assertNotIn("sk-secret", str(logs.output))
        self.assertNotIn("sk-secret", self.replies()[-1].text)
        self.assertIn("try again", self.replies()[-1].text)
        self.dialogue.reply.side_effect = None
        await self.message("Retry")
        self.assertEqual(self.replies()[-1].text, "Hello <Anna> 💋")

    async def test_long_responses_are_split_losslessly(self) -> None:
        text = "a" * 3999 + "💋" * 3000 + "<plain text>"
        self.dialogue.reply.return_value = text
        await self.message("Long answer please")
        chunks = [reply.text for reply in self.replies()]
        self.assertEqual("".join(chunks), text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.encode("utf-16-le")) // 2, 4096)
        self.assertEqual(list(text_chunks("")), [])
