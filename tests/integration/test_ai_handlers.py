import json
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, nullcontext
from datetime import UTC, datetime
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ag2 import Agent
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage, SendRichMessage, TelegramMethod
from aiogram.types import File, Message, Update, User

from dashulya.database import Database
from dashulya.domain.groups import Group
from dashulya.domain.memory import MessageRecorder
from dashulya.domain.messages import ImageOmission, Reply
from dashulya.main import create_container
from dashulya.main.config import Settings
from dashulya.storage.groups import GroupStore
from tests.support import (
    TEST_API_KEY,
    BotOverride,
    DecisionOverride,
    DialogueOverride,
    RecorderOverride,
)

BOT_TOKEN = "123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN"
BOT_USER: dict[str, Any] = {"id": 123456789, "is_bot": True, "first_name": "Dashulya"}
USER: dict[str, Any] = {"id": 42, "is_bot": False, "first_name": "Anna"}
PRIVATE: dict[str, Any] = {"id": 42, "type": "private", "first_name": "Anna"}
GROUP: dict[str, Any] = {"id": -123, "type": "supergroup", "title": "Test"}
PHOTO = [{"file_id": "photo", "file_unique_id": "unique", "width": 2, "height": 2}]
PNG = b"\x89PNG\r\n\x1a\nimage-data"
ANIMATION: dict[str, Any] = {
    "file_id": "animation",
    "file_unique_id": "animation",
    "width": 320,
    "height": 240,
    "duration": 3,
    "thumbnail": {**PHOTO[0], "file_id": "thumbnail"},
}


@pytest.mark.usefixtures("_setup")
class _AIHandlerFixture:
    public_group_ids: tuple[int, ...] = ()
    active_group = True

    @pytest.fixture
    async def _setup(self, database: Database, cleanup: AsyncExitStack) -> None:
        self.database = database
        self.cleanup = cleanup
        self.directory = TemporaryDirectory()
        self.cleanup.callback(self.directory.cleanup)
        self.bot = Bot(BOT_TOKEN)
        self.cleanup.push_async_callback(self.bot.session.close)
        me_patch = patch.object(
            self.bot,
            "me",
            AsyncMock(return_value=User(**BOT_USER, username="dashulya_bot")),
        )
        me_patch.start()
        self.cleanup.callback(me_patch.stop)
        self.sent_id = 1000
        self.requests = AsyncMock(side_effect=self.request)
        request_patch = patch.object(self.bot.session, "make_request", self.requests)
        request_patch.start()
        self.cleanup.callback(request_patch.stop)
        self.dialogue = AsyncMock()
        self.dialogue.delivery = MagicMock(return_value=nullcontext())
        self.dialogue.reply.return_value = Reply("Hello <Anna> 💋")
        self.dialogue.recent_context.return_value = []
        self.decision_agent = AsyncMock(spec=Agent)
        self.decision_agent.ask.return_value.content = AsyncMock(return_value=False)
        self.recorder = AsyncMock(spec=MessageRecorder)
        self.container = create_container(
            Settings(
                bot_token=BOT_TOKEN,
                openai_api_key=TEST_API_KEY,
                owner_user_id=USER["id"],
                public_group_ids=self.public_group_ids,
            ),
            BotOverride(self.bot),
            RecorderOverride(self.recorder),
            DialogueOverride(self.dialogue),
            DecisionOverride(self.decision_agent),
        )
        self.cleanup.push_async_callback(self.container.close)
        self.groups = await self.container.get(GroupStore)
        if self.active_group:
            await self.groups.remember(Group(GROUP["id"], GROUP["title"]))
            await self.groups.activate(GROUP["id"])
        self.dispatcher = await self.container.get(Dispatcher)
        self.update_id = 0
        self.get_file = AsyncMock(
            return_value=File(
                file_id="photo", file_unique_id="unique", file_path="photos/image.png"
            )
        )

        async def stream(*args: Any, **kwargs: Any) -> AsyncGenerator[bytes]:
            yield PNG

        self.download = MagicMock(side_effect=stream)
        for target, name, replacement in (
            (self.bot, "get_file", self.get_file),
            (self.bot.session, "stream_content", self.download),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.cleanup.callback(patcher.stop)

    async def request(
        self, bot: Bot, method: TelegramMethod[Any], **kwargs: Any
    ) -> Message | bool:
        if isinstance(method, (SendMessage, SendRichMessage)):
            self.sent_id += 1
            return Message.model_validate({
                "message_id": self.sent_id,
                "date": 1700000000,
                "chat": GROUP if method.chat_id == GROUP["id"] else PRIVATE,
                "from": BOT_USER,
                "text": method.text if isinstance(method, SendMessage) else None,
            })
        return True

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


class TestAIHandler(_AIHandlerFixture):
    async def test_private_text_invokes_dialogue_and_replies_as_plain_text(self) -> None:
        await self.message("Hi")
        self.dialogue.reply.assert_awaited_once_with(
            (42, None),
            "Hi",
            user_id=42,
            user_name="Anna",
            sent_at=datetime.fromtimestamp(1700000000, UTC),
            message_id=1,
            group_context=None,
            images=(),
            image_loader=None,
        )
        self.decision_agent.ask.assert_not_awaited()
        reply = self.replies()[-1]
        assert reply.text == "Hello <Anna> 💋"
        assert reply.parse_mode is None
        assert reply.reply_parameters is not None
        assert reply.reply_parameters.message_id == 1

    async def test_other_users_cannot_use_ai_or_commands_in_any_chat(
        self, subtests: pytest.Subtests
    ) -> None:
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
                with subtests.test(chat=chat["type"], text=text):
                    await self.message(text, chat=chat, sender=outsider)
        self.dialogue.reply.assert_not_awaited()
        self.dialogue.clear.assert_not_awaited()
        self.decision_agent.ask.assert_not_awaited()
        self.requests.assert_not_awaited()
        assert (await self.groups.list_groups())[0].chat_id == GROUP["id"]

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
        assert "/groups" in self.replies()[-1].text
        assert "/clear works only in private chats" in self.replies()[-1].text
        self.dialogue.reply.assert_not_awaited()

    async def test_commands_media_bots_and_unaddressed_group_text_skip_ai(self) -> None:
        for text in ("/start", "/help", "/groups", "/unknown", "/help@other_bot"):
            await self.message(text)
        await self.message(
            chat=GROUP,
            sticker={
                "file_id": "x",
                "file_unique_id": "x",
                "type": "regular",
                "width": 1,
                "height": 1,
                "is_animated": False,
                "is_video": False,
            },
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
        assert (await self.groups.list_groups())[0].chat_id == -123

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
        assert self.dialogue.reply.await_count == 3
        assert self.dialogue.reply.call_args.args[0] == (-123, 7)
        assert len(self.replies()) == 3
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
        assert context["reply_to"]["sender_id"] == BOT_USER["id"]
        assert context["reply_to"]["text"] == "Hi"
        reply_parameters = self.replies()[-1].reply_parameters
        assert reply_parameters is not None
        assert reply_parameters.message_id == 1

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
        assert self.decision_agent.ask.await_count == 3
        assert self.dialogue.reply.await_count == 3
        key = self.dialogue.recent_context.call_args.args[0]
        context = json.loads(self.decision_agent.ask.call_args.args[0])["current_message"]
        assert key == (-123, None)
        assert self.dialogue.reply.call_args.args[0] == key
        assert context["text"] == "DASHULYA, help"
        assert context["sender_id"] == 42

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
        assert context["reply_to"]["sender_id"] == 43
        assert context["reply_to"]["text"] == "Let's discuss it"
        self.dialogue.reply.assert_not_awaited()

    async def test_classifier_failure_is_silent_and_direct_ping_still_works(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.decision_agent.ask.side_effect = RuntimeError("sk-secret provider data")
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.handlers.dialogue"):
            await self.message("Dashulya, help", chat=GROUP)
        assert any(
            record.name == "dashulya.bot.handlers.dialogue"
            or record.name.startswith("dashulya.bot.handlers.dialogue.")
            for record in caplog.records
        )
        assert "sk-secret" not in str(caplog.messages)
        assert caplog.records[-1].__dict__["addressing_reason"] == "classifier_error"
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
        for text in (
            "dashulya_bot",
            "megadashulya",
            "Дашуля123",  # ruff: ignore[ambiguous-unicode-character-string] -- Cyrillic name next to digits.
            "супердашуль",
            "дашуль123",  # ruff: ignore[ambiguous-unicode-character-string] -- Cyrillic name next to digits.
            "дашуль_бот",
        ):
            await self.message(text, chat=GROUP)
        self.decision_agent.ask.assert_not_awaited()
        self.dialogue.reply.assert_not_awaited()

    async def test_negative_decisions_log_the_actual_refusal_reason(
        self, caplog: pytest.LogCaptureFixture, subtests: pytest.Subtests
    ) -> None:
        for text, answer, reason in (
            ("Hello", False, "name_not_found"),
            ("дашуль, привет", False, "classifier_rejected"),
            ("ДАШУЛЬ, привет", None, "classifier_no_answer"),
        ):
            with subtests.test(reason=reason):
                self.decision_agent.ask.return_value.content.return_value = answer
                caplog.clear()
                with caplog.at_level("INFO", logger="dashulya.bot.handlers.dialogue"):
                    await self.message(text, chat=GROUP)
                assert any(
                    record.name == "dashulya.bot.handlers.dialogue"
                    or record.name.startswith("dashulya.bot.handlers.dialogue.")
                    for record in caplog.records
                )
                record = caplog.records[-1]
                assert record.__dict__["addressing_reason"] == reason
                assert not record.__dict__["should_reply"]
        assert self.decision_agent.ask.await_count == 2
        self.dialogue.reply.assert_not_awaited()
        self.requests.assert_not_awaited()

    async def test_context_error_is_logged_separately_without_classifier_call(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self.dialogue.recent_context.side_effect = RuntimeError("PRIVATE CONTEXT")
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.handlers.dialogue"):
            await self.message("дашуль, помоги", chat=GROUP)
        assert any(
            record.name == "dashulya.bot.handlers.dialogue"
            or record.name.startswith("dashulya.bot.handlers.dialogue.")
            for record in caplog.records
        )
        assert caplog.records[-1].__dict__["addressing_reason"] == "context_error"
        assert "PRIVATE CONTEXT" not in str(caplog.messages)
        self.decision_agent.ask.assert_not_awaited()
        self.dialogue.reply.assert_not_awaited()
        self.requests.assert_not_awaited()

    async def test_lowercase_vocative_is_classified_and_replied_to(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.decision_agent.ask.return_value.content.return_value = True
        caplog.clear()
        with caplog.at_level("INFO", logger="dashulya.bot.handlers.dialogue"):
            await self.message("дашуль, передай привет", chat=GROUP)
        assert any(
            record.name == "dashulya.bot.handlers.dialogue"
            or record.name.startswith("dashulya.bot.handlers.dialogue.")
            for record in caplog.records
        )
        assert caplog.records[-1].__dict__["addressing_reason"] == "classifier_accepted"
        self.decision_agent.ask.assert_awaited_once()
        self.dialogue.reply.assert_awaited_once()

    async def test_direct_mention_logs_bypass_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.clear()
        with caplog.at_level("INFO", logger="dashulya.bot.handlers.dialogue"):
            await self.message(
                "@dashulya_bot hi",
                chat=GROUP,
                entities=[
                    {
                        "type": "mention",
                        "offset": 0,
                        "length": 13,
                    }
                ],
            )
        assert any(
            record.name == "dashulya.bot.handlers.dialogue"
            or record.name.startswith("dashulya.bot.handlers.dialogue.")
            for record in caplog.records
        )
        assert caplog.records[-1].__dict__["addressing_reason"] == "explicit_mention"
        self.decision_agent.ask.assert_not_awaited()

    async def test_clear_targets_private_chat_and_never_calls_ai(self) -> None:
        await self.message("/clear")
        self.dialogue.clear.assert_awaited_once_with((42, None))
        self.dialogue.reply.assert_not_awaited()
        assert "cleared" in self.replies()[-1].text

    async def test_clear_in_groups_is_ignored(self) -> None:
        for chat_type in ("group", "supergroup"):
            await self.message("/clear@dashulya_bot", chat={**GROUP, "type": chat_type})
        await self.message(
            "/clear", chat=GROUP, message_thread_id=7, is_topic_message=True
        )
        self.dialogue.clear.assert_not_awaited()
        self.dialogue.reply.assert_not_awaited()
        self.requests.assert_not_awaited()

    async def test_errors_are_redacted_and_next_message_still_works(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.dialogue.reply.side_effect = RuntimeError("sk-secret provider response")
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.replies"):
            await self.message("Hi")
        assert any(
            record.name == "dashulya.bot.replies"
            or record.name.startswith("dashulya.bot.replies.")
            for record in caplog.records
        )
        assert "sk-secret" not in str(caplog.messages)
        assert "sk-secret" not in self.replies()[-1].text
        assert "try again" in self.replies()[-1].text
        self.dialogue.reply.side_effect = None
        await self.message("Retry")
        assert self.replies()[-1].text == "Hello <Anna> 💋"

    async def test_private_photo_and_image_document_reach_dialogue_as_bytes(self) -> None:
        for fields in (
            {"photo": PHOTO},
            {
                "document": {
                    "file_id": "document",
                    "file_unique_id": "doc",
                    "mime_type": "image/png",
                },
                "caption": "Read the text",
            },
        ):
            await self.message(**fields)
            image = self.dialogue.reply.call_args.kwargs["images"][0]
            assert image.data == PNG
            assert image.media_type == "image/png"
        assert self.dialogue.reply.await_count == 2
        assert self.dialogue.reply.call_args.args[1] == "Read the text"
        self.decision_agent.ask.assert_not_awaited()

    async def test_caption_mentions_and_names_preserve_group_addressing(self) -> None:
        await self.message(
            chat=GROUP,
            photo=PHOTO,
            caption="@dashulya_bot inspect",
            caption_entities=[{"type": "mention", "offset": 0, "length": 13}],
        )
        self.dialogue.reply.assert_awaited_once()
        self.decision_agent.ask.assert_not_awaited()
        self.decision_agent.ask.return_value.content.return_value = True
        await self.message(chat=GROUP, photo=PHOTO, caption="Dashulya, inspect")
        self.decision_agent.ask.assert_awaited_once()
        assert self.dialogue.reply.await_count == 2

    async def test_outsider_and_rejected_group_photos_are_never_downloaded(self) -> None:
        await self.message(photo=PHOTO, sender={**USER, "id": 43})
        await self.message(chat=GROUP, photo=PHOTO, caption="Dashulya said this")
        await self.message(chat=GROUP, photo=PHOTO)
        self.get_file.assert_not_awaited()
        self.download.assert_not_called()
        self.dialogue.reply.assert_not_awaited()

    async def test_text_request_can_inspect_its_reply_image(self) -> None:
        await self.message(
            "@dashulya_bot explain",
            chat=GROUP,
            entities=[{"type": "mention", "offset": 0, "length": 13}],
            reply_to_message={
                "message_id": 100,
                "date": 1700000000,
                "chat": GROUP,
                "from": USER,
                "photo": PHOTO,
            },
        )
        images = self.dialogue.reply.call_args.kwargs["images"]
        assert images[0].message_id == 100
        assert images[0].data == PNG

    async def test_oversized_image_reaches_agent_as_omission_without_download(
        self,
    ) -> None:
        await self.message(photo=[{**PHOTO[0], "file_size": 20_000_001}])
        self.get_file.assert_not_awaited()
        self.download.assert_not_called()
        self.dialogue.reply.assert_awaited_once()
        assert self.dialogue.reply.call_args.kwargs["images"] == (ImageOmission(1),)

    async def test_image_download_failure_does_not_expose_token(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.get_file.side_effect = RuntimeError(self.bot.token)
        caplog.clear()
        with caplog.at_level("WARNING", logger="dashulya.bot.replies"):
            await self.message(photo=PHOTO)
        assert any(
            record.name == "dashulya.bot.replies"
            or record.name.startswith("dashulya.bot.replies.")
            for record in caplog.records
        )
        self.dialogue.reply.assert_not_awaited()
        assert self.bot.token not in str(caplog.messages)
        assert self.bot.token not in self.replies()[-1].text

    async def test_stickers_of_every_format_invoke_private_reply(self) -> None:
        for format_flag in (None, "is_animated", "is_video"):
            sticker = {
                "file_id": "sticker",
                "file_unique_id": "unique",
                "type": "regular",
                "width": 512,
                "height": 512,
                "is_animated": format_flag == "is_animated",
                "is_video": format_flag == "is_video",
                "thumbnail": {**PHOTO[0], "file_id": "thumbnail"},
            }
            await self.message(sticker=sticker)
            images = self.dialogue.reply.call_args.kwargs["images"]
            assert images[0].preview_kind == (
                "sticker" if format_flag is not None else None
            )
            assert (
                self.dialogue.reply.call_args.args[1]
                == "Please respond to this media in the context of our conversation."
            )
        assert self.dialogue.reply.await_count == 3

    async def test_unaddressed_and_outsider_stickers_are_never_downloaded(self) -> None:
        sticker = {
            "file_id": "sticker",
            "file_unique_id": "unique",
            "type": "regular",
            "width": 512,
            "height": 512,
            "is_animated": False,
            "is_video": False,
        }
        await self.message(chat=GROUP, sticker=sticker)
        await self.message(sticker=sticker, sender={**USER, "id": 43})
        self.dialogue.reply.assert_not_awaited()
        self.get_file.assert_not_awaited()
        self.download.assert_not_called()

    async def test_group_sticker_reply_to_bot_and_question_about_sticker(self) -> None:
        sticker = {
            "file_id": "sticker",
            "file_unique_id": "unique",
            "type": "regular",
            "width": 512,
            "height": 512,
            "is_animated": True,
            "is_video": False,
            "thumbnail": {**PHOTO[0], "file_id": "thumbnail"},
        }
        await self.message(
            chat=GROUP,
            sticker=sticker,
            reply_to_message={
                "message_id": 100,
                "date": 1700000000,
                "chat": GROUP,
                "from": BOT_USER,
                "text": "Hello",
            },
        )
        self.dialogue.reply.assert_awaited_once()
        self.decision_agent.ask.assert_not_awaited()
        await self.message(
            "@dashulya_bot what does this mean?",
            chat=GROUP,
            entities=[{"type": "mention", "offset": 0, "length": 13}],
            reply_to_message={
                "message_id": 101,
                "date": 1700000000,
                "chat": GROUP,
                "from": USER,
                "sticker": sticker,
            },
        )
        assert self.dialogue.reply.await_count == 2
        assert self.dialogue.reply.call_args.kwargs["images"][0].message_id == 101
        assert self.get_file.await_args_list[-1].args == ("thumbnail",)

    async def test_private_and_caption_addressed_gif_use_animation_previews(self) -> None:
        await self.message(animation=ANIMATION)
        assert (
            self.dialogue.reply.call_args.kwargs["images"][0].preview_kind == "animation"
        )
        await self.message(
            chat=GROUP,
            animation=ANIMATION,
            caption="@dashulya_bot explain",
            caption_entities=[{"type": "mention", "offset": 0, "length": 13}],
        )
        assert self.dialogue.reply.await_count == 2
        self.decision_agent.ask.assert_not_awaited()
        assert self.get_file.await_args_list[-1].args == ("thumbnail",)

    async def test_unaddressed_and_outsider_gifs_never_download(self) -> None:
        await self.message(animation=ANIMATION, sender={**USER, "id": 43})
        await self.message(chat=GROUP, animation=ANIMATION)
        self.get_file.assert_not_awaited()
        self.download.assert_not_called()
        self.dialogue.reply.assert_not_awaited()

    async def test_question_replying_to_gif_and_decode_failure_preserve_dialogue(
        self,
    ) -> None:
        await self.message(
            "@dashulya_bot explain",
            chat=GROUP,
            entities=[{"type": "mention", "offset": 0, "length": 13}],
            reply_to_message={
                "message_id": 100,
                "date": 1700000000,
                "chat": GROUP,
                "from": USER,
                "animation": ANIMATION,
            },
        )
        assert self.dialogue.reply.call_args.kwargs["images"][0].message_id == 100
        self.get_file.reset_mock()
        await self.message(animation={**ANIMATION, "thumbnail": None})
        self.get_file.assert_awaited_once_with("animation")
        assert self.dialogue.reply.call_args.kwargs["images"] == (
            ImageOmission(self.update_id, "decode_failed"),
        )


class TestGroupActivation(_AIHandlerFixture):
    active_group = False
    public_group_ids = (-123,)

    async def test_inactive_groups_stay_silent_for_owner_and_participants(self) -> None:
        for chat_type in ("group", "supergroup"):
            for sender in (USER, {**USER, "id": 99}):
                for text in ("@dashulya_bot hello", "Дашуль, помоги", "/help", "/groups"):
                    await self.message(
                        text,
                        chat={**GROUP, "type": chat_type},
                        sender=sender,
                        entities=[{"type": "mention", "offset": 0, "length": 13}]
                        if text.startswith("@")
                        else [],
                    )
                await self.message(
                    chat={**GROUP, "type": chat_type},
                    sender=sender,
                    photo=PHOTO,
                    caption="@dashulya_bot explain",
                    caption_entities=[{"type": "mention", "offset": 0, "length": 13}],
                )
                await self.message(
                    "Follow-up",
                    chat={**GROUP, "type": chat_type},
                    sender=sender,
                    reply_to_message={
                        "message_id": 100,
                        "date": 1700000000,
                        "chat": GROUP,
                        "from": BOT_USER,
                        "text": "Earlier answer",
                    },
                )
        self.requests.assert_not_awaited()
        self.dialogue.reply.assert_not_awaited()
        self.decision_agent.ask.assert_not_awaited()
        self.get_file.assert_not_awaited()
        self.recorder.record.assert_awaited()
        assert not await self.groups.is_active(GROUP["id"])

    async def test_only_actual_human_owner_can_activate_group(self) -> None:
        for sender in ({**USER, "id": 99}, {**USER, "is_bot": True}):
            await self.message(
                "/start@dashulya_bot",
                chat=GROUP,
                sender=sender,
                forward_origin={"type": "user", "date": 1700000000, "sender_user": USER},
            )
        await self.message("/start", chat=GROUP, **{"from": None})
        await self.message("/start@other_bot", chat=GROUP)
        assert not await self.groups.is_active(GROUP["id"])
        self.requests.assert_not_awaited()
        await self.message("/start@dashulya_bot", chat=GROUP, message_thread_id=7)
        assert await self.groups.is_active(GROUP["id"])
        assert "active in this group" in self.replies()[-1].text
        self.dialogue.reply.assert_not_awaited()
        self.decision_agent.ask.assert_not_awaited()
        for sender in (USER, {**USER, "id": 99}):
            await self.message(
                "@dashulya_bot hello",
                chat=GROUP,
                sender=sender,
                entities=[{"type": "mention", "offset": 0, "length": 13}],
            )
        assert self.dialogue.reply.await_count == 2

    async def test_private_start_and_activation_of_another_group_do_not_unlock_group(
        self,
    ) -> None:
        await self.message("/start")
        assert "Hi!" in self.replies()[-1].text
        assert await self.groups.list_groups() == []
        await self.message("/start", chat={**GROUP, "id": -456})
        assert await self.groups.is_active(-456)
        self.requests.reset_mock()
        await self.message("Дашуль, помоги", chat=GROUP)
        self.requests.assert_not_awaited()
        self.decision_agent.ask.assert_not_awaited()
        self.dialogue.reply.assert_not_awaited()


class TestPublicGroupHandler(_AIHandlerFixture):
    public_group_ids = (-123,)

    async def test_other_participant_can_mention_bot_with_their_own_identity(
        self,
    ) -> None:
        await self.message(
            "@dashulya_bot hello",
            chat=GROUP,
            sender={**USER, "id": 99, "first_name": "Boris"},
            entities=[{"type": "mention", "offset": 0, "length": 13}],
        )
        self.dialogue.reply.assert_awaited_once()
        assert self.dialogue.reply.call_args.args[0] == (-123, None)
        assert self.dialogue.reply.call_args.kwargs["user_id"] == 99
        assert self.dialogue.reply.call_args.kwargs["user_name"] == "Boris"
        self.decision_agent.ask.assert_not_awaited()
        assert self.replies()[-1].chat_id == -123

    async def test_other_participant_can_reply_to_bot(self) -> None:
        await self.message(
            "Follow-up",
            chat=GROUP,
            sender={**USER, "id": 99},
            reply_to_message={
                "message_id": 100,
                "date": 1700000000,
                "chat": GROUP,
                "from": BOT_USER,
                "text": "Answer",
            },
        )
        self.dialogue.reply.assert_awaited_once()
        self.decision_agent.ask.assert_not_awaited()

    async def test_name_addressing_still_uses_decision_and_passive_chatter_stays_silent(
        self,
    ) -> None:
        await self.message("Group chatter", chat=GROUP, sender={**USER, "id": 99})
        self.dialogue.reply.assert_not_awaited()
        self.decision_agent.ask.assert_not_awaited()
        await self.message("Дашуль, помоги", chat=GROUP, sender={**USER, "id": 99})
        self.decision_agent.ask.assert_awaited_once()
        self.dialogue.reply.assert_not_awaited()
        self.decision_agent.ask.return_value.content.return_value = True
        await self.message("Дашуль, помоги", chat=GROUP, sender={**USER, "id": 99})
        self.dialogue.reply.assert_awaited_once()

    async def test_other_private_chats_groups_and_bot_senders_remain_blocked(
        self,
    ) -> None:
        for chat, sender in (
            ({**PRIVATE, "id": 99}, {**USER, "id": 99}),
            ({**GROUP, "id": -456}, {**USER, "id": 99}),
            (GROUP, {**USER, "id": 99, "is_bot": True}),
        ):
            await self.message(
                "@dashulya_bot hello",
                chat=chat,
                sender=sender,
                entities=[{"type": "mention", "offset": 0, "length": 13}],
            )
        await self.message("@dashulya_bot hello", chat=GROUP, **{"from": None})
        self.dialogue.reply.assert_not_awaited()
        self.decision_agent.ask.assert_not_awaited()
        self.requests.assert_not_awaited()

    async def test_group_participants_do_not_gain_owner_commands(self) -> None:
        for text in (
            "/start",
            "/help",
            "/groups",
            "/groups@dashulya_bot",
            " /groups",
            "/clear",
        ):
            await self.message(text, chat=GROUP, sender={**USER, "id": 99})
        await self.message(
            chat=GROUP, sender={**USER, "id": 99}, photo=PHOTO, caption="/groups"
        )
        self.requests.assert_not_awaited()
        self.dialogue.reply.assert_not_awaited()
        self.dialogue.clear.assert_not_awaited()
        await self.message("/help", chat=GROUP)
        assert "/groups" in self.replies()[-1].text
