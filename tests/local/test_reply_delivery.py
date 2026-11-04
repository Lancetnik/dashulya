import json
from contextlib import AsyncExitStack, nullcontext
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import (
    GetStickerSet,
    SendMessage,
    SendRichMessage,
    SendSticker,
    TelegramMethod,
)
from aiogram.types import Message, MessageEntity

from dashulya.ai.dialogue import Dialogue
from dashulya.bot.images import TelegramImages
from dashulya.bot.middlewares.memory import CaptureSentMessagesMiddleware
from dashulya.bot.replies import ReplyDelivery
from dashulya.bot.stickers import TelegramStickers
from dashulya.domain.memory import MessageRecorder
from dashulya.domain.messages import ImageOmission, Reply, ReplyBatch
from dashulya.observability import JsonLogFormatter
from tests.support import activated_group_registry, dashulya_sticker_set

PRIVATE = {"id": 42, "type": "private", "first_name": "Anna"}


@pytest.mark.usefixtures("_setup")
class TestReplyDelivery:
    @pytest.fixture
    async def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.bot = Bot("123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN")
        self.cleanup.push_async_callback(self.bot.session.close)
        self.requests = AsyncMock(return_value=True)
        requests_patch = patch.object(self.bot.session, "make_request", self.requests)
        requests_patch.start()
        self.cleanup.callback(requests_patch.stop)
        self.dialogue = AsyncMock(spec=Dialogue)
        self.dialogue.delivery = MagicMock(return_value=nullcontext())
        self.delivery = ReplyDelivery(
            self.bot,
            self.dialogue,
            TelegramImages(self.bot, max_bytes=20_000_000),
            TelegramStickers(self.bot),
        )

    async def message(self, text: str) -> None:
        incoming = Message.model_validate({
            "message_id": 1,
            "date": datetime(2026, 10, 27, tzinfo=UTC),
            "chat": PRIVATE,
            "from": {"id": 42, "is_bot": False, "first_name": "Anna"},
            "text": text,
        }).as_(self.bot)
        await self.delivery.reply(incoming)

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

    @staticmethod
    def collapse_span(reply: SendMessage) -> MessageEntity:
        spans = [
            entity
            for entity in reply.entities or []
            if entity.type == "expandable_blockquote"
        ]
        assert len(spans) == 1, "Expected one native collapsible quotation"
        return spans[0]

    async def test_sticker_and_text_keep_their_order_and_reply_targets(self) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply("First"),
            Reply(sticker="love", reply_to_message_id=20),
            Reply("Last"),
        ))

        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> Any:
            return dashulya_sticker_set() if isinstance(method, GetStickerSet) else True

        self.requests.side_effect = request
        await self.message("Hi")
        sent = [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], (SendMessage, SendSticker))
        ]
        assert [
            item.text if isinstance(item, SendMessage) else item.sticker for item in sent
        ] == ["First", "love-file", "Last"]
        assert [
            item.reply_parameters.message_id if item.reply_parameters else None
            for item in sent
        ] == [1, 20, None]

    async def test_sticker_only_reply_uses_current_topic_and_message(self) -> None:
        self.dialogue.reply.return_value = ReplyBatch((Reply(sticker="joy"),))
        incoming = Message.model_validate({
            "message_id": 10,
            "date": datetime(2026, 10, 27, tzinfo=UTC),
            "chat": {
                "id": -123,
                "type": "supergroup",
                "title": "Group",
                "is_forum": True,
            },
            "from": {"id": 42, "is_bot": False, "first_name": "Anna"},
            "is_topic_message": True,
            "message_thread_id": 7,
            "text": "Congratulations!",
        }).as_(self.bot)
        with patch.object(
            self.bot, "get_sticker_set", AsyncMock(return_value=dashulya_sticker_set())
        ):
            await self.delivery.reply(incoming)
        sent = [
            call.args[1]
            for call in self.requests.call_args_list
            if isinstance(call.args[1], SendSticker)
        ]
        assert len(sent) == 1
        assert sent[0].sticker == "joy-file"
        assert sent[0].chat_id == -123
        assert sent[0].message_thread_id == 7
        assert sent[0].reply_parameters is not None
        assert sent[0].reply_parameters.message_id == 10
        assert self.replies() == []

    async def test_ambiguous_sticker_send_is_not_retried_and_stops_the_batch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply("First"),
            Reply(sticker="sad"),
            Reply("Must not be sent"),
        ))

        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, SendSticker):
                raise TelegramNetworkError(method, "PRIVATE response after sending")
            return True

        self.requests.side_effect = request
        with (
            patch.object(
                self.bot,
                "get_sticker_set",
                AsyncMock(return_value=dashulya_sticker_set()),
            ),
            pytest.raises(TelegramNetworkError),
        ):
            await self.message("Hi")
        assert [item.text for item in self.replies()] == ["First"]
        assert (
            sum(
                isinstance(call.args[1], SendSticker)
                for call in self.requests.call_args_list
            )
            == 1
        )
        event = json.loads(JsonLogFormatter().format(caplog.records[-1]))
        assert event["delivery_stage"] == "send"
        assert event["chunk_count"] == 1
        assert "PRIVATE" not in json.dumps(event)

    async def test_lookup_failure_sends_no_sticker_or_remaining_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply(sticker="sad"),
            Reply("Later"),
        ))
        with (
            patch.object(
                self.bot,
                "get_sticker_set",
                AsyncMock(side_effect=TimeoutError("PRIVATE")),
            ),
            pytest.raises(TimeoutError),
        ):
            await self.message("Hi")
        assert not any(
            isinstance(call.args[1], (SendSticker, SendMessage))
            for call in self.requests.call_args_list
        )
        event = json.loads(JsonLogFormatter().format(caplog.records[-1]))
        assert event["delivery_stage"] == "sticker_lookup"
        assert event["chunk_count"] == 0
        assert "PRIVATE" not in json.dumps(event)

    async def test_messages_keep_order_formatting_and_individual_reply_targets(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply("**First**", 10),
            Reply("[Second](https://example.com)", 20),
            Reply("Third"),
        ))
        with caplog.at_level("INFO", logger="dashulya.bot.replies"):
            await self.message("Hi")
        sent = self.replies()
        assert [item.text for item in sent] == ["First", "Second", "Third"]
        assert [
            item.reply_parameters.message_id if item.reply_parameters else None
            for item in sent
        ] == [10, 20, None]
        assert [[e.type for e in item.entities or []] for item in sent] == [
            ["bold"],
            ["text_link"],
            [],
        ]
        event = json.loads(JsonLogFormatter().format(caplog.records[-1]))
        assert event["message"] == "AI reply delivered"
        assert event["message_count"] == 3
        assert event["chunk_count"] == 3

    async def test_failure_stops_the_batch_without_resending_delivered_messages(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply("First"),
            Reply("Second"),
            Reply("Third"),
        ))

        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, SendMessage) and method.text == "Second":
                raise TelegramNetworkError(method, "Connection lost after sending")
            return True

        self.requests.side_effect = request
        with pytest.raises(TelegramNetworkError):
            await self.message("Hi")
        assert [item.text for item in self.replies()] == ["First", "Second"]
        event = json.loads(JsonLogFormatter().format(caplog.records[-1]))
        assert event["message"] == "AI reply delivery failed"
        assert event["message_count"] == 3
        assert event["chunk_count"] == 1
        assert event["delivery_stage"] == "send"

    async def test_render_failure_is_logged_before_any_send(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self.dialogue.reply.return_value = ReplyBatch((Reply("First"),))
        with (
            patch(
                "dashulya.bot.replies.render_markdown", side_effect=ValueError("PRIVATE")
            ),
            pytest.raises(ValueError, match=r"^PRIVATE$"),
        ):
            await self.message("Hi")
        assert self.replies() == []
        event = json.loads(JsonLogFormatter().format(caplog.records[-1]))
        assert event["message"] == "AI reply delivery failed"
        assert event["delivery_stage"] == "render"
        assert event["chunk_count"] == 0
        assert "PRIVATE" not in json.dumps(event)

    async def test_failed_image_download_still_reaches_the_reply_agent(self) -> None:
        incoming = Message.model_validate({
            "message_id": 1,
            "date": datetime(2026, 10, 27, tzinfo=UTC),
            "chat": PRIVATE,
            "from": {"id": 42, "is_bot": False, "first_name": "Anna"},
            "caption": "Read this picture",
            "photo": [
                {"file_id": "photo", "file_unique_id": "unique", "width": 2, "height": 2}
            ],
        }).as_(self.bot)
        self.dialogue.reply.return_value = ReplyBatch((
            Reply("I couldn't load the picture. Please resend it."),
        ))
        with patch.object(self.bot, "get_file", AsyncMock(side_effect=TimeoutError)):
            await self.delivery.reply(incoming)
        request = self.dialogue.reply.await_args.args[0]
        assert request.images == (ImageOmission(1, "unavailable"),)
        assert request.text == "Read this picture"
        assert [reply.text for reply in self.replies()] == [
            "I couldn't load the picture. Please resend it."
        ]

    async def test_markdown_bold_and_source_links_use_explicit_entities(
        self,
    ) -> None:
        link_label = "Обсуждение"
        self.dialogue.reply.return_value = ReplyBatch((
            Reply(
                "Вчера обсуждали **Codex и Claude** 😄\n\n"
                f"[{link_label}](https://t.me/fastnewsdev_chat/19008)"
            ),
        ))
        await self.message("Что обсуждали?")
        reply = self.replies()[-1]
        assert reply.text == f"Вчера обсуждали Codex и Claude 😄\n\n{link_label}"
        assert [e.type for e in reply.entities or []] == ["bold", "text_link"]
        assert reply.chat_id == PRIVATE["id"]
        assert reply.reply_parameters is not None
        assert reply.reply_parameters.message_id == 1
        assert self.rich_replies() == []

    async def test_tables_are_collapsed_with_monospace_formatting(self) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply("| a | b |\n| --- | --- |\n| 1 | 2 |"),
        ))
        await self.message("Compare models")
        assert self.rich_replies() == []
        reply = self.replies()[0]
        assert reply.text == "a | b\n--+--\n1 | 2"
        assert [
            (entity.type, entity.offset, entity.length) for entity in reply.entities or []
        ] == [("expandable_blockquote", 0, 17), ("pre", 0, 17)]

    async def test_rejected_entity_message_is_not_retried_as_plain_text(self) -> None:
        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, SendMessage):
                raise TelegramBadRequest(method, "Bad Request: can't parse entities")
            return True

        self.requests.side_effect = request
        self.dialogue.reply.return_value = ReplyBatch((
            Reply("**Codex** [source](https://example.com)"),
        ))
        with pytest.raises(TelegramBadRequest):
            await self.message("Hi")
        assert self.rich_replies() == []
        assert len(self.replies()) == 1
        reply = self.replies()[-1]
        assert reply.text == "Codex source"
        assert [entity.type for entity in reply.entities or []] == ["bold", "text_link"]

    async def test_entity_network_failure_is_not_retried_as_another_message(self) -> None:
        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, SendMessage):
                raise TelegramNetworkError(method, "Connection lost after sending")
            return True

        self.requests.side_effect = request
        self.dialogue.reply.return_value = ReplyBatch((Reply("**Codex**"),))
        with pytest.raises(TelegramNetworkError):
            await self.message("Hi")
        assert self.rich_replies() == []
        assert len(self.replies()) == 1

    async def test_long_answer_is_collapsed_and_keeps_formatting_in_every_chunk(
        self,
    ) -> None:
        text = "💋" * 16500
        self.dialogue.reply.return_value = ReplyBatch((Reply(f"**{text}**", 25),))
        await self.message("Long answer")
        sent = self.replies()
        assert self.rich_replies() == []
        assert len(sent) == 9
        assert "".join(reply.text for reply in sent) == text
        for reply in sent:
            assert [entity.type for entity in reply.entities or []] == [
                "expandable_blockquote",
                "bold",
            ]
            assert all(
                entity.extract_from(reply.text) == reply.text
                for entity in reply.entities or []
            )
            assert reply.reply_parameters is not None
            assert reply.reply_parameters.message_id == 25

    @pytest.mark.parametrize(
        ("lead", "offset"),
        [
            ("**Summary** 💋\nA second line.", 27),
            ("a" * 240, 242),
            ("a" * 241, 0),
            ("One\nTwo\nThree", 0),
            ("## Heading", 0),
            ("Heading\n=======", 0),
            ("Heading\n-------", 0),
            ("- first\n- second", 0),
            ("> quoted", 0),
        ],
    )
    async def test_only_a_short_prose_lead_stays_outside_collapse(
        self, lead: str, offset: int
    ) -> None:
        self.dialogue.reply.return_value = ReplyBatch((Reply(lead + "\n\n" + "b" * 601),))
        await self.message("Details please")
        assert self.rich_replies() == []
        reply = self.replies()[0]
        quote = self.collapse_span(reply)
        assert quote.offset == offset
        assert quote.extract_from(reply.text).endswith("b" * 601)
        if offset:
            assert quote.extract_from(reply.text) == "b" * 601

    async def test_collapsed_code_preserves_markup_and_literal_html(self) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply(
                "**Summary** 💋\n\n"
                "[source](https://example.com/?x=1&y=2) ^^under^^ ||hidden|| with `inline`\n\n"
                "> quoted **bold**\n\n"
                '```python\nprint("<b> & 💋")\nline two\n```\n\n'
                "<details open>literal</details>"
            ),
        ))
        await self.message("Show code")
        assert self.rich_replies() == []
        reply = self.replies()[0]
        assert reply.text == (
            "Summary 💋\n\nsource under hidden with inline\n\nquoted bold\n\n"
            'print("<b> & 💋")\nline two\n\n<details open>literal</details>'
        )
        entities = reply.entities or []
        quote = self.collapse_span(reply)
        assert quote.offset == 12
        assert quote.extract_from(reply.text).startswith("source under hidden")
        assert [
            entity.extract_from(reply.text)
            for entity in entities
            if entity.type == "bold"
        ] == ["Summary", "bold"]
        assert [
            entity.extract_from(reply.text)
            for entity in entities
            if entity.type == "italic"
        ] == ["quoted bold"]
        assert not any(entity.type == "blockquote" for entity in entities)
        assert (
            next(entity for entity in entities if entity.type == "text_link").url
            == "https://example.com/?x=1&y=2"
        )
        assert (
            next(
                entity for entity in entities if entity.type == "underline"
            ).extract_from(reply.text)
            == "under"
        )
        assert (
            next(entity for entity in entities if entity.type == "spoiler").extract_from(
                reply.text
            )
            == "hidden"
        )
        assert [
            entity.extract_from(reply.text)
            for entity in entities
            if entity.type == "code"
        ] == ["inline"]
        code = next(entity for entity in entities if entity.type == "pre")
        assert code.language == "python"
        assert code.extract_from(reply.text) == 'print("<b> & 💋")\nline two'

    @pytest.mark.parametrize(("chat_id", "topic"), [(42, None), (-123, 7), (-123, None)])
    async def test_collapsed_delivery_records_original_text_and_routes_each_message(
        self, chat_id: int, topic: int | None
    ) -> None:
        recorded = AsyncMock(spec=MessageRecorder)
        self.bot.session.middleware(
            CaptureSentMessagesMiddleware(
                recorded, activated_group_registry(), owner_user_id=42
            )
        )
        original = "Summary.\n\n" + "💋" * 601 + '\n\nprint("<summary> & 💋")'
        self.dialogue.reply.return_value = ReplyBatch((
            Reply(
                "Summary.\n\n"
                + "💋" * 601
                + '\n\n```python\nprint("<summary> & 💋")\n```',
                25,
            ),
            Reply("Short follow-up"),
        ))

        next_id = 100

        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> Any:
            nonlocal next_id
            if isinstance(method, SendMessage):
                message_id = next_id
                next_id += 1
                return Message.model_validate({
                    "message_id": message_id,
                    "date": 1700000000,
                    "chat": {
                        "id": chat_id,
                        "type": "private" if chat_id > 0 else "supergroup",
                        "first_name": "Owner",
                        "title": "Group",
                    },
                    "from": {"id": bot.id, "is_bot": True, "first_name": "Dashulya"},
                    "text": method.text,
                })
            return True

        self.requests.side_effect = request
        incoming = Message.model_validate({
            "message_id": 1,
            "date": 1700000000,
            "chat": {
                "id": chat_id,
                "type": "private" if chat_id > 0 else "supergroup",
                "first_name": "Owner",
                "title": "Group",
                "is_forum": topic is not None,
            },
            "from": {"id": 42, "is_bot": False, "first_name": "Owner"},
            "is_topic_message": topic is not None,
            "message_thread_id": topic
            if topic is not None
            else 88
            if chat_id < 0
            else None,
            "text": "Answer",
        }).as_(self.bot)
        await self.delivery.reply(incoming)
        sent = self.replies()[0]
        assert self.rich_replies() == []
        assert sent.chat_id == chat_id
        assert sent.message_thread_id == topic
        assert sent.reply_parameters is not None
        assert sent.reply_parameters.message_id == 25
        assert self.replies()[1].reply_parameters is None
        rows = [call.args[0] for call in recorded.record.await_args_list]
        assert [row.text for row in rows] == [original, "Short follow-up"]
        assert [row.message_id for row in rows] == [100, 101]
        assert rows[0].reply_to_id == 25
        assert rows[0].thread_id == topic

    @pytest.mark.parametrize(
        ("body", "collapsed"),
        [
            ("a" * 600, False),
            ("a" * 601, True),
            ("💋" * 600, False),
            ("💋" * 601, True),
            ("[Short](https://example.com/" + "a" * 650 + ")", False),
            ("first\nsecond\nthird\nfourth\nfifth\nsixth", False),
        ],
    )
    async def test_collapse_threshold_uses_visible_unicode_characters(
        self, body: str, collapsed: bool
    ) -> None:
        self.dialogue.reply.return_value = ReplyBatch((Reply(body),))
        await self.message("Answer")
        assert self.rich_replies() == []
        sent = self.replies()
        assert len(sent) == 1
        assert (
            any(
                entity.type == "expandable_blockquote"
                for entity in sent[0].entities or []
            )
            is collapsed
        )

    async def test_only_first_chunk_can_have_a_visible_lead(self) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply("Lead.\n\n" + "a" * 3993 + "Tail.\n\nEnd."),
        ))
        await self.message("Long answer")
        sent = self.replies()
        assert self.rich_replies() == []
        assert [reply.text for reply in sent] == [
            "Lead.\n\n" + "a" * 3993,
            "Tail.\n\nEnd.",
        ]
        assert [
            (entity.offset, entity.length)
            for reply in sent
            for entity in reply.entities or []
            if entity.type == "expandable_blockquote"
        ] == [(7, 3993), (0, 11)]

    @pytest.mark.parametrize("error_type", [TelegramBadRequest, TelegramNetworkError])
    async def test_failed_collapsed_send_is_not_retried_and_stops_the_batch(
        self, error_type: type[TelegramBadRequest | TelegramNetworkError]
    ) -> None:
        self.dialogue.reply.return_value = ReplyBatch((Reply("a" * 8001), Reply("Later")))
        attempts = []

        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, SendMessage):
                attempts.append(method)
                if error_type is TelegramBadRequest:
                    raise TelegramBadRequest(method, "PRIVATE failure")
                raise TelegramNetworkError(method, "PRIVATE failure")
            return True

        self.requests.side_effect = request
        with pytest.raises(error_type):
            await self.message("Long answer")
        assert len(attempts) == 1
        assert len(self.replies()) == 1
        assert self.rich_replies() == []

    @pytest.mark.parametrize("lead", ["", "**Short lead**\n\n"])
    async def test_collapsed_body_uses_ordinary_message_with_only_authored_bold(
        self, lead: str
    ) -> None:
        prose = "Ordinary text " + "x" * 601
        self.dialogue.reply.return_value = ReplyBatch((
            Reply(
                lead + prose + "\n\n**Only this label** stays bold; the rest is ordinary."
            ),
        ))
        await self.message("Long answer")
        assert self.rich_replies() == []
        reply = self.replies()[0]
        assert reply.parse_mode is None
        entities = reply.entities or []
        quote = self.collapse_span(reply)
        assert quote.offset == (12 if lead else 0)
        assert (
            quote.extract_from(reply.text)
            == prose + "\n\nOnly this label stays bold; the rest is ordinary."
        )
        assert [
            entity.extract_from(reply.text)
            for entity in entities
            if entity.type == "bold"
        ] == (["Short lead", "Only this label"] if lead else ["Only this label"])

    async def test_formatted_paragraphs_use_entities_without_implicit_bold(self) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply("Summary: *one note*.\n\nOrdinary text. [Source](https://example.com)"),
        ))
        await self.message("Question")
        sent = [*self.replies(), *self.rich_replies()]
        assert len(sent) == 1
        assert isinstance(sent[0], SendMessage)
        assert sent[0].text == "Summary: one note.\n\nOrdinary text. Source"
        assert sent[0].parse_mode is None
        assert [e.type for e in sent[0].entities or []] == ["italic", "text_link"]

    async def test_literal_bullet_lines_preserve_links_and_emphasis_in_delivery(
        self,
    ) -> None:
        self.dialogue.reply.return_value = ReplyBatch((
            Reply(
                "Summary:\n\n• **First:** original. [Source](https://example.com/1)\n"
                "• **Second:** reply 😀. [Source](https://example.com/2)"
            ),
        ))
        await self.message("Summarize today")
        messages = self.replies()
        assert len(messages) == 1
        assert not any(
            isinstance(c.args[1], SendRichMessage) for c in self.requests.call_args_list
        )
        sent = messages[0]
        assert (
            sent.text
            == "Summary:\n\n• First: original. Source\n• Second: reply 😀. Source"
        )
        assert sent.parse_mode is None
        assert [
            (entity.type, entity.extract_from(sent.text))
            for entity in sent.entities or []
        ] == [
            ("bold", "First:"),
            ("text_link", "Source"),
            ("bold", "Second:"),
            ("text_link", "Source"),
        ]
