from contextlib import AsyncExitStack, nullcontext
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import SendMessage, SendRichMessage, TelegramMethod
from aiogram.types import Message

from dashulya.ai.dialogue import Dialogue
from dashulya.bot.images import TelegramImages
from dashulya.bot.replies import ReplyDelivery
from dashulya.domain.messages import Reply

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
            self.bot, self.dialogue, TelegramImages(self.bot, max_bytes=20_000_000)
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

    async def test_markdown_bold_and_source_links_use_explicit_entities(
        self,
    ) -> None:
        link_label = "Обсуждение"
        self.dialogue.reply.return_value = Reply(
            "Вчера обсуждали **Codex и Claude** 😄\n\n"
            f"[{link_label}](https://t.me/fastnewsdev_chat/19008)"
        )
        await self.message("Что обсуждали?")
        reply = self.replies()[-1]
        assert reply.text == f"Вчера обсуждали Codex и Claude 😄\n\n{link_label}"
        assert [e.type for e in reply.entities or []] == ["bold", "text_link"]
        assert reply.chat_id == PRIVATE["id"]
        assert reply.reply_parameters is not None
        assert reply.reply_parameters.message_id == 1
        assert self.rich_replies() == []

    async def test_tables_use_monospace_entities(self) -> None:
        text = "## Models\n\n| Model | Cost |\n| --- | --- |\n| Luna | Low |\n\n**More information**"
        self.dialogue.reply.return_value = Reply(text)
        await self.message("Compare models")
        assert self.rich_replies() == []
        reply = self.replies()[-1]
        assert [e.type for e in reply.entities or []] == ["bold", "pre", "bold"]
        assert "Model" in reply.text
        assert "Luna" in reply.text

    async def test_rejected_entity_message_is_not_retried_as_plain_text(self) -> None:
        async def request(bot: Bot, method: TelegramMethod[Any], **kwargs: Any) -> bool:
            if isinstance(method, SendMessage):
                raise TelegramBadRequest(method, "Bad Request: can't parse entities")
            return True

        self.requests.side_effect = request
        self.dialogue.reply.return_value = Reply(
            "**Codex** [source](https://example.com)"
        )
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
        self.dialogue.reply.return_value = Reply("**Codex**")
        with pytest.raises(TelegramNetworkError):
            await self.message("Hi")
        assert self.rich_replies() == []
        assert len(self.replies()) == 1

    async def test_long_answer_keeps_formatting_in_text_chunks(self) -> None:
        text = "a" * 33000
        self.dialogue.reply.return_value = Reply(f"**{text}**")
        await self.message("Long answer")
        assert self.rich_replies() == []
        assert "".join(reply.text for reply in self.replies()) == text
        for reply in self.replies():
            assert [entity.type for entity in reply.entities or []] == ["bold"]

    async def test_formatted_paragraphs_use_entities_without_implicit_bold(self) -> None:
        self.dialogue.reply.return_value = Reply(
            "Summary: *one note*.\n\nOrdinary text. [Source](https://example.com)"
        )
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
        self.dialogue.reply.return_value = Reply(
            "Summary:\n\n• **First:** original. [Source](https://example.com/1)\n"
            "• **Second:** reply 😀. [Source](https://example.com/2)"
        )
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
