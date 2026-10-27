import unittest
from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aiogram import Bot
from aiogram.methods import SendMessage, SendRichMessage
from aiogram.types import Message, MessageEntity
from aiogram.utils.formatting import sizeof

from dashulya.bot.formatting import render_markdown
from dashulya.bot.images import TelegramImages
from dashulya.bot.reactions import TelegramReactions
from dashulya.bot.replies import ReplyDelivery
from dashulya.domain.messages import Reply


class MarkdownFormattingTests(unittest.TestCase):
    def test_literal_bullet_lines_keep_their_visual_line_breaks(self) -> None:
        markdown = "Summary:\n\n• **First:** original.\n• **Second:** reply."
        text, entities = self.render(markdown)
        self.assertEqual(text, "Summary:\n\n• First: original.\n• Second: reply.")
        self.assertEqual(
            [entity.extract_from(text) for entity in entities], ["First:", "Second:"]
        )

    def test_slopwriter_span_markup(self) -> None:
        for markdown, kind, body in (
            ("**bold**", "bold", "bold"),
            ("__bold__", "bold", "bold"),
            ("*italic*", "italic", "italic"),
            ("_italic_", "italic", "italic"),
            ("~~old~~", "strikethrough", "old"),
            ("^^under^^", "underline", "under"),
            ("||hidden||", "spoiler", "hidden"),
        ):
            with self.subTest(markdown=markdown):
                text, entities = self.render(markdown)
                self.assertEqual(text, body)
                self.assertEqual(
                    [(e.type, e.offset, e.length) for e in entities],
                    [(kind, 0, len(body))],
                )

    def test_slopwriter_tables_are_aligned_monospace(self) -> None:
        text, entities = self.render("| a | b |\n| --- | --- |\n| longer | x |")
        self.assertEqual(text, "a      | b\n-------+--\nlonger | x")
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].type, "pre")
        self.assertEqual(entities[0].extract_from(text), text)

    def test_nested_lists_hashtags_and_autolinks_match_slopwriter(self) -> None:
        text, entities = self.render(
            "intro:\n- outer\n  - **inner**\n\n#python #telegram\n\nhttps://example.com/x"
        )
        self.assertEqual(
            text,
            "intro:\n\n• outer\n  • inner\n\n#python #telegram\n\nhttps://example.com/x",
        )
        self.assertEqual(
            [(e.type, e.extract_from(text)) for e in entities],
            [("bold", "inner"), ("text_link", "https://example.com/x")],
        )

    def test_ordinary_paragraphs_are_not_bold(self) -> None:
        text, entities = self.render(
            "Summary: *no shared decision*.\n\nOrdinary paragraph with `pytest` and a [source](https://example.com).\n\n*Conclusion:* use selectively."
        )
        self.assertIn("Ordinary paragraph", text)
        self.assertEqual(sum(e.type == "bold" for e in entities), 0)
        self.assertEqual(sum(e.type == "italic" for e in entities), 2)

    def render(self, markdown: str) -> tuple[str, list[MessageEntity]]:
        chunks = list(render_markdown(markdown))
        self.assertEqual(len(chunks), 1)
        return chunks[0]

    def test_nested_emphasis_emoji_and_links_have_utf16_offsets(self) -> None:
        text, entities = self.render(
            "💋 **bold and *italic*** [source](https://example.com/a_(b)?x=1&y=2)"
        )
        self.assertEqual(text, "💋 bold and italic source")
        self.assertEqual(
            [entity.type for entity in entities], ["bold", "italic", "text_link"]
        )
        self.assertEqual(entities[0].offset, 3)
        self.assertEqual(entities[0].extract_from(text), "bold and italic")
        self.assertEqual(entities[1].extract_from(text), "italic")
        self.assertEqual(entities[2].extract_from(text), "source")
        self.assertEqual(entities[2].url, "https://example.com/a_(b)?x=1&y=2")

    def test_literal_html_symbols_and_escaped_markdown_remain_literal(self) -> None:
        text, entities = self.render(r"<b>literal</b> & 2 < 3; snake_case; \*not bold\*")
        self.assertEqual(text, "<b>literal</b> & 2 < 3; snake_case; *not bold*")
        self.assertEqual(entities, [])

    def test_code_blocks_preserve_language_and_literal_markup(self) -> None:
        code = 'print("**literal** <tag> & value")'
        text, entities = self.render(f"```python\n{code}\n```")
        self.assertEqual(text, code)
        self.assertEqual(entities[0].type, "pre")
        self.assertEqual(entities[0].language, "python")
        self.assertEqual(entities[0].extract_from(text), code)

    def test_code_inside_emphasis_has_no_forbidden_entity_overlap(self) -> None:
        text, entities = self.render("**before `x_y` after**")
        self.assertEqual(text, "before x_y after")
        code = next(entity for entity in entities if entity.type == "code")
        for entity in entities:
            if entity is not code:
                self.assertTrue(
                    entity.offset + entity.length <= code.offset
                    or entity.offset >= code.offset + code.length
                )

    def test_code_in_link_label_keeps_the_link(self) -> None:
        text, entities = self.render("[`search_memory`](https://example.com)")
        self.assertEqual(text, "search_memory")
        self.assertEqual([entity.type for entity in entities], ["text_link"])

    def test_headings_and_lists_are_readable_and_preserve_numbering(self) -> None:
        text, entities = self.render(
            "# Heading\n\n- first\n- **second**\n\n3. third\n4. fourth"
        )
        self.assertEqual(text, "Heading\n\n• first\n• second\n\n3. third\n4. fourth")
        self.assertEqual(
            [entity.extract_from(text) for entity in entities], ["Heading", "second"]
        )

    def test_quotes_and_strikethrough(self) -> None:
        text, entities = self.render("> ~~old~~ new\n\nOutside")
        self.assertEqual(text, "old new\n\nOutside")
        self.assertEqual(
            {entity.type for entity in entities}, {"blockquote", "strikethrough"}
        )
        quote = next(entity for entity in entities if entity.type == "blockquote")
        self.assertEqual(quote.extract_from(text), "old new")

    def test_nested_quotes_do_not_create_nested_blockquote_entities(self) -> None:
        _, entities = self.render("> Outer\n>\n> > Inner")
        self.assertEqual(sum(entity.type == "blockquote" for entity in entities), 1)

    def test_long_bold_and_links_survive_chunk_boundaries(self) -> None:
        bold = "a" * 3999 + "💋" * 100
        label = "b" * 5000
        chunks = list(render_markdown(f"**{bold}** [{label}](https://example.com)"))
        self.assertEqual("".join(text for text, _ in chunks), f"{bold} {label}")
        self.assertEqual(
            "".join(
                entity.extract_from(text)
                for text, entities in chunks
                for entity in entities
                if entity.type == "bold"
            ),
            bold,
        )
        self.assertEqual(
            "".join(
                entity.extract_from(text)
                for text, entities in chunks
                for entity in entities
                if entity.type == "text_link"
            ),
            label,
        )
        for text, entities in chunks:
            self.assertLessEqual(sizeof(text), 4000)
            for entity in entities:
                self.assertGreater(entity.length, 0)
                self.assertGreaterEqual(entity.offset, 0)
                self.assertLessEqual(entity.offset + entity.length, sizeof(text))

    def test_long_code_chunks_keep_language(self) -> None:
        code = "💋" * 5000
        chunks = list(render_markdown(f"```text\n{code}\n```"))
        self.assertEqual("".join(text for text, _ in chunks), code)
        for text, entities in chunks:
            self.assertEqual(len(entities), 1)
            self.assertEqual(entities[0].type, "pre")
            self.assertEqual(entities[0].language, "text")
            self.assertEqual(entities[0].extract_from(text), text)

    def test_unclosed_markdown_is_readable_and_empty_input_is_ignored(self) -> None:
        text, entities = self.render("**unfinished [link](broken")
        self.assertEqual(text, "**unfinished [link](broken")
        self.assertEqual(entities, [])
        self.assertEqual(list(render_markdown("")), [])


class ReplyFormattingTests(unittest.IsolatedAsyncioTestCase):
    async def test_formatted_paragraphs_use_entities_without_implicit_bold(self) -> None:
        bot = Bot("123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN")
        self.addAsyncCleanup(bot.session.close)
        dialogue = AsyncMock()
        dialogue.delivery = MagicMock(return_value=nullcontext())
        dialogue.reply.return_value = Reply(
            "Summary: *one note*.\n\nOrdinary text. [Source](https://example.com)"
        )
        incoming = Message.model_validate({
            "message_id": 1,
            "date": 1700000000,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "is_bot": False, "first_name": "Example"},
            "text": "Question",
        }).as_(bot)
        requests = AsyncMock(return_value=True)
        with patch.object(bot.session, "make_request", requests):
            await ReplyDelivery(
                bot,
                dialogue,
                TelegramImages(bot, max_bytes=20_000_000),
                TelegramReactions(bot),
            ).reply(incoming)
        sent = [
            c.args[1]
            for c in requests.call_args_list
            if isinstance(c.args[1], (SendMessage, SendRichMessage))
        ]
        self.assertEqual(len(sent), 1)
        assert isinstance(sent[0], SendMessage)
        self.assertEqual(sent[0].text, "Summary: one note.\n\nOrdinary text. Source")
        self.assertIsNone(sent[0].parse_mode)
        self.assertEqual(
            [e.type for e in sent[0].entities or []], ["italic", "text_link"]
        )

    async def test_literal_bullet_lines_preserve_links_and_emphasis_in_delivery(
        self,
    ) -> None:
        bot = Bot("123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN")
        self.addAsyncCleanup(bot.session.close)
        dialogue = AsyncMock()
        dialogue.delivery = MagicMock(return_value=nullcontext())
        dialogue.reply.return_value = Reply(
            "Summary:\n\n• **First:** original. [Source](https://example.com/1)\n"
            "• **Second:** reply 😀. [Source](https://example.com/2)"
        )
        incoming = Message.model_validate({
            "message_id": 1,
            "date": datetime(2026, 10, 27, tzinfo=UTC),
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "is_bot": False, "first_name": "Example"},
            "text": "Summarize today",
        }).as_(bot)
        requests = AsyncMock(return_value=True)
        with patch.object(bot.session, "make_request", requests):
            await ReplyDelivery(
                bot,
                dialogue,
                TelegramImages(bot, max_bytes=20_000_000),
                TelegramReactions(bot),
            ).reply(incoming)
        messages = [
            c.args[1]
            for c in requests.call_args_list
            if isinstance(c.args[1], SendMessage)
        ]
        self.assertEqual(len(messages), 1)
        self.assertFalse(
            any(isinstance(c.args[1], SendRichMessage) for c in requests.call_args_list)
        )
        sent = messages[0]
        self.assertEqual(
            sent.text, "Summary:\n\n• First: original. Source\n• Second: reply 😀. Source"
        )
        self.assertIsNone(sent.parse_mode)
        self.assertEqual(
            [
                (entity.type, entity.extract_from(sent.text))
                for entity in sent.entities or []
            ],
            [
                ("bold", "First:"),
                ("text_link", "Source"),
                ("bold", "Second:"),
                ("text_link", "Source"),
            ],
        )
