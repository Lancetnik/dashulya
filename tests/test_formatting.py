import unittest

from aiogram.types import MessageEntity
from aiogram.utils.formatting import sizeof

from dashulya.bot.formatting import render_markdown, rich_message


class MarkdownFormattingTests(unittest.TestCase):
    def test_native_rich_formatting_preserves_markdown(self) -> None:
        for markdown in (
            "**bold**",
            "# Heading",
            "| A | B |\n| --- | --- |\n| 1 | 2 |",
            "<details><summary>Title</summary>\n\nBody\n\n</details>",
            "`<html>`",
        ):
            with self.subTest(markdown=markdown):
                rich = rich_message(markdown)
                assert rich is not None
                self.assertEqual(rich.markdown, markdown)

    def test_plain_text_unsupported_html_and_media_use_text_transport(self) -> None:
        for markdown in (
            "",
            "Hello",
            "**bold** <script>text</script>",
            "![alt](https://example.com/image.png)",
            "<details><summary>Unclosed",
        ):
            with self.subTest(markdown=markdown):
                self.assertIsNone(rich_message(markdown))

    def test_rich_limits_fall_back_before_sending(self) -> None:
        wide_table = "|" + "A|" * 21 + "\n|" + "---|" * 21
        for markdown in (
            "**" + "a" * 32768 + "**",
            "**paragraph**\n\n" * 501,
            "<details>" * 17 + "text" + "</details>" * 17,
            wide_table,
        ):
            with self.subTest(length=len(markdown)):
                self.assertIsNone(rich_message(markdown))

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
