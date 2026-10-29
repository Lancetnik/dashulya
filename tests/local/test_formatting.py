import pytest
from aiogram.types import MessageEntity
from aiogram.utils.formatting import sizeof
from hypothesis import example, given, settings, strategies as st

from dashulya.bot.formatting import render_markdown, text_chunks


@settings(max_examples=50, derandomize=True)
@example(fragment="💋", repetitions=2001)
@example(fragment="a💋\u0301\n", repetitions=1000)
@given(
    fragment=st.text(
        alphabet=st.characters(exclude_categories=["Cs"]), min_size=1, max_size=30
    ),
    repetitions=st.integers(min_value=1, max_value=500),
)
def test_text_chunks_preserve_unicode_and_telegram_limits(
    fragment: str, repetitions: int
) -> None:
    text = fragment * repetitions
    chunks = list(text_chunks(text))
    assert "".join(chunks) == text
    for chunk in chunks:
        assert chunk
        assert len(chunk.encode("utf-16-le")) // 2 <= 4000


class TestMarkdownFormatting:
    def test_slopwriter_span_markup(self, subtests: pytest.Subtests) -> None:
        for markdown, kind, body in (
            ("**bold**", "bold", "bold"),
            ("__bold__", "bold", "bold"),
            ("*italic*", "italic", "italic"),
            ("_italic_", "italic", "italic"),
            ("~~old~~", "strikethrough", "old"),
            ("^^under^^", "underline", "under"),
            ("||hidden||", "spoiler", "hidden"),
        ):
            with subtests.test(markdown=markdown):
                text, entities = self.render(markdown)
                assert text == body
                assert [(e.type, e.offset, e.length) for e in entities] == [
                    (kind, 0, len(body))
                ]

    def test_slopwriter_tables_are_aligned_monospace(self) -> None:
        text, entities = self.render("| a | b |\n| --- | --- |\n| longer | x |")
        assert text == "a      | b\n-------+--\nlonger | x"
        assert len(entities) == 1
        assert entities[0].type == "pre"
        assert entities[0].extract_from(text) == text

    def test_spoilers_keep_distinct_utf16_spans(self, subtests: pytest.Subtests) -> None:
        for markdown, expected, spans in (
            ("||first|| between ||second||", "first between second", [(0, 5), (14, 6)]),
            (
                "💋 ||first|| between ||second💋||",
                "💋 first between second💋",
                [(3, 5), (17, 8)],
            ),
            ("||💋||||next||", "💋next", [(0, 2), (2, 4)]),
            ("||closed|| and ||unfinished", "closed and ||unfinished", [(0, 6)]),
        ):
            with subtests.test(markdown=markdown):
                text, entities = self.render(markdown)
                assert text == expected
                assert [(e.type, e.offset, e.length) for e in entities] == [
                    ("spoiler", offset, length) for offset, length in spans
                ]

    def test_nested_lists_hashtags_and_autolinks_match_slopwriter(self) -> None:
        text, entities = self.render(
            "intro:\n- outer\n  - **inner**\n\n#python #telegram\n\nhttps://example.com/x"
        )
        assert (
            text
            == "intro:\n\n• outer\n  • inner\n\n#python #telegram\n\nhttps://example.com/x"
        )
        assert [(e.type, e.extract_from(text)) for e in entities] == [
            ("bold", "inner"),
            ("text_link", "https://example.com/x"),
        ]

    def test_ordinary_paragraphs_are_not_bold(self) -> None:
        text, entities = self.render(
            "Summary: *no shared decision*.\n\nOrdinary paragraph with `pytest` and a [source](https://example.com).\n\n*Conclusion:* use selectively."
        )
        assert "Ordinary paragraph" in text
        assert sum(e.type == "bold" for e in entities) == 0
        assert sum(e.type == "italic" for e in entities) == 2

    def render(self, markdown: str) -> tuple[str, list[MessageEntity]]:
        chunks = list(render_markdown(markdown))
        assert len(chunks) == 1
        return chunks[0]

    def test_nested_emphasis_emoji_and_links_have_utf16_offsets(self) -> None:
        text, entities = self.render(
            "💋 **bold and *italic*** [source](https://example.com/a_(b)?x=1&y=2)"
        )
        assert text == "💋 bold and italic source"
        assert [entity.type for entity in entities] == ["bold", "italic", "text_link"]
        assert entities[0].offset == 3
        assert entities[0].extract_from(text) == "bold and italic"
        assert entities[1].extract_from(text) == "italic"
        assert entities[2].extract_from(text) == "source"
        assert entities[2].url == "https://example.com/a_(b)?x=1&y=2"

    def test_literal_html_symbols_and_escaped_markdown_remain_literal(self) -> None:
        text, entities = self.render(r"<b>literal</b> & 2 < 3; snake_case; \*not bold\*")
        assert text == "<b>literal</b> & 2 < 3; snake_case; *not bold*"
        assert entities == []

    def test_code_blocks_preserve_language_and_literal_markup(self) -> None:
        code = 'print("**literal** <tag> & value")'
        text, entities = self.render(f"```python\n{code}\n```")
        assert text == code
        assert entities[0].type == "pre"
        assert entities[0].language == "python"
        assert entities[0].extract_from(text) == code

    def test_code_inside_emphasis_has_no_forbidden_entity_overlap(self) -> None:
        text, entities = self.render("**before `x_y` after**")
        assert text == "before x_y after"
        code = next(entity for entity in entities if entity.type == "code")
        for entity in entities:
            if entity is not code:
                assert (
                    entity.offset + entity.length <= code.offset
                    or entity.offset >= code.offset + code.length
                )

    def test_code_in_link_label_keeps_the_link(self) -> None:
        text, entities = self.render("[`search_memory`](https://example.com)")
        assert text == "search_memory"
        assert [entity.type for entity in entities] == ["text_link"]

    def test_headings_and_lists_are_readable_and_preserve_numbering(self) -> None:
        text, entities = self.render(
            "# Heading\n\n- first\n- **second**\n\n3. third\n4. fourth"
        )
        assert text == "Heading\n\n• first\n• second\n\n3. third\n4. fourth"
        assert [entity.extract_from(text) for entity in entities] == ["Heading", "second"]

    def test_quotes_and_strikethrough(self) -> None:
        text, entities = self.render("> ~~old~~ new\n\nOutside")
        assert text == "old new\n\nOutside"
        assert {entity.type for entity in entities} == {"blockquote", "strikethrough"}
        quote = next(entity for entity in entities if entity.type == "blockquote")
        assert quote.extract_from(text) == "old new"

    def test_nested_quotes_do_not_create_nested_blockquote_entities(self) -> None:
        _, entities = self.render("> Outer\n>\n> > Inner")
        assert sum(entity.type == "blockquote" for entity in entities) == 1

    def test_long_bold_and_links_survive_chunk_boundaries(self) -> None:
        bold = "a" * 3999 + "💋" * 100
        label = "b" * 5000
        chunks = list(render_markdown(f"**{bold}** [{label}](https://example.com)"))
        assert "".join((text for text, _ in chunks)) == f"{bold} {label}"
        assert (
            "".join(
                (
                    entity.extract_from(text)
                    for text, entities in chunks
                    for entity in entities
                    if entity.type == "bold"
                )
            )
            == bold
        )
        assert (
            "".join(
                (
                    entity.extract_from(text)
                    for text, entities in chunks
                    for entity in entities
                    if entity.type == "text_link"
                )
            )
            == label
        )
        for text, entities in chunks:
            assert sizeof(text) <= 4000
            for entity in entities:
                assert entity.length > 0
                assert entity.offset >= 0
                assert entity.offset + entity.length <= sizeof(text)

    def test_long_code_chunks_keep_language(self) -> None:
        code = "💋" * 5000
        chunks = list(render_markdown(f"```text\n{code}\n```"))
        assert "".join((text for text, _ in chunks)) == code
        for text, entities in chunks:
            assert len(entities) == 1
            assert entities[0].type == "pre"
            assert entities[0].language == "text"
            assert entities[0].extract_from(text) == text

    def test_unclosed_markdown_is_readable_and_empty_input_is_ignored(self) -> None:
        text, entities = self.render("**unfinished [link](broken")
        assert text == "**unfinished [link](broken"
        assert entities == []
        assert list(render_markdown("")) == []
