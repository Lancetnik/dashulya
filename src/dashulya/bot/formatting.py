"""Markdown to Telegram text/entities, adapted from slop-writer 0.5.1.

Source: Lancetnik/slop-writer, commit e80d5b871cfa4816e9db999d55e30c812364cc6d,
src/slop_writer/markdown.py. Apache-2.0; see slop-writer.LICENSE alongside this file.
Adaptations use aiogram entities, enforce Bot API overlap rules and chunk replies.
"""

from collections.abc import Iterator
from re import Match
from typing import Any, cast

import mistune
from aiogram.types import MessageEntity

type Token = dict[str, Any]


def _u16len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _parse_spoiler(
    inline: mistune.InlineParser, match: Match[str], state: mistune.InlineState
) -> int | None:
    position = match.end()
    end = state.src.find("||", position)
    if end == -1:
        return None
    inner = state.copy()
    inner.src = state.src[position:end]
    state.append_token({"type": "spoiler", "children": inline.render(inner)})
    return end + 2


def _spoiler_plugin(md: mistune.Markdown) -> None:
    md.inline.register("spoiler", r"\|\|(?=[^\s|])", _parse_spoiler, before="link")


_MARKDOWN = mistune.create_markdown(
    renderer=None,
    plugins=["strikethrough", "insert", "table", "url", _spoiler_plugin],
)
_SPANS = {
    "strong": "bold",
    "emphasis": "italic",
    "strikethrough": "strikethrough",
    "insert": "underline",
    "spoiler": "spoiler",
}


class _Builder:
    """Accumulate plain text and entity spans using a UTF-16 cursor."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.entities: list[MessageEntity] = []

    def text(self, text: str) -> None:
        if text:
            self.parts.append(text)
            self.length += _u16len(text)

    def span(
        self,
        kind: str,
        start: int,
        *,
        url: str | None = None,
        language: str | None = None,
    ) -> None:
        if self.length > start:
            self.entities.append(
                MessageEntity(
                    type=kind,
                    offset=start,
                    length=self.length - start,
                    url=url,
                    language=language,
                )
            )


def _inline(builder: _Builder, tokens: list[Token], *, in_link: bool = False) -> None:
    for token in tokens:
        kind = token["type"]
        if kind == "text":
            builder.text(token["raw"])
        elif kind in _SPANS:
            start = builder.length
            _inline(builder, token["children"], in_link=in_link)
            builder.span(_SPANS[kind], start)
        elif kind == "codespan":
            start = builder.length
            builder.text(token["raw"])
            if not in_link:
                builder.span("code", start)
        elif kind == "link":
            url = token["attrs"]["url"]
            start = builder.length
            _inline(
                builder,
                token.get("children") or [{"type": "text", "raw": url}],
                in_link=True,
            )
            builder.span("text_link", start, url=url)
        elif kind in {"softbreak", "linebreak"}:
            builder.text("\n")
        elif "children" in token:
            _inline(builder, token["children"], in_link=in_link)
        elif "raw" in token:
            builder.text(token["raw"])


def _cell_text(cell: Token) -> str:
    builder = _Builder()
    _inline(builder, cell.get("children", []))
    return "".join(builder.parts).replace("\n", " ")


def _render_table(builder: _Builder, token: Token) -> None:
    head: list[str] = []
    rows: list[list[str]] = []
    for section in token["children"]:
        if section["type"] == "table_head":
            head = [_cell_text(cell) for cell in section["children"]]
        elif section["type"] == "table_body":
            rows = [
                [_cell_text(cell) for cell in row["children"]]
                for row in section["children"]
            ]
    columns = max([len(head), *(len(row) for row in rows)], default=0)
    widths = [0] * columns
    for cells in [head, *rows]:
        for index in range(columns):
            widths[index] = max(
                widths[index], len(cells[index]) if index < len(cells) else 0
            )

    def formatted(cells: list[str]) -> str:
        return " | ".join(
            (cells[index] if index < len(cells) else "").ljust(widths[index])
            for index in range(columns)
        )

    lines = [
        formatted(head),
        "-+-".join("-" * width for width in widths),
        *(formatted(row) for row in rows),
    ]
    start = builder.length
    builder.text("\n".join(lines))
    builder.span("pre", start, language="")


def _render_list(
    builder: _Builder, token: Token, depth: int = 0, *, quoted: bool = False
) -> None:
    ordered = token["attrs"]["ordered"]
    number = token["attrs"].get("start", 1) or 1
    indent = "  " * depth
    for index, item in enumerate(token["children"]):
        if index:
            builder.text("\n")
        builder.text(f"{indent}{number}. " if ordered else f"{indent}• ")
        number += 1
        for child in item["children"]:
            kind = child["type"]
            if kind in {"block_text", "paragraph"}:
                _inline(builder, child["children"])
            elif kind == "list":
                builder.text("\n")
                _render_list(builder, child, depth + 1, quoted=quoted)
            else:
                _block(builder, child, quoted=quoted)


def _block(builder: _Builder, token: Token, *, quoted: bool = False) -> None:
    kind = token["type"]
    if kind in {"paragraph", "block_text"}:
        _inline(builder, token["children"])
    elif kind == "heading":
        start = builder.length
        _inline(builder, token["children"])
        builder.span("bold", start)
    elif kind == "block_code":
        language = (token.get("attrs") or {}).get("info") or ""
        language = language.split()[0] if language else ""
        start = builder.length
        builder.text(token["raw"].rstrip("\n"))
        builder.span("pre", start, language=language)
    elif kind == "block_quote":
        start = builder.length
        _blocks(builder, token["children"], quoted=True)
        if not quoted:
            builder.span("blockquote", start)
    elif kind == "list":
        _render_list(builder, token, quoted=quoted)
    elif kind == "table":
        _render_table(builder, token)
    elif kind == "thematic_break":
        builder.text("—")
    elif "children" in token:
        _inline(builder, token["children"])
    elif "raw" in token:
        builder.text(token["raw"])


def _blocks(builder: _Builder, tokens: list[Token], *, quoted: bool = False) -> None:
    first = True
    for token in tokens:
        if token["type"] == "blank_line":
            continue
        if not first:
            builder.text("\n\n")
        first = False
        _block(builder, token, quoted=quoted)


def text_chunks(text: str) -> Iterator[str]:
    # Split before an astral character when its two UTF-16 units would overflow.
    chunk = ""
    units = 0
    for character in text:
        size = 2 if ord(character) > 0xFFFF else 1
        if units + size > 4000:
            yield chunk
            chunk, units = "", 0
        chunk += character
        units += size
    if chunk:
        yield chunk


def _compatible_entities(
    text: str, entities: list[MessageEntity]
) -> Iterator[MessageEntity]:
    code_ranges = [
        (entity.offset, entity.offset + entity.length)
        for entity in entities
        if entity.type in {"code", "pre"}
    ]
    for entity in entities:
        if entity.type in {"code", "pre"}:
            yield entity
            continue
        length = _u16len(entity.extract_from(text).rstrip())
        ranges = [(entity.offset, entity.offset + length)]
        for code_start, code_end in code_ranges:
            remaining = []
            for start, end in ranges:
                if code_end <= start or code_start >= end:
                    remaining.append((start, end))
                else:
                    if start < code_start:
                        remaining.append((start, code_start))
                    if code_end < end:
                        remaining.append((code_end, end))
            ranges = remaining
        for start, end in ranges:
            if end > start:
                yield entity.model_copy(update={"offset": start, "length": end - start})


def render_markdown(markdown: str) -> Iterator[tuple[str, list[MessageEntity]]]:
    """Render once, then preserve UTF-16 entity spans across Telegram-sized chunks."""
    builder = _Builder()
    _blocks(builder, cast("list[Token]", _MARKDOWN(markdown)))
    text = "".join(builder.parts).rstrip("\n")
    entities = sorted(
        _compatible_entities(text, builder.entities),
        key=lambda entity: (entity.offset, -entity.length),
    )
    offset = 0
    for chunk in text_chunks(text):
        end = offset + _u16len(chunk)
        chunk_entities = []
        for entity in entities:
            start, stop = (
                max(offset, entity.offset),
                min(end, entity.offset + entity.length),
            )
            if stop > start:
                chunk_entities.append(
                    entity.model_copy(
                        update={"offset": start - offset, "length": stop - start}
                    )
                )
        yield chunk, chunk_entities
        offset = end
