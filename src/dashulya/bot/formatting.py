from collections.abc import Iterator

from aiogram.types import MessageEntity
from aiogram.utils.formatting import (
    BlockQuote,
    Bold,
    Code,
    Italic,
    Pre,
    Strikethrough,
    Text,
    TextLink,
    sizeof,
)
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

_MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("strikethrough")
_STYLES: dict[str, type[Text]] = {"strong": Bold, "em": Italic, "s": Strikethrough}


def text_chunks(text: str) -> Iterator[str]:
    # Telegram measures offsets and lengths in UTF-16 units; keep emoji intact.
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


def _ancestors(node: SyntaxTreeNode) -> Iterator[SyntaxTreeNode]:
    parent = node.parent
    while parent is not None:
        yield parent
        parent = parent.parent


def _render(node: SyntaxTreeNode) -> Text:
    if node.type in {"text", "html_inline", "html_block"}:
        return Text(node.content)
    if node.type in {"softbreak", "hardbreak"}:
        return Text("\n")
    if node.type == "code_inline":
        # Telegram cannot combine a code entity with an inline link.
        return (
            Text(node.content)
            if any(parent.type == "link" for parent in _ancestors(node))
            else Code(node.content)
        )
    if node.type in {"fence", "code_block"}:
        language = node.info.split()[0] if node.info.strip() else None
        return Text(Pre(node.content.removesuffix("\n"), language=language), "\n\n")
    if node.type in {"bullet_list", "ordered_list"}:
        start = int(str(node.attrs.get("start", 1)))
        depth = sum(
            parent.type in {"bullet_list", "ordered_list"} for parent in _ancestors(node)
        )
        items = [
            Text(
                "  " * depth,
                f"{index}. " if node.type == "ordered_list" else "• ",
                _render(item),
            )
            for index, item in enumerate(node.children, start=start)
        ]
        return Text(*items, "\n")
    body = [_render(child) for child in node.children]
    if node.type in _STYLES:
        return _STYLES[node.type](*body)
    if node.type == "link":
        return TextLink(*body, url=str(node.attrs["href"]))
    if node.type == "heading":
        return Text(Bold(*body), "\n\n")
    if node.type == "paragraph":
        return Text(*body, "\n" if node.hidden else "\n\n")
    if node.type == "blockquote":
        if any(parent.type == "blockquote" for parent in _ancestors(node)):
            return Text(*body)
        return BlockQuote(*body)
    if node.type == "hr":
        return Text("—\n\n")
    if body:
        return Text(*body)
    return Text(node.content if node.token is not None else "")


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
        # Code/pre cannot overlap any other entity, even when Markdown nests them.
        length = sizeof(entity.extract_from(text).rstrip())
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
    document = SyntaxTreeNode(_MARKDOWN.parse(markdown))
    text, raw_entities = _render(document).render()
    text = text.rstrip("\n")
    entities = sorted(
        _compatible_entities(text, raw_entities), key=lambda entity: entity.offset
    )
    offset = 0
    for chunk in text_chunks(text):
        end = offset + sizeof(chunk)
        chunk_entities = []
        for entity in entities:
            start = max(offset, entity.offset)
            stop = min(end, entity.offset + entity.length)
            if stop > start:
                chunk_entities.append(
                    entity.model_copy(
                        update={"offset": start - offset, "length": stop - start}
                    )
                )
        yield chunk, chunk_entities
        offset = end
