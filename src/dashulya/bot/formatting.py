import re
from collections.abc import Iterator

from aiogram.types import InputRichMessage, MessageEntity
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
_RICH_MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable([
    "table",
    "strikethrough",
])
_STYLES: dict[str, type[Text]] = {"strong": Bold, "em": Italic, "s": Strikethrough}
_RICH_FORMATS = {
    "strong",
    "em",
    "s",
    "code_inline",
    "fence",
    "code_block",
    "link",
    "heading",
    "bullet_list",
    "ordered_list",
    "blockquote",
    "table",
    "hr",
}
_RICH_BLOCKS = {
    "paragraph",
    "heading",
    "fence",
    "code_block",
    "bullet_list",
    "ordered_list",
    "list_item",
    "blockquote",
    "table",
    "tr",
    "hr",
}


def rich_message(markdown: str) -> InputRichMessage | None:
    # Native Rich Markdown supports 32768 characters, 500 blocks, and 16 levels.
    if not markdown.strip() or len(markdown) > 32768:
        return None
    nodes = list(SyntaxTreeNode(_RICH_MARKDOWN.parse(markdown)).walk())
    stack: list[str] = []
    html_blocks = 0
    formatted = any(node.type in _RICH_FORMATS for node in nodes)
    for node in nodes:
        if node.type == "image" or sum(1 for _ in _ancestors(node)) > 16:
            return None
        if node.type == "tr" and len(node.children) > 20:
            return None
        if node.type != "text":
            continue
        for tag in re.findall(r"<[^>\n]*>", node.content):
            match = re.fullmatch(r"<(\/)?(details|summary|u|ins|sub|sup)( open)?>", tag)
            if match is None:
                return None
            closing, name, attribute = match.groups()
            if attribute and (closing or name != "details"):
                return None
            formatted = True
            if closing:
                if not stack or stack.pop() != name:
                    return None
            else:
                stack.append(name)
                html_blocks += name in {"details", "summary"}
                if len(stack) > 16:
                    return None
    if stack or not formatted:
        return None
    if sum(node.type in _RICH_BLOCKS for node in nodes) + html_blocks > 500:
        return None
    return InputRichMessage(markdown=markdown)


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
