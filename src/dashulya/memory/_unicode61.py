"""SQLite Unicode 6.1 token categories and folding without a runtime SQLite store.

unicode61.json contains separator boundaries and non-identity folds generated from
SQLite's public-domain ext/fts5/fts5_unicode2.c, version-3.46.1. Category zero
(unassigned in Unicode 6.1) is a token character, including many newer emoji.
"""

import json
from bisect import bisect_right
from pathlib import Path

_TABLES = json.loads(Path(__file__).with_name("unicode61.json").read_text())
_SEPARATORS: list[int] = _TABLES["separators"]
_FOLD: dict[int, str] = {
    int(code): chr(folded) for code, folded in _TABLES["fold"].items()
}


def tokens(text: str) -> list[str]:
    result = []
    current: list[str] = []
    for character in text:
        code = ord(character)
        if not bisect_right(_SEPARATORS, code) % 2:
            current.append(_FOLD.get(code, character))
        elif (
            current
            and 768 <= code <= 817
            and ((0x08029FDF if code < 800 else 0x000361F8) & (1 << ((code - 768) % 32)))
        ):
            # FTS5 ignores these combining marks inside any token.
            continue
        elif current:
            result.append("".join(current))
            current = []
    if current:
        result.append("".join(current))
    return result
