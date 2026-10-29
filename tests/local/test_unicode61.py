import random
import sqlite3
from contextlib import closing

from dashulya.memory._unicode61 import tokens


class TestUnicode61:
    def test_unicode61_categories_folding_and_combining_marks_match_reference(
        self,
    ) -> None:
        # Sample the Unicode range independently of the generated lookup tables.
        rng = random.Random(61)
        text = " ".join(
            chr(code)
            for code in [*range(1, 2048), *(rng.randrange(0x110000) for _ in range(4000))]
            if not 0xD800 <= code <= 0xDFFF
        )
        with closing(sqlite3.connect(":memory:")) as reference:
            reference.execute(
                "CREATE VIRTUAL TABLE docs USING fts5(body,tokenize='unicode61 remove_diacritics 2')"
            )
            reference.execute(
                "CREATE VIRTUAL TABLE terms USING fts5vocab(docs,'instance')"
            )
            reference.execute("INSERT INTO docs(body) VALUES (?)", (text,))
            expected = [
                row[0]
                for row in reference.execute("SELECT term FROM terms ORDER BY offset")
            ]
        assert tokens(text) == expected
