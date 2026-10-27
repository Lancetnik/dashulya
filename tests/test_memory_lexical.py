import random
import sqlite3

from dashulya.memory._unicode61 import tokens
from dashulya.memory.lexical import LexicalIndex
from tests.postgres import PostgresTestCase


class LexicalIndexTests(PostgresTestCase):
    async def test_long_tokens_remain_searchable_beyond_postgres_index_tuple_limit(
        self,
    ) -> None:
        index = LexicalIndex(self.database)
        text = "prefix" + "x" * 12000
        await index.prepare([text])
        keys = [index.key(text)]
        self.assertEqual(await index.search(keys, keys, ["prefix"], limit=1), keys)

    async def test_postgres_bm25_matches_sqlite_reference_over_authorized_corpus(
        self,
    ) -> None:
        rng = random.Random(31)
        words = [
            "redis",
            "redirection",
            "postgres",
            "project",
            "café",
            "инструменты",
            "инструментами",
            "bb",
        ]
        texts = [" ".join(rng.choices(words, k=rng.randint(2, 45))) for _ in range(18)]
        texts += [
            "naïve CAFÉ cafe\u0301 Straße STRASSE",
            "Какие инструменты, SQLite_json и BB?",
            "project🤯 tools🤷 привет\u0301 ΙΣΟΣ ὈΔΥΣΣΕΎΣ",
            "\u0301start separate 👩\u200d💻 combining\u036f",
        ]
        index = LexicalIndex(self.database)
        await index.prepare(texts)
        # Unrelated cached documents must not affect corpus df/avgdl.
        await index.prepare(["redis " * 100 + "private unrelated"] * 2)
        keys = [index.key(text) for text in texts]
        with sqlite3.connect(":memory:") as reference:
            reference.execute(
                "CREATE VIRTUAL TABLE docs USING fts5(body,tokenize='unicode61 remove_diacritics 2')"
            )
            reference.executemany(
                "INSERT INTO docs(body) VALUES (?)", [(text,) for text in texts]
            )
            for prefixes in [
                ["red"],
                ["postgres"],
                ["red", "postgres"],
                ["инструмент"],
                ["café"],
                ["naïve"],
                ["strasse"],
                ["project"],
                ["привет"],
            ]:
                expression = " OR ".join('"' + term + '"*' for term in prefixes)
                ranked = [
                    row[0] - 1
                    for row in reference.execute(
                        "SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs)",
                        (expression,),
                    )
                ]
                for eligible in [
                    list(range(len(texts))),
                    list(range(1, 18, 2)),
                    [2, 4, 9],
                ]:
                    with self.subTest(prefixes=prefixes, eligible=eligible):
                        expected = [keys[i] for i in ranked if i in eligible]
                        self.assertEqual(
                            await index.search(
                                keys, [keys[i] for i in eligible], prefixes, limit=160
                            ),
                            expected,
                        )

    async def test_unicode61_categories_folding_and_combining_marks_match_reference(
        self,
    ) -> None:
        # Sample the Unicode range independently of the generated lookup tables.
        rng = random.Random(61)
        text = " ".join(
            chr(code)
            for code in [*range(1, 2048), *(rng.randrange(0x110000) for _ in range(4000))]
            if not 0xD800 <= code <= 0xDFFF
        )
        with sqlite3.connect(":memory:") as reference:
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
        self.assertEqual(tokens(text), expected)
