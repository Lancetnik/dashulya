import random
import sqlite3
from contextlib import closing

import pytest

from dashulya.database import Database
from dashulya.memory.lexical import LexicalIndex


@pytest.mark.usefixtures("_setup")
class TestLexicalIndex:
    @pytest.fixture
    def _setup(self, database: Database) -> None:
        self.database = database

    async def test_long_tokens_remain_searchable_beyond_postgres_index_tuple_limit(
        self,
    ) -> None:
        index = LexicalIndex(self.database)
        text = "prefix" + "x" * 12000
        await index.prepare([text])
        keys = [index.key(text)]
        assert await index.search(keys, keys, ["prefix"], limit=1) == keys

    async def test_postgres_bm25_matches_sqlite_reference_over_authorized_corpus(
        self,
        subtests: pytest.Subtests,
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
        with closing(sqlite3.connect(":memory:")) as reference:
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
                    with subtests.test(prefixes=prefixes, eligible=eligible):
                        expected = [keys[i] for i in ranked if i in eligible]
                        assert (
                            await index.search(
                                keys, [keys[i] for i in eligible], prefixes, limit=160
                            )
                            == expected
                        )
