"""PostgreSQL postings and BM25 over the caller-authorized document snapshot."""

import hashlib
from collections import Counter
from collections.abc import Sequence

from dashulya.database import Database
from dashulya.memory._unicode61 import tokens


class LexicalIndex:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(("unicode61-v1:" + text).encode()).hexdigest()

    async def prepare(self, texts: Sequence[str]) -> None:
        keyed = {self.key(text): text for text in texts}
        if not keyed:
            return
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                "SELECT document_key FROM lexical_documents WHERE document_key=ANY(%s)",
                (list(keyed),),
            )
            saved = {row["document_key"] for row in await cursor.fetchall()}
            for key in keyed.keys() - saved:
                frequencies = Counter(tokens(keyed[key]))
                async with connection.transaction():
                    inserted = await connection.execute(
                        "INSERT INTO lexical_documents VALUES (%s,%s) ON CONFLICT DO NOTHING RETURNING document_key",
                        (key, sum(frequencies.values())),
                    )
                    if await inserted.fetchone() is None:
                        continue
                    async with connection.cursor() as writer:
                        await writer.executemany(
                            "INSERT INTO lexical_terms VALUES (%s,%s,%s)",
                            [(key, term, count) for term, count in frequencies.items()],
                        )

    async def search(
        self,
        corpus: Sequence[str],
        eligible: Sequence[str],
        prefixes: Sequence[str],
        *,
        limit: int,
    ) -> list[str]:
        normalized = [term for prefix in prefixes for term in tokens(prefix)]
        if not corpus or not eligible or not normalized:
            return []
        async with self.database.connection() as connection:
            cursor = await connection.execute(
                """WITH corpus AS MATERIALIZED (
                    SELECT d.*,a.position FROM lexical_documents d
                    JOIN unnest(%(corpus)s::text[]) WITH ORDINALITY a(document_key,position)
                        ON d.document_key=a.document_key
                ), stats AS (
                    SELECT count(*)::float8 AS n,avg(word_count)::float8 AS avgdl FROM corpus
                ), phrases AS (
                    SELECT * FROM unnest(%(prefixes)s::text[]) WITH ORDINALITY q(prefix,phrase)
                ), matches AS MATERIALIZED (
                    SELECT t.document_key,q.phrase,sum(t.frequency)::float8 AS tf
                    FROM lexical_terms t JOIN corpus c USING(document_key)
                    JOIN phrases q ON t.term LIKE q.prefix || '%%'
                    GROUP BY t.document_key,q.phrase
                ), frequencies AS (
                    SELECT phrase,count(*)::float8 AS df FROM matches GROUP BY phrase
                ), scores AS (
                    SELECT m.document_key,c.position,
                        sum((CASE WHEN ln((s.n-f.df+0.5)/(f.df+0.5))<=0 THEN 1e-6
                            ELSE ln((s.n-f.df+0.5)/(f.df+0.5)) END)
                            *m.tf*2.2/(m.tf+1.2*(0.25+0.75*c.word_count/NULLIF(s.avgdl,0)))) AS score
                    FROM matches m JOIN corpus c USING(document_key)
                    JOIN frequencies f USING(phrase) CROSS JOIN stats s
                    GROUP BY m.document_key,c.position
                ) SELECT document_key FROM scores WHERE document_key=ANY(%(eligible)s)
                    ORDER BY score DESC,position LIMIT %(limit)s""",
                {
                    "corpus": list(corpus),
                    "eligible": list(eligible),
                    "prefixes": normalized,
                    "limit": limit,
                },
            )
            return [row["document_key"] for row in await cursor.fetchall()]
