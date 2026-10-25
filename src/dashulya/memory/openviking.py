"""Semantic candidate retrieval with evidence reconstructed from local SQLite."""

import hashlib
import json
from pathlib import Path
from typing import Any

import openviking_sdk  # type: ignore[import-untyped]

from dashulya.memory._database import _read_connection
from dashulya.memory.search import MemorySearch, validate_search


def index_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class OpenVikingMemory:
    """Use OpenViking L0/L1/L2 candidates without trusting their citation metadata."""

    def __init__(
        self,
        evidence: MemorySearch,
        manifest: Path,
        *,
        url: str,
        api_key: str = "",
    ) -> None:
        self.evidence = evidence
        self.url = url.rstrip("/")
        self.api_key = api_key
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data["version"] != 1 or data["index_sha256"] != index_digest(evidence.path):
            error_message = (
                "OpenViking export does not match MEMORY_PATH; export and ingest again."
            )
            raise ValueError(error_message)
        if not data.get("import_coverage"):
            error_message = "Complete OpenViking ingestion and resource mapping before selecting this backend."
            raise ValueError(error_message)
        self.root: str = data["root_uri"]
        self.records: dict[str, str] = data["records"]
        self.conversations: dict[str, str] = data["conversations"]

    async def check_ready(self) -> None:
        """Require a healthy server and the imported corpus before agent startup."""
        client = openviking_sdk.AsyncHTTPClient(
            url=self.url,
            api_key=self.api_key,
            timeout=10,
            account="default",
            user="default",
        )
        try:
            try:
                await client.initialize()
                if not await client.health():
                    raise RuntimeError
                await client.stat(self.root)
            finally:
                await client.close()
        except Exception:  # Redact SDK error bodies and credentials.
            error_message = (
                "OpenViking is unavailable or its imported corpus is missing. "
                "Start docker compose up -d --wait openviking and check the import and service key."
            )
            raise RuntimeError(error_message) from None

    async def search(
        self,
        query: str = "",
        *,
        memory_id: str | None = None,
        author_id: int | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 3,
    ) -> dict[str, Any]:
        _, after, before = validate_search(query, memory_id, after, before, limit)
        if memory_id:
            return await self.evidence.search(
                query,
                memory_id=memory_id,
                author_id=author_id,
                after=after,
                before=before,
                limit=limit,
            )
        eligible = await self._eligible(author_id, after, before)
        uris = [uri for uri, key in self.records.items() if key in eligible]
        result: dict[str, Any] = {"coverage": self.evidence.coverage, "hits": []}
        if not uris:
            return result
        try:
            client = openviking_sdk.AsyncHTTPClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30,
                account="default",
                user="default",
            )
            try:
                await client.initialize()
                candidates = (
                    await client.find(
                        query,
                        target_uri=self.root,
                        limit=100,
                        options={
                            "filter": {
                                "op": "must",
                                "field": "uri",
                                "conds": uris,
                                "para": "-d=0",
                            },
                            "include_provenance": True,
                        },
                    )
                )["resources"]
            finally:
                await client.close()
            if not isinstance(candidates, list) or any(
                not isinstance(candidate, dict)
                or not isinstance(candidate.get("uri"), str)
                for candidate in candidates
            ):
                raise TypeError
        except Exception:  # Redact SDK error bodies and credentials.
            error_message = (
                "OpenViking search failed; check the local server and imported corpus."
            )
            raise RuntimeError(error_message) from None
        seen: set[str] = set()
        for candidate in candidates:
            key = self.records.get(candidate["uri"].rstrip("/"))
            if key not in eligible or key is None:
                continue
            conversation = self.conversations[key]
            if conversation in seen:
                continue
            expanded = (
                await self.evidence.search(
                    memory_id=key,
                    author_id=author_id,
                    after=after,
                    before=before,
                )
            )["hits"]
            if not expanded:
                continue
            seen.add(conversation)
            result["hits"].extend(expanded)
            if len(result["hits"]) == limit:
                break
        return result

    async def _eligible(
        self, author: int | None, after: str | None, before: str | None
    ) -> set[str]:
        async with _read_connection(self.evidence.path) as connection:
            rows = await connection.execute_fetchall(
                """
                SELECT n.key FROM memories n WHERE NOT EXISTS (
                    SELECT 1 FROM sources s JOIN messages m ON m.id = s.message_id
                    WHERE s.memory_id = n.id AND (
                        (? IS NOT NULL AND (m.user_id IS NULL OR m.user_id != ?))
                        OR (? IS NOT NULL AND (m.sent_at IS NULL OR julianday(m.sent_at) < julianday(?)))
                        OR (? IS NOT NULL AND (m.sent_at IS NULL OR julianday(m.sent_at) > julianday(?)))
                    )
                )
            """,
                (author, author, after, after, before, before),
            )
            return {row["key"] for row in rows}
