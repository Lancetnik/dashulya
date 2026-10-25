"""Offline, reproducible resource export; never modify the maintained index."""

import hashlib
import json
from contextlib import closing
from pathlib import Path
from typing import Any

from dashulya.memory._database import _connect
from dashulya.memory.openviking import index_digest


def export_resources(index: Path, destination: Path, *, root_uri: str) -> Path:  # ruff: ignore[too-many-locals] -- Keep the export and its exact URI map together.
    """Export all record kinds, bounded discussion directories and a citation map."""
    if not root_uri.startswith("viking://") or root_uri.endswith("/"):
        error_message = (
            "Use an exact viking:// resource directory URI without a trailing slash."
        )
        raise ValueError(error_message)
    destination.mkdir(parents=True, exist_ok=False)
    resources = destination / "resources"
    resources.mkdir()
    digest = index_digest(index)
    with closing(_connect(index)) as connection:
        nodes = connection.execute("SELECT * FROM memories ORDER BY id").fetchall()
        messages = {
            row["id"]: dict(row) for row in connection.execute("SELECT * FROM messages")
        }
        sources: dict[int, list[int]] = {}
        for row in connection.execute("SELECT * FROM sources ORDER BY message_id"):
            sources.setdefault(row["memory_id"], []).append(row["message_id"])
    records: dict[str, str] = {}
    conversations = {node["key"]: node["conversation"] for node in nodes}
    folders: dict[int, str] = {}
    for node in nodes:
        if node["kind"] in {"discussion", "window"}:
            folder = f"discussion-{node['id']}"
            records[f"{root_uri}/{folder}"] = node["key"]
            folders[node["id"]] = folder
            for message_id in sources[node["id"]]:
                folders[message_id] = folder
    for node in nodes:
        key = node["key"]
        name = hashlib.sha256(key.encode()).hexdigest()[:24]
        folder = folders.get(node["id"], f"record-{name}")
        directory = resources / folder
        directory.mkdir(exist_ok=True)
        records.setdefault(f"{root_uri}/{folder}", key)
        records[f"{root_uri}/{folder}/{name}.md"] = key
        evidence = [messages[source] for source in sources[node["id"]]]
        text = (
            f"# Telegram {node['kind']}: {key}\n\n"
            f"Conversation: {node['conversation']}\n\n"
            f"{node['body']}\n\n## Original sources\n\n"
            + "\n\n".join(
                f"Message ID: {row['id']}\nSent at: {row['sent_at']}\n"
                f"Author: {row['author']} (user_id={row['user_id']})\n"
                f"Reply to: {row['reply_to_id']}\nURL: {row['url']}\n\n{row['body']}"
                for row in evidence
            )
        )
        (directory / f"{name}.md").write_text(text + "\n")
    if digest != index_digest(index):
        error_message = "Index changed during export; retry from a stable index."
        raise ValueError(error_message)
    manifest: dict[str, Any] = {
        "version": 1,
        "index_sha256": digest,
        "root_uri": root_uri,
        "records": records,
        "conversations": conversations,
        "message_count": len(messages),
        "memory_count": len(nodes),
    }
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path


def reconcile_resources(manifest: Path, entries: list[dict[str, Any]]) -> dict[str, int]:
    """Map actual parser output, including wrapped files and split chunks, to sources."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    exported: dict[str, str] = data.get("exported_records", data["records"])
    anchors = {
        uri.removesuffix(".md"): key
        for uri, key in exported.items()
        if uri.endswith(".md")
    }
    records: dict[str, str] = {}
    leaf_keys: set[str] = set()
    for entry in entries:
        uri = entry["uri"].rstrip("/")
        key = exported.get(uri)
        ancestor = uri
        while key is None and ancestor.startswith(data["root_uri"] + "/"):
            key = anchors.get(ancestor)
            ancestor = ancestor.rpartition("/")[0]
        if key is not None:
            records[uri] = key
            if not entry["isDir"]:
                leaf_keys.add(key)
    if missing := set(anchors.values()) - leaf_keys:
        error_message = f"OpenViking import is incomplete: {len(missing)} records have no mapped original leaf."
        raise ValueError(error_message)
    data["exported_records"] = exported
    data["records"] = records
    coverage = {"mapped_uris": len(records), "records_with_leaves": len(leaf_keys)}
    data["import_coverage"] = coverage
    manifest.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return coverage
