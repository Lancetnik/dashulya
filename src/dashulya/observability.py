import json
import logging
import os
import re
import traceback
from collections.abc import Iterable
from copy import copy
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_CONTEXT_FIELDS = (
    "chat_id",
    "message_id",
    "duration_ms",
    "error_type",
    "status_code",
    "error_code",
    "error_param",
    "request_id",
    "model",
    "should_reply",
    "addressing_reason",
    "requested_reply_to_message_id",
    "reply_to_message_id",
    "reply_target_reason",
    "context_message_ids",
    "exceptions",
)


def _exception_trace(error: BaseException) -> list[dict[str, Any]]:
    current: BaseException | None = error
    exceptions = []
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        exceptions.append({
            "type": type(current).__name__,
            "frames": [
                {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(current.__traceback__)
            ],
        })
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return exceptions


def exception_fields(error: BaseException) -> dict[str, object]:
    fields: dict[str, object] = {
        "error_type": type(error).__name__,
        "exceptions": _exception_trace(error),
    }
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        fields["status_code"] = status
    for attribute, field in (
        ("code", "error_code"),
        ("param", "error_param"),
        ("request_id", "request_id"),
    ):
        value = getattr(error, attribute, None)
        # Provider metadata is useful; free-form error messages may contain user data.
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,200}", value):
            fields[field] = value
    return fields


class JsonLogFormatter(logging.Formatter):
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = sorted(
            {secret for secret in secrets if secret}, key=len, reverse=True
        )

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "<REDACTED>")
        text = re.sub(r"sk-[A-Za-z0-9_.*-]+", "<REDACTED>", text)
        text = re.sub(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b", "<REDACTED>", text)
        return re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer <REDACTED>", text)

    def format(self, record: logging.LogRecord) -> str:
        safe_record = copy(record)
        if isinstance(record.args, tuple):
            safe_record.args = tuple(
                type(arg).__name__ if isinstance(arg, BaseException) else arg
                for arg in record.args
            )
        elif isinstance(record.args, dict):
            safe_record.args = {
                key: type(arg).__name__ if isinstance(arg, BaseException) else arg
                for key, arg in record.args.items()
            }
        if isinstance(safe_record.msg, BaseException):
            safe_record.msg = type(safe_record.msg).__name__
        event: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "pid": os.getpid(),
            "message": safe_record.getMessage(),
        }
        for field in _CONTEXT_FIELDS:
            if hasattr(record, field):
                event[field] = getattr(record, field)
        if record.exc_info and record.exc_info[1] is not None:
            event["exceptions"] = _exception_trace(record.exc_info[1])
        return self._redact(json.dumps(event, ensure_ascii=False))


def configure_logging(
    path: Path,
    *,
    secrets: Iterable[str] = (),
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = JsonLogFormatter(secrets)
    file_handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True,
    )
    # SDK debug logs may include HTTP request bodies and credentials.
    for name in ("openai", "httpx", "httpx2", "httpcore", "httpcore2"):
        logging.getLogger(name).setLevel(logging.WARNING)
