import json
import logging
from contextlib import AsyncExitStack
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx2
import pytest
from openai import BadRequestError

from dashulya.observability import JsonLogFormatter, configure_logging, exception_fields


@pytest.mark.usefixtures("_setup")
class TestLogging:
    @pytest.fixture
    def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        self.directory = TemporaryDirectory()
        self.cleanup.callback(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nested" / "dashulya.log"
        self.root = logging.getLogger(f"test-observability-{id(self)}")
        self.root.propagate = False
        root_patch = patch.object(logging, "root", self.root)
        root_patch.start()
        self.cleanup.callback(root_patch.stop)
        self.cleanup.callback(self.close_handlers)

    def close_handlers(self) -> None:
        for handler in self.root.handlers:
            handler.close()

    def test_file_and_console_include_utc_context_and_redact_secrets(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        token = "123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN"
        configure_logging(self.path, secrets=(token, "test-api-key"))
        self.root.info("Started %s %s", token, "test-api-key", extra={"chat_id": 42})
        self.root.info("Unknown credential sk-other-secret Bearer opaque-credential")
        lines = self.path.read_text().splitlines()
        event = json.loads(lines[0])
        assert event["level"] == "INFO"
        assert event["chat_id"] == 42
        assert event["timestamp"].endswith("+00:00")
        for output in (self.path.read_text(), capsys.readouterr().err):
            assert token not in output
            assert "test-api-key" not in output
            assert "sk-other-secret" not in output
            assert "opaque-credential" not in output
            assert "<REDACTED>" in output
        assert logging.getLogger("openai").level == logging.WARNING

    def test_api_error_preserves_diagnostics_without_message_or_payload(self) -> None:
        configure_logging(self.path)
        private_text = "PRIVATE CONVERSATION CONTENT"
        response = httpx2.Response(
            400,
            request=httpx2.Request("POST", "https://api.openai.com/v1/responses"),
            headers={"x-request-id": "req-test-123"},
        )
        try:
            raise BadRequestError(
                private_text,
                response=response,
                body={
                    "message": private_text,
                    "code": "invalid_value",
                    "param": "reasoning.effort",
                },
            )
        except BadRequestError as error:
            self.root.warning("AI reply failed", extra=exception_fields(error))
        event = json.loads(self.path.read_text())
        assert event["error_type"] == "BadRequestError"
        assert event["status_code"] == 400
        assert event["error_param"] == "reasoning.effort"
        assert event["error_code"] == "invalid_value"
        assert event["request_id"] == "req-test-123"
        assert event["exceptions"][0]["frames"]
        assert private_text not in self.path.read_text()

    def test_json_logs_preserve_addressing_decision_reason(self) -> None:
        configure_logging(self.path)
        self.root.info(
            "Group addressing decision completed",
            extra={
                "chat_id": -123,
                "message_id": 42,
                "should_reply": False,
                "addressing_reason": "name_not_found",
            },
        )
        event = json.loads(self.path.read_text())
        assert not event["should_reply"]
        assert event["addressing_reason"] == "name_not_found"

    def test_exception_chains_and_exception_arguments_hide_contents(self) -> None:
        configure_logging(self.path)
        private_text = "PRIVATE USER TEXT"
        provider_body = "SECRET PROVIDER BODY"
        try:
            try:
                raise ValueError(private_text)
            except ValueError as cause:
                raise RuntimeError(provider_body) from cause
        except RuntimeError as error:
            self.root.exception("Unexpected failure: %s", error)
        text = self.path.read_text()
        assert "PRIVATE USER TEXT" not in text
        assert "SECRET PROVIDER BODY" not in text
        event = json.loads(text)
        assert [item["type"] for item in event["exceptions"]] == [
            "RuntimeError",
            "ValueError",
        ]

    def test_json_logs_preserve_requested_and_effective_reply_targets(self) -> None:
        configure_logging(self.path)
        self.root.info(
            "Group reply target selected",
            extra={
                "chat_id": -123,
                "message_id": 5,
                "requested_reply_to_message_id": 999,
                "reply_to_message_id": 5,
                "reply_target_reason": "outside_context",
                "context_message_ids": [1, 5],
            },
        )
        event = json.loads(self.path.read_text())
        assert event["requested_reply_to_message_id"] == 999
        assert event["reply_to_message_id"] == 5
        assert event["reply_target_reason"] == "outside_context"
        assert event["context_message_ids"] == [1, 5]

    def test_rotation_keeps_bounded_backups_with_valid_json_lines(self) -> None:
        configure_logging(self.path, max_bytes=500, backup_count=2)
        for index in range(30):
            self.root.info("Rotation event %s", index)
        files = list(self.path.parent.glob("dashulya.log*"))
        assert len(files) == 3
        assert "29" in json.loads(self.path.read_text().splitlines()[-1])["message"]
        for path in files:
            for line in path.read_text().splitlines():
                assert "timestamp" in json.loads(line)

    def test_reconfiguration_replaces_handlers_without_duplicates(self) -> None:
        configure_logging(self.path)
        previous = next(
            handler
            for handler in self.root.handlers
            if isinstance(handler, RotatingFileHandler)
        )
        configure_logging(self.path)
        self.root.info("Only once")
        assert len(self.root.handlers) == 2
        assert previous.stream is None
        assert len(self.path.read_text().splitlines()) == 1

    def test_formatter_does_not_mutate_exception_arguments_for_other_handlers(
        self,
    ) -> None:
        error = RuntimeError("private")
        record = logging.LogRecord(
            "test", logging.WARNING, __file__, 1, "%s", (error,), None
        )
        JsonLogFormatter().format(record)
        assert record.args == (error,)
