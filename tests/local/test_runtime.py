import json
from contextlib import AsyncExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest

from dashulya.main.config import Settings
from dashulya.main.runtime import main
from dashulya.observability import JsonLogFormatter
from tests.support import TEST_API_KEY


@pytest.mark.usefixtures("_setup")
class TestEntrypoint:
    @pytest.fixture
    def _setup(self, cleanup: AsyncExitStack) -> None:
        self.cleanup = cleanup
        directory = TemporaryDirectory()
        self.cleanup.callback(directory.cleanup)
        self.settings = Settings(
            bot_token="123:TEST",
            openai_api_key=TEST_API_KEY,
            owner_user_id=42,
            exa_api_key="exa-test-only",  # pragma: allowlist secret -- Fake test key.
            database_url="postgresql://user:synthetic-password@localhost/test",  # pragma: allowlist secret -- Fake test database.
            log_path=Path(directory.name) / "bot.log",
        )

    def test_invalid_settings_exit_before_logging_or_startup(self) -> None:
        with (
            patch(
                "dashulya.main.runtime.load_settings",
                side_effect=ValueError("Invalid settings"),
            ),
            patch("dashulya.main.runtime.configure_logging") as logging,
            patch("dashulya.main.runtime.run", new_callable=AsyncMock) as run,
            pytest.raises(SystemExit) as raised,
        ):
            main()
        assert str(raised.value) == "Invalid settings"
        assert raised.value.__suppress_context__
        logging.assert_not_called()
        run.assert_not_awaited()

    def test_success_and_keyboard_interrupt_report_normal_shutdown(
        self, caplog: pytest.LogCaptureFixture, subtests: pytest.Subtests
    ) -> None:
        for error in (None, KeyboardInterrupt()):
            caplog.clear()
            with (
                subtests.test(error=error),
                patch("dashulya.main.runtime.load_settings", return_value=self.settings),
                patch("dashulya.main.runtime.configure_logging") as logging,
                patch(
                    "dashulya.main.runtime.run", new_callable=AsyncMock, side_effect=error
                ) as run,
                caplog.at_level("INFO", logger="dashulya.main.runtime"),
            ):
                main()
            assert any(
                record.name == "dashulya.main.runtime"
                or record.name.startswith("dashulya.main.runtime.")
                for record in caplog.records
            )
            run.assert_awaited_once_with(self.settings)
            logging.assert_called_once_with(
                self.settings.log_path,
                secrets=(
                    self.settings.bot_token,
                    TEST_API_KEY,
                    self.settings.exa_api_key,
                    self.settings.database_url,
                    "synthetic-password",
                ),
            )
            assert [record.getMessage() for record in caplog.records] == ["Bot stopped"]

    def test_unexpected_failure_exits_with_safe_log_location(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.clear()
        with (
            patch("dashulya.main.runtime.load_settings", return_value=self.settings),
            patch("dashulya.main.runtime.configure_logging"),
            patch(
                "dashulya.main.runtime.run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("PRIVATE PROVIDER BODY"),
            ) as run,
            caplog.at_level("ERROR", logger="dashulya.main.runtime"),
            pytest.raises(SystemExit) as raised,
        ):
            main()
        assert any(
            record.name == "dashulya.main.runtime"
            or record.name.startswith("dashulya.main.runtime.")
            for record in caplog.records
        )
        run.assert_awaited_once_with(self.settings)
        assert (
            str(raised.value)
            == f"Bot stopped unexpectedly. See {self.settings.log_path}."
        )
        assert raised.value.__suppress_context__
        assert [record.getMessage() for record in caplog.records] == [
            "Bot stopped unexpectedly"
        ]
        assert vars(caplog.records[0]).get("error_type") == "RuntimeError"
        serialized = JsonLogFormatter().format(caplog.records[0])
        assert [error["type"] for error in json.loads(serialized)["exceptions"]] == [
            "RuntimeError"
        ]
        assert "PRIVATE PROVIDER BODY" not in serialized
