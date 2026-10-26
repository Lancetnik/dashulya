import os
import unittest
from pathlib import Path
from unittest.mock import patch

from dashulya.main.config import Settings, load_settings

BOT_TOKEN = "123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN"


class SettingsTests(unittest.TestCase):
    @patch("dashulya.main.config.load_dotenv")
    def test_database_configuration_and_secret_redaction(
        self, mock_load_dotenv: unittest.mock.MagicMock
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret -- Synthetic key.
            "OWNER_USER_ID": "42",
            "DATABASE_URL": "postgresql://test:test-only@localhost/dashulya",  # pragma: allowlist secret -- Synthetic database.
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()
            self.assertEqual(settings.database_url, env["DATABASE_URL"])
            self.assertNotIn("test-only", repr(settings))
            for invalid in ("", "sqlite:///old.db", "not-a-url", "postgresql://host"):
                os.environ["DATABASE_URL"] = invalid
                with self.assertRaisesRegex(ValueError, "DATABASE_URL"):
                    load_settings()

    @patch("dashulya.main.config.load_dotenv")
    def test_missing_openai_key_has_actionable_error(
        self, mock_load_dotenv: unittest.mock.MagicMock
    ) -> None:
        with (
            patch.dict(os.environ, {"BOT_TOKEN": BOT_TOKEN}, clear=True),
            self.assertRaisesRegex(ValueError, "Set OPENAI_API_KEY"),
        ):
            load_settings()

    @patch("dashulya.main.config.load_dotenv")
    def test_model_defaults_to_luna_and_can_be_overridden(
        self, mock_load_dotenv: unittest.mock.MagicMock
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": " test-key ",
            "OWNER_USER_ID": "42",
            "DATABASE_URL": "postgresql://localhost/dashulya",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()
            self.assertEqual(settings.openai_model, "gpt-6-luna")
            self.assertEqual(settings.openai_api_key, "test-key")
            self.assertEqual(settings.log_path, Path("var/logs/dashulya.log"))
            os.environ["LOG_PATH"] = "custom/bot.log"
            self.assertEqual(load_settings().log_path, Path("custom/bot.log"))
            os.environ["OPENAI_MODEL"] = "custom-model"
            self.assertEqual(load_settings().openai_model, "custom-model")
            os.environ["OPENAI_MODEL"] = " "
            self.assertEqual(load_settings().openai_model, "gpt-6-luna")
            self.assertEqual(settings.openai_decision_model, "gpt-6-luna")
            os.environ["OPENAI_DECISION_MODEL"] = " decision-model "
            self.assertEqual(load_settings().openai_decision_model, "decision-model")
            os.environ["OPENAI_DECISION_MODEL"] = " "
            self.assertEqual(load_settings().openai_decision_model, "gpt-6-luna")

    def test_settings_repr_hides_both_credentials(self) -> None:
        settings = Settings(
            bot_token=BOT_TOKEN,
            openai_api_key="test-secret",  # pragma: allowlist secret -- Fake credential.
            owner_user_id=42,
        )
        self.assertNotIn(BOT_TOKEN, repr(settings))
        self.assertNotIn("test-secret", repr(settings))

    @patch("dashulya.main.config.load_dotenv")
    def test_local_timezone_defaults_and_validation(
        self, mock_load_dotenv: unittest.mock.MagicMock
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret -- Synthetic key.  # pragma: allowlist secret -- Fake credential.  # pragma: allowlist secret -- Fake credential.
            "OWNER_USER_ID": "42",
            "DATABASE_URL": "postgresql://localhost/dashulya",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(load_settings().bot_timezone, "Europe/Moscow")
            os.environ["BOT_TIMEZONE"] = " America/New_York "
            self.assertEqual(load_settings().bot_timezone, "America/New_York")
            os.environ["BOT_TIMEZONE"] = " "
            self.assertEqual(load_settings().bot_timezone, "Europe/Moscow")
            for value in ("Not/A_Timezone", "/Europe/Moscow"):
                with self.subTest(value=value):
                    os.environ["BOT_TIMEZONE"] = value
                    with self.assertRaisesRegex(ValueError, "BOT_TIMEZONE"):
                        load_settings()

    @patch("dashulya.main.config.load_dotenv")
    def test_owner_id_is_required_and_must_be_positive(
        self, mock_load_dotenv: unittest.mock.MagicMock
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret -- Synthetic key.  # pragma: allowlist secret -- Fake credential.  # pragma: allowlist secret -- Fake credential.
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "Set OWNER_USER_ID"):
                load_settings()
            for value in ("", " ", "0", "-42", "@owner", "42.0", "invalid"):
                with self.subTest(value=value):
                    os.environ["OWNER_USER_ID"] = value
                    with self.assertRaisesRegex(ValueError, "OWNER_USER_ID"):
                        load_settings()
            os.environ["DATABASE_URL"] = "postgresql://localhost/dashulya"
            os.environ["OWNER_USER_ID"] = " 1234567890123 "
            self.assertEqual(load_settings().owner_user_id, 1234567890123)
