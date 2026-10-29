import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dashulya.main.config import Settings, load_settings

BOT_TOKEN = "123456789:TEST_ONLY_NOT_A_REAL_BOT_TOKEN"


class TestSettings:
    @patch("dashulya.main.config.load_dotenv")
    def test_public_group_ids_are_explicit_negative_ids(
        self, mock_load_dotenv: MagicMock, subtests: pytest.Subtests
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret
            "OWNER_USER_ID": "42",
            "DATABASE_URL": "postgresql://localhost/dashulya",
        }
        with patch.dict(os.environ, env, clear=True):
            assert load_settings().public_group_ids == ()
            os.environ["PUBLIC_GROUP_IDS"] = " -100200, -300, -100200 "
            assert load_settings().public_group_ids == (-100200, -300)
            for invalid in ("0", "42", "-100200,42", "@chat", "1.5", "-100200,invalid"):
                os.environ["PUBLIC_GROUP_IDS"] = invalid
                with (
                    subtests.test(invalid=invalid),
                    pytest.raises(ValueError, match="PUBLIC_GROUP_IDS"),
                ):
                    load_settings()

    @patch("dashulya.main.config.load_dotenv")
    def test_explicit_owner_author_ids_do_not_change_access_identity(
        self, mock_load_dotenv: MagicMock, subtests: pytest.Subtests
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret -- Synthetic key.
            "OWNER_USER_ID": "42",
            "DATABASE_URL": "postgresql://localhost/dashulya",
        }
        with patch.dict(os.environ, env, clear=True):
            assert load_settings().owner_author_ids == ()
            os.environ["OWNER_AUTHOR_IDS"] = " -100200, 200, -100200 "
            settings = load_settings()
            assert settings.owner_author_ids == (-100200, 200)
            assert settings.owner_user_id == 42
            for invalid in ("@channel", "0", "200,invalid", "1.5"):
                os.environ["OWNER_AUTHOR_IDS"] = invalid
                with (
                    subtests.test(invalid=invalid),
                    pytest.raises(ValueError, match="OWNER_AUTHOR_IDS"),
                ):
                    load_settings()

    @patch("dashulya.main.config.load_dotenv")
    def test_database_configuration_and_secret_redaction(
        self, mock_load_dotenv: MagicMock
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret -- Synthetic key.
            "OWNER_USER_ID": "42",
            "DATABASE_URL": "postgresql://test:test-only@localhost/dashulya",  # pragma: allowlist secret -- Synthetic database.
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()
            assert settings.database_url == env["DATABASE_URL"]
            assert "test-only" not in repr(settings)
            for invalid in ("", "sqlite:///old.db", "not-a-url", "postgresql://host"):
                os.environ["DATABASE_URL"] = invalid
                with pytest.raises(ValueError, match="DATABASE_URL"):
                    load_settings()

    @patch("dashulya.main.config.load_dotenv")
    def test_missing_openai_key_has_actionable_error(
        self, mock_load_dotenv: MagicMock
    ) -> None:
        with (
            patch.dict(os.environ, {"BOT_TOKEN": BOT_TOKEN}, clear=True),
            pytest.raises(ValueError, match="Set OPENAI_API_KEY"),
        ):
            load_settings()

    @patch("dashulya.main.config.load_dotenv")
    def test_model_defaults_to_luna_and_can_be_overridden(
        self, mock_load_dotenv: MagicMock
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": " test-key ",
            "OWNER_USER_ID": "42",
            "DATABASE_URL": "postgresql://localhost/dashulya",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()
            assert settings.openai_model == "gpt-6-luna"
            assert settings.openai_api_key == "test-key"  # pragma: allowlist secret
            assert settings.log_path == Path("var/logs/dashulya.log")
            os.environ["LOG_PATH"] = "custom/bot.log"
            assert load_settings().log_path == Path("custom/bot.log")
            os.environ["OPENAI_MODEL"] = "custom-model"
            assert load_settings().openai_model == "custom-model"
            os.environ["OPENAI_MODEL"] = " "
            assert load_settings().openai_model == "gpt-6-luna"
            assert settings.openai_decision_model == "gpt-6-luna"
            os.environ["OPENAI_DECISION_MODEL"] = " decision-model "
            assert load_settings().openai_decision_model == "decision-model"
            os.environ["OPENAI_DECISION_MODEL"] = " "
            assert load_settings().openai_decision_model == "gpt-6-luna"

    def test_settings_repr_hides_both_credentials(self) -> None:
        settings = Settings(
            bot_token=BOT_TOKEN,
            openai_api_key="test-secret",  # pragma: allowlist secret -- Fake credential.
            owner_user_id=42,
        )
        assert BOT_TOKEN not in repr(settings)
        assert "test-secret" not in repr(settings)

    @patch("dashulya.main.config.load_dotenv")
    def test_local_timezone_defaults_and_validation(
        self, mock_load_dotenv: MagicMock, subtests: pytest.Subtests
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret -- Synthetic key.  # pragma: allowlist secret -- Fake credential.  # pragma: allowlist secret -- Fake credential.
            "OWNER_USER_ID": "42",
            "DATABASE_URL": "postgresql://localhost/dashulya",
        }
        with patch.dict(os.environ, env, clear=True):
            assert load_settings().bot_timezone == "Europe/Moscow"
            os.environ["BOT_TIMEZONE"] = " America/New_York "
            assert load_settings().bot_timezone == "America/New_York"
            os.environ["BOT_TIMEZONE"] = " "
            assert load_settings().bot_timezone == "Europe/Moscow"
            for value in ("Not/A_Timezone", "/Europe/Moscow"):
                with subtests.test(value=value):
                    os.environ["BOT_TIMEZONE"] = value
                    with pytest.raises(ValueError, match="BOT_TIMEZONE"):
                        load_settings()

    @patch("dashulya.main.config.load_dotenv")
    def test_owner_id_is_required_and_must_be_positive(
        self, mock_load_dotenv: MagicMock, subtests: pytest.Subtests
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret -- Synthetic key.  # pragma: allowlist secret -- Fake credential.  # pragma: allowlist secret -- Fake credential.
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Set OWNER_USER_ID"):
                load_settings()
            for value in ("", " ", "0", "-42", "@owner", "42.0", "invalid"):
                with subtests.test(value=value):
                    os.environ["OWNER_USER_ID"] = value
                    with pytest.raises(ValueError, match="OWNER_USER_ID"):
                        load_settings()
            os.environ["DATABASE_URL"] = "postgresql://localhost/dashulya"
            os.environ["OWNER_USER_ID"] = " 1234567890123 "
            assert load_settings().owner_user_id == 1234567890123

    @patch("dashulya.main.config.load_dotenv")
    def test_exa_key_is_optional_trimmed_and_hidden(
        self, mock_load_dotenv: MagicMock
    ) -> None:
        env = {
            "BOT_TOKEN": BOT_TOKEN,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret -- Synthetic key.
            "OWNER_USER_ID": "42",
            "DATABASE_URL": "postgresql://localhost/dashulya",
        }
        with patch.dict(os.environ, env, clear=True):
            assert load_settings().exa_api_key == ""
            os.environ["EXA_API_KEY"] = " exa-test-only "
            settings = load_settings()
            assert settings.exa_api_key == "exa-test-only"  # pragma: allowlist secret
            assert "exa-test-only" not in repr(settings)
            os.environ["EXA_API_KEY"] = " "
            assert load_settings().exa_api_key == ""
