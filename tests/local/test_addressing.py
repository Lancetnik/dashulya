import json
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from ag2 import Agent

from dashulya.ai.addressing import GroupAddressing
from dashulya.domain.messages import GroupMessage

MESSAGE = GroupMessage(
    message_id=1,
    text="Hello",
    sent_at=datetime(2026, 10, 25, tzinfo=UTC),
    sender_id=42,
    sender_name="Anna",
)


@pytest.mark.usefixtures("_setup")
class TestGroupAddressing:
    @pytest.fixture
    def _setup(self) -> None:
        self.agent = AsyncMock(spec=Agent)
        self.agent.ask.return_value.content = AsyncMock(return_value=True)
        self.addressing = GroupAddressing(
            self.agent, bot_id=123, bot_username="dashulya_bot"
        )

    async def test_direct_mentions_and_replies_work_without_classifier(
        self, subtests: pytest.Subtests
    ) -> None:
        self.agent.ask.side_effect = RuntimeError("Classifier unavailable")
        for message in (
            replace(MESSAGE, mentioned_usernames=("DASHULYA_BOT",)),
            replace(MESSAGE, mentioned_user_ids=(123,)),
            replace(
                MESSAGE, reply_to=replace(MESSAGE, sender_id=123, sender_is_bot=True)
            ),
        ):
            with subtests.test(message=message):
                assert await self.addressing.should_reply(message, recent_conversation=[])
        self.agent.ask.assert_not_awaited()

    async def test_unrelated_text_and_other_bots_do_not_call_classifier(
        self, subtests: pytest.Subtests
    ) -> None:
        for message in (
            MESSAGE,
            replace(MESSAGE, mentioned_usernames=("dashulya_bot_other",)),
            replace(MESSAGE, mentioned_user_ids=(456,)),
            replace(
                MESSAGE, reply_to=replace(MESSAGE, sender_id=456, sender_is_bot=True)
            ),
            replace(
                MESSAGE, reply_to=replace(MESSAGE, sender_id=123, sender_is_bot=False)
            ),
        ):
            with subtests.test(message=message):
                assert not await self.addressing.should_reply(
                    message, recent_conversation=[]
                )
        self.agent.ask.assert_not_awaited()

    async def test_name_decision_receives_current_message_and_recent_context(
        self,
    ) -> None:
        recent = [{"role": "assistant", "text": "An earlier answer"}]
        message = replace(MESSAGE, text="Dashulya, help", reply_to=MESSAGE)
        assert await self.addressing.should_reply(message, recent_conversation=recent)
        evidence = json.loads(self.agent.ask.call_args.args[0])
        assert evidence["current_message"] == message.context()
        assert evidence["recent_conversation"] == recent
        assert "stream" not in self.agent.ask.call_args.kwargs

    async def test_classifier_errors_propagate_to_the_delivery_handler(self) -> None:
        self.agent.ask.side_effect = TimeoutError
        with pytest.raises(TimeoutError):
            await self.addressing.should_reply(
                replace(MESSAGE, text="Dashulya, help"), recent_conversation=[]
            )

    async def test_vocative_name_reaches_classifier_in_any_case(
        self, subtests: pytest.Subtests
    ) -> None:
        for name in ("дашуль", "Дашуль", "ДАШУЛЬ"):
            with subtests.test(name=name):
                self.agent.ask.reset_mock()
                result = await self.addressing.should_reply(
                    replace(MESSAGE, text=f"{name}, ты теперь и меня игноришь?"),
                    recent_conversation=[],
                )
                self.agent.ask.assert_awaited_once()
                assert result

    async def test_decision_explains_each_direct_and_model_path(
        self, subtests: pytest.Subtests
    ) -> None:
        for message, expected in (
            (MESSAGE, (False, "name_not_found")),
            (
                replace(MESSAGE, mentioned_usernames=("DASHULYA_BOT",)),
                (True, "explicit_mention"),
            ),
            (
                replace(
                    MESSAGE, reply_to=replace(MESSAGE, sender_id=123, sender_is_bot=True)
                ),
                (True, "reply_to_bot"),
            ),
        ):
            with subtests.test(expected=expected):
                decision = await self.addressing.decide(message, recent_conversation=[])
                assert (decision.should_reply, decision.reason) == expected
        self.agent.ask.assert_not_awaited()
        for answer, reason in (
            (True, "classifier_accepted"),
            (False, "classifier_rejected"),
            (None, "classifier_no_answer"),
        ):
            with subtests.test(answer=answer):
                self.agent.ask.return_value.content.return_value = answer
                decision = await self.addressing.decide(
                    replace(MESSAGE, text="дашуль, помоги"), recent_conversation=[]
                )
                assert decision.should_reply == (answer is True)
                assert decision.reason == reason
