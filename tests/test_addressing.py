import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

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


class GroupAddressingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.agent = AsyncMock(spec=Agent)
        self.agent.ask.return_value.content = AsyncMock(return_value=True)
        self.addressing = GroupAddressing(
            self.agent, bot_id=123, bot_username="dashulya_bot"
        )

    async def test_direct_mentions_and_replies_work_without_classifier(self) -> None:
        self.agent.ask.side_effect = RuntimeError("Classifier unavailable")
        for message in (
            replace(MESSAGE, mentioned_usernames=("DASHULYA_BOT",)),
            replace(MESSAGE, mentioned_user_ids=(123,)),
            replace(
                MESSAGE, reply_to=replace(MESSAGE, sender_id=123, sender_is_bot=True)
            ),
        ):
            with self.subTest(message=message):
                self.assertTrue(
                    await self.addressing.should_reply(message, recent_conversation=[])
                )
        self.agent.ask.assert_not_awaited()

    async def test_unrelated_text_and_other_bots_do_not_call_classifier(self) -> None:
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
            with self.subTest(message=message):
                self.assertFalse(
                    await self.addressing.should_reply(message, recent_conversation=[])
                )
        self.agent.ask.assert_not_awaited()

    async def test_name_decision_receives_current_message_and_recent_context(
        self,
    ) -> None:
        recent = [{"role": "assistant", "text": "An earlier answer"}]
        message = replace(MESSAGE, text="Dashulya, help", reply_to=MESSAGE)
        self.assertTrue(
            await self.addressing.should_reply(message, recent_conversation=recent)
        )
        evidence = json.loads(self.agent.ask.call_args.args[0])
        self.assertEqual(evidence["current_message"], message.context())
        self.assertEqual(evidence["recent_conversation"], recent)
        self.assertNotIn("stream", self.agent.ask.call_args.kwargs)

    async def test_negative_or_missing_decision_does_not_authorize_reply(self) -> None:
        for decision in (False, None):
            self.agent.ask.return_value.content.return_value = decision
            self.assertFalse(
                await self.addressing.should_reply(
                    replace(MESSAGE, text="Дашуля вчера сказала"), recent_conversation=[]
                )
            )

    async def test_classifier_errors_propagate_to_the_delivery_handler(self) -> None:
        self.agent.ask.side_effect = TimeoutError
        with self.assertRaises(TimeoutError):
            await self.addressing.should_reply(
                replace(MESSAGE, text="Dashulya, help"), recent_conversation=[]
            )
