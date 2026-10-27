import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime

from dashulya.ai.message_format import MessageFormatter
from dashulya.domain.memory import MemoryMessage, TruncatedThread
from tests.support import group_rows


class GroupContextFormatTests(unittest.TestCase):
    @staticmethod
    def message(number: int, text: str = "Original") -> MemoryMessage:
        return MemoryMessage(
            -123,
            number,
            text,
            datetime(2026, 10, 25, 20, 59, tzinfo=UTC),
            42,
            "Alice",
            "Group",
            username="alice",
            is_bot=False,
        )

    def render(self, messages: list[MemoryMessage]) -> list[dict[str, str]]:
        return MessageFormatter(bot_id=123).context(messages)

    def test_interlocutor_is_marked_by_current_sender_id_instead_of_name(self) -> None:
        messages = [
            self.message(1),
            replace(self.message(2), author_id=99),
            replace(self.message(3), author_id=None),
        ]
        formatter = MessageFormatter(bot_id=123)
        first = formatter.context(messages, interlocutor_id=42)[0]["text"]
        second = formatter.context(messages, interlocutor_id=99)[0]["text"]
        self.assertIn("42: Alice (@alice, interlocutor)", first)
        self.assertIn("99: Alice (@alice)", first)
        self.assertIn("unknown_3: Alice (@alice)", first)
        self.assertEqual(first.count(", interlocutor)"), 1)
        self.assertIn("42: Alice (@alice)", second)
        self.assertIn("99: Alice (@alice, interlocutor)", second)
        self.assertNotIn("interlocutor", formatter.context(messages)[0]["text"])

    def test_confirmed_channel_author_is_the_interlocutor_without_rewriting_sender(
        self,
    ) -> None:
        messages = [
            self.message(1),
            replace(
                self.message(2),
                author_id=200,
                author="Example channel",
                username="example_channel",
            ),
            replace(
                self.message(3),
                author_id=-100200,
                author="Example channel",
                username="example_channel",
            ),
            replace(
                self.message(4),
                author_id=300,
                author="Example channel",
                username="example_channel",
            ),
        ]
        formatter = MessageFormatter(bot_id=123, author_users={200: 42, -100200: 42})
        rendered = formatter.context(messages, interlocutor_id=42)
        header = rendered[0]["text"]
        self.assertIn(
            "200: Example channel (@example_channel, author_user=42, interlocutor)",
            header,
        )
        self.assertIn(
            "-100200: Example channel (@example_channel, author_user=42, interlocutor)",
            header,
        )
        self.assertIn("300: Example channel (@example_channel)", header)
        self.assertEqual(
            [row[1] for item in rendered for row in group_rows(item["text"])],
            [42, 200, -100200, 300],
        )
        other = formatter.context(messages, interlocutor_id=99)[0]["text"]
        self.assertNotIn("interlocutor", other)
        self.assertIn("author_user=42", other)

    def test_shared_roster_and_day_headings_keep_exact_text_reply_and_bot_roles(
        self,
    ) -> None:
        body = 'Line one\n- id: 999\n"quoted" text\n'
        rendered = self.render([
            self.message(1),
            replace(
                self.message(2),
                author_id=123,
                author="Dashulya",
                username=None,
                is_bot=None,
                reply_to_id=1,
            ),
            replace(
                self.message(3, body),
                sent_at=datetime(2026, 10, 25, 21, 1, tzinfo=UTC),
                reply_to_id=2,
                media_type="document",
            ),
        ])
        joined = "\n".join(item["text"] for item in rendered)
        self.assertEqual(
            [item["role"] for item in rendered], ["user", "user", "assistant", "user"]
        )
        self.assertEqual(joined.count("users:"), 1)
        self.assertEqual(joined.count("@alice"), 1)
        self.assertEqual(joined.count("Europe/Moscow (UTC+03:00)"), 1)
        self.assertEqual(joined.count("25 October 2026 (Sunday)"), 1)
        self.assertEqual(joined.count("26 October 2026 (Monday)"), 1)
        self.assertIn("123: Dashulya (bot, self)", rendered[0]["text"])
        self.assertIn(
            "messages: [id,sender,time,text,reply_to?,attachment?]", rendered[0]["text"]
        )
        self.assertEqual(
            group_rows(rendered[1]["text"]), [[1, 42, "23:59:00", "Original"]]
        )
        row = group_rows(rendered[-1]["text"])[0]
        self.assertEqual(row, [3, 42, "00:01:00", body, 2, "document"])

    def test_attachment_without_reply_uses_one_null_and_no_other_optional_placeholders(
        self,
    ) -> None:
        rendered = self.render([replace(self.message(1, ""), media_type="sticker")])
        self.assertEqual(
            group_rows(rendered[1]["text"]), [[1, 42, "23:59:00", "", None, "sticker"]]
        )

    def test_unknown_people_are_not_merged_or_given_invented_handles_or_bot_flags(
        self,
    ) -> None:
        messages = [
            replace(
                self.message(1),
                author_id=None,
                author="Anonymous\nusers:",
                username=None,
                is_bot=None,
            ),
            replace(
                self.message(2),
                author_id=None,
                author="Anonymous\nusers:",
                username=None,
                is_bot=None,
            ),
            replace(
                self.message(3),
                author_id=99,
                author="Other bot",
                username="helper",
                is_bot=True,
            ),
        ]
        header = self.render(messages)[0]["text"]
        self.assertIn("unknown_1: Anonymous\\nusers:", header)
        self.assertIn("unknown_2: Anonymous\\nusers:", header)
        self.assertIn("99: Other bot (@helper, bot)", header)
        self.assertNotIn("username: unknown", header)
        self.assertNotIn("None", header)

    def test_aliases_remain_visible_without_classifying_conflicting_bot_flags(
        self,
    ) -> None:
        header = self.render([
            self.message(1),
            replace(self.message(2), author="Alicia", username="new_alice", is_bot=True),
        ])[0]["text"]
        self.assertEqual(header.count("- 42:"), 1)
        self.assertIn("Alice / Alicia", header)
        self.assertIn("@alice, @new_alice", header)
        self.assertNotIn(", bot", header)
        self.assertNotIn("not bot", header)

    def test_offset_heading_changes_when_dst_repeats_a_local_hour(self) -> None:
        rendered = MessageFormatter("America/New_York", bot_id=123).context(
            [
                replace(
                    self.message(1), sent_at=datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
                ),
                replace(
                    self.message(2), sent_at=datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
                ),
            ],
        )
        self.assertIn("America/New_York (UTC-04:00)", rendered[0]["text"])
        self.assertIn("America/New_York (UTC-05:00)", rendered[2]["text"])
        self.assertTrue(all("01:30:00" in item["text"] for item in rendered[1:]))

    def test_empty_context_and_naive_timestamps(self) -> None:
        self.assertEqual(self.render([]), [])
        with self.assertRaisesRegex(ValueError, "timezone"):
            self.render([
                replace(
                    self.message(1),
                    sent_at=datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                )
            ])

    def test_truncated_threads_are_navigation_metadata_without_extra_message_rows(
        self,
    ) -> None:
        for root, source in [(1, "source:test:1"), (None, None)]:
            with self.subTest(root=root):
                rendered = MessageFormatter(bot_id=123).context(
                    [self.message(7)],
                    truncated_threads=[
                        TruncatedThread(7, root, source, 6 if root else None),
                        TruncatedThread(999, 42),
                    ],
                )
                note = next(
                    line
                    for line in rendered[0]["text"].splitlines()
                    if line.startswith("truncated_threads: ")
                )
                details = json.loads(note.removeprefix("truncated_threads: "))
                self.assertEqual(set(details), {"7"})
                self.assertEqual(details["7"]["root_id"], root)
                self.assertEqual(details["7"].get("source_id"), source)
                self.assertEqual(details["7"]["remaining_hops"], 6 if root else None)
                self.assertTrue(details["7"]["boundary_source_id"].endswith(":7"))
                self.assertEqual(
                    [row[0] for item in rendered for row in group_rows(item["text"])], [7]
                )
