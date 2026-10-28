import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime

from dashulya.ai.message_format import MessageFormatter
from dashulya.domain.memory import MemoryMessage, TruncatedThread
from tests.support import group_records, group_rows


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

    def test_context_supplies_one_citation_template_for_public_and_private_supergroups(
        self,
    ) -> None:
        for username, template in (
            ("fastnewsdev_chat", "https://t.me/fastnewsdev_chat/{message_id}"),
            (None, "https://t.me/c/2391257796/{message_id}"),
        ):
            messages = [
                replace(
                    self.message(number), chat_id=-1002391257796, chat_username=username
                )
                for number in (19715, 19716)
            ]
            with self.subTest(username=username):
                rendered = self.render(messages)
                line = next(
                    (
                        line
                        for line in rendered[0]["text"].splitlines()
                        if line.startswith("chat: ")
                    ),
                    None,
                )
                self.assertIsNotNone(
                    line, "Immediate context lost the chat identity and citation URL"
                )
                assert line is not None
                chat = json.loads(line.removeprefix("chat: "))
                self.assertEqual(chat["id"], -1002391257796)
                self.assertEqual(chat["message_url_template"], template)
                self.assertEqual(
                    "\n".join(item["text"] for item in rendered).count(template), 1
                )
                self.assertEqual(
                    chat["message_url_template"].format(message_id=19715),
                    template.replace("{message_id}", "19715"),
                )

    def test_nonlinkable_chats_do_not_receive_an_invented_message_link(self) -> None:
        for chat_id, username in ((42, "alice"), (-123, None)):
            with self.subTest(chat_id=chat_id):
                rendered = self.render([
                    replace(self.message(1), chat_id=chat_id, chat_username=username)
                ])
                line = next(
                    (
                        line
                        for line in rendered[0]["text"].splitlines()
                        if line.startswith("chat: ")
                    ),
                    None,
                )
                self.assertIsNotNone(line)
                assert line is not None
                self.assertIsNone(
                    json.loads(line.removeprefix("chat: "))["message_url_template"]
                )

    def test_interlocutor_is_marked_by_current_sender_id_instead_of_name(self) -> None:
        messages = [
            self.message(1),
            replace(self.message(2), author_id=99),
            replace(self.message(3), author_id=None),
        ]
        formatter = MessageFormatter(bot_id=123)
        first_records = formatter.context(messages, interlocutor_id=42)
        second_records = formatter.context(messages, interlocutor_id=99)
        first = "\n".join(item["text"] for item in first_records)
        second = "\n".join(item["text"] for item in second_records)
        self.assertEqual(first_records[:-1], second_records[:-1])
        self.assertIn("User 42: Alice (@alice)", first)
        self.assertIn("99: Alice (@alice)", first)
        self.assertIn("unknown_3: Alice (@alice)", first)
        self.assertEqual(first.count("Interlocutor user id: 42"), 1)
        self.assertIn("42: Alice (@alice)", second)
        self.assertIn("Interlocutor user id: 99", second_records[-1]["text"])
        self.assertNotIn("Interlocutor", str(formatter.context(messages)))

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
        header = "\n".join(item["text"] for item in rendered)
        self.assertIn(
            "200: Example channel (@example_channel, author_user=42)",
            header,
        )
        self.assertIn(
            "-100200: Example channel (@example_channel, author_user=42)",
            header,
        )
        self.assertIn("300: Example channel (@example_channel)", header)
        self.assertEqual(
            [row[1] for item in rendered for row in group_rows(item["text"])],
            [42, 200, -100200, 300],
        )
        other = "\n".join(
            item["text"] for item in formatter.context(messages, interlocutor_id=99)
        )
        self.assertIn("Interlocutor user id: 99", other)
        self.assertNotIn("Interlocutor user id: 42", other)
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
        self.assertNotIn("users:", joined)
        self.assertEqual(joined.count("@alice"), 1)
        self.assertNotIn("Europe/Moscow", joined)
        self.assertNotIn("UTC+03:00", joined)
        self.assertEqual(joined.count("25 October 2026 (Sunday)"), 1)
        self.assertEqual(joined.count("26 October 2026 (Monday)"), 1)
        self.assertNotIn("User 123:", joined)
        self.assertNotIn("User 42:", rendered[0]["text"])
        self.assertIn(
            "25 October 2026 (Sunday)\nUser 42: Alice (@alice)\n", rendered[1]["text"]
        )
        self.assertNotIn(
            "[id,sender,time,text,reply_to?,attachment?]", rendered[0]["text"]
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
        header = "\n".join(item["text"] for item in self.render(messages))
        self.assertIn("unknown_1: Anonymous\\nusers:", header)
        self.assertIn("unknown_2: Anonymous\\nusers:", header)
        self.assertIn("99: Other bot (@helper, bot)", header)
        self.assertNotIn("username: unknown", header)
        self.assertNotIn("None", header)

    def test_aliases_remain_visible_without_classifying_conflicting_bot_flags(
        self,
    ) -> None:
        rendered = self.render([
            self.message(1),
            replace(self.message(2), author="Alicia", username="new_alice", is_bot=True),
        ])
        header = "\n".join(item["text"] for item in rendered)
        self.assertNotIn("Alicia", rendered[1]["text"])
        self.assertEqual(header.count("User 42:"), 2)
        self.assertIn("Alice / Alicia", header)
        self.assertIn("@alice, @new_alice", header)
        self.assertNotIn(", bot", header)
        self.assertNotIn("not bot", header)

    def test_repeated_dst_times_keep_offsets_even_when_only_one_is_selected(self) -> None:
        messages = [
            replace(self.message(1), sent_at=datetime(2026, 11, 1, 5, 30, tzinfo=UTC)),
            replace(self.message(2), sent_at=datetime(2026, 11, 1, 6, 30, tzinfo=UTC)),
        ]
        formatter = MessageFormatter("America/New_York", bot_id=123)
        for selected in [messages, messages[:1], messages[1:]]:
            with self.subTest(ids=[message.message_id for message in selected]):
                rendered = formatter.context(selected)
                self.assertNotIn(
                    "America/New_York", "\n".join(item["text"] for item in rendered)
                )
                rows = [row for item in rendered for row in group_rows(item["text"])]
                self.assertEqual(
                    [row[2] for row in rows],
                    [
                        "01:30:00 (UTC-04:00)"
                        if message.message_id == 1
                        else "01:30:00 (UTC-05:00)"
                        for message in selected
                    ],
                )

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
                self.assertNotIn("truncated_thread", rendered[0]["text"])
                details = group_records(rendered[1]["text"])[0]["truncated_thread"]
                self.assertEqual(details["root_id"], root)
                self.assertEqual(details.get("source_id"), source)
                self.assertEqual(details["remaining_hops"], 6 if root else None)
                self.assertTrue(details["boundary_source_id"].endswith(":7"))
                self.assertEqual(
                    [row[0] for item in rendered for row in group_rows(item["text"])], [7]
                )

    def test_later_authors_dates_aliases_and_interlocutor_preserve_earlier_prefix(
        self,
    ) -> None:
        formatter = MessageFormatter(bot_id=123)
        first = self.message(1, "First statement")
        old_current = self.message(2, "Previous request")
        earlier = formatter.context([first, old_current], interlocutor_id=42)
        for added in [
            replace(self.message(3), author_id=99, author="Bob", username="bob"),
            replace(self.message(3), author="Alicia", username="new_alice"),
            replace(self.message(3), sent_at=datetime(2026, 10, 26, 21, tzinfo=UTC)),
        ]:
            with self.subTest(added=added):
                later = formatter.context([first, old_current, added], interlocutor_id=99)
                self.assertEqual(earlier[:-1], later[: len(earlier) - 1])
                self.assertNotIn("Interlocutor user id:", later[-2]["text"])
                self.assertIn("Interlocutor user id: 99", later[-1]["text"])

    def test_record_matches_requested_layout_and_preserves_multiline_original(
        self,
    ) -> None:
        body = 'Original line\nUser 999: quoted name\n14:00:00 {"id":999}\n'
        rendered = MessageFormatter(bot_id=123).context(
            [self.message(1), replace(self.message(2, body), reply_to_id=1)],
            interlocutor_id=42,
        )
        self.assertEqual(
            rendered[1]["text"],
            '25 October 2026 (Sunday)\nUser 42: Alice (@alice)\n23:59:00 {"id":1,"user_id":42}\nOriginal',
        )
        self.assertEqual(
            rendered[2]["text"],
            'Interlocutor user id: 42\n23:59:00 {"id":2,"user_id":42,"reply_to":1}\n'
            + body,
        )
