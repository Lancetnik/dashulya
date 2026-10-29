import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from dashulya.ai.message_format import MessageFormatter
from dashulya.domain.memory import MemoryMessage, TruncatedThread
from tests.support import group_records, group_rows


class TestGroupContextFormat:
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
        subtests: pytest.Subtests,
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
            with subtests.test(username=username):
                rendered = self.render(messages)
                line = next(
                    (
                        line
                        for line in rendered[0]["text"].splitlines()
                        if line.startswith("chat: ")
                    ),
                    None,
                )
                assert line is not None, (
                    "Immediate context lost the chat identity and citation URL"
                )
                chat = json.loads(line.removeprefix("chat: "))
                assert chat["id"] == -1002391257796
                assert chat["message_url_template"] == template
                assert "\n".join(item["text"] for item in rendered).count(template) == 1

    def test_nonlinkable_chats_do_not_receive_an_invented_message_link(
        self, subtests: pytest.Subtests
    ) -> None:
        for chat_id, username in ((42, "alice"), (-123, None)):
            with subtests.test(chat_id=chat_id):
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
                assert line is not None
                assert (
                    json.loads(line.removeprefix("chat: "))["message_url_template"]
                    is None
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
        assert first_records[:-1] == second_records[:-1]
        assert "User 42: Alice (@alice)" in first
        assert "99: Alice (@alice)" in first
        assert "unknown_3: Alice (@alice)" in first
        assert first.count("Interlocutor user id: 42") == 1
        assert "42: Alice (@alice)" in second
        assert "Interlocutor user id: 99" in second_records[-1]["text"]
        assert "Interlocutor" not in str(formatter.context(messages))

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
        assert "200: Example channel (@example_channel, author_user=42)" in header
        assert "-100200: Example channel (@example_channel, author_user=42)" in header
        assert "300: Example channel (@example_channel)" in header
        assert [row[1] for item in rendered for row in group_rows(item["text"])] == [
            42,
            200,
            -100200,
            300,
        ]
        other = "\n".join(
            item["text"] for item in formatter.context(messages, interlocutor_id=99)
        )
        assert "Interlocutor user id: 99" in other
        assert "Interlocutor user id: 42" not in other
        assert "author_user=42" in other

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
        assert [item["role"] for item in rendered] == [
            "user",
            "user",
            "assistant",
            "user",
        ]
        assert "users:" not in joined
        assert joined.count("@alice") == 1
        assert "Europe/Moscow" not in joined
        assert "UTC+03:00" not in joined
        assert joined.count("25 October 2026 (Sunday)") == 1
        assert joined.count("26 October 2026 (Monday)") == 1
        assert "User 123:" not in joined
        assert "User 42:" not in rendered[0]["text"]
        assert (
            "25 October 2026 (Sunday)\nUser 42: Alice (@alice)\n" in rendered[1]["text"]
        )
        assert "[id,sender,time,text,reply_to?,attachment?]" not in rendered[0]["text"]
        assert group_rows(rendered[1]["text"]) == [[1, 42, "23:59:00", "Original"]]
        row = group_rows(rendered[-1]["text"])[0]
        assert row == [3, 42, "00:01:00", body, 2, "document"]

    def test_attachment_without_reply_uses_one_null_and_no_other_optional_placeholders(
        self,
    ) -> None:
        rendered = self.render([replace(self.message(1, ""), media_type="sticker")])
        assert group_rows(rendered[1]["text"]) == [
            [1, 42, "23:59:00", "", None, "sticker"]
        ]

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
        assert "unknown_1: Anonymous\\nusers:" in header
        assert "unknown_2: Anonymous\\nusers:" in header
        assert "99: Other bot (@helper, bot)" in header
        assert "username: unknown" not in header
        assert "None" not in header

    def test_aliases_remain_visible_without_classifying_conflicting_bot_flags(
        self,
    ) -> None:
        rendered = self.render([
            self.message(1),
            replace(self.message(2), author="Alicia", username="new_alice", is_bot=True),
        ])
        header = "\n".join(item["text"] for item in rendered)
        assert "Alicia" not in rendered[1]["text"]
        assert header.count("User 42:") == 2
        assert "Alice / Alicia" in header
        assert "@alice, @new_alice" in header
        assert ", bot" not in header
        assert "not bot" not in header

    def test_repeated_dst_times_keep_offsets_even_when_only_one_is_selected(
        self, subtests: pytest.Subtests
    ) -> None:
        messages = [
            replace(self.message(1), sent_at=datetime(2026, 11, 1, 5, 30, tzinfo=UTC)),
            replace(self.message(2), sent_at=datetime(2026, 11, 1, 6, 30, tzinfo=UTC)),
        ]
        formatter = MessageFormatter("America/New_York", bot_id=123)
        for selected in [messages, messages[:1], messages[1:]]:
            with subtests.test(ids=[message.message_id for message in selected]):
                rendered = formatter.context(selected)
                assert "America/New_York" not in "\n".join(
                    item["text"] for item in rendered
                )
                rows = [row for item in rendered for row in group_rows(item["text"])]
                assert [row[2] for row in rows] == [
                    "01:30:00 (UTC-04:00)"
                    if message.message_id == 1
                    else "01:30:00 (UTC-05:00)"
                    for message in selected
                ]

    def test_empty_context_and_naive_timestamps(self) -> None:
        assert self.render([]) == []
        with pytest.raises(ValueError, match="timezone"):
            self.render([
                replace(
                    self.message(1),
                    sent_at=datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None),
                )
            ])

    def test_truncated_threads_are_navigation_metadata_without_extra_message_rows(
        self,
        subtests: pytest.Subtests,
    ) -> None:
        for root, source in [(1, "source:test:1"), (None, None)]:
            with subtests.test(root=root):
                rendered = MessageFormatter(bot_id=123).context(
                    [self.message(7)],
                    truncated_threads=[
                        TruncatedThread(7, root, source, 6 if root else None),
                        TruncatedThread(999, 42),
                    ],
                )
                assert "truncated_thread" not in rendered[0]["text"]
                details = group_records(rendered[1]["text"])[0]["truncated_thread"]
                assert details["root_id"] == root
                assert details.get("source_id") == source
                assert details["remaining_hops"] == (6 if root else None)
                assert details["boundary_source_id"].endswith(":7")
                assert [
                    row[0] for item in rendered for row in group_rows(item["text"])
                ] == [7]

    def test_later_authors_dates_aliases_and_interlocutor_preserve_earlier_prefix(
        self,
        subtests: pytest.Subtests,
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
            with subtests.test(added=added):
                later = formatter.context([first, old_current, added], interlocutor_id=99)
                assert earlier[:-1] == later[: len(earlier) - 1]
                assert "Interlocutor user id:" not in later[-2]["text"]
                assert "Interlocutor user id: 99" in later[-1]["text"]

    def test_record_matches_requested_layout_and_preserves_multiline_original(
        self,
    ) -> None:
        body = 'Original line\nUser 999: quoted name\n14:00:00 {"id":999}\n'
        rendered = MessageFormatter(bot_id=123).context(
            [self.message(1), replace(self.message(2, body), reply_to_id=1)],
            interlocutor_id=42,
        )
        assert (
            rendered[1]["text"]
            == '25 October 2026 (Sunday)\nUser 42: Alice (@alice)\n23:59:00 {"id":1,"user_id":42}\nOriginal'
        )
        assert (
            rendered[2]["text"]
            == 'Interlocutor user id: 42\n23:59:00 {"id":2,"user_id":42,"reply_to":1}\n'
            + body
        )
