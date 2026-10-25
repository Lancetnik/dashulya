from dataclasses import dataclass
from datetime import datetime

ConversationKey = tuple[int, int | None]


@dataclass(frozen=True)
class GroupMessage:
    message_id: int
    text: str
    sent_at: datetime
    sender_id: int | None
    sender_name: str | None
    sender_is_bot: bool = False
    mentioned_usernames: tuple[str, ...] = ()
    mentioned_user_ids: tuple[int, ...] = ()
    reply_to: "GroupMessage | None" = None

    def context(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "text": self.text,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "sender_is_bot": self.sender_is_bot,
            "sent_at": self.sent_at.isoformat(),
            "reply_to": self.reply_to.context() if self.reply_to is not None else None,
        }
