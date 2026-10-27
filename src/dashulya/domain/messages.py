from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ConversationKey = tuple[int, int | None]


@dataclass(frozen=True)
class ImageAttachment:
    data: bytes = field(repr=False)
    media_type: Literal["image/jpeg", "image/png", "image/webp"]
    message_id: int
    preview_kind: Literal["sticker", "animation"] | None = None
    first_frame: bool = False


@dataclass(frozen=True)
class ImageOmission:
    message_id: int
    reason: Literal["too_large", "preview_unavailable", "decode_failed"] = "too_large"


@dataclass(frozen=True)
class AddressingDecision:
    should_reply: bool
    reason: Literal[
        "reply_to_bot",
        "explicit_mention",
        "name_not_found",
        "classifier_accepted",
        "classifier_rejected",
        "classifier_no_answer",
    ]


@dataclass(frozen=True)
class Reply:
    text: str
    reply_to_message_id: int | None = None


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
