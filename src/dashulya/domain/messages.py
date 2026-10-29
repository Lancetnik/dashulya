from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ConversationKey = tuple[int, int | None]


@dataclass(frozen=True)
class ImageFile:
    """Telegram visual-media references, without downloaded content or file URLs."""

    file_id: str = field(repr=False)
    file_size: int | None = None
    preview_kind: Literal["sticker", "animation"] | None = None
    thumbnail_file_id: str | None = field(default=None, repr=False)
    thumbnail_file_size: int | None = None


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
    reason: Literal[
        "too_large", "preview_unavailable", "decode_failed", "unavailable"
    ] = "too_large"


@dataclass(frozen=True)
class ReplyRequest:
    """One authorized message and its already downloaded visual inputs.

    Negative chat IDs select fresh group context; positive IDs retain private
    history. Group requests require message_id. Reply parents refer to originals
    in that conversation, never to an independent snapshot supplied by a caller.
    """

    key: ConversationKey
    text: str
    user_id: int
    user_name: str
    sent_at: datetime
    message_id: int | None = None
    reply_to_id: int | None = None
    images: tuple[ImageAttachment | ImageOmission, ...] = ()


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
