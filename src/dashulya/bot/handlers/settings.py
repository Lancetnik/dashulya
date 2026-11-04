from typing import Literal

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dishka.integrations.aiogram import FromDishka, inject

from dashulya.bot.middlewares.group_tracking import GROUP_TYPES
from dashulya.domain.groups import Group, GroupRegistry, WhisperAudience


class AudienceChoice(CallbackData, prefix="audience"):
    kind: Literal["reply", "whisper"]
    audience: WhisperAudience


def panel(group: Group) -> tuple[str, InlineKeyboardMarkup]:
    labels: dict[WhisperAudience, str] = {
        "all": "Всем участникам",
        "admins": "Только администраторам",
        "disabled": "Выключен",
    }
    text = (
        "Настройки Дашули 💋\n\n"
        f"Ответы: {'включены' if group.active else 'приостановлены'}\n"
        f"Обычные обращения: {labels[group.reply_audience].lower()}\n"
        f"Шёпот (/whisper): {labels[group.whisper_audience].lower()}\n\n"
        "Выбери, кому отвечать. Верхний ряд — обычные обращения, нижний — шёпот."
    )
    rows = []
    for kind, selected in (
        ("reply", group.reply_audience),
        ("whisper", group.whisper_audience),
    ):
        rows.append([
            InlineKeyboardButton(
                text=f"{'✓ ' if selected == value else ''}{label}",
                callback_data=AudienceChoice(kind=kind, audience=value).pack(),
            )
            for value, label in labels.items()
            if kind == "whisper" or value != "disabled"
        ])
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


@inject
async def show_settings(message: Message, groups: FromDishka[GroupRegistry]) -> None:
    group = await groups.get(message.chat.id)
    if group is None:
        return
    text, keyboard = panel(group)
    await message.answer(text, reply_markup=keyboard)


@inject
async def set_audience(
    callback: CallbackQuery,
    callback_data: AudienceChoice,
    groups: FromDishka[GroupRegistry],
) -> None:
    assert isinstance(callback.message, Message)
    chat_id = callback.message.chat.id
    if await groups.get(chat_id) is None:
        await callback.answer("Эта группа больше не зарегистрирована.", show_alert=True)
        return
    if callback_data.kind == "reply":
        if callback_data.audience == "disabled":
            await callback.answer(
                "Для паузы всех ответов используй /stop.", show_alert=True
            )
            return
        await groups.set_reply_audience(chat_id, callback_data.audience)
    else:
        await groups.set_whisper_audience(chat_id, callback_data.audience)
    group = await groups.get(chat_id)
    if group is None:
        await callback.answer("Эта группа больше не зарегистрирована.", show_alert=True)
        return
    text, keyboard = panel(group)
    await callback.answer("Настройка сохранена 💋")
    try:
        await callback.message.edit_text(text, reply_markup=keyboard)
    except TelegramBadRequest as error:
        # Repeated selections and older panels can already show this state.
        if "message is not modified" not in error.message:
            raise


def create_router() -> Router:
    router = Router(name=__name__)
    router.message.register(
        show_settings, Command("settings"), F.chat.type.in_(GROUP_TYPES)
    )
    router.callback_query.register(set_audience, AudienceChoice.filter())
    return router
