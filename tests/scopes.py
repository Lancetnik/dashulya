import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ag2.tools import tool
from aiogram.types import Message, TelegramObject
from dishka import Provider, provide
from dishka.integrations.aiogram import inject as inject_aiogram
from dishka_ag2 import AG2Scope, FromDishka, inject


@dataclass
class TurnState:
    pass


@dataclass
class CallState:
    pass


@dataclass
class ScopeTrace:
    uses: list[tuple[TurnState, CallState]] = field(default_factory=list)
    closed_turns: list[TurnState] = field(default_factory=list)
    closed_calls: list[CallState] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    telegram_calls: list[tuple[Message, TelegramObject, CallState]] = field(
        default_factory=list
    )


class ScopeProvider(Provider):
    @provide(scope=AG2Scope.APP)
    def trace(self) -> ScopeTrace:
        return ScopeTrace()

    @provide(scope=AG2Scope.SESSION)
    async def turn(self, trace: ScopeTrace) -> AsyncIterator[TurnState]:
        state = TurnState()
        try:
            yield state
        finally:
            trace.closed_turns.append(state)

    @provide(scope=AG2Scope.REQUEST)
    async def call(self, trace: ScopeTrace) -> AsyncIterator[CallState]:
        state = CallState()
        try:
            yield state
        finally:
            trace.closed_calls.append(state)


@tool
@inject
async def inspect_scope(
    turn: FromDishka[TurnState],
    call: FromDishka[CallState],
    trace: FromDishka[ScopeTrace],
    *,
    block: bool = False,
) -> str:
    trace.uses.append((turn, call))
    if block:
        trace.started.set()
        await trace.release.wait()
    return "Inspected"


@inject_aiogram
async def record_update(
    message: Message,
    event: FromDishka[TelegramObject],
    call: FromDishka[CallState],
    trace: FromDishka[ScopeTrace],
) -> None:
    trace.telegram_calls.append((message, event, call))
