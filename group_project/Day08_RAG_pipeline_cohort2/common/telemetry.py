"""Lightweight request-scoped telemetry for Stage 4 and Stage 5."""

from __future__ import annotations

import contextvars
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

TelemetrySink = Callable[[dict[str, Any]], Awaitable[None]]

_sink: contextvars.ContextVar[TelemetrySink | None] = contextvars.ContextVar(
    "telemetry_sink", default=None
)
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "telemetry_trace_id", default=""
)
_telemetry_url: contextvars.ContextVar[str] = contextvars.ContextVar(
    "telemetry_url", default=""
)


def configure(
    *,
    trace_id: str,
    sink: TelemetrySink | None = None,
    telemetry_url: str = "",
) -> None:
    _trace_id.set(trace_id)
    _sink.set(sink)
    _telemetry_url.set(telemetry_url)


async def emit(
    agent: str,
    status: str,
    message: str,
    *,
    latency: float | None = None,
    detail: str = "",
    kind: str = "lifecycle",
    tool: str = "",
    tool_input: Any = None,
    tool_output: Any = None,
    trace_id: str = "",
    telemetry_url: str = "",
) -> None:
    event = {
        "type": "trace",
        "trace_id": trace_id or _trace_id.get(),
        "agent": agent,
        "status": status,
        "message": message,
        "detail": detail,
        "kind": kind,
        "timestamp": time.time(),
    }
    if tool:
        event["tool"] = tool
    if tool_input is not None:
        event["tool_input"] = tool_input
    if tool_output is not None:
        event["tool_output"] = tool_output
    if latency is not None:
        event["latency"] = round(latency, 3)

    sink = _sink.get()
    if sink is not None:
        await sink(event)
        return

    callback_url = telemetry_url or _telemetry_url.get()
    if callback_url:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.post(callback_url, json=event)
        except Exception:
            # Telemetry must never break the agent workflow.
            pass


class span:
    def __init__(
        self,
        agent: str,
        start_message: str,
        end_message: str,
        *,
        detail: str = "",
        trace_id: str = "",
        telemetry_url: str = "",
    ) -> None:
        self.agent = agent
        self.start_message = start_message
        self.end_message = end_message
        self.detail = detail
        self.trace_id = trace_id
        self.telemetry_url = telemetry_url
        self.started_at = 0.0

    async def __aenter__(self) -> "span":
        self.started_at = time.perf_counter()
        await emit(
            self.agent,
            "running",
            self.start_message,
            detail=self.detail,
            trace_id=self.trace_id,
            telemetry_url=self.telemetry_url,
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        latency = time.perf_counter() - self.started_at
        if exc is None:
            await emit(
                self.agent,
                "completed",
                self.end_message,
                latency=latency,
                trace_id=self.trace_id,
                telemetry_url=self.telemetry_url,
            )
        else:
            await emit(
                self.agent,
                "error",
                f"{self.agent} failed",
                latency=latency,
                detail=str(exc),
                trace_id=self.trace_id,
                telemetry_url=self.telemetry_url,
            )
        return False


async def emit_tool_messages(
    agent: str,
    messages: list[Any],
    *,
    trace_id: str = "",
    telemetry_url: str = "",
) -> None:
    """Emit completed tool calls captured in a LangGraph message history."""
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            call_id = str(call.get("id", ""))
            calls[call_id] = {
                "name": call.get("name", "unknown_tool"),
                "args": call.get("args", {}),
            }

    for message in messages:
        call_id = str(getattr(message, "tool_call_id", ""))
        if not call_id:
            continue
        call = calls.get(call_id, {})
        await emit(
            agent,
            "completed",
            f"Tool {call.get('name', getattr(message, 'name', 'unknown_tool'))} returned",
            kind="tool",
            tool=call.get("name", getattr(message, "name", "unknown_tool")),
            tool_input=call.get("args", {}),
            tool_output=str(getattr(message, "content", ""))[:1600],
            trace_id=trace_id,
            telemetry_url=telemetry_url,
        )
