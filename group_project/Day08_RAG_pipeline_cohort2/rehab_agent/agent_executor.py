"""Rehab Agent — AgentExecutor bridge between A2A SDK and LangGraph."""

from __future__ import annotations

import logging
from uuid import uuid4

from langchain_core.messages import HumanMessage

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart

from rehab_agent.graph import create_graph

logger = logging.getLogger(__name__)

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph


class RehabAgentExecutor(AgentExecutor):
    """Bridges A2A RequestContext to the Rehab LangGraph agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = self._extract_question(context)
        context_id = context.context_id or str(uuid4())
        task_id = context.task_id or str(uuid4())
        metadata = context.message.metadata or {} if context.message else {}
        trace_id = metadata.get("trace_id", str(uuid4()))
        depth = int(metadata.get("delegation_depth", 0))
        telemetry_url = metadata.get("telemetry_url", "")

        logger.info(
            "RehabAgent executing | task=%s context=%s trace=%s depth=%d",
            task_id, context_id, trace_id, depth,
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.submit()
        await updater.start_work()

        try:
            from common.telemetry import emit, emit_tool_messages, span

            async with span(
                "rehab",
                "Analyzing rehabilitation and administrative measures",
                "Rehabilitation analysis completed",
                trace_id=trace_id,
                telemetry_url=telemetry_url,
            ):
                result = await _get_graph().ainvoke(
                    {"messages": [HumanMessage(content=question)]},
                    config={"configurable": {"thread_id": context_id}},
                )
            await emit_tool_messages(
                "rehab",
                result.get("messages", []),
                trace_id=trace_id,
                telemetry_url=telemetry_url,
            )

            # Extract the last AI message
            answer = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "content") and msg.content:
                    if not isinstance(msg, HumanMessage):
                        answer = msg.content
                        break

            if not answer:
                answer = "I was unable to generate a compliance analysis at this time."
            await emit(
                "rehab",
                "completed",
                "Rehab Agent produced analysis",
                kind="output",
                detail=answer[:1600],
                trace_id=trace_id,
                telemetry_url=telemetry_url,
            )

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=answer))],
                name="compliance_analysis",
            )
            await updater.complete()

        except Exception as exc:
            logger.exception("RehabAgent execution error: %s", exc)
            await updater.failed(
                updater.new_agent_message(
                    parts=[Part(root=TextPart(text=f"Rehab analysis failed: {exc}"))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    @staticmethod
    def _extract_question(context: RequestContext) -> str:
        if context.message and context.message.parts:
            parts = []
            for part in context.message.parts:
                inner = getattr(part, "root", part)
                text = getattr(inner, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return ""
