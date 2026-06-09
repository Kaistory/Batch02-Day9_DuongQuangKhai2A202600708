"""Law Agent LangGraph StateGraph definition.

Graph topology:
    analyze_law → check_routing → (parallel) call_criminal + call_rehab → aggregate → END

The parallel branches (call_criminal / call_rehab) are dispatched via LangGraph's
Send API so that both sub-agent calls happen concurrently.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_llm

logger = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 3


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a


class LawState(TypedDict):
    question: str
    context_id: str
    trace_id: str
    delegation_depth: int
    telemetry_url: str
    law_analysis: str
    needs_criminal: bool
    needs_rehab: bool
    # Annotated so parallel branches can both write without conflict
    criminal_result: Annotated[str, _last_wins]
    rehab_result: Annotated[str, _last_wins]
    final_answer: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def analyze_law(state: LawState) -> dict:
    """LLM analysis from a contract / general law perspective."""
    from common.telemetry import emit, span

    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                "You are a senior corporate litigation attorney specialising in contract law, "
                "tort law, and general business law. Analyse the legal aspects of the question "
                "thoroughly, covering relevant statutes, case law principles, and liability exposure."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    async with span(
        "law",
        "Analyzing the legal question",
        "Legal analysis completed",
        trace_id=state["trace_id"],
        telemetry_url=state.get("telemetry_url", ""),
    ):
        result = await llm.ainvoke(messages)
    await emit(
        "law",
        "completed",
        "Law Agent produced general legal analysis",
        kind="output",
        detail=result.content[:1600],
        trace_id=state["trace_id"],
        telemetry_url=state.get("telemetry_url", ""),
    )
    return {"law_analysis": result.content}


async def check_routing(state: LawState) -> dict:
    """Determine whether criminal and/or rehab sub-agents are needed.

    Returns updated state flags so the routing function can read them.
    If delegation depth is already at the max, skip further delegation.
    """
    from common.telemetry import emit, span

    depth = state.get("delegation_depth", 0)
    if depth >= MAX_DELEGATION_DEPTH:
        logger.info("Max delegation depth reached (%d); skipping sub-agents", depth)
        return {"needs_criminal": False, "needs_rehab": False}

    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                'You are a legal routing expert. Based on the question, decide whether '
                'specialist sub-agents are needed.\n'
                'Reply with ONLY valid JSON — no markdown, no extra text:\n'
                '{"needs_criminal": <true|false>, "needs_rehab": <true|false>}\n\n'
                'needs_criminal = true  → question involves trách nhiệm hình sự, tội phạm ma túy, buôn bán, vận chuyển, tàng trữ, phạt tù\n'
                'needs_rehab = true → question involves cai nghiện, quản lý sau cai nghiện, xử lý vi phạm hành chính về ma túy, cơ sở cai nghiện'
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    async with span(
        "router",
        "Selecting specialist agents",
        "Routing decision completed",
        trace_id=state["trace_id"],
        telemetry_url=state.get("telemetry_url", ""),
    ):
        result = await llm.ainvoke(messages)
    raw = result.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Routing LLM returned non-JSON: %r — defaulting to both=True", raw)
        parsed = {"needs_criminal": True, "needs_rehab": True}

    needs_criminal = bool(parsed.get("needs_criminal", True))
    needs_rehab = bool(parsed.get("needs_rehab", True))
    await emit(
        "router",
        "completed",
        "Router selected downstream agents",
        kind="decision",
        detail=(
            "Criminal Agent handles criminal liability; Rehab Agent handles "
            "rehabilitation and administrative measures."
        ),
        tool_output={
            "Criminal Agent": needs_criminal,
            "Rehab Agent": needs_rehab,
            "Law Agent": True,
        },
        trace_id=state["trace_id"],
        telemetry_url=state.get("telemetry_url", ""),
    )
    logger.info("Routing decision: needs_criminal=%s needs_rehab=%s", needs_criminal, needs_rehab)
    return {"needs_criminal": needs_criminal, "needs_rehab": needs_rehab}


def route_to_subagents(state: LawState) -> list[Send]:
    """Routing function: dispatch parallel Send objects based on routing flags.

    This function is used with add_conditional_edges; it returns a list of
    Send objects which LangGraph executes as parallel branches.
    """
    sends: list[Send] = []
    # Always analyze law in parallel with sub-agents to reduce latency
    sends.append(Send("analyze_law", state))
    if state.get("needs_criminal"):
        sends.append(Send("call_criminal", state))
    if state.get("needs_rehab"):
        sends.append(Send("call_rehab", state))
    return sends


async def call_criminal(state: LawState) -> dict:
    """Delegate to the Criminal Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover
    from common.telemetry import emit

    try:
        await emit(
            "law",
            "running",
            "Discovering Criminal Agent",
            kind="tool",
            tool="registry.discover",
            tool_input={"task": "criminal_liability"},
            trace_id=state["trace_id"],
            telemetry_url=state.get("telemetry_url", ""),
        )
        endpoint = await discover("criminal_liability")
        await emit(
            "law",
            "running",
            "Delegating to Criminal Agent",
            kind="tool",
            tool="a2a.delegate",
            tool_input={"endpoint": endpoint, "question": state["question"]},
            trace_id=state["trace_id"],
            telemetry_url=state.get("telemetry_url", ""),
        )
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
            telemetry_url=state.get("telemetry_url", ""),
        )
        await emit(
            "law",
            "completed",
            "Criminal Agent response received",
            kind="tool",
            tool="a2a.delegate",
            tool_input={"endpoint": endpoint},
            tool_output=result[:1600],
            trace_id=state["trace_id"],
            telemetry_url=state.get("telemetry_url", ""),
        )
        logger.info("Criminal Agent returned %d chars", len(result))
        return {"criminal_result": result}
    except Exception as exc:
        logger.exception("call_criminal failed: %s", exc)
        return {"criminal_result": f"[Criminal analysis unavailable: {exc}]"}


async def call_rehab(state: LawState) -> dict:
    """Delegate to the Rehab Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover
    from common.telemetry import emit

    try:
        await emit(
            "law",
            "running",
            "Discovering Rehab Agent",
            kind="tool",
            tool="registry.discover",
            tool_input={"task": "rehab_rehab"},
            trace_id=state["trace_id"],
            telemetry_url=state.get("telemetry_url", ""),
        )
        endpoint = await discover("rehab_rehab")
        await emit(
            "law",
            "running",
            "Delegating to Rehab Agent",
            kind="tool",
            tool="a2a.delegate",
            tool_input={"endpoint": endpoint, "question": state["question"]},
            trace_id=state["trace_id"],
            telemetry_url=state.get("telemetry_url", ""),
        )
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
            telemetry_url=state.get("telemetry_url", ""),
        )
        await emit(
            "law",
            "completed",
            "Rehab Agent response received",
            kind="tool",
            tool="a2a.delegate",
            tool_input={"endpoint": endpoint},
            tool_output=result[:1600],
            trace_id=state["trace_id"],
            telemetry_url=state.get("telemetry_url", ""),
        )
        logger.info("Rehab Agent returned %d chars", len(result))
        return {"rehab_result": result}
    except Exception as exc:
        logger.exception("call_rehab failed: %s", exc)
        return {"rehab_result": f"[Rehab analysis unavailable: {exc}]"}


async def aggregate(state: LawState) -> dict:
    """Combine law_analysis, criminal_result, and rehab_result into a final answer."""
    from common.telemetry import emit, span

    llm = get_llm()

    sections: list[str] = []
    if state.get("law_analysis"):
        sections.append(f"## Legal Analysis\n{state['law_analysis']}")
    if state.get("criminal_result"):
        sections.append(f"## Criminal Analysis\n{state['criminal_result']}")
    if state.get("rehab_result"):
        sections.append(f"## Regulatory Rehab Analysis\n{state['rehab_result']}")

    combined = "\n\n---\n\n".join(sections)

    messages = [
        SystemMessage(
            content=(
                "You are a senior legal counsel synthesising specialist analyses into a "
                "comprehensive, well-structured response for the client. Combine the following "
                "analyses into a cohesive answer with clear sections. Avoid redundancy. "
                "End with a brief disclaimer that the analysis is educational and the client "
                "should consult licensed attorneys for their specific situation."
            )
        ),
        HumanMessage(content=combined),
    ]
    async with span(
        "aggregator",
        "Synthesizing specialist analyses",
        "Final answer synthesized",
        trace_id=state["trace_id"],
        telemetry_url=state.get("telemetry_url", ""),
    ):
        result = await llm.ainvoke(messages)
    await emit(
        "aggregator",
        "completed",
        "Aggregator produced final response",
        kind="output",
        detail=result.content[:1600],
        trace_id=state["trace_id"],
        telemetry_url=state.get("telemetry_url", ""),
    )
    return {"final_answer": result.content}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the Law Agent StateGraph."""
    graph = StateGraph(LawState)

    graph.add_node("analyze_law", analyze_law)
    graph.add_node("check_routing", check_routing)
    graph.add_node("call_criminal", call_criminal)
    graph.add_node("call_rehab", call_rehab)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("check_routing")

    # Conditional parallel dispatch: after check_routing, route_to_subagents
    # returns a list of Send objects (to analyze_law, call_criminal, call_rehab)
    graph.add_conditional_edges(
        "check_routing",
        route_to_subagents,
        ["analyze_law", "call_criminal", "call_rehab"],
    )

    graph.add_edge("analyze_law", "aggregate")

    graph.add_edge("call_criminal", "aggregate")
    graph.add_edge("call_rehab", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()
