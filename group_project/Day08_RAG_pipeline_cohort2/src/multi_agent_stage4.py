"""Stage 4: named multi-agent workflow running in one Python process."""

from __future__ import annotations

import asyncio
import json
import operator
import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.telemetry import emit, span
from src.task9_retrieval_pipeline import retrieve


def _last_wins(current: str, new: str) -> str:
    return new if new else current


class Stage4State(TypedDict):
    question: str
    needs_criminal: bool
    needs_rehab: bool
    law_result: Annotated[str, _last_wins]
    criminal_result: Annotated[str, _last_wins]
    rehab_result: Annotated[str, _last_wins]
    rag_result: Annotated[str, _last_wins]
    final_answer: str


def get_stage4_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


def _preview(value: object, limit: int = 900) -> str:
    text = str(value).strip()
    return text if len(text) <= limit else text[:limit] + "..."


async def _retrieve_for(agent: str, query: str, top_k: int = 3) -> list[dict]:
    started = asyncio.get_running_loop().time()
    await emit(
        agent,
        "running",
        "Calling retrieval pipeline",
        kind="tool",
        tool="retrieve",
        tool_input={"query": query, "top_k": top_k},
    )
    results = await asyncio.to_thread(retrieve, query, top_k)
    compact = [
        {
            "source": item.get("source", ""),
            "score": round(float(item.get("score", 0)), 3),
            "content": _preview(item.get("content", ""), 240),
        }
        for item in results
    ]
    await emit(
        agent,
        "completed",
        f"retrieve returned {len(results)} documents",
        latency=asyncio.get_running_loop().time() - started,
        kind="tool",
        tool="retrieve",
        tool_input={"query": query, "top_k": top_k},
        tool_output=compact,
    )
    return results


def _evidence(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{item.get('source', 'unknown')} | {item.get('score', 0):.3f}] "
        f"{item.get('content', '')[:1400]}"
        for item in results
    )


async def customer_agent(state: Stage4State) -> dict:
    await emit(
        "customer",
        "running",
        "Received the user question",
        kind="decision",
        detail="The request is a substantive legal question and will be delegated to the legal workflow.",
    )
    await emit(
        "customer",
        "completed",
        "Delegated request to Law Agent workflow",
        kind="output",
        detail=_preview(state["question"], 300),
    )
    return {}


async def router(state: Stage4State) -> dict:
    llm = get_stage4_llm()
    prompt = (
        "Classify this Vietnamese drug-law question. Return JSON only: "
        '{"needs_criminal": true|false, "needs_rehab": true|false, "reason": "short public rationale"}. '
        "Criminal covers possession, trafficking, organizing use, prison and criminal liability. "
        "Rehab covers addiction treatment, administrative handling, post-rehab management.\n\n"
        f"Question: {state['question']}"
    )
    try:
        async with span("router", "Evaluating routing rules", "Routing decision completed"):
            response = await llm.ainvoke([SystemMessage(content=prompt)])
        raw = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        decision = json.loads(raw)
    except Exception as exc:
        question = state["question"].lower()
        decision = {
            "needs_criminal": any(
                word in question
                for word in ["hình sự", "tàng trữ", "mua bán", "vận chuyển", "tổ chức sử dụng", "phạt tù"]
            ),
            "needs_rehab": any(
                word in question
                for word in ["nghiện", "cai nghiện", "sau cai", "hành chính"]
            ),
            "reason": f"Keyword fallback was used because the classifier failed: {exc}",
        }
    await emit(
        "router",
        "completed",
        "Selected agents for parallel execution",
        kind="decision",
        detail=decision.get("reason", ""),
        tool_output={
            "Law Agent": True,
            "Criminal Agent": bool(decision.get("needs_criminal", True)),
            "Rehab Agent": bool(decision.get("needs_rehab", True)),
            "RAG Agent": True,
        },
    )
    return {
        "needs_criminal": bool(decision.get("needs_criminal", True)),
        "needs_rehab": bool(decision.get("needs_rehab", True)),
    }


def dispatch_agents(state: Stage4State) -> list[Send]:
    sends = [Send("law_agent", state), Send("rag_agent", state)]
    if state.get("needs_criminal"):
        sends.append(Send("criminal_agent", state))
    if state.get("needs_rehab"):
        sends.append(Send("rehab_agent", state))
    return sends


async def law_agent(state: Stage4State) -> dict:
    llm = get_stage4_llm()
    results = await _retrieve_for("law", state["question"])
    messages = [
        SystemMessage(
            content=(
                "Bạn là Law Agent chuyên Luật Phòng, chống ma túy Việt Nam. "
                "Phân tích căn cứ pháp lý tổng quát từ bằng chứng RAG, ngắn gọn và có nguồn."
            )
        ),
        HumanMessage(content=f"Câu hỏi: {state['question']}\n\nBằng chứng:\n{_evidence(results)}"),
    ]
    async with span("law", "Analyzing general legal framework", "Law Agent analysis completed"):
        response = await llm.ainvoke(messages)
    await emit("law", "completed", "Law Agent produced analysis", kind="output", detail=_preview(response.content))
    return {"law_result": response.content}


async def criminal_agent(state: Stage4State) -> dict:
    llm = get_stage4_llm()
    results = await _retrieve_for("criminal", state["question"])
    messages = [
        SystemMessage(
            content=(
                "Bạn là Criminal Agent chuyên trách nhiệm hình sự về ma túy tại Việt Nam. "
                "Phân biệt sử dụng, tàng trữ, vận chuyển, mua bán và tổ chức sử dụng. "
                "Chỉ kết luận dựa trên dữ kiện và bằng chứng được cung cấp."
            )
        ),
        HumanMessage(content=f"Câu hỏi: {state['question']}\n\nBằng chứng:\n{_evidence(results)}"),
    ]
    async with span("criminal", "Analyzing criminal liability", "Criminal Agent analysis completed"):
        response = await llm.ainvoke(messages)
    await emit("criminal", "completed", "Criminal Agent produced analysis", kind="output", detail=_preview(response.content))
    return {"criminal_result": response.content}


async def rehab_agent(state: Stage4State) -> dict:
    llm = get_stage4_llm()
    results = await _retrieve_for("rehab", state["question"])
    messages = [
        SystemMessage(
            content=(
                "Bạn là Rehab Agent chuyên cai nghiện, quản lý sau cai và xử lý hành chính "
                "theo pháp luật Việt Nam. Phân tích dựa trên bằng chứng RAG được cung cấp."
            )
        ),
        HumanMessage(content=f"Câu hỏi: {state['question']}\n\nBằng chứng:\n{_evidence(results)}"),
    ]
    async with span("rehab", "Analyzing rehabilitation measures", "Rehab Agent analysis completed"):
        response = await llm.ainvoke(messages)
    await emit("rehab", "completed", "Rehab Agent produced analysis", kind="output", detail=_preview(response.content))
    return {"rehab_result": response.content}


async def rag_agent(state: Stage4State) -> dict:
    results = await _retrieve_for("rag", state["question"], top_k=5)
    summary = _evidence(results)
    await emit(
        "rag",
        "completed",
        "RAG Agent prepared shared evidence",
        kind="output",
        detail=_preview(summary),
    )
    return {"rag_result": summary}


async def aggregator(state: Stage4State) -> dict:
    llm = get_stage4_llm()
    sections = [
        ("Law Agent", state.get("law_result", "")),
        ("Criminal Agent", state.get("criminal_result", "")),
        ("Rehab Agent", state.get("rehab_result", "")),
        ("RAG Agent", state.get("rag_result", "")),
    ]
    combined = "\n\n".join(f"## {name}\n{text}" for name, text in sections if text)
    messages = [
        SystemMessage(
            content=(
                "Bạn là Aggregator nội bộ của Law Agent. Tổng hợp các kết quả thành câu trả lời "
                "tiếng Việt rõ ràng, không bịa thêm căn cứ và giữ trích dẫn nguồn."
            )
        ),
        HumanMessage(content=f"Câu hỏi: {state['question']}\n\nKết quả agent:\n{combined}"),
    ]
    async with span("aggregator", "Combining named-agent outputs", "Final answer synthesized"):
        response = await llm.ainvoke(messages)
    await emit("aggregator", "completed", "Aggregator produced final response", kind="output", detail=_preview(response.content))
    return {"final_answer": response.content}


builder = StateGraph(Stage4State)
builder.add_node("customer_agent", customer_agent)
builder.add_node("router", router)
builder.add_node("law_agent", law_agent)
builder.add_node("criminal_agent", criminal_agent)
builder.add_node("rehab_agent", rehab_agent)
builder.add_node("rag_agent", rag_agent)
builder.add_node("aggregator", aggregator)
builder.set_entry_point("customer_agent")
builder.add_edge("customer_agent", "router")
builder.add_conditional_edges(
    "router",
    dispatch_agents,
    ["law_agent", "criminal_agent", "rehab_agent", "rag_agent"],
)
builder.add_edge("law_agent", "aggregator")
builder.add_edge("criminal_agent", "aggregator")
builder.add_edge("rehab_agent", "aggregator")
builder.add_edge("rag_agent", "aggregator")
builder.add_edge("aggregator", END)
graph = builder.compile()
