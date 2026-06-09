from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from common.telemetry import configure, emit

app = FastAPI(title="Drug Law Multi-Agent Observatory")

BASE_DIR = Path(__file__).resolve().parent
TELEMETRY_BASE_URL = os.getenv("TELEMETRY_BASE_URL", "http://127.0.0.1:8000")
CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:10000")

trace_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}


class ChatRequest(BaseModel):
    query: str
    mode: str


class TelemetryEvent(BaseModel):
    type: str = "trace"
    trace_id: str
    agent: str
    status: str
    message: str
    detail: str = ""
    kind: str = "lifecycle"
    tool: str = ""
    tool_input: Any = None
    tool_output: Any = None
    timestamp: float
    latency: float | None = None


@app.get("/")
async def get_ui() -> FileResponse:
    return FileResponse(BASE_DIR / "frontend" / "index.html")


@app.post("/api/telemetry")
async def receive_telemetry(event: TelemetryEvent) -> dict[str, str]:
    queue = trace_queues.get(event.trace_id)
    if queue is not None:
        await queue.put(event.model_dump(exclude_none=True))
    return {"status": "ok"}


async def run_stage4(query: str, trace_id: str, queue: asyncio.Queue) -> str:
    from src.multi_agent_stage4 import graph

    async def sink(event: dict[str, Any]) -> None:
        await queue.put(event)

    configure(trace_id=trace_id, sink=sink)
    await emit("client", "running", "Request entered the in-process graph")
    started_at = time.perf_counter()
    result = await graph.ainvoke(
        {
            "question": query,
            "needs_criminal": False,
            "needs_rehab": False,
            "law_result": "",
            "criminal_result": "",
            "rehab_result": "",
            "rag_result": "",
            "final_answer": "",
        },
        config={"recursion_limit": 50},
    )
    await emit(
        "client",
        "completed",
        "Stage 4 response returned to the user",
        latency=time.perf_counter() - started_at,
    )
    return result["final_answer"]


def extract_a2a_text(response: object) -> str:
    if hasattr(response, "root"):
        response = response.root
    result = getattr(response, "result", None)
    if result is None:
        return ""

    text = ""
    for artifact in getattr(result, "artifacts", None) or []:
        for part in getattr(artifact, "parts", None) or []:
            text += getattr(getattr(part, "root", part), "text", "") or ""
    if text:
        return text

    for part in getattr(result, "parts", None) or []:
        text += getattr(getattr(part, "root", part), "text", "") or ""
    if text:
        return text

    status = getattr(result, "status", None)
    status_message = getattr(status, "message", None)
    for part in getattr(status_message, "parts", None) or []:
        text += getattr(getattr(part, "root", part), "text", "") or ""
    if text:
        return text

    for history_message in getattr(result, "history", None) or []:
        for part in getattr(history_message, "parts", None) or []:
            text += getattr(getattr(part, "root", part), "text", "") or ""
    return text


async def run_stage5(query: str, trace_id: str, queue: asyncio.Queue) -> str:
    from a2a.client import A2AClient
    from a2a.types import (
        AgentCard,
        Message,
        MessageSendParams,
        Part,
        Role,
        SendMessageRequest,
        TextPart,
    )

    async def local_event(
        agent: str,
        status: str,
        message: str,
        *,
        latency: float | None = None,
        detail: str = "",
    ) -> None:
        event = {
            "type": "trace",
            "trace_id": trace_id,
            "agent": agent,
            "status": status,
            "message": message,
            "detail": detail,
            "timestamp": time.time(),
        }
        if latency is not None:
            event["latency"] = round(latency, 3)
        await queue.put(event)

    started_at = time.perf_counter()
    await local_event("client", "running", "Preparing A2A request")

    async with httpx.AsyncClient(timeout=300.0) as client:
        registry_started = time.perf_counter()
        await local_event("registry", "running", "Checking service registry")
        registry_response = await client.get(f"{REGISTRY_URL}/health")
        registry_response.raise_for_status()
        await local_event(
            "registry",
            "completed",
            "Registry is healthy",
            latency=time.perf_counter() - registry_started,
            detail=f"{registry_response.json().get('agent_count', 0)} agents registered",
        )

        card_response = await client.get(
            f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
        )
        card_response.raise_for_status()
        agent_card = AgentCard.model_validate(card_response.json())
        a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=query))],
            message_id=str(uuid4()),
            metadata={
                "trace_id": trace_id,
                "delegation_depth": 0,
                "telemetry_url": f"{TELEMETRY_BASE_URL}/api/telemetry",
            },
        )
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message),
        )
        response = await a2a_client.send_message(request)

    answer = extract_a2a_text(response)
    if not answer:
        raise RuntimeError("Stage 5 returned no text artifact")

    await local_event(
        "client",
        "completed",
        "A2A response rendered",
        latency=time.perf_counter() - started_at,
    )
    return answer


async def execute_request(
    req: ChatRequest,
    trace_id: str,
    queue: asyncio.Queue[dict[str, Any]],
) -> None:
    try:
        if req.mode == "stage4":
            answer = await run_stage4(req.query, trace_id, queue)
        elif req.mode == "stage5":
            answer = await run_stage5(req.query, trace_id, queue)
        else:
            raise HTTPException(status_code=400, detail="Unsupported execution mode")
        await queue.put(
            {
                "type": "result",
                "trace_id": trace_id,
                "response": answer,
                "timestamp": time.time(),
            }
        )
    except Exception as exc:
        await queue.put(
            {
                "type": "error",
                "trace_id": trace_id,
                "message": str(exc),
                "detail": traceback.format_exc(),
                "timestamp": time.time(),
            }
        )
    finally:
        await queue.put({"type": "done", "trace_id": trace_id})


@app.post("/api/chat/stream")
async def stream_chat(req: ChatRequest) -> StreamingResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    trace_id = str(uuid4())
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    trace_queues[trace_id] = queue
    worker = asyncio.create_task(execute_request(req, trace_id, queue))

    async def generate():
        try:
            yield json.dumps(
                {"type": "meta", "trace_id": trace_id, "mode": req.mode}
            ) + "\n"
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] == "done":
                    break
        finally:
            trace_queues.pop(trace_id, None)
            if not worker.done():
                worker.cancel()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
