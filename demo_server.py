"""Bridge server cho demo_agents.html.

Cho phép nhập câu hỏi từ giao diện và chạy multi-agent system THẬT (Stage 4
in-process), trả về: quyết định routing, thời gian từng node, và câu trả lời.

Chạy:
    uv run python demo_server.py
Rồi mở: http://localhost:8800
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from stages.stage_4_milti_agent.main import (
    LegalState,
    analyze_law,
    call_tax_specialist,
    call_compliance_specialist,
    aggregate,
    create_graph,
)

HERE = Path(__file__).parent


# --- Graph tối ưu (fan-out song song + keyword routing) ---
def _fan_out(state: LegalState):
    q = state["question"].lower()
    sends = [Send("analyze_law", state)]
    if any(k in q for k in ["tax", "irs", "thuế", "avoid", "trốn thuế", "evasion"]):
        sends.append(Send("call_tax_specialist", state))
    if any(k in q for k in ["compliance", "sec", "sox", "regulat", "contract",
                            "breach", "gdpr", "privacy", "data", "dữ liệu", "hợp đồng"]):
        sends.append(Send("call_compliance_specialist", state))
    return sends


def build_optimized():
    g = StateGraph(LegalState)
    g.add_node("analyze_law", analyze_law)
    g.add_node("call_tax_specialist", call_tax_specialist)
    g.add_node("call_compliance_specialist", call_compliance_specialist)
    g.add_node("aggregate", aggregate)
    g.set_conditional_entry_point(
        _fan_out, ["analyze_law", "call_tax_specialist", "call_compliance_specialist"]
    )
    g.add_edge("analyze_law", "aggregate")
    g.add_edge("call_tax_specialist", "aggregate")
    g.add_edge("call_compliance_specialist", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


app = FastAPI(title="Multi-Agent Demo Bridge")


@app.get("/")
def index():
    return HTMLResponse(HERE.joinpath("demo_agents.html").read_text(encoding="utf-8"))


class Ask(BaseModel):
    question: str
    mode: str = "baseline"  # "baseline" | "optimized"


@app.post("/ask")
async def ask(req: Ask):
    graph = build_optimized() if req.mode == "optimized" else create_graph()
    init = {
        "question": req.question,
        "law_analysis": "",
        "needs_tax": False,
        "needs_compliance": False,
        "tax_result": "",
        "compliance_result": "",
        "final_answer": "",
    }
    timings, routing, answer = [], {}, ""
    t0 = time.perf_counter()
    try:
        async for chunk in graph.astream(init, stream_mode="updates"):
            now = round(time.perf_counter() - t0, 2)
            for node, delta in chunk.items():
                timings.append({"node": node, "t": now})
                if node == "check_routing" and isinstance(delta, dict):
                    routing = {
                        "needs_tax": delta.get("needs_tax"),
                        "needs_compliance": delta.get("needs_compliance"),
                    }
                if isinstance(delta, dict) and delta.get("final_answer"):
                    answer = delta["final_answer"]
    except Exception as exc:  # vd: 429 hết quota Gemini
        return JSONResponse(
            {"error": str(exc), "timings": timings,
             "total": round(time.perf_counter() - t0, 2)},
            status_code=200,
        )
    return {
        "question": req.question,
        "mode": req.mode,
        "total": round(time.perf_counter() - t0, 2),
        "routing": routing,
        "agents": [x["node"] for x in timings],
        "timings": timings,
        "answer": answer or "(không có nội dung trả về)",
    }


if __name__ == "__main__":
    import uvicorn

    print("Demo bridge chạy tại: http://localhost:8800")
    uvicorn.run(app, host="127.0.0.1", port=8800, log_level="warning")
