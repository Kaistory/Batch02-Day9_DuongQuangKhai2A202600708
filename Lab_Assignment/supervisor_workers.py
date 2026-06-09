"""Lab Assignment Day09 — Supervisor–Workers Multi-Agent System.

Cải tiến agent (Day08) sang pattern **Supervisor–Workers**:

- 1 **Supervisor**: điều phối ĐỘNG — mỗi vòng nhìn vào kết quả đã tích lũy rồi
  quyết định gọi worker nào TIẾP THEO, hoặc FINISH khi đã đủ thông tin.
- 3 **Workers** chuyên môn (mỗi worker là 1 ReAct agent có tool riêng):
    1. legal_worker      — hợp đồng, trách nhiệm dân sự
    2. tax_worker        — thuế, IRS, FBAR/FATCA
    3. compliance_worker — SEC, SOX, GDPR, FCPA

Khác biệt với fan-out song song (Stage 4):
- Stage 4: router chạy 1 lần, bắn song song tất cả worker cần thiết (cố định).
- Supervisor–Workers: LẶP, mỗi vòng chỉ 1 worker, supervisor xem kết quả rồi
  quyết tiếp → linh hoạt hơn (có thể bỏ qua/ gọi thêm worker tùy ngữ cảnh).

Chạy:
    uv run python Lab_Assignment/supervisor_workers.py
"""

import asyncio
import os
import sys
from typing import Annotated, Literal, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from common.llm import get_llm

# Nạp .env NGAY khi import — vì các worker (ReAct agent) được tạo ở cấp module,
# cần OPENAI_API_KEY có sẵn trước khi gọi get_llm().
load_dotenv()

WORKERS = ["legal_worker", "tax_worker", "compliance_worker"]
MAX_STEPS = 6  # chặn vòng lặp vô hạn của supervisor


# ---------------------------------------------------------------------------
# Tools cho từng worker (knowledge base giả lập — production dùng vector store)
# ---------------------------------------------------------------------------

@tool
def search_legal_db(query: str) -> str:
    """Tra cứu luật hợp đồng / dân sự (UCC, remedies, damages)."""
    return (
        "UCC Article 2: expectation damages, consequential damages (Hadley v. Baxendale), "
        "specific performance, cover damages. Statute of limitations 4 years (UCC § 2-725)."
    )


@tool
def search_tax_db(query: str) -> str:
    """Tra cứu luật thuế (IRS, tax evasion, FBAR/FATCA)."""
    return (
        "Tax evasion (26 U.S.C. § 7201): felony, phạt tới $250K + 5 năm tù. Gian lận dân sự: "
        "75% số thuế thiếu (IRC § 6663). FBAR: phạt tới $100K hoặc 50% số dư tài khoản/vi phạm."
    )


@tool
def search_compliance_db(query: str) -> str:
    """Tra cứu tuân thủ (SEC, SOX, GDPR, FCPA)."""
    return (
        "GDPR: phạt tới 4% doanh thu toàn cầu hoặc EUR 20M. CCPA: $7,500/vi phạm cố ý. "
        "SOX § 906: chứng nhận sai — tới $5M + 20 năm tù. FCPA: hối lộ nước ngoài, tới $2M (DN)."
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    question: str
    findings: dict           # {worker_name: kết quả}
    next: str                # worker tiếp theo hoặc "FINISH"
    steps: int               # số vòng supervisor đã chạy
    final: str               # câu trả lời tổng hợp


# ---------------------------------------------------------------------------
# Supervisor — định tuyến động bằng structured output
# ---------------------------------------------------------------------------

class Route(BaseModel):
    """Quyết định của supervisor."""
    next: Literal["legal_worker", "tax_worker", "compliance_worker", "FINISH"] = Field(
        description="Worker cần gọi tiếp theo, hoặc FINISH nếu đã đủ thông tin."
    )
    reason: str = Field(description="Lý do ngắn gọn cho quyết định.")


def supervisor(state: SupervisorState) -> dict:
    """Nhìn vào findings đã có → chọn worker tiếp theo hoặc FINISH."""
    steps = state.get("steps", 0)
    done = list(state.get("findings", {}).keys())

    # Chặn vòng lặp: hết ngân sách bước → buộc FINISH
    if steps >= MAX_STEPS:
        print(f"  [Supervisor] Đạt MAX_STEPS ({MAX_STEPS}) → FINISH")
        return {"next": "FINISH", "steps": steps + 1}

    llm = get_llm()
    system = (
        "Bạn là SUPERVISOR điều phối 3 worker chuyên môn:\n"
        "- legal_worker: hợp đồng, trách nhiệm dân sự\n"
        "- tax_worker: thuế, IRS, trốn thuế, FBAR/FATCA\n"
        "- compliance_worker: SEC, SOX, GDPR, FCPA, quyền riêng tư\n\n"
        f"Các worker ĐÃ báo cáo: {done or 'chưa có'}.\n"
        "Dựa vào câu hỏi và kết quả đã có, chọn worker TIẾP THEO liên quan mà CHƯA chạy. "
        "Khi đã thu thập đủ mọi khía cạnh liên quan, trả về FINISH. "
        "KHÔNG gọi lại worker đã chạy trừ khi thật sự cần."
    )
    human = f"Câu hỏi: {state['question']}\n\nKết quả hiện có: {state.get('findings', {}) or '(trống)'}"
    route = get_llm().with_structured_output(Route).invoke(
        [SystemMessage(content=system), HumanMessage(content=human)]
    )
    print(f"  [Supervisor] (bước {steps + 1}) → {route.next}  | {route.reason[:70]}")
    return {"next": route.next, "steps": steps + 1}


# ---------------------------------------------------------------------------
# Workers — mỗi worker là 1 ReAct agent có tool riêng
# ---------------------------------------------------------------------------

def _make_worker(name: str, prompt: str, tools: list):
    agent = create_react_agent(model=get_llm(), tools=tools, prompt=prompt)

    async def worker(state: SupervisorState) -> dict:
        print(f"  [Worker:{name}] đang xử lý...")
        res = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
        out = res["messages"][-1].content
        findings = dict(state.get("findings", {}))
        findings[name] = out
        print(f"  [Worker:{name}] xong ({len(out)} ký tự)")
        return {"findings": findings}

    return worker


legal_worker = _make_worker(
    "legal_worker",
    "Bạn là luật sư hợp đồng. Dùng search_legal_db để phân tích khía cạnh hợp đồng/dân sự. "
    "Trả lời dưới 150 từ.",
    [search_legal_db],
)
tax_worker = _make_worker(
    "tax_worker",
    "Bạn là chuyên gia thuế. Dùng search_tax_db để phân tích khía cạnh thuế. Trả lời dưới 150 từ.",
    [search_tax_db],
)
compliance_worker = _make_worker(
    "compliance_worker",
    "Bạn là chuyên gia tuân thủ. Dùng search_compliance_db để phân tích SEC/SOX/GDPR/FCPA. "
    "Trả lời dưới 150 từ.",
    [search_compliance_db],
)


async def finalize(state: SupervisorState) -> dict:
    """Tổng hợp toàn bộ findings thành câu trả lời cuối."""
    print("  [Finalize] tổng hợp báo cáo cuối...")
    llm = get_llm()
    parts = [f"### {k}\n{v}" for k, v in state.get("findings", {}).items()]
    combined = "\n\n".join(parts) if parts else "(không có dữ liệu)"
    messages = [
        SystemMessage(content=(
            "Bạn là luật sư trưởng. Tổng hợp các phân tích chuyên môn thành một báo cáo "
            "pháp lý mạch lạc, có cấu trúc, dưới 400 từ. Kết thúc bằng miễn trừ trách nhiệm ngắn."
        )),
        HumanMessage(content=f"Câu hỏi: {state['question']}\n\nCác phân tích:\n{combined}"),
    ]
    res = await llm.ainvoke(messages)
    return {"final": res.content}


# ---------------------------------------------------------------------------
# Định tuyến + dựng graph
# ---------------------------------------------------------------------------

def route_supervisor(state: SupervisorState) -> str:
    """Supervisor → worker tiếp theo, hoặc → finalize khi FINISH."""
    return "finalize" if state["next"] == "FINISH" else state["next"]


def build_graph():
    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor)
    g.add_node("legal_worker", legal_worker)
    g.add_node("tax_worker", tax_worker)
    g.add_node("compliance_worker", compliance_worker)
    g.add_node("finalize", finalize)

    g.add_edge(START, "supervisor")
    # Supervisor quyết định đi đâu (định tuyến động)
    g.add_conditional_edges("supervisor", route_supervisor,
                            [*WORKERS, "finalize"])
    # Mỗi worker xong → QUAY LẠI supervisor (vòng lặp đặc trưng của pattern này)
    for w in WORKERS:
        g.add_edge(w, "supervisor")
    g.add_edge("finalize", END)
    return g.compile()


QUESTION = (
    "Một công ty vừa vi phạm hợp đồng cung cấp, vừa trốn thuế thu nhập ở nước ngoài, "
    "vừa làm rò rỉ dữ liệu khách hàng. Hãy phân tích toàn bộ hậu quả pháp lý."
)


async def main():
    load_dotenv()
    print("=" * 70)
    print("SUPERVISOR–WORKERS MULTI-AGENT (Lab Assignment Day09)")
    print("=" * 70)
    print(f"\nCâu hỏi: {QUESTION}\n")
    print("Luồng: START → supervisor ⇄ workers (lặp) → finalize → END\n")
    print("-" * 70)

    graph = build_graph()
    result = await graph.ainvoke({
        "question": QUESTION,
        "findings": {},
        "next": "",
        "steps": 0,
        "final": "",
    })

    print("\n" + "=" * 70)
    print(f"WORKERS ĐÃ CHẠY: {list(result['findings'].keys())}")
    print("=" * 70)
    print("\nBÁO CÁO CUỐI:\n")
    print(result["final"])


if __name__ == "__main__":
    asyncio.run(main())
