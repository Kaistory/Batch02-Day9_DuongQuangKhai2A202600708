# Lab-Solution — Day09 Multi-Agent / MCP / A2A

Lời giải các bài Lab trên lớp (Codelab "Xây Dựng Hệ Thống Multi-Agent với A2A Protocol").

Mỗi bài gồm: **yêu cầu → lời giải (code) → nơi code thật nằm trong repo → điểm rút ra**.
Lời giải chi tiết hơn (kèm output thực tế) nằm trong [`CODELAB.md`](CODELAB.md) và
[`exercises/README.md`](exercises/README.md); file này là bản tổng hợp gọn để nộp.

> ⚠️ **Chạy thật cần `GEMINI_API_KEY`** trong `.env` (free tier 20 request/model/ngày).
> Trên Windows nếu gặp `UnicodeEncodeError` với tiếng Việt, ép UTF-8:
> `$env:PYTHONIOENCODING="utf-8"; uv run python <file>`

---

## Tổng quan 5 Stage

| Stage | Pattern | File chính | Độ phức tạp |
|---|---|---|---|
| 1 | Direct LLM | `stages/stage_1_direct_llm/main.py` | ⭐ |
| 2 | LLM + RAG & Tools | `stages/stage_2_rag_tools/main.py` | ⭐⭐ |
| 3 | ReAct Single Agent | `stages/stage_3_single_agent/main.py` | ⭐⭐⭐ |
| 4 | Multi-Agent (in-process) | `stages/stage_4_milti_agent/main.py` | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | `registry/`, `*_agent/`, `customer_agent/` | ⭐⭐⭐⭐⭐ |

---

## Phần 1 — Direct LLM

### Bài 1.1 — Đổi câu hỏi
**Yêu cầu:** Sửa biến `QUESTION` trong `stages/stage_1_direct_llm/main.py` sang câu hỏi pháp lý khác và chạy lại.

```python
QUESTION = "Người lao động bị sa thải trái pháp luật có những quyền lợi gì theo Bộ luật Lao động Việt Nam?"
```

```bash
$env:PYTHONIOENCODING="utf-8"; uv run python stages/stage_1_direct_llm/main.py
```

**Điểm rút ra:** Stage 1 trả lời hoàn toàn dựa vào *training data* — không tra cứu văn bản luật thật, không kiểm chứng nguồn, có thể trích sai số điều luật. Đây là lý do Stage 2 thêm RAG + Tools để "neo" câu trả lời vào dữ liệu thật.

### Bài 1.2 — Temperature control
**Yêu cầu:** Thêm `temperature=0.3` vào `get_llm()` trong `common/llm.py`.

```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        openai_api_key=os.getenv("GEMINI_API_KEY"),
        openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "1024")),
        temperature=0.3,  # ← thêm
    )
```

**Điểm rút ra:** `temperature` thấp (0.0–0.3) cho output **ổn định, ít bịa** — phù hợp pháp lý/tài chính/code; cao (1.2–2.0) cho sáng tạo. Hệ pháp lý cần chính xác và tái lập → chọn `0.3`. (Nâng cao: `temperature=float(os.getenv("GEMINI_TEMPERATURE","0.3"))` để chỉnh qua `.env`.)

---

## Phần 2 — LLM + RAG & Tools

> Code đáp án đầy đủ: [`exercises/exercise_2_tools.py`](exercises/exercise_2_tools.py)

### Bài 2.1 — Thêm knowledge base entry (luật lao động)
**Yêu cầu:** Thêm 1 entry vào `LEGAL_KNOWLEDGE`.

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "thôi việc", "hợp đồng lao động",
                 "labor", "employment", "dismissal"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019: Người sử dụng lao động chỉ được đơn phương "
        "chấm dứt HĐLĐ trong các trường hợp luật định (Điều 36)... Thời hiệu yêu cầu giải "
        "quyết tranh chấp lao động cá nhân là 1 năm (Điều 190)."
    ),
}
```

### Bài 2.2 — Tạo tool mới `check_statute_of_limitations`
**Yêu cầu:** Tạo `@tool` nhận `case_type` → trả về thời hiệu khởi kiện, đăng ký vào `tools`.

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, labor, property).
    """
    limits = {
        "contract": "Hợp đồng (UCC): 4 năm (UCC § 2-725)",
        "tort": "Bồi thường ngoài hợp đồng: 2-3 năm tùy bang",
        "labor": "Tranh chấp lao động cá nhân (VN): 1 năm (BLLĐ 2019, Điều 190)",
        "property": "Tranh chấp tài sản: 5 năm",
    }
    return limits.get(case_type.lower().strip(), f"Không xác định thời hiệu cho '{case_type}'")
```

**3 bước bắt buộc khi thêm tool:**
1. Định nghĩa `@tool` (docstring = LLM đọc để biết *khi nào* gọi; type hint = schema tham số).
2. Thêm vào danh sách `tools` để `bind_tools` thấy. *(Thiếu → LLM không biết tool tồn tại.)*
3. Thêm nhánh `elif tool_call["name"] == ...` để thực thi khi LLM gọi. *(Thiếu → tool được gọi nhưng không chạy.)*

**Điểm rút ra:** Stage 2 vẫn là vòng lặp tool **thủ công, chỉ 1 vòng**. Nếu LLM cần tra cứu thêm sau khi xem kết quả thì không tự làm được → Stage 3 dùng ReAct agent tự động hóa.

---

## Phần 3 — ReAct Single Agent

> Code: [`stages/stage_3_single_agent/main.py`](stages/stage_3_single_agent/main.py)

### Bài 3.1 — Thêm tool tra cứu án lệ
```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.

    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```
Đăng ký: `TOOLS = [search_legal_database, calculate_penalty, check_compliance_requirements, search_case_law]`.

**Điểm rút ra:** Khác Stage 2, ở Stage 3 chỉ cần **thêm tool vào `TOOLS`** — agent **tự nhận ra** câu hỏi cần "case law" và tự gọi, không phải sửa logic điều phối.

### Bài 3.2 — Debug agent reasoning
**Đính chính:** Đề viết `verbose=True` theo API LangChain cũ → **lỗi** với LangGraph hiện tại:
```
TypeError: create_react_agent() got unexpected keyword arguments: {'verbose': True}
```
**Cách đúng** — dùng `debug=True`:
```python
graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT, debug=True)
```
Kiểm chứng signature: `python -c "from langgraph.prebuilt import create_react_agent; import inspect; print(inspect.signature(create_react_agent))"`.

**Điểm rút ra:** `debug=True` in ra `[values]`/`[updates]`, `token_usage`/`cost`, `tool_calls`+`id`, `finish_reason` — soi rõ ReAct loop. Bài học: **tài liệu có thể lỗi thời so với thư viện** → dùng `inspect.signature()` để biết API thật.

---

## Phần 4 — Multi-Agent In-Process

> Code đáp án đầy đủ: [`exercises/exercise_4_multiagent.py`](exercises/exercise_4_multiagent.py).
> Code thật của repo trong `stages/stage_4_milti_agent/main.py` đặt tên khác (`LegalState`, `analyze_law`, `call_tax_specialist`, `route_to_specialists`, `aggregate`).

### Bài 4.1 — Thêm `privacy_agent`
```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về bảo vệ dữ liệu cá nhân. Phân tích khía cạnh quyền riêng tư:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: GDPR, CCPA, data breach notification, quyền của chủ thể dữ liệu, mức phạt."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```
Đồng bộ **4 chỗ**: thêm field `privacy_analysis: Annotated[str, _last_wins]` vào State → hàm agent → `add_node` + `add_edge("privacy_agent","aggregate_results")` → đọc trong `aggregate_results`.

### Bài 4.2 — Conditional routing cho privacy
```python
if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu", "rò rỉ", "breach", "khách hàng"]):
    tasks.append(Send("privacy_agent", state))
```

**Lỗi quan trọng trong scaffold (phải sửa code mới chạy):** `check_routing` trả `list[Send]` → **KHÔNG** được `add_node` (node phải trả `dict`, nếu không → `InvalidUpdateError`). Phải dùng nó làm **hàm conditional-edge**:
```python
graph.add_edge(START, "law_agent")
graph.add_conditional_edges(
    "law_agent", check_routing,
    ["tax_agent", "compliance_agent", "privacy_agent", "aggregate_results"],
)
graph.add_edge("tax_agent", "aggregate_results")
graph.add_edge("compliance_agent", "aggregate_results")
graph.add_edge("privacy_agent", "aggregate_results")
graph.add_edge("aggregate_results", END)
```

**Điểm rút ra:**
- **Node** trả `dict` (ghi state); **conditional-edge** trả tên node hoặc `Send` (điều phối).
- Trả về **list nhiều `Send`** → chạy **song song**; `aggregate` đợi mọi nhánh hội tụ.
- **Reducer `_last_wins`** bắt buộc trên field ghi song song, nếu không → lỗi "concurrent update".
- Conditional routing chỉ gọi agent *thực sự cần* → tiết kiệm thời gian & quota.

---

## Phần 5 — Distributed A2A

> Khởi động: `./start_all.ps1` (Windows) — Registry (10000), Customer (10100), Law (10101), Tax (10102), Compliance (10103). Test: `uv run python test_client.py`.

### Bài 5.1 — Trace request flow
`trace_id` (UUID) được sinh **một lần** ở Customer Agent rồi **truyền nguyên vẹn** qua mọi delegation (trong state + A2A metadata) → lọc log theo 1 `trace_id` để dựng lại toàn bộ hành trình.
```
test_client → Customer(10100) → discover(legal_question) → Law(10101)
   → analyze_law → check_routing(needs_tax=True) → discover(tax_question)
   → Tax(10102) [Gemini + search_tax_law] → tax_result → aggregate → final_answer
```
**Điểm rút ra:** đây là nền tảng *observability* trong hệ phân tán — distributed tracing.

### Bài 5.2 — Dynamic discovery / fault tolerance
Dừng Tax Agent rồi gọi lại → `call_tax` bắt exception và **degrade nhẹ nhàng**:
```python
async def call_tax(state):
    try:
        endpoint = await discover("tax_question")
        result = await delegate(endpoint=endpoint, ...)
        return {"tax_result": result}
    except Exception as exc:
        logger.exception("call_tax failed: %s", exc)
        return {"tax_result": f"[Tax analysis unavailable: {exc}]"}
```
**Điểm rút ra:** lỗi 1 agent **không sập cả hệ thống** — `aggregate` vẫn tổng hợp phần còn lại (**graceful degradation**). Phân biệt lỗi A2A (`discover`/`delegate`) với lỗi LLM (`POST .../chat/completions`, vd Gemini 503).

### Bài 5.3 — Modify agent behavior
Sửa system prompt trong `tax_agent/graph.py`, **restart CHỈ Tax Agent**:
```python
tax_prompt = (
    "You are a specialist tax attorney. Answer in AT MOST 3 bullet points, "
    "each under 20 words. Use the search_tax_law tool. Be extremely concise."
)
```
```powershell
Get-NetTCPConnection -LocalPort 10102 -State Listen | Select -Expand OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
$env:PYTHONIOENCODING="utf-8"; uv run python -m tax_agent
```
**Điểm rút ra:** sửa & deploy **riêng từng agent**, không downtime phần còn lại — chính là lý do tồn tại của kiến trúc A2A so với monolith Stage 4.

---

## Câu Hỏi Ôn Tập

1. **Single vs multi-agent?** Single khi 1 lĩnh vực, cần đơn giản/latency thấp. Multi khi nhiều chuyên môn, cần song song, cần scale/deploy độc lập. *Quy tắc: bắt đầu single, chỉ tách khi 1 prompt "ôm" quá nhiều việc.*
2. **A2A vs gRPC/REST?** A2A = chuẩn **trên HTTP** cho agent: có **Agent Card** (`/.well-known/agent-card.json`) tự mô tả, **task lifecycle** chuẩn hóa (task/message/artifact/context_id/trace_id), JSON dễ debug. Chậm hơn gRPC chút nhưng *interoperable* và dễ phát triển.
3. **Chống vòng lặp delegation vô hạn?** **Depth limiting** — mỗi lần ủy thác +1 `delegation_depth`, khi `>= MAX_DELEGATION_DEPTH` (=3) thì router từ chối ủy thác tiếp. Bổ sung: theo dõi `trace_id` phát hiện chu trình, timeout tổng, visited set.
4. **Tại sao cần Registry?** Dynamic discovery theo *task* (không hardcode URL), decoupling, mở rộng (agent mới tự đăng ký), nền tảng load-balancing/health-check. Hardcode URL chạy được với demo nhỏ nhưng là anti-pattern khi số agent tăng.

---

## Bài Tập Nâng Cao (hướng giải)

- **Memory:** dùng `context_id` làm khóa lưu lịch sử, hoặc `checkpointer` (`MemorySaver`/`SqliteSaver`) + `thread_id=context_id`.
- **Auth:** `APIKeyMiddleware` kiểm tra header `Authorization: Bearer $A2A_API_KEY` (chừa `/.well-known/` public để vẫn discover).
- **Retry:** `tenacity` với `wait_exponential` — chỉ retry lỗi *transient* (5xx, ConnectError), KHÔNG retry 4xx.
- **Observability:** LangSmith (chỉ cần env var `LANGCHAIN_TRACING_V2=true`) hoặc Prometheus `/metrics` + Grafana, kết hợp `trace_id`.

---

## Assignment — Supervisor–Workers

Phần cải tiến Agent Day08 sang pattern **Supervisor–Workers** nằm trong
[`Lab_Assignment/`](Lab_Assignment/) — xem [`Lab_Assignment/README.md`](Lab_Assignment/README.md).
