# Bài Tập Thực Hành

Thư mục này chứa các bài tập thực hành cho codelab A2A Multi-Agent.

## Danh Sách Bài Tập

### Exercise 2: Tools và Knowledge Base
**File:** `exercise_2_tools.py`  
**Thời gian:** 10 phút  
**Mục tiêu:** Học cách thêm tools và knowledge base vào LLM

**Nhiệm vụ:**
1. Thêm entry về luật lao động vào `LEGAL_KNOWLEDGE`
2. Tạo tool `check_statute_of_limitations` để kiểm tra thời hiệu khởi kiện
3. Test với câu hỏi về thời hiệu

**Chạy:**
```bash
uv run python exercises/exercise_2_tools.py
```

---

### Exercise 4: Multi-Agent với Privacy Agent
**File:** `exercise_4_multiagent.py`  
**Thời gian:** 15 phút  
**Mục tiêu:** Mở rộng multi-agent system với agent mới

**Nhiệm vụ:**
1. Implement `privacy_agent` function
2. Thêm conditional routing cho privacy agent
3. Thêm privacy_agent vào graph
4. Test với câu hỏi về data breach

**Chạy:**
```bash
uv run python exercises/exercise_4_multiagent.py
```

---

## Đáp Án Chi Tiết (đã hoàn thành + kiểm thử)

**⚠️ Lưu ý:** Hãy cố gắng **tự làm trước** khi xem đáp án bên dưới! Code đáp án đã được điền sẵn vào 2 file `.py` (các chỗ đánh dấu `# ✅`).

---

### ✅ Đáp Án Exercise 2 — `exercise_2_tools.py`

**Nhiệm vụ 1 — Thêm entry luật lao động vào `LEGAL_KNOWLEDGE`:**

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

**Nhiệm vụ 2 — Tạo tool `check_statute_of_limitations`:**

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

**Nhiệm vụ 3 — Đăng ký tool + xử lý kết quả:**

```python
tools = [search_legal_knowledge, check_statute_of_limitations]   # ① thêm vào list
# ... trong vòng lặp tool_calls:
elif tool_call["name"] == "check_statute_of_limitations":         # ② thêm nhánh xử lý
    tool_result = check_statute_of_limitations.invoke(tool_call["args"])
```

**Giải thích:**
- `@tool` tự biến hàm thành tool: lấy **tên** từ tên hàm, **mô tả** từ docstring (LLM đọc để biết *khi nào* gọi), **schema tham số** từ type hint (`case_type: str`).
- Tool tra cứu KB (`search_legal_knowledge`) khớp **keyword** — tăng/giảm độ phủ bằng cách thêm từ khóa vào `keywords`.
- **3 bước bắt buộc** khi thêm tool: (1) định nghĩa `@tool` → (2) thêm vào `tools` list (để `bind_tools` thấy) → (3) thêm nhánh `elif` xử lý khi LLM gọi. **Thiếu bước 2 → LLM không biết tool; thiếu bước 3 → tool được gọi nhưng không thực thi.**

**Kết quả test** (tool là keyword/dict thuần, test trực tiếp không cần LLM):
```
search("người lao động bị sa thải")  → [labor_law] Theo Bộ luật Lao động VN 2019...
check_statute("contract")            → Hợp đồng (UCC): 4 năm (UCC § 2-725)
check_statute("labor")               → Tranh chấp lao động cá nhân (VN): 1 năm...
check_statute("xyz")                 → Không xác định thời hiệu cho 'xyz'
```

---

### ✅ Đáp Án Exercise 4 — `exercise_4_multiagent.py`

**Nhiệm vụ 1 — Implement `privacy_agent`** (theo mẫu `tax_agent`/`compliance_agent`):

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về bảo vệ dữ liệu cá nhân. Phân tích khía cạnh quyền riêng tư:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: GDPR, CCPA, data breach notification, quyền của chủ thể dữ liệu, mức phạt."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}   # ghi vào field privacy_analysis
```

**Nhiệm vụ 2 — Routing cho privacy** (trong `check_routing`):

```python
if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu", "rò rỉ", "breach", "khách hàng"]):
    tasks.append(Send("privacy_agent", state))
```

**Nhiệm vụ 3 — Thêm node + edge** (trong `build_graph`):

```python
graph.add_node("privacy_agent", privacy_agent)
graph.add_edge("privacy_agent", "aggregate_results")
```

**Nhiệm vụ 4 — Thêm vào aggregate:**

```python
if state.get("privacy_analysis"):
    sections.append(f"🔒 PHÂN TÍCH QUYỀN RIÊNG TƯ:\n{state['privacy_analysis']}")
```

#### ⚠️ Lỗi quan trọng trong scaffold (phải sửa thì code mới chạy!)

Scaffold gốc nối graph như sau — **bị lỗi**:

```python
graph.add_node("check_routing", check_routing)        # ❌ check_routing trả list[Send]
graph.add_edge("law_agent", "check_routing")
graph.add_conditional_edges("check_routing", lambda x: x)
```

Chạy sẽ báo:
```
InvalidUpdateError: Expected dict, got [Send(node='tax_agent', ...), Send(node='privacy_agent', ...)]
During task 'check_routing'
```

**Nguyên nhân:** một **node** trong LangGraph phải trả về `dict` (cập nhật state). Nhưng `check_routing` trả về `list[Send]` để *điều phối song song* — đó là việc của **hàm conditional-edge**, KHÔNG phải node.

**Cách sửa đúng** — bỏ `check_routing` khỏi danh sách node, dùng nó làm hàm conditional-edge gắn ngay sau `law_agent`:

```python
graph.add_edge(START, "law_agent")
graph.add_conditional_edges(
    "law_agent",
    check_routing,                                    # ✅ hàm trả list[Send]
    ["tax_agent", "compliance_agent", "privacy_agent", "aggregate_results"],
)
graph.add_edge("tax_agent", "aggregate_results")
graph.add_edge("compliance_agent", "aggregate_results")
graph.add_edge("privacy_agent", "aggregate_results")
graph.add_edge("aggregate_results", END)
```

**Giải thích cơ chế:**
- **Node** = một bước xử lý, **trả `dict`** để ghi vào state (vd `law_agent` trả `{"law_analysis": ...}`).
- **Conditional edge** = hàm quyết định đi node nào tiếp theo, **trả tên node hoặc `Send`**. `Send("privacy_agent", state)` nghĩa là "chạy node `privacy_agent` với `state` này".
- Trả về **list nhiều `Send`** → LangGraph chạy chúng **song song** (tax + privacy cùng lúc). `aggregate_results` đợi tất cả nhánh hội tụ (vì có nhiều `add_edge(... , "aggregate_results")`).
- **Reducer `_last_wins`** trên các field (`Annotated[str, _last_wins]`) là bắt buộc: vì nhiều agent chạy song song cùng ghi state, LangGraph cần biết cách gộp — không có reducer sẽ báo lỗi "concurrent update".

**Kết quả test** (dùng **mock LLM** để khỏi tốn quota Gemini — chỉ kiểm tra routing + luồng graph):
```
routing("rò rỉ dữ liệu khách hàng và trốn thuế")  → ['tax_agent', 'privacy_agent']
routing("vi phạm hợp đồng đơn thuần")             → ['aggregate_results']   (không match keyword → đi thẳng aggregate)
full graph: law=True  tax=True  privacy=True  compliance=False   ← chạy SONG SONG đúng 3 nhánh cần
```
→ Câu hỏi *"rò rỉ dữ liệu + trốn thuế"* kích hoạt **tax + privacy** (không có keyword compliance như sec/sox nên Compliance bị bỏ — đúng thiết kế *conditional routing*).

---

## 💡 Bài học rút ra từ 2 exercise

1. **Thêm tool (Ex2):** luôn đủ 3 bước — định nghĩa `@tool` → thêm vào `tools` list → xử lý khi gọi.
2. **Thêm agent (Ex4):** đồng bộ 4 chỗ — state field (có reducer) → hàm agent → node + edge → aggregate.
3. **Node vs Conditional-edge:** node trả `dict`; muốn fan-out song song bằng `Send` thì dùng **conditional-edge**, không nhét vào node.
4. **Conditional routing** chỉ gọi agent *thực sự cần* → tiết kiệm thời gian & quota (xem thêm phần Latency trong `CODELAB.md`).

> 📌 **Lưu ý chạy thật:** 2 file dùng `get_llm()` → cần `GEMINI_API_KEY` trong `.env` và còn quota (free tier 20 request/model/ngày). Ex2 có thể test **tool trực tiếp** không cần LLM; Ex4 có thể test **routing + graph** bằng mock LLM (như trên) khi hết quota.

---

## Hướng Dẫn Làm Bài

### 1. Đọc TODO Comments
Mỗi file có các comment `# TODO:` chỉ ra chỗ cần điền code.

### 2. Tìm Gợi Ý
Các comment `# Gợi ý:` cho biết hướng làm.

### 3. Tham Khảo Stages
Code trong `stages/*` là examples tốt để tham khảo.

### 4. Test Thường Xuyên
Sau mỗi thay đổi, chạy lại để kiểm tra.

### 5. Debug
Nếu lỗi:
- Đọc error message cẩn thận
- Check syntax (dấu ngoặc, indentation)
- Thêm `print()` để xem giá trị biến
- So sánh với code trong stages

---

## Bài Tập Nâng Cao (Optional)

Sau khi hoàn thành 2 bài tập chính, bạn có thể thử:

### Challenge 1: Financial Agent
Thêm `financial_agent` vào multi-agent system để phân tích thiệt hại tài chính.

### Challenge 2: Conversation Memory
Implement memory để agent nhớ các câu hỏi trước đó.

### Challenge 3: Custom Tool
Tạo tool gọi API thực (ví dụ: tra cứu luật từ database online).

### Challenge 4: Error Handling
Thêm try-catch và retry logic khi tool fails.

---

## Câu Hỏi Thường Gặp

**Q: Làm sao biết code đúng chưa?**  
A: Chạy file và xem output. Nếu không có error và có kết quả hợp lý là OK.

**Q: Tool không được gọi?**  
A: Check xem đã thêm vào `tools` list và `.bind_tools()` chưa.

**Q: Agent không chạy song song?**  
A: Đảm bảo dùng `Send()` API và các agents không phụ thuộc lẫn nhau.

**Q: Import error?**  
A: Chạy `uv sync` để cài đặt dependencies.

---

## Hỗ Trợ

Nếu gặp khó khăn:
1. Đọc lại phần lý thuyết trong `CODELAB.md`
2. Xem `QUICK_REFERENCE.md` cho syntax
3. Hỏi bạn bè hoặc giảng viên
4. Xem **"Đáp Án Chi Tiết"** ở trên (last resort!)

**Chúc bạn làm bài tốt! 💪**
