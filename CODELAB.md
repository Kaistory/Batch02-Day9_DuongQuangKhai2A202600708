# Codelab: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Thời gian:** 2 giờ  
**Ngôn ngữ:** Python 3.11+  
**Công nghệ:** LangGraph, LangChain, A2A SDK

## Mục Tiêu Học Tập

Sau khi hoàn thành codelab này, bạn sẽ:
- Hiểu cách LLM hoạt động từ cơ bản đến nâng cao
- Biết cách tích hợp tools và RAG vào LLM
- Xây dựng được single agent với ReAct pattern
- Tạo multi-agent system với LangGraph
- Triển khai distributed agents với A2A protocol

## Chuẩn Bị

### Yêu Cầu Hệ Thống
- Python 3.11 trở lên
- [uv](https://docs.astral.sh/uv/) package manager
- API key từ [Google AI Studio](https://aistudio.google.com/apikey) (Gemini)

### Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd legal_multiagent

# Cài đặt dependencies
uv sync

# Cấu hình environment
cp .env.example .env
# Sửa file .env, thêm GEMINI_API_KEY của bạn (lấy tại https://aistudio.google.com/apikey)
# và đặt GEMINI_MODEL=gemini-2.5-flash
```

---

## Phần 1: Direct LLM Calling (20 phút)

### Lý Thuyết

LLM (Large Language Model) ở dạng cơ bản nhất là một API nhận input text và trả về output text. Không có memory, không có tools, chỉ dựa vào training data.

**Ưu điểm:**
- Đơn giản, dễ implement
- Phản hồi nhanh

**Nhược điểm:**
- Không có kiến thức real-time
- Không thể tra cứu database
- Không có context giữa các lần gọi

### Thực Hành

**Bước 1:** Chạy demo Stage 1

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**Bước 2:** Đọc và hiểu code

Mở file `stages/stage_1_direct_llm/main.py` và trả lời:

1. LLM được khởi tạo như thế nào? (Tìm hàm `get_llm()`)
2. Message được gửi đến LLM có cấu trúc gì?
3. Tại sao cần có `SystemMessage` và `HumanMessage`?

#### 💡 Giải Đáp

**1. LLM được khởi tạo như thế nào? (`get_llm()`)**

Hàm `get_llm()` nằm trong `common/llm.py` — một *factory function* dùng chung cho tất cả các stage. Nó trả về một client `ChatOpenAI` (trỏ vào **Google Gemini** qua endpoint tương thích OpenAI):

```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        openai_api_key=os.getenv("GEMINI_API_KEY"),
        openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "1024")),
        temperature=0.3,
    )
```

Các tham số quan trọng:
- `model` — tên model, đọc từ `GEMINI_MODEL` (vd `gemini-2.5-flash`).
- `openai_api_key` — API key Gemini, đọc từ `GEMINI_API_KEY` trong `.env`.
- `openai_api_base` — **điểm mấu chốt**: trỏ về endpoint **OpenAI-compatible của Gemini** (`https://generativelanguage.googleapis.com/v1beta/openai/`). Vì Gemini cung cấp API *tương thích OpenAI*, ta vẫn dùng được class `ChatOpenAI` của LangChain — chỉ đổi `base_url` + key + tên model là chuyển nhà cung cấp, **không phải viết lại code**.
- `max_tokens` — giới hạn token câu trả lời (mặc định 1024). ⚠️ Với `gemini-2.5-flash` có "thinking mode" ăn token, nên đặt cao hơn (vd `GEMINI_MAX_TOKENS=2048`) kẻo câu trả lời bị cụt.

Lý do tách ra factory: mọi agent gọi `get_llm()` để có cấu hình LLM **thống nhất** — sửa một chỗ, áp dụng toàn hệ thống. (Chính nhờ vậy, việc **đổi từ OpenRouter sang Gemini** chỉ cần sửa đúng 1 hàm này + file `.env`.)

**2. Message gửi đến LLM có cấu trúc gì?**

LLM không nhận một chuỗi text phẳng, mà nhận một **danh sách các message có vai trò (role)**. Trong `stage_1_direct_llm/main.py`:

```python
messages = [
    SystemMessage(content="You are a legal expert. Provide a clear, concise analysis..."),
    HumanMessage(content=QUESTION),
]
response = await llm.ainvoke(messages)
```

Cấu trúc này tương ứng với format chat chuẩn của OpenAI:

```json
[
  { "role": "system", "content": "You are a legal expert..." },
  { "role": "user",   "content": "What are the legal consequences..." }
]
```

Mỗi message gồm 2 phần: **role** (vai trò: system / user / assistant) và **content** (nội dung). LangChain bọc chúng thành các class `SystemMessage`, `HumanMessage`, `AIMessage` cho dễ đọc và type-safe.

**3. Tại sao cần `SystemMessage` và `HumanMessage`?**

Hai loại message này có **vai trò khác nhau**, và LLM xử lý chúng khác nhau:

| Message | Role | Mục đích |
|---|---|---|
| `SystemMessage` | `system` | Thiết lập **bối cảnh, vai trò, quy tắc** cho LLM. Đây là "chỉ thị nền" định hình *cách* LLM trả lời (ví dụ: "Bạn là chuyên gia pháp lý, trả lời dưới 300 từ"). |
| `HumanMessage` | `user` | **Câu hỏi/yêu cầu thực tế** của người dùng (ví dụ: câu hỏi về hậu quả vi phạm NDA). |

Vì sao phải tách:
- **Phân tách chỉ thị và dữ liệu** — `SystemMessage` định nghĩa "tính cách & luật chơi" một lần, còn `HumanMessage` thay đổi theo từng câu hỏi. LLM ưu tiên tuân thủ system prompt mạnh hơn.
- **Bảo mật & kiểm soát** — chỉ thị quan trọng (vai trò, giới hạn) nằm ở `system`, khó bị input người dùng ghi đè hơn.
- **Đa lượt hội thoại** — khi mở rộng, lịch sử hội thoại là chuỗi xen kẽ `HumanMessage` (người hỏi) và `AIMessage` (LLM trả lời), trong khi `SystemMessage` giữ nguyên ở đầu. Đây là nền tảng cho memory ở các stage sau.

**Bài Tập 1.1:** Thay đổi câu hỏi

Sửa biến `QUESTION` thành câu hỏi pháp lý khác (tiếng Việt hoặc tiếng Anh) và chạy lại.

#### 💡 Lời Giải Bài Tập 1.1

**Bước 1 — Sửa biến `QUESTION`** trong `stages/stage_1_direct_llm/main.py`:

```python
QUESTION = "Người lao động bị sa thải trái pháp luật có những quyền lợi gì theo Bộ luật Lao động Việt Nam?"
```

**Bước 2 — Chạy lại:**

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**⚠️ Lưu ý quan trọng trên Windows — lỗi `UnicodeEncodeError`**

Nếu câu hỏi/câu trả lời có ký tự tiếng Việt, trên Windows bạn có thể gặp lỗi:

```
UnicodeEncodeError: 'charmap' codec can't encode characters... maps to <undefined>
```

Đây **không phải lỗi code** mà do console Windows mặc định dùng bảng mã `cp1252`, không in được Unicode/tiếng Việt. Khắc phục bằng cách ép Python xuất UTF-8:

```bash
# PowerShell
$env:PYTHONIOENCODING="utf-8"; uv run python stages/stage_1_direct_llm/main.py

# Git Bash / Linux / macOS
PYTHONIOENCODING=utf-8 uv run python stages/stage_1_direct_llm/main.py
```

**Bước 3 — Kết quả & quan sát:**

LLM trả lời mạch lạc về quyền lợi người lao động bị sa thải trái luật (quyền phục hồi công việc, trả lương, bồi thường, thời hiệu khởi kiện 01 năm...).

Điều cần rút ra — đây chính là **giới hạn của Stage 1**:
- Câu trả lời hoàn toàn dựa vào **training data**, LLM tự "nhớ" về Bộ luật Lao động 2019.
- Nó **không tra cứu** văn bản luật thật → có thể trích sai số điều luật, sai con số, hoặc lỗi thời nếu luật đã sửa đổi.
- Không có cách nào **kiểm chứng nguồn** (cite điều khoản cụ thể từ database).

→ Đây là lý do **Stage 2** thêm RAG + Tools để "neo" (ground) câu trả lời vào dữ liệu thật.

**Bài Tập 1.2:** Thêm temperature control

Thêm parameter `temperature=0.3` vào hàm `get_llm()` trong `common/llm.py` để làm output ổn định hơn.

#### 💡 Lời Giải Bài Tập 1.2

**Bước 1 — Thêm `temperature=0.3`** vào `get_llm()` trong `common/llm.py`:

```python
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        openai_api_key=os.getenv("GEMINI_API_KEY"),
        openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "1024")),
        temperature=0.3,  # ← thêm dòng này
    )
```

**Bước 2 — Chạy lại:**

```bash
# Windows PowerShell
$env:PYTHONIOENCODING="utf-8"; uv run python stages/stage_1_direct_llm/main.py
```

**`temperature` là gì?**

`temperature` kiểm soát **mức độ ngẫu nhiên / sáng tạo** của LLM khi chọn từ tiếp theo. Giá trị thường từ `0.0` đến `2.0`:

| Temperature | Hành vi | Phù hợp với |
|---|---|---|
| `0.0` – `0.3` | **Ổn định, nhất quán**, ít ngẫu nhiên. Cùng câu hỏi → câu trả lời gần như giống nhau. | Pháp lý, tài chính, code, dữ kiện cần chính xác |
| `0.7` – `1.0` | Cân bằng giữa chính xác và đa dạng | Chatbot, hỏi đáp thông thường |
| `1.2` – `2.0` | **Sáng tạo, đa dạng cao**, đôi khi "bịa" nhiều hơn | Viết truyện, brainstorm ý tưởng |

**Vì sao dùng `0.3` cho hệ thống pháp lý?**

Câu trả lời pháp lý cần **chính xác và đáng tin cậy**, không cần "sáng tạo". Đặt `temperature=0.3`:
- Giảm việc LLM bịa ra điều luật/con số khác nhau giữa các lần chạy.
- Cho output **ổn định hơn** — dễ kiểm thử (test) và tái lập (reproduce) kết quả.
- Vẫn giữ chút linh hoạt về cách diễn đạt (không cứng nhắc như `0.0`).

**Quan sát:** Thử chạy chương trình **vài lần** với cùng câu hỏi. Ở `temperature` thấp, các lần trả lời sẽ có nội dung và cấu trúc rất giống nhau; nếu tăng lên `temperature=1.5` rồi chạy lại, bạn sẽ thấy câu trả lời thay đổi nhiều hơn giữa các lần.

> 📝 **Nâng cao (tùy chọn):** Có thể biến `temperature` thành cấu hình động giống `max_tokens`:
> `temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3"))` — để chỉnh qua file `.env` mà không phải sửa code.

---

## Phần 2: LLM + RAG & Tools (30 phút)

### Lý Thuyết

**RAG (Retrieval-Augmented Generation):** Cho phép LLM tra cứu knowledge base trước khi trả lời.

**Tools:** Các function mà LLM có thể gọi để thực hiện tác vụ cụ thể (tính toán, query database, gọi API).

**Function Calling Flow:**
1. LLM nhận câu hỏi + danh sách tools
2. LLM quyết định gọi tool nào (hoặc không gọi)
3. Tool được execute, trả về kết quả
4. LLM nhận kết quả và tạo câu trả lời cuối cùng

### Thực Hành

**Bước 1:** Chạy demo Stage 2

```bash
uv run python stages/stage_2_rag_tools/main.py
```

**Bước 2:** Phân tích code

Mở `stages/stage_2_rag_tools/main.py` và tìm:

1. Hàm `@tool` decorator được dùng ở đâu?
2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
3. LLM được bind với tools ra sao? (Tìm `.bind_tools()`)

#### 💡 Giải Đáp

**1. `@tool` decorator được dùng ở đâu?**

Trong `stages/stage_2_rag_tools/main.py`, `@tool` (import từ `langchain_core.tools`) được đặt **phía trên định nghĩa hàm** để biến một hàm Python thường thành **tool mà LLM có thể gọi**. File này có 2 tool:

```python
@tool
def search_legal_database(query: str) -> str:
    """Search the legal knowledge base for relevant statutes, case law, and legal principles."""
    ...

@tool
def calculate_damages(breach_type: str, contract_value: float) -> str:
    """Calculate estimated damages for a contract breach based on type and contract value."""
    ...
```

Decorator `@tool` tự động làm 3 việc quan trọng:
- **Lấy tên tool** từ tên hàm (`search_legal_database`, `calculate_damages`).
- **Lấy mô tả** từ docstring (`"""..."""`) — đây chính là phần LLM đọc để *quyết định khi nào gọi tool*. → Docstring viết rõ ràng = LLM gọi tool đúng lúc.
- **Suy ra schema tham số** từ type hints (`query: str`, `contract_value: float`) → LLM biết phải truyền argument gì, kiểu gì.

Cả hai tool được gom vào danh sách: `TOOLS = [search_legal_database, calculate_damages]`.

**2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?**

Là một **list các dict** — đóng vai trò "cơ sở tri thức giả lập" (trong production sẽ là vector database thật). Mỗi entry có 3 trường:

```python
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",                                  # mã định danh nguồn
        "keywords": ["breach", "contract", "remedies", ...], # từ khóa để tìm kiếm
        "text": "Under the Uniform Commercial Code...",      # nội dung pháp lý đầy đủ
    },
    ...
]
```

| Trường | Vai trò |
|---|---|
| `id` | Mã định danh ngắn của nguồn (vd `ucc_breach`, `nda_trade_secret`) — dùng để trích dẫn trong kết quả: `[ucc_breach] ...` |
| `keywords` | Danh sách từ khóa. Tool `search_legal_database` **so khớp** từ khóa này với câu query để chấm điểm độ liên quan |
| `text` | Nội dung pháp lý thật mà LLM sẽ dùng làm căn cứ trả lời |

Cách tìm kiếm hoạt động (trong `search_legal_database`): tách query thành các từ → đếm số từ khóa trùng (`overlap`) với từng entry → sắp xếp giảm dần → lấy **2 entry điểm cao nhất**. Đây là dạng *keyword matching* đơn giản; hệ thống thật sẽ dùng *semantic search* (so khớp theo ngữ nghĩa bằng embedding).

**3. LLM được bind với tools ra sao? (`.bind_tools()`)**

Tại dòng 158 trong `main()`:

```python
llm = get_llm()
llm_with_tools = llm.bind_tools(TOOLS)   # ← gắn danh sách tools vào LLM
tool_map = {t.name: t for t in TOOLS}    # map tên → hàm, để gọi lại sau
```

`.bind_tools(TOOLS)` **không gọi tool** — nó chỉ "khai báo" cho LLM biết *có những tool nào, tên gì, nhận tham số gì*. LLM sau đó tự quyết định có gọi tool hay không.

Luồng 3 bước (đúng như khi chạy thực tế ở trên):

```
Step 1: llm_with_tools.ainvoke(messages)
        → LLM trả về response.tool_calls (vd: gọi search_legal_database
          với query="breach of NDA remedies")  ← LLM tự chọn tool & args

Step 2: Code lặp qua response.tool_calls, dùng tool_map để tìm hàm,
        thực thi tool, rồi đẩy kết quả vào messages dưới dạng ToolMessage

Step 3: llm_with_tools.ainvoke(messages)  ← gọi LLM lần 2, lần này
        có thêm kết quả tool → LLM tạo câu trả lời cuối được "neo" vào dữ liệu
```

**Điểm mấu chốt — đây là giới hạn của Stage 2:** vòng lặp gọi tool (Step 1→2→3) **do ta viết thủ công**, và chỉ chạy **một vòng**. Nếu LLM cần tra cứu thêm sau khi xem kết quả, nó không tự làm được. → **Stage 3** dùng ReAct agent để **tự động hóa** vòng lặp này (LLM tự lặp Think → Act → Observe đến khi đủ thông tin).

**Bài Tập 2.1:** Thêm knowledge base entry

Thêm một entry mới vào `LEGAL_KNOWLEDGE` về luật lao động:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

#### 💡 Lời Giải Bài Tập 2.1

**Bước 1 — Thêm entry vào list `LEGAL_KNOWLEDGE`** trong `stages/stage_2_rag_tools/main.py` (dán dict ở trên vào, ngay sau entry `injunctive_relief`). Lưu ý dấu phẩy `,` ngăn cách giữa các entry.

**Bước 2 — Test (không cần tốn credit LLM):** Tool `search_legal_database` chỉ là **keyword matching** thuần Python, nên có thể gọi trực tiếp để kiểm tra:

```bash
$env:PYTHONIOENCODING="utf-8"; uv run python -c "import sys; sys.path.insert(0,'.'); from stages.stage_2_rag_tools.main import search_legal_database, LEGAL_KNOWLEDGE; print('Tổng entry:', len(LEGAL_KNOWLEDGE)); print(search_legal_database.invoke({'query': 'termination labor sa thải'}))"
```

**Kết quả:**

```
Tổng entry: 6
[labor_law] Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể
đơn phương chấm dứt hợp đồng trong các trường hợp: (1)...
```

**Quan sát & điểm rút ra:**

- Knowledge base tăng từ 5 → **6 entry**, tool tìm đúng `labor_law` khi query chứa từ khóa khớp (`labor`, `termination`, `sa thải`).
- Cơ chế khớp là **giao tập hợp từ khóa** (xem lại Giải Đáp Stage 2): query được `.split()` thành các từ → đếm số từ trùng với `keywords` của entry → entry nào điểm cao nhất được chọn.
- ⚠️ **Hạn chế của keyword matching:** chỉ khớp **đúng chữ**. Nếu người dùng hỏi "đuổi việc nhân viên" (không có chữ "sa thải" hay "termination") thì sẽ **không tìm thấy** dù cùng nghĩa. Đây là lý do hệ thống thật dùng **semantic search** (embedding) để khớp theo *ngữ nghĩa* thay vì từ khóa cứng.

**Bài Tập 2.2:** Tạo tool mới

Tạo một tool `@tool` mới tên `check_statute_of_limitations` nhận vào `case_type` (string) và trả về thời hiệu khởi kiện:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm tool này vào danh sách tools và test.

#### 💡 Lời Giải Bài Tập 2.2

**Bước 1 — Thêm tool mới** vào `stages/stage_2_rag_tools/main.py` (đặt sau `calculate_damages`):

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

**Bước 2 — Đăng ký tool vào danh sách `TOOLS`** (rất quan trọng, nếu quên thì LLM không thấy tool):

```python
TOOLS = [search_legal_database, calculate_damages, check_statute_of_limitations]
```

**Bước 3 — Đổi câu hỏi để kích hoạt tool mới** (câu hỏi cũ về NDA không liên quan đến thời hiệu nên LLM sẽ không gọi tool này):

```python
QUESTION = "What is the statute of limitations for a contract breach, and what damages apply to a willful breach of a $100,000 contract?"
```

**Bước 4 — Chạy test:**

```bash
$env:PYTHONIOENCODING="utf-8"; uv run python stages/stage_2_rag_tools/main.py
```

**Kết quả — LLM tự gọi cả 3 tool:**

```
>>> Step 2: LLM requested 3 tool call(s):

  Tool: check_statute_of_limitations
  Args: {'case_type': 'contract'}
  Result: 4 năm (UCC § 2-725)          ← tool MỚI hoạt động!

  Tool: calculate_damages
  Args: {'breach_type': 'willful breach', 'contract_value': 100000}
  Result: ... Total estimated exposure: $215,000.00

  Tool: search_legal_database
  Args: {'query': 'statute of limitations contract breach damages...'}
  Result: [ucc_breach] Under the Uniform Commercial Code...
```

**Quan sát & điểm rút ra:**

- LLM **tự quyết định** gọi `check_statute_of_limitations` với `case_type='contract'` — ta không hề chỉ định. Nó đọc **docstring** ("Kiểm tra thời hiệu khởi kiện...") và **type hint** (`case_type: str`) để biết tool làm gì và truyền argument nào.
- Câu hỏi chứa **2 ý** (thời hiệu + bồi thường) → LLM gọi nhiều tool **song song** trong một lượt, rồi tổng hợp tất cả vào câu trả lời cuối.
- Đây là minh chứng cho 3 việc mà `@tool` tự động hóa (đã giải thích ở phần Giải Đáp Stage 2): tên tool, mô tả từ docstring, schema tham số từ type hint.

> ⚠️ **Lỗi thường gặp:** Định nghĩa tool xong nhưng **quên thêm vào `TOOLS`** → LLM hoàn toàn không biết tool tồn tại và sẽ không bao giờ gọi. Luôn nhớ đăng ký vào danh sách `TOOLS`.

---

## Phần 3: Single Agent với ReAct (25 phút)

### Lý Thuyết

**ReAct Pattern:** Reasoning + Acting

Agent tự động lặp lại chu trình:
1. **Think:** Suy nghĩ cần làm gì
2. **Act:** Gọi tool
3. **Observe:** Nhận kết quả
4. Lặp lại cho đến khi có câu trả lời cuối cùng

LangGraph cung cấp `create_react_agent` để tự động hóa pattern này.

### Thực Hành

**Bước 1:** Chạy demo Stage 3

```bash
uv run python stages/stage_3_single_agent/main.py
```

**Bước 2:** Quan sát output

Chú ý cách agent tự động:
- Quyết định tool nào cần gọi
- Gọi nhiều tools liên tiếp
- Tổng hợp kết quả

**Bước 3:** Đọc code

Mở `stages/stage_3_single_agent/main.py`:

1. Tìm `create_react_agent()` — đây là magic function
2. So sánh với Stage 2: không còn manual tool loop
3. Xem `agent_executor.invoke()` — chỉ cần gọi một lần

#### 💡 Giải Đáp

**1. `create_react_agent()` — "magic function"**

Nằm trong `main()` của `stages/stage_3_single_agent/main.py`:

```python
from langgraph.prebuilt import create_react_agent

llm = get_llm()
graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)
```

Chỉ **một dòng** này thay thế toàn bộ vòng lặp tool thủ công của Stage 2. Nó nhận 3 thứ và trả về một **agent graph** hoàn chỉnh:
- `model` — LLM (từ `get_llm()`).
- `tools` — danh sách tool (`TOOLS`).
- `prompt` — system prompt định hình vai trò agent.

Bên trong, LangGraph tự dựng một **đồ thị 2 node lặp vòng**: node `agent` (LLM suy nghĩ + quyết định gọi tool) ↔ node `tools` (thực thi tool). Agent tự lặp **Think → Act → Observe** cho đến khi không cần gọi tool nữa thì trả lời cuối.

**2. So sánh với Stage 2: không còn manual tool loop**

| | **Stage 2** (thủ công) | **Stage 3** (`create_react_agent`) |
|---|---|---|
| Bind tool | `llm.bind_tools(TOOLS)` | Tự động bên trong |
| Vòng lặp gọi tool | **Ta tự viết** Step 1→2→3, lặp qua `tool_calls`, tạo `ToolMessage` | **Agent tự lo** toàn bộ |
| Số vòng lặp | **Cố định 1 vòng** — không tra cứu lại được | **Lặp bao nhiêu lần tùy ý** đến khi đủ thông tin |
| `tool_map` thủ công | Cần `{t.name: t for t in TOOLS}` | Không cần |
| Dòng code điều phối | ~40 dòng | ~2 dòng |

Ở Stage 2 ta phải tự đóng vai "người điều phối": nhận `tool_calls`, gọi hàm, nhét kết quả về. Stage 3 giao toàn bộ trách nhiệm đó cho agent — đó là sự khác biệt cốt lõi giữa "LLM + tools" và "agent".

**3. Gọi agent — "chỉ cần gọi một lần"**

> ⚠️ **Lưu ý:** Giáo trình viết `agent_executor.invoke()`, nhưng **code thực tế trong repo** dùng cách hiện đại hơn của LangGraph — `graph.astream()` để **stream từng bước** (giúp ta in ra Think/Act/Observe). Bản chất giống nhau: ta **chỉ "khởi động" agent một lần**, mọi vòng lặp bên trong agent tự xử lý.

```python
inputs = {"messages": [{"role": "user", "content": QUESTION}]}

async for chunk in graph.astream(inputs, stream_mode="updates"):
    ...  # in ra từng bước Think / Act / Observe
```

Nếu chỉ cần kết quả cuối (không cần xem từng bước), có thể gọi gọn hơn:

```python
result = await graph.ainvoke(inputs)   # gọi MỘT lần, agent tự lặp bên trong
print(result["messages"][-1].content)
```

Khác hẳn Stage 2 (phải gọi LLM thủ công 2 lần ở Step 1 và Step 3) — ở đây **một lệnh duy nhất** và agent tự quyết định lặp bao nhiêu vòng.

**Quan sát khi chạy thực tế** (câu hỏi có 2 vi phạm: data privacy + thuế):

```
[Step 1] THINK + ACT → agent gọi 5 tool cùng lúc:
         search_legal_database (data privacy), search_legal_database (tax),
         check_compliance_requirements, calculate_penalty x2
[Step 2-6] OBSERVE → nhận kết quả từng tool
[Step 7] FINAL ANSWER → tổng hợp toàn bộ thành phân tích pháp lý hoàn chỉnh
```

→ Agent **tự bóc tách** câu hỏi phức tạp thành nhiều sub-task, gọi đúng tool cho từng phần, rồi tổng hợp — tất cả **không cần ta viết một dòng điều phối nào**.

**Giới hạn của Stage 3 (dẫn sang Stage 4):** vẫn là **một agent duy nhất** xử lý mọi lĩnh vực (luật, thuế, compliance) với cùng một system prompt → không chuyên môn hóa, và các tool gọi tuần tự. **Stage 4** tách thành nhiều agent chuyên biệt chạy **song song**.

**Bài Tập 3.1:** Thêm tool tra cứu án lệ

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

Thêm vào tools list và test với câu hỏi về breach of contract.

#### 💡 Lời Giải Bài Tập 3.1

**Bước 1 — Thêm tool** `search_case_law` vào `stages/stage_3_single_agent/main.py` (sau `check_compliance_requirements`).

**Bước 2 — Đăng ký vào `TOOLS`:**

```python
TOOLS = [search_legal_database, calculate_penalty, check_compliance_requirements, search_case_law]
```

**Bước 3 — Đổi câu hỏi sang chủ đề breach of contract** để kích hoạt tool:

```python
QUESTION = (
    "A company breached a supply contract worth $200,000. What are the legal remedies, "
    "relevant case law, and estimated penalties?"
)
```

**Bước 4 — Chạy test:**

```bash
$env:PYTHONIOENCODING="utf-8"; uv run python stages/stage_3_single_agent/main.py
```

**Kết quả — agent tự gọi tool mới cùng các tool khác:**

```
[Step 1] THINK + ACT → agent gọi 3 tool:
  search_legal_database (query: contract breach remedies...)
  search_case_law       (keywords: supply contract breach damages)  ← tool MỚI
  calculate_penalty     (violation_type: contract_breach...)

[Step 2] OBSERVE → search_case_law trả về:
  "Hadley v. Baxendale (1854) - Consequential damages"   ← khớp keyword "breach"

[Step 5] FINAL ANSWER → agent dẫn chiếu án lệ Hadley v. Baxendale ngay
         trong phần phân tích Consequential Damages
```

**Quan sát & điểm rút ra:**

- Khác với Stage 2 (ta phải tự đổi câu hỏi *và* tự gọi tool), ở Stage 3 ta chỉ cần **thêm tool vào `TOOLS`** — agent **tự nhận ra** câu hỏi có ý "case law" và tự gọi `search_case_law`. Không cần sửa logic điều phối.
- Tool dùng **keyword matching đơn giản**: hàm lặp qua dict `cases`, trả về án lệ đầu tiên có keyword khớp. Câu query chứa "breach" → khớp ngay `"breach": "Hadley v. Baxendale..."`.
- Agent **kết hợp** kết quả của cả 3 tool (luật + án lệ + tiền phạt) vào một câu trả lời mạch lạc — minh chứng cho sức mạnh của ReAct: tự bóc tách và tổng hợp đa nguồn.

> 💡 **Mẹo:** Muốn chắc chắn agent gọi tool mới, hãy đưa **từ khóa khớp** vào câu hỏi (ví dụ "case law", "breach"). Nếu câu hỏi không liên quan, agent có quyền **không gọi** tool đó — đúng như thiết kế.

**Bài Tập 3.2:** Debug agent reasoning

Thêm `verbose=True` vào `create_react_agent()` để xem chi tiết quá trình suy nghĩ của agent.

#### 💡 Lời Giải Bài Tập 3.2

**⚠️ Đính chính quan trọng — `verbose=True` KHÔNG dùng được!**

Đề bài viết theo API LangChain cũ (`AgentExecutor`). Với **LangGraph hiện tại**, `create_react_agent()` **không có** tham số `verbose`. Thử thêm `verbose=True` sẽ báo lỗi ngay:

```python
graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT, verbose=True)
```
```
TypeError: create_react_agent() got unexpected keyword arguments: {'verbose': True}
```

**✅ Cách đúng — dùng `debug=True`:**

```python
graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT, debug=True)
```

(Kiểm chứng signature: `uv run python -c "from langgraph.prebuilt import create_react_agent; import inspect; print(inspect.signature(create_react_agent))"` → thấy `debug: bool = False`, không hề có `verbose`.)

**Chạy lại:**

```bash
$env:PYTHONIOENCODING="utf-8"; uv run python stages/stage_3_single_agent/main.py
```

**Kết quả — `debug=True` in ra toàn bộ "nội tâm" của agent ở mỗi bước:**

```
[values] {'messages': [HumanMessage(content='A company breached...')]}   ← state đầu vào

[updates] {'agent': {'messages': [AIMessage(
    content="I'll help you analyze...",
    response_metadata={'token_usage': {'completion_tokens': 210, 'prompt_tokens': 1100, ...},
                       'cost': 0.00645, 'model_name': 'anthropic/claude-4.5-sonnet...'},
    tool_calls=[{'name': 'search_legal_database', 'args': {...}, 'id': 'toolu_...'},
                {'name': 'search_case_law', ...},
                {'name': 'calculate_penalty', ...}]   ← agent quyết định gọi 3 tool
)]}}

[updates] {'tools': {'messages': [ToolMessage(
    content='Hadley v. Baxendale (1854)...', name='search_case_law', tool_call_id='toolu_...'
)]}}                                                  ← kết quả tool trả về
```

**`debug=True` cho thấy những gì mà output thường ẩn đi:**

| Thông tin | Ý nghĩa khi debug |
|---|---|
| `[values]` | **Toàn bộ state** (lịch sử messages) trước mỗi node — thấy agent "nhớ" gì |
| `[updates]` | Mỗi node (`agent` / `tools`) **thêm message gì** vào state |
| `token_usage`, `cost` | Số token & **chi phí thực tế** từng lượt gọi LLM → hữu ích để tối ưu credit |
| `tool_calls` (kèm `id`) | Agent **quyết định** gọi tool nào, args gì, và `tool_call_id` để khớp kết quả |
| `finish_reason` | `'tool_calls'` (cần gọi tool tiếp) hay `'stop'` (đã xong) |

**Điểm rút ra:**
- `debug=True` là công cụ **soi cơ chế bên trong** ReAct loop — thấy rõ chu trình `agent` (Think+Act) ↔ `tools` (Observe) luân phiên, và *vì sao* agent quyết định như vậy.
- Trong code mẫu, ta để mặc định **`debug=False`** (kèm comment hướng dẫn) để output gọn gàng khi chạy bình thường — chỉ bật khi cần gỡ lỗi.
- Đây là bài học thực tế quan trọng: **tài liệu/giáo trình có thể lỗi thời so với thư viện**. Khi gặp `TypeError` về tham số, hãy kiểm tra `inspect.signature()` để biết API thật.

> 📝 **Bonus — cách debug khác không cần sửa tham số:** Bản thân code đã dùng `graph.astream(..., stream_mode="updates")` để in từng bước Think/Act/Observe thủ công (đẹp và gọn hơn `debug=True`). Đó cũng là một dạng "debug reasoning" do chính tác giả repo viết.

---

## Phần 4: Multi-Agent In-Process (30 phút)

### Lý Thuyết

**Multi-Agent System:** Nhiều agents chuyên môn hóa cùng làm việc.

**Ưu điểm:**
- Mỗi agent tập trung vào domain riêng
- Có thể chạy song song (parallel execution)
- Dễ maintain và mở rộng

**LangGraph StateGraph:**
- Định nghĩa state (dữ liệu chia sẻ giữa các nodes)
- Tạo nodes (các bước xử lý)
- Định nghĩa edges (luồng điều khiển)

**Send API:** Cho phép dispatch nhiều tasks song song.

### Thực Hành

**Bước 1:** Chạy demo Stage 4

```bash
uv run python stages/stage_4_milti_agent/main.py
```

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state
2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`
3. Tìm `Send()` API — dispatch parallel tasks
4. Xem `graph.add_node()` và `graph.add_edge()`

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

#### 💡 Giải Đáp (Bước 2 + Bước 3)

> ⚠️ **Lưu ý:** Tên trong giáo trình là *minh họa lý thuyết*; **code thật** trong `stage_4_milti_agent/main.py` đặt tên khác. Bảng dưới ánh xạ "tên giáo trình → tên code thật".

**1. Shared state — `class State(TypedDict)`**

Code thật đặt tên là **`LegalState`** (dòng 112). Đây là "bộ nhớ dùng chung" mà mọi node đọc/ghi:

```python
class LegalState(TypedDict):
    question: str
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    tax_result: Annotated[str, _last_wins]         # ← có reducer
    compliance_result: Annotated[str, _last_wins]  # ← có reducer
    final_answer: str
```

Điểm tinh tế nhất: `tax_result` và `compliance_result` dùng **`Annotated[str, _last_wins]`**. Vì 2 specialist chạy **song song** và cùng ghi vào state, LangGraph cần một **reducer** (`_last_wins`) để biết cách gộp giá trị — nếu không sẽ báo lỗi "concurrent update". Các field thường (như `law_analysis`) không cần vì chỉ một node ghi.

**2. Các agent functions**

| Tên giáo trình | Tên thật trong code | Vai trò |
|---|---|---|
| `law_agent` | **`analyze_law`** (dòng 126) | Luật sư trưởng phân tích khía cạnh pháp lý |
| — (router) | **`check_routing`** (dòng 145) | Hỏi LLM trả về JSON `{needs_tax, needs_compliance}` |
| `tax_agent` | **`call_tax_specialist`** (dòng 193) | Chuyên gia thuế — chạy như **ReAct agent con** (`create_react_agent`) |
| `compliance_agent` | **`call_compliance_specialist`** (dòng 216) | Chuyên gia tuân thủ — cũng là ReAct agent con |
| `aggregate_results` | **`aggregate`** (dòng 238) | Gộp 3 phân tích thành câu trả lời cuối |

Điểm hay: mỗi specialist **không chỉ là một LLM call** mà là **cả một ReAct agent** (có tool riêng `search_tax_law` / `search_compliance_law`) — tức Stage 4 *lồng* Stage 3 vào trong mỗi node.

**3. `Send()` API — dispatch song song**

Nằm trong hàm **`route_to_specialists`** (dòng 181) — đây là *conditional routing function*:

```python
def route_to_specialists(state: LegalState) -> list[Send]:
    sends = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))        # gửi task tới node tax
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state)) # gửi task tới node compliance
    if not sends:
        sends.append(Send("aggregate", state))
    return sends
```

`Send("tên_node", state)` = "gửi `state` này tới node đó để chạy". Trả về **list nhiều `Send`** → LangGraph chạy chúng **song song**. Đây chính là lý do tax + compliance specialist chạy đồng thời (xem output Bước 1: cả hai "starting" liên tiếp rồi mới "Done").

**4. `graph.add_node()` và `graph.add_edge()`** — trong `create_graph()` (dòng 273):

```python
graph = StateGraph(LegalState)
graph.add_node("analyze_law", analyze_law)            # đăng ký node
graph.add_node("check_routing", check_routing)
graph.add_node("call_tax_specialist", call_tax_specialist)
...
graph.set_entry_point("analyze_law")                  # điểm bắt đầu
graph.add_edge("analyze_law", "check_routing")        # cạnh cố định A→B
graph.add_conditional_edges(                          # cạnh ĐỘNG (rẽ nhánh)
    "check_routing", route_to_specialists,
    ["call_tax_specialist", "call_compliance_specialist", "aggregate"],
)
graph.add_edge("call_tax_specialist", "aggregate")    # specialist → gộp
graph.add_edge("call_compliance_specialist", "aggregate")
graph.add_edge("aggregate", END)
```

- **`add_edge(A, B)`** — cạnh **cố định**: chạy xong A luôn sang B.
- **`add_conditional_edges(node, fn, [...])`** — cạnh **động**: gọi `fn` (chính là `route_to_specialists`) để quyết định đi node nào tiếp theo (có thể nhiều node song song).

**Bước 3 — Vẽ graph:**

> ⚠️ Lệnh `draw_mermaid_png()` cần **IPython + kết nối internet** (gọi server mermaid.ink) nên thường lỗi khi chạy script `.py` thuần. Hai cách thay thế **không cần internet**:

```python
# Cách 1: in mã Mermaid ra text (dán vào https://mermaid.live để xem)
print(graph.get_graph().draw_mermaid())

# Cách 2: in sơ đồ ASCII ngay trong terminal
graph.get_graph().print_ascii()
```

Topology thu được (đã in sẵn trong demo, dòng 312):

```
analyze_law → check_routing → ┬→ call_tax_specialist ───────┐
                              └→ call_compliance_specialist ─┴→ aggregate → END
                                  (chạy song song qua Send)
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
    
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
    
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

#### 💡 Lời Giải Bài Tập 4.1

> ⚠️ Code mẫu trong đề dùng tên lý thuyết (`State`, `aggregate_results`, `privacy_analysis`). Dưới đây là phiên bản **chạy được với code thật** (`LegalState`, `aggregate`), theo đúng phong cách các specialist sẵn có.

**Bước 1 — Thêm field vào `LegalState`** (nhớ reducer `_last_wins` vì node này có thể chạy song song):

```python
class LegalState(TypedDict):
    question: str
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool                                # ← thêm
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    privacy_result: Annotated[str, _last_wins]         # ← thêm (có reducer)
    final_answer: str
```

**Bước 2 — Thêm tool + node specialist** (theo đúng mẫu `call_tax_specialist` — dùng ReAct agent, không phải LLM call trần):

```python
@tool
def search_privacy_law(query: str) -> str:
    """Search data-privacy knowledge base (GDPR, CCPA, etc.).

    Args:
        query: Natural language query about data privacy law.
    """
    knowledge = [
        (["gdpr", "eu", "data subject"],
         "GDPR: fines up to 4% global revenue or EUR 20M. Requires lawful basis, "
         "consent, data subject rights (access, erasure, portability)."),
        (["ccpa", "california", "consumer"],
         "CCPA/CPRA: up to $7,500 per intentional violation; consumer private right "
         "of action $100-$750 per incident for data breaches."),
    ]
    q = query.lower()
    hits = [t for kws, t in knowledge if any(k in q for k in kws)]
    return "\n\n".join(hits) if hits else "No specific privacy law matches found."


async def call_privacy_specialist(state: LegalState) -> dict:
    from langgraph.prebuilt import create_react_agent
    print("\n  [Node: call_privacy_specialist] Privacy specialist agent starting...")
    prompt = (
        "You are a data-privacy counsel specialising in GDPR, CCPA/CPRA, and global "
        "data-protection law. Use the search_privacy_law tool to ground your analysis. "
        "Keep your response under 200 words."
    )
    agent = create_react_agent(model=get_llm(), tools=[search_privacy_law], prompt=prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    final_msg = result["messages"][-1].content
    print(f"  [Node: call_privacy_specialist] Done ({len(final_msg)} chars)")
    return {"privacy_result": final_msg}
```

**Bước 3 — Đăng ký node + nối cạnh tới `aggregate`** (trong `create_graph()`):

```python
graph.add_node("call_privacy_specialist", call_privacy_specialist)
graph.add_edge("call_privacy_specialist", "aggregate")
# và thêm "call_privacy_specialist" vào list của add_conditional_edges(...)
```

**Bước 4 — Cho `aggregate` đọc kết quả mới** (thêm vào hàm `aggregate`):

```python
if state.get("privacy_result"):
    sections.append(f"## Data Privacy Analysis\n{state['privacy_result']}")
```

**Bước 5 — Khởi tạo field trong `graph.ainvoke({...})`** ở `main()`: thêm `"needs_privacy": False, "privacy_result": ""`.

> 💡 **Điểm rút ra:** thêm một agent chuyên môn = 5 chỗ phải đồng bộ: **state field → tool → node → cạnh graph → aggregate**. Đây vừa là sức mạnh (mở rộng dễ) vừa là gánh nặng (nhiều chỗ phải sửa) của in-process multi-agent — và chính là động lực để Stage 5 tách mỗi agent thành **service độc lập** (chỉ cần đăng ký với Registry, không phải sửa graph trung tâm).

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    
    return tasks if tasks else [Send("aggregate_results", state)]
```

#### 💡 Lời Giải Bài Tập 4.2

**Quan trọng — phân biệt 2 cách routing.** Code thật chia routing làm **2 bước** (khác với đề bài gộp làm 1):

| | Đề bài (1 bước) | Code thật (2 bước) |
|---|---|---|
| Cách quyết định | **Keyword matching** trong hàm routing | **Hỏi LLM** (node `check_routing`) → set cờ `needs_*` |
| Hàm dispatch | `check_routing` trả `list[Send]` | `route_to_specialists` đọc cờ → trả `list[Send]` |

Cách của đề (keyword) **nhanh & rẻ** (không tốn LLM call) nhưng cứng nhắc. Cách code thật (LLM router) **thông minh hơn** (hiểu ngữ cảnh, vd "IRS audit" → cần tax dù không có chữ "tax") nhưng tốn thêm 1 lượt gọi LLM.

**Phương án A — Thêm privacy vào router LLM hiện có (khuyến nghị, đúng kiến trúc repo):**

Sửa `check_routing` để LLM trả thêm cờ `needs_privacy`:

```python
# trong system prompt của check_routing, đổi JSON yêu cầu thành:
'{"needs_tax": <bool>, "needs_compliance": <bool>, "needs_privacy": <bool>}'
'needs_privacy = true → câu hỏi liên quan dữ liệu cá nhân, GDPR, CCPA, quyền riêng tư'
# và đọc thêm: needs_privacy = bool(parsed.get("needs_privacy", False))
# return {..., "needs_privacy": needs_privacy}
```

Rồi thêm nhánh vào `route_to_specialists`:

```python
def route_to_specialists(state: LegalState) -> list[Send]:
    sends = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state))
    if state.get("needs_privacy"):                              # ← thêm
        sends.append(Send("call_privacy_specialist", state))
    if not sends:
        sends.append(Send("aggregate", state))
    return sends
```

**Phương án B — Routing bằng keyword (đúng tinh thần đề bài, không tốn LLM):**

Nếu muốn bỏ hẳn node `check_routing` LLM và quyết định bằng từ khóa ngay trong hàm dispatch:

```python
def route_to_specialists(state: LegalState) -> list[Send]:
    q = state["question"].lower()
    sends = []
    if any(kw in q for kw in ["tax", "irs", "thuế", "offshore"]):
        sends.append(Send("call_tax_specialist", state))
    if any(kw in q for kw in ["compliance", "sec", "sox", "regulation"]):
        sends.append(Send("call_compliance_specialist", state))
    if any(kw in q for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        sends.append(Send("call_privacy_specialist", state))
    return sends if sends else [Send("aggregate", state)]
```

> ⚠️ **Lưu ý tên node:** đề bài viết `Send("tax_agent", ...)`, `Send("aggregate_results", ...)` — nhưng **tên node thật** là `Send("call_tax_specialist", ...)` và `Send("aggregate", ...)`. Dùng sai tên → LangGraph báo lỗi "node not found". Luôn khớp với tên đã `add_node()`.

**Điểm rút ra:**
- **Conditional routing** = chỉ chạy agent khi *thực sự cần* → tiết kiệm thời gian & credit (câu hỏi chỉ về thuế thì không gọi compliance/privacy).
- Trade-off **LLM router vs keyword router** là một quyết định thiết kế kinh điển: thông minh & linh hoạt **vs** nhanh & rẻ. Repo này chọn LLM router để xử lý câu hỏi ngôn ngữ tự nhiên phức tạp.

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**
- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**
```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

**Bước 3:** Quan sát logs

Mở 5 terminal tabs và xem logs của từng service:
- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

#### 💡 Lời Giải Bài Tập 5.1 — Trace request flow

Mỗi request được gắn một **`trace_id`** (UUID) đi xuyên suốt mọi agent — đây là kỹ thuật *distributed tracing*. Tìm trong log bằng:

```bash
# Git Bash
grep -iE "trace|routing|discover|returned" .logs/*.log
```

**Dữ liệu thật từ một lần chạy:**

```
[law_agent]  Routing decision: needs_tax=True needs_compliance=False
[law_agent]  GET http://localhost:10000/discover/tax_question → 200 OK     (hỏi Registry tax ở đâu)
[tax_agent]  TaxAgent executing | task=f204282c... context=37ccda06... trace=84602eb0... depth=2
[tax_agent]  POST https://...gemini.../chat/completions → 200 OK           (Gemini xử lý)
[law_agent]  Tax Agent returned 1883 chars
```

**Sequence diagram** (theo đúng luồng quan sát được):

```
test_client    Customer(10100)   Registry(10000)   Law(10101)    Tax(10102)
    │                │                  │               │             │
    │──question────►│                  │               │             │
    │                │──discover(legal_question)───────►│             │
    │                │◄──endpoint:10101─┤               │             │
    │                │──delegate(q, trace_id, depth=1)─►│             │
    │                │                  │     [analyze_law: Gemini]   │
    │                │                  │     [check_routing→tax=True]│
    │                │                  │◄─discover(tax_question)     │
    │                │                  │──endpoint:10102────────────►│
    │                │                  │──delegate(q, trace, depth=2)►│
    │                │                  │               │   [Gemini + search_tax_law]
    │                │                  │◄──────tax_result─────────────┤
    │                │                  │     [aggregate: Gemini]     │
    │                │◄────final_answer─┤               │             │
    │◄──memo─────────┤                  │               │             │
```

**Điểm rút ra:** `trace_id` được sinh **một lần** ở Customer Agent rồi **truyền nguyên vẹn** qua mọi delegation (nằm trong `LawState.trace_id` và A2A message metadata). Nhờ vậy, dù request đi qua 4 service, ta vẫn lọc log theo 1 `trace_id` để dựng lại toàn bộ hành trình — bài học cốt lõi của observability trong hệ phân tán.

#### 💡 Lời Giải Bài Tập 5.2 — Test dynamic discovery (fault tolerance)

**Thực hiện:** dừng Tax Agent rồi gửi lại câu hỏi:

```powershell
# Dừng tax agent (Windows)
Get-NetTCPConnection -LocalPort 10102 -State Listen | `
  Select -Expand OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force }

# Chạy lại client
uv run python test_client.py
```

**Hệ thống xử lý lỗi thế nào — xem code `law_agent/graph.py` (`call_tax`):**

```python
async def call_tax(state: LawState) -> dict:
    try:
        endpoint = await discover("tax_question")   # ① hỏi Registry
        result = await delegate(endpoint=endpoint, ...)  # ② gọi Tax Agent qua HTTP
        return {"tax_result": result}
    except Exception as exc:                          # ③ Tax chết → bắt lỗi
        logger.exception("call_tax failed: %s", exc)
        return {"tax_result": f"[Tax analysis unavailable: {exc}]"}   # ④ degrade nhẹ nhàng
```

Khi Tax Agent down, một trong hai chỗ sẽ ném exception:
- `discover("tax_question")` — nếu Tax đã **unregister** khỏi Registry → không tìm thấy endpoint.
- `delegate(...)` — nếu endpoint còn trong Registry nhưng port đã chết → `httpx.ConnectError`.

Điểm thiết kế quan trọng: lỗi **không làm sập cả hệ thống**. `call_tax` trả về chuỗi `[Tax analysis unavailable: ...]`, node `aggregate` vẫn chạy và tổng hợp phần Legal + Compliance còn lại. Đây là **graceful degradation** — đặc tính sống còn của microservices.

> ⚠️ **Lưu ý thực nghiệm:** Khi mình test, Gemini đôi lúc trả `503 "high demand"` (lỗi *tạm thời* phía Google, không liên quan A2A). Đừng nhầm lỗi quota/503 của LLM với lỗi discovery — đọc kỹ log để phân biệt: lỗi A2A nằm ở dòng `discover`/`delegate`, lỗi LLM nằm ở dòng `POST .../chat/completions`.

#### 💡 Lời Giải Bài Tập 5.3 — Modify agent behavior

**Bước 1 — Sửa system prompt** của Tax Agent. Prompt nằm trong `tax_agent/graph.py` (node tạo `create_react_agent`). Ví dụ ép trả lời cực ngắn:

```python
tax_prompt = (
    "You are a specialist tax attorney. Answer in AT MOST 3 bullet points, "
    "each under 20 words. Use the search_tax_law tool to ground your analysis. "
    "Be extremely concise — no preamble."
)
```

**Bước 2 — Restart CHỈ Tax Agent** (đây là sức mạnh của A2A: sửa 1 agent không cần đụng các agent khác):

```powershell
# Dừng tax agent cũ
Get-NetTCPConnection -LocalPort 10102 -State Listen | `
  Select -Expand OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force }
# Khởi động lại
$env:PYTHONIOENCODING="utf-8"; uv run python -m tax_agent
```

**Bước 3 — Test lại** `uv run python test_client.py` → phần "Tax Analysis" trong memo sẽ ngắn gọn hẳn, trong khi Law và Compliance **giữ nguyên**.

**Điểm rút ra — đây chính là lý do tồn tại của Stage 5:** ở Stage 4 (monolith) muốn đổi hành vi tax phải sửa file chung và chạy lại toàn bộ. Ở Stage 5, mỗi agent là **service độc lập** → sửa & deploy riêng từng agent, không downtime cho phần còn lại. Đổi lại, ta phải quản lý nhiều process + Registry + giao tiếp HTTP (xem bảng so sánh Stage 4 vs 5 ở output demo).

---

### 📊 Phân Tích Latency (đo thật với `gemini-2.5-flash-lite`)

> 🖥️ **Demo trực quan:** mở file [`demo_agents.html`](demo_agents.html) trong trình duyệt — có nút chuyển **Stage 4 ⟷ Stage 5**, animation luồng tương tác giữa các agent, nhật ký real-time, và biểu đồ so sánh latency. Tick *"Bật tối ưu (song song)"* ở Stage 4 để thấy đồng hồ về đích sớm hơn, và bấm **"⚡ Chạy đối chiếu A/B"** để xem 2 thanh baseline vs tối ưu chạy đua trực tiếp — kết quả hiện rõ **2.26s nhanh hơn (−28%): 8.09s → 5.83s**.
>
> 🤖 **Nhập câu hỏi của bạn (chạy agent thật):** chạy `uv run python demo_server.py` rồi mở **http://localhost:8800** — gõ câu hỏi vào ô *"Nhập câu hỏi của bạn"*, bấm **"🚀 Hỏi agent"** để chạy multi-agent thật (Stage 4 in-process), xem **routing thật + thời gian từng node + câu trả lời thật**. (Cần `GEMINI_API_KEY` còn quota.)

#### Câu hỏi 1 — Latency là bao nhiêu giây?

Đo bằng `time.perf_counter()` quanh lúc gửi request → nhận response:

| Cấu hình | Latency | Ghi chú |
|---|---|---|
| **Stage 5** (Distributed A2A) | **23.59 s** | 5 service, qua HTTP + Registry discovery, status `completed`, 2353 ký tự |
| **Stage 4** (In-Process, baseline) | **8.09 s** | Cùng logic, gọi hàm trực tiếp trong 1 process |

→ **Chênh ~15.5 giây** chính là **chi phí của kiến trúc phân tán**: mỗi hop (`test_client → Customer → Law → Tax ‖ Compliance → aggregate`) là một lời gọi HTTP + serialize JSON + một vòng Gemini, cộng thêm các ReAct sub-agent (mỗi cái 2 lượt LLM).

#### Câu hỏi 2 — Đề xuất giảm latency + demo kết quả

**Phân tích per-node baseline Stage 4** (đo bằng `graph.astream`) cho thấy thủ phạm là **chuỗi tuần tự**:

```
analyze_law 2.65s → check_routing(LLM) 0.93s → [tax‖compliance] 1.93s → aggregate 2.59s = 8.09s
```

**Hai phương án (đã đo thật):**

1. **Bỏ LLM router** → thay bằng **keyword routing** (tức thời, 0 lời gọi LLM) → cắt thẳng **0.93s** khỏi critical path.
2. **Chạy `analyze_law` SONG SONG** với tax/compliance — vì cả 3 node chỉ cần `question`, **không phụ thuộc** kết quả của nhau. Dùng `set_conditional_entry_point` + `Send()` để fan-out ngay từ entry:

```python
def fan_out(state) -> list[Send]:
    q = state["question"].lower()
    sends = [Send("analyze_law", state)]              # ← law analysis chạy song song
    if any(k in q for k in ["tax", "irs", "thuế", "avoid"]):
        sends.append(Send("call_tax_specialist", state))
    if any(k in q for k in ["compliance", "sec", "sox", "contract", "breach"]):
        sends.append(Send("call_compliance_specialist", state))
    return sends

g.set_conditional_entry_point(fan_out, ["analyze_law", "call_tax_specialist", "call_compliance_specialist"])
g.add_edge("analyze_law", "aggregate")               # aggregate đợi cả 3 hội tụ
g.add_edge("call_tax_specialist", "aggregate")
g.add_edge("call_compliance_specialist", "aggregate")
```

**Kết quả đo:**

| Pha | Baseline (tuần tự) | Tối ưu (fan-out) | Cải thiện |
|---|---|---|---|
| Tới khi bắt đầu `aggregate` | 5.51 s | **3.24 s** | **−41%** |
| `aggregate` | 2.59 s | ~2.59 s (chiếu từ baseline) | — |
| **Tổng** | **8.09 s** | **≈5.8 s** | **−28%** |

> ⚠️ `aggregate` của bản tối ưu được **chiếu từ baseline** vì quota Gemini free/ngày (20 request/model) đã cạn khi đo — phần fan-out song song (3.24s) là **số đo thật**, đã chứng minh 3 node `analyze_law`/`tax`/`compliance` khởi động đồng thời thay vì xếp hàng.

**Các phương án khác** (không cần đo, nêu để tham khảo):
- **Đổi model nhẹ hơn**: `gemini-2.5-flash` → `gemini-2.5-flash-lite` (đã làm — lite nhanh hơn rõ rệt).
- **Streaming response** về client thay vì đợi memo hoàn chỉnh → giảm *perceived latency*.
- **Bỏ qua agent không cần** qua conditional routing (đã có): câu hỏi chỉ về thuế thì không gọi compliance.
- **Cache** câu hỏi/embedding lặp lại; **gộp** analyze + aggregate vào 1 lời gọi nếu chấp nhận giảm chất lượng.

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern | Use Case | Complexity |
|---|---|---|---|
| 1 | Direct LLM | Câu hỏi đơn giản, không cần tools | ⭐ |
| 2 | LLM + Tools | Cần tra cứu data hoặc tính toán | ⭐⭐ |
| 3 | ReAct Agent | Tự động orchestration, multi-step | ⭐⭐⭐ |
| 4 | Multi-Agent | Nhiều domains, parallel processing | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | Production, scalable, fault-tolerant | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
3. Làm thế nào để prevent infinite delegation loops trong A2A?
4. Tại sao cần Registry service? Có thể hardcode URLs không?

#### 💡 Giải Đáp Câu Hỏi Ôn Tập

**1. Khi nào dùng single agent thay vì multi-agent?**

Dùng **single agent** (Stage 3) khi:
- Bài toán **một lĩnh vực**, một bộ tool là đủ (vd chỉ tra cứu luật hợp đồng).
- Cần **đơn giản, dễ debug**, latency thấp (một LLM, không overhead điều phối).
- Lưu lượng nhỏ, không cần scale từng phần.

Dùng **multi-agent** (Stage 4/5) khi:
- Bài toán **nhiều chuyên môn** khác nhau (luật + thuế + compliance) → mỗi agent có prompt & tool riêng cho chất lượng sâu hơn.
- Cần **chạy song song** để giảm thời gian (tax + compliance đồng thời).
- Các phần cần **scale / deploy / sửa độc lập**.

> Quy tắc thực dụng: **bắt đầu bằng single agent**, chỉ tách multi-agent khi một prompt "ôm" quá nhiều việc khiến chất lượng giảm hoặc latency tăng. Đừng "over-engineer" sớm.

**2. Ưu điểm của A2A so với gRPC / REST thông thường?**

A2A **không thay thế** HTTP/REST — nó là **một chuẩn (convention) chạy TRÊN HTTP** dành riêng cho agent. Khác biệt chính:
- **Agent Card** (`/.well-known/agent-card.json`) — mỗi agent **tự mô tả** khả năng (tên, version, tasks, skills) để agent khác *khám phá tự động*. REST/gRPC thuần không có chuẩn self-description này.
- **Task lifecycle chuẩn hóa** — A2A định nghĩa sẵn khái niệm task/message/artifact/context_id/trace_id, hợp với hội thoại nhiều lượt & tác vụ chạy lâu. Với REST thuần bạn phải tự thiết kế.
- **Tương thích sẵn LLM tooling** — A2A SDK lo phần streaming, message parts (text/file/data) đúng format agent cần.
- So với **gRPC**: gRPC nhanh (binary/HTTP2) nhưng cần `.proto` + codegen, khó debug bằng mắt. A2A dùng **JSON/HTTP** → dễ đọc, dễ test bằng `curl`, không cần biên dịch schema.

Tóm lại: A2A = "REST + chuẩn mô tả & vòng đời tác vụ cho agent". Đánh đổi: chậm hơn gRPC một chút, nhưng **interoperable** (agent của tổ chức khác cắm vào được) và dễ phát triển.

**3. Làm sao chống vòng lặp delegation vô hạn?**

Repo dùng **giới hạn độ sâu (depth limiting)** — xem `law_agent/graph.py`:

```python
MAX_DELEGATION_DEPTH = 3

async def check_routing(state):
    depth = state.get("delegation_depth", 0)
    if depth >= MAX_DELEGATION_DEPTH:                 # ① chạm trần → ngừng ủy thác
        return {"needs_tax": False, "needs_compliance": False}
    ...

async def call_tax(state):
    result = await delegate(..., depth=state.get("delegation_depth", 0) + 1)  # ② tăng depth mỗi lần
```

Cơ chế: mỗi lần một agent ủy thác cho agent con, nó **+1 vào `delegation_depth`** và truyền qua A2A. Khi depth ≥ `MAX_DELEGATION_DEPTH` (=3), router **từ chối ủy thác tiếp** → buộc agent tự trả lời. Trong log thật ta thấy Tax Agent nhận `depth=2` — còn dưới ngưỡng nên hợp lệ.

Các kỹ thuật bổ sung (production): theo dõi `trace_id` để phát hiện chu trình A→B→A, timeout tổng cho mỗi request, và danh sách "đã thăm" (visited set).

**4. Tại sao cần Registry? Có thể hardcode URL không?**

**Có thể** hardcode (`TAX_URL = "http://localhost:10102"`), và với demo nhỏ thì chạy được. Nhưng Registry giải quyết các vấn đề thực tế:
- **Dynamic discovery** — agent tìm nhau theo **task** (`discover("tax_question")`) chứ không theo địa chỉ cứng. Đổi port/host của Tax Agent → chỉ Registry biết, các agent khác không cần sửa code.
- **Tách rời (decoupling)** — Law Agent không cần biết Tax Agent ở đâu lúc viết code; nó hỏi Registry lúc *runtime*.
- **Khả năng mở rộng** — thêm agent mới (vd Privacy Agent) chỉ cần nó *tự đăng ký* với Registry; không phải sửa file config của mọi agent đang chạy.
- **Nền tảng cho load-balancing / health-check** — Registry có thể trả endpoint khỏe mạnh, bỏ endpoint chết.

Trong code: agent **tự đăng ký** lúc khởi động (`register()` trong `common/registry_client.py`), và tra cứu lúc cần (`discover(task)` → `GET /discover/{task}`). Hardcode URL = mất hết những lợi ích trên, và là anti-pattern khi số agent tăng.

### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

#### 💡 Hướng Giải Challenge 1–4

> Đây là bài tập nâng cao tự học — dưới đây là **thiết kế + code mẫu** để bạn bắt đầu, không phải lời giải duy nhất.

**Challenge 1 — Conversation memory**

Hệ thống hiện **stateless**: mỗi request độc lập. Để agent nhớ hội thoại, dùng **`context_id`** đã có sẵn (đang truyền qua A2A nhưng chưa dùng để lưu lịch sử) làm khóa lưu trữ:

```python
# common/memory.py — bộ nhớ đơn giản (production: dùng Redis)
_HISTORY: dict[str, list[dict]] = {}

def get_history(context_id: str) -> list[dict]:
    return _HISTORY.get(context_id, [])

def append_history(context_id: str, role: str, content: str) -> None:
    _HISTORY.setdefault(context_id, []).append({"role": role, "content": content})
```

Rồi trong node `analyze_law`, nạp lịch sử vào messages trước câu hỏi mới:

```python
history = get_history(state["context_id"])
messages = [SystemMessage(content=...), *history, HumanMessage(content=state["question"])]
# sau khi có kết quả:
append_history(state["context_id"], "user", state["question"])
append_history(state["context_id"], "assistant", result.content)
```

Cách "đúng chuẩn LangGraph" hơn: dùng **`checkpointer`** (`MemorySaver`/`SqliteSaver`) + `thread_id=context_id` khi `compile()` — LangGraph tự lưu/khôi phục state theo thread.

**Challenge 2 — API key authentication cho A2A endpoints**

Thêm middleware kiểm tra header `Authorization` ở mỗi agent server (đều là Starlette/FastAPI app bên dưới A2A SDK):

```python
# common/auth.py
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # cho phép agent card công khai để vẫn discover được
        if request.url.path.startswith("/.well-known/"):
            return await call_next(request)
        if request.headers.get("Authorization") != f"Bearer {os.getenv('A2A_API_KEY')}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)
```

Gắn middleware khi dựng app (`app.add_middleware(APIKeyMiddleware)`), và phía gọi (`common/a2a_client.py`) thêm header `Authorization: Bearer $A2A_API_KEY` vào mỗi `delegate()`. Lưu key trong `.env` (`A2A_API_KEY=...`).

**Challenge 3 — Retry với exponential backoff**

Bọc lời gọi A2A (và cả lời gọi LLM, vì ta đã thấy Gemini trả `503` tạm thời) bằng retry. Dùng thư viện `tenacity`:

```python
# uv add tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),   # 1s → 2s → 4s...
    retry=retry_if_exception_type((httpx.ConnectError, httpx.HTTPStatusError)),
)
async def delegate_with_retry(endpoint, **kwargs):
    return await delegate(endpoint, **kwargs)
```

Điều này biến lỗi *tạm thời* (Tax Agent đang restart, Gemini 503) thành tự hồi phục thay vì fail ngay. Lưu ý: chỉ retry lỗi **transient** (5xx, connect error), KHÔNG retry lỗi **4xx** (401/400 — retry vô ích).

**Challenge 4 — Monitoring & Observability**

*Cách nhanh nhất — LangSmith* (chỉ cần biến môi trường, không sửa code vì repo đã dùng LangChain):

```bash
# thêm vào .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=legal-multiagent
```

→ Mọi lời gọi LLM/agent tự động hiện trên dashboard LangSmith (cây trace, token, latency, cost).

*Cách Prometheus* — expose metrics endpoint ở mỗi agent:

```python
# uv add prometheus-client
from prometheus_client import Counter, Histogram, make_asgi_app

REQUESTS = Counter("agent_requests_total", "Total requests", ["agent", "status"])
LATENCY = Histogram("agent_request_seconds", "Request latency", ["agent"])

# mount /metrics: app.mount("/metrics", make_asgi_app())
# trong node: with LATENCY.labels("tax").time(): ...; REQUESTS.labels("tax","ok").inc()
```

Rồi Prometheus scrape `:1010x/metrics`, Grafana vẽ dashboard. Kết hợp với `trace_id` (Bài 5.1) để có **distributed tracing** đầy đủ.

> 💡 **Thứ tự ưu tiên thực tế:** Challenge 3 (retry) cho **độ bền** ngay lập tức → Challenge 4 (observability) để **nhìn thấy** vấn đề → Challenge 1 (memory) khi cần hội thoại nhiều lượt → Challenge 2 (auth) trước khi đưa ra ngoài internet.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [Gemini API (OpenAI compatibility)](https://ai.google.dev/gemini-api/docs/openai)
- [Lấy Gemini API key](https://aistudio.google.com/apikey)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:
1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

**Chúc các bạn học tốt! 🚀**
