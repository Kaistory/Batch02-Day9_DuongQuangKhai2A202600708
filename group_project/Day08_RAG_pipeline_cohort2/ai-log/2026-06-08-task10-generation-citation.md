# Log phiên làm việc — 2026-06-08 (Task 10)

**Agent:** Claude Code (Opus 4.8) · **Branch:** `2A202600708`
**Phạm vi:** Task 10 — Generation có Citation (4 điểm)

---

## 1. Yêu cầu
- Reorder chunks chống "lost in the middle", format context, inject prompt, LLM trả lời
  **có citation**; thiếu evidence → "I cannot verify this information". Chọn top_k/top_p (giải thích).

## 2. Tiêu chí pass test (`TestTask10`)
- `reorder_for_llm()` giữ nguyên số lượng, chunk tốt nhất vẫn ở đầu.
- `format_context()` chứa thông tin source.
- `generate_with_citation()` trả dict có `answer` (string, > 0). Test skip nếu lỗi API.

## 3. Quyết định tham số
| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `TOP_K` | 5 | Đủ evidence (luật + tin), context không quá dài → giảm nhiễu |
| `TOP_P` | 0.9 | Nucleus sampling đủ đa dạng diễn đạt, vẫn bám context |
| `TEMPERATURE` | 0.2 | RAG cần factual, hạn chế bịa, output ổn định |

## 4. Triển khai (`src/task10_generation.py`)
- **`reorder_for_llm`**: `evens + odds[::-1]` → `[0,1,2,3,4] → [0,2,4,3,1]`
  (quan trọng nhất ở đầu & cuối; chunk[0] luôn đứng đầu).
- **`format_context`**: mỗi chunk gắn `[Document i | Source: <file> | Type: <legal/news>]`
  để LLM cite chính xác.
- **`generate_with_citation`**: retrieve (Task 9) → reorder → format → prompt.
  - Có `OPENAI_API_KEY` → gọi `client.chat.completions.create` (model `gpt-4o-mini`,
    temp 0.2, top_p 0.9) với SYSTEM_PROMPT yêu cầu citation `[Nguồn, Năm/Loại]`.
  - Không có key (hoặc API lỗi) → **extractive fallback**: tổng hợp đoạn liên quan + citation.
- SYSTEM_PROMPT yêu cầu: chỉ dùng context, mọi claim phải có citation, thiếu evidence →
  "Tôi không thể xác minh thông tin này từ nguồn hiện có".

## 5. Kết quả
- Test `TestTask10`: **3/3 PASSED** (reorder_function_exists, format_context_includes_source,
  generate_returns_dict_with_answer — gọi OpenAI `gpt-4o-mini` thật, trả answer có nội dung).
- `.env` đã có `OPENAI_API_KEY` (do người dùng điền) → generation dùng LLM thật.
  (Key nằm trong `.env`, đã `.gitignore`, không commit, không log.)

## ▶ Cách chạy & kiểm tra
> Yêu cầu: đã build index (Task 4). `OPENAI_API_KEY` trong `.env` (thiếu → extractive fallback).

```powershell
# 1) Chạy thử (in câu trả lời có citation cho 3 câu hỏi)
python -m src.task10_generation

# 2) Chạy test
python -m pytest tests/test_individual.py::TestTask10 -v   # kỳ vọng 3/3 PASSED

# 3) Check cấu trúc trả về + reorder
python -c "from src.task10_generation import generate_with_citation as g, reorder_for_llm as ro; print(g('Hình phạt tàng trữ ma túy?').keys()); c=[{'content':f'C{i}','score':1-i*.1} for i in range(5)]; print([x['content'] for x in ro(c)])"
```
**Đạt khi:** dict có `answer`/`sources`/`retrieval_source`; `answer` không rỗng & có citation `[Nguồn, ...]`;
reorder cho `['C0','C2','C4','C3','C1']` (C0 vẫn đầu).

## 6. Ghi chú
- "Lost in the middle" (Liu et al. 2023): LLM nhớ tốt đầu/cuối context → reorder tăng độ chính xác.
