# Log phiên làm việc — 2026-06-08 (Task 6)

**Agent:** Claude Code (Opus 4.8) · **Branch:** `2A202600708`
**Phạm vi:** Task 6 — Lexical Search / BM25 (6 điểm)

---

## 1. Yêu cầu
- `lexical_search(query, top_k)` → list `{content, score, metadata}` sorted desc. Mặc định BM25.

## 2. Tiêu chí pass test (`TestTask6`)
- Trả về `list`; item có `content`, `score`.
- Sorted theo BM25 score giảm dần.
- Có ít nhất 1 score > 0 với query khớp keyword.

## 3. Triển khai (`src/task6_lexical_search.py`)
- Cài `rank-bm25`; dùng `BM25Okapi` (k1=1.5, b=0.75 — mặc định thư viện).
- Corpus lấy từ `data/index/chunks.json` (Task 4 tạo) → **cùng tập chunk với dense search**.
- Tokenize: regex `\w+` (Unicode) + lowercase — giữ nguyên dấu tiếng Việt; đơn giản,
  không cần thư viện tách từ nặng (underthesea).
- `_load_corpus()` cache (`lru_cache`) corpus + BM25 index để khỏi build lại mỗi query.
- `get_scores(tokenize(query))` → sort giảm dần → loại score ≤ 0 → cắt `top_k`.

## 4. Kết quả
- Test `TestTask6`: **4/4 PASSED** (returns_list, required_keys, sorted_descending, keyword_match_scores_higher).

## ▶ Cách chạy & kiểm tra
> Yêu cầu: đã build index (Task 4) → có `data/index/chunks.json`

```powershell
# 1) Chạy thử module
python src/task6_lexical_search.py

# 2) Chạy test
python -m pytest tests/test_individual.py::TestTask6 -v   # kỳ vọng 4/4 PASSED

# 3) Check nhanh: BM25 score > 0 và giảm dần với query khớp keyword
python -c "from src.task6_lexical_search import lexical_search as s; r=s('Điều 248 tàng trữ trái phép chất ma túy', top_k=3); print([round(x['score'],2) for x in r])"
```
**Đạt khi:** có ít nhất 1 score > 0, kết quả sorted giảm dần.

## 5. Ghi chú
- BM25: TF cao + từ hiếm (IDF lớn) + chuẩn hoá độ dài → điểm cao. Mạnh ở khớp
  thuật ngữ chính xác ("Điều 248", "tiền chất") mà dense đôi khi bỏ sót.
