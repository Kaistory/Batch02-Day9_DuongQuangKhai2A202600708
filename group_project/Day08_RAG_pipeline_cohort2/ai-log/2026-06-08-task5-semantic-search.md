# Log phiên làm việc — 2026-06-08 (Task 5)

**Agent:** Claude Code (Opus 4.8) · **Branch:** `2A202600708`
**Phạm vi:** Task 5 — Semantic Search / dense retrieval (6 điểm)

---

## 1. Yêu cầu
- `semantic_search(query, top_k)` → list `{content, score, metadata}`, sorted desc,
  tương thích embedding model + vector store ở Task 4.

## 2. Tiêu chí pass test (`TestTask5`)
- Trả về `list`; mỗi item có `content`, `score`.
- Sorted theo `score` giảm dần.
- Không vượt quá `top_k`.

## 3. Triển khai (`src/task5_semantic_search.py`)
- Tái dùng `embed_texts()` + `get_chroma_collection()` của Task 4 (cùng model/space).
- Embed query → `collection.query(query_embeddings=..., n_results=top_k, include=[documents, metadatas, distances])`.
- Chuyển **cosine distance → similarity**: `score = 1 - distance`.
- Sort giảm dần, cắt `top_k`.
- An toàn: nếu collection chưa tồn tại (chưa chạy Task 4) → trả `[]` thay vì crash.

## 4. Kết quả
- Test `TestTask5`: **4/4 PASSED** (returns_list, required_keys, sorted_descending, respects_top_k).

## ▶ Cách chạy & kiểm tra
> Yêu cầu: đã build index (Task 4) — `python -m src.task4_chunking_indexing`

```powershell
# 1) Chạy thử module (in 5 kết quả: [score] (source) snippet)
python src/task5_semantic_search.py

# 2) Chạy test
python -m pytest tests/test_individual.py::TestTask5 -v   # kỳ vọng 4/4 PASSED

# 3) Check nhanh: score giảm dần, mỗi item có content/score/metadata
python -c "from src.task5_semantic_search import semantic_search as s; r=s('hình phạt tàng trữ ma túy', top_k=3); print(len(r)); print([round(x['score'],3) for x in r])"
```
**Đạt khi:** trả về ≤ top_k item, score giảm dần, content khớp chủ đề ma tuý.

## 5. Ghi chú
- Embeddings normalize ở Task 4 nên cosine ổn định; score ∈ ~[0,1], dùng làm
  "confidence" cho ngưỡng fallback ở Task 9.
