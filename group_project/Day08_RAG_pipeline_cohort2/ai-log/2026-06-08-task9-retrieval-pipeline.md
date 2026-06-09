# Log phiên làm việc — 2026-06-08 (Task 9)

**Agent:** Claude Code (Opus 4.8) · **Branch:** `2A202600708`
**Phạm vi:** Task 9 — Retrieval Pipeline hoàn chỉnh (7 điểm)

---

## 1. Yêu cầu
- Gộp semantic + lexical, merge (RRF/weighted), rerank, fallback PageIndex khi
  score < threshold. `retrieve()` trả kết quả gắn `source`.

## 2. Tiêu chí pass test (`TestTask9`)
- Trả `list`; item có `content`, `score`, `source` ∈ {`hybrid`, `pageindex`}.
- Tôn trọng `top_k`.
- Có fallback logic, không crash với query vô nghĩa / threshold cao.

## 3. Triển khai (`src/task9_retrieval_pipeline.py`)
1. `semantic_search(top_k*2)` + `lexical_search(top_k*2)`.
2. **confidence** = cosine cao nhất của dense (thang 0..1 → dễ ngưỡng hoá).
3. Merge bằng **RRF** (`rerank_rrf`), gắn `source='hybrid'`.
4. **Rerank** (`rerank`, cross_encoder) → giữ `source`.
5. **Fallback**: nếu `not final` hoặc `confidence < score_threshold (0.3)` →
   `pageindex_search`; nếu fallback có kết quả → trả nó, ngược lại giữ hybrid.
6. Cắt `top_k`.

## 4. Kết quả
- Test `TestTask9`: **4/4 PASSED** (returns_list, required_keys, respects_top_k, fallback_logic_exists).

## ▶ Cách chạy & kiểm tra
> Yêu cầu: đã build index (Task 4).

```powershell
# 1) Chạy thử pipeline trên 3 query mẫu
python -m src.task9_retrieval_pipeline

# 2) Chạy test
python -m pytest tests/test_individual.py::TestTask9 -v   # kỳ vọng 4/4 PASSED

# 3) Check source hợp lệ + fallback không crash
python -c "from src.task9_retrieval_pipeline import retrieve as r; x=r('luật phòng chống ma túy', top_k=3); print([i['source'] for i in x]); print(r('xyzabc123', top_k=3, score_threshold=0.99).__class__.__name__)"
```
**Đạt khi:** mỗi item có `source` ∈ {`hybrid`,`pageindex`}, query vô nghĩa vẫn trả `list` (không lỗi).

## 5. Ghi chú
- Tách bạch "score hiển thị" (RRF/rerank) với "confidence để quyết định fallback"
  (cosine dense) → ngưỡng 0.3 có ý nghĩa ổn định, tránh fallback nhầm do RRF score nhỏ.
