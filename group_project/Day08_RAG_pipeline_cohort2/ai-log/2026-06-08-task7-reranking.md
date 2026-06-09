# Log phiên làm việc — 2026-06-08 (Task 7)

**Agent:** Claude Code (Opus 4.8) · **Branch:** `2A202600708`
**Phạm vi:** Task 7 — Reranking (6 điểm)

---

## 1. Yêu cầu
- `rerank(query, candidates, top_k)` chấm lại độ liên quan, re-sort. Chọn 1 trong
  cross-encoder / MMR / RRF (ở đây implement cả 3).

## 2. Tiêu chí pass test (`TestTask7`)
- `rerank()` trả `list`; tôn trọng `top_k`; item có `score`.

## 3. Triển khai (`src/task7_reranking.py`)
- **`rerank_cross_encoder`**: neural cross-encoder (`sentence-transformers CrossEncoder`).
  - Bật bằng env `ENABLE_CROSS_ENCODER=1` (+ tuỳ chọn `RERANK_MODEL`).
  - Mặc định **fallback token-overlap** (Jaccard-weighted) → test nhanh, không tải model
    bất ngờ, chạy được offline. Trên lỗi model cũng tự fallback.
- **`rerank_mmr`**: `MMR = λ*sim(q,d) - (1-λ)*max sim(d, selected)`; cosine thuần Python;
  cần `embedding` trong candidate (nếu thiếu → dùng score gốc + diversity=0).
- **`rerank_rrf`**: `RRF(d) = Σ 1/(k + rank)`, k=60 (Cormack 2009); gộp nhiều ranked list,
  dedup theo `content`.
- **`rerank()`**: giao diện thống nhất, mặc định `method="cross_encoder"`.

## 4. Kết quả
- Test `TestTask7`: **3/3 PASSED** (returns_list, respects_top_k, has_score).

## ▶ Cách chạy & kiểm tra
```powershell
# 1) Chạy thử (rerank dữ liệu dummy bằng fallback token-overlap)
python src/task7_reranking.py

# 2) Bật CrossEncoder thật (tải model ~80MB lần đầu)
$env:ENABLE_CROSS_ENCODER="1"; python src/task7_reranking.py

# 3) Chạy test
python -m pytest tests/test_individual.py::TestTask7 -v   # kỳ vọng 3/3 PASSED

# 4) Check RRF gộp 2 ranked list
python -c "from src.task7_reranking import rerank_rrf as f; a=[{'content':'x'},{'content':'y'}]; b=[{'content':'y'},{'content':'z'}]; print([r['content'] for r in f([a,b], top_k=3)])"
```
**Đạt khi:** output re-sorted, mỗi item có `score`, không quá top_k.

## 5. Ghi chú
- RRF được Task 9 dùng để gộp dense + sparse; cross_encoder dùng để rerank cuối.
- Lý do gate CrossEncoder: tránh `pytest` mặc định phải tải ~80–470MB model.
