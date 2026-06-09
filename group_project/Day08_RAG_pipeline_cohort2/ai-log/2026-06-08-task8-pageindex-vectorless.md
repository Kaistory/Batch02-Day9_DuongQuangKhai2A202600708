# Log phiên làm việc — 2026-06-08 (Task 8)

**Agent:** Claude Code (Opus 4.8) · **Branch:** `2A202600708`
**Phạm vi:** Task 8 — PageIndex Vectorless RAG (4 điểm)

---

## 1. Yêu cầu
- `pageindex_search(query, top_k)` retrieval **không dùng vector**, làm fallback cho Task 9.
- Kết quả gắn `source='pageindex'`.

## 2. Tiêu chí pass test (`TestTask8`)
- `pageindex_search` là callable.
- Trả `list`; nếu có kết quả, item[0]`['source'] == 'pageindex'` (test skip nếu lỗi).

## 3. Triển khai (`src/task8_pageindex_vectorless.py`)
- **Có `PAGEINDEX_API_KEY` + SDK** → gọi PageIndex thật (`pi.query`).
- **Không có key** → **vectorless cục bộ**:
  - "Structural understanding": tách tài liệu theo ranh giới `^#{1,6} ...` hoặc `^Điều N. ...`
    → các *section*.
  - Xếp hạng section bằng **độ trùng từ khoá** (tf chuẩn hoá theo `len^0.5`) — KHÔNG embedding.
  - Trả top_k, gắn `source='pageindex'`.
- `upload_documents()` chỉ chạy khi có key (no-op + cảnh báo khi không).

## 4. Kết quả
- Test `TestTask8`: **2/2 PASSED** (function_exists, returns_list_with_source_marker — fallback cục bộ trả `source='pageindex'`).

## ▶ Cách chạy & kiểm tra
> Không cần index Chroma — đọc trực tiếp `data/standardized/`.

```powershell
# 1) Chạy thử (vectorless cục bộ khi chưa có PAGEINDEX_API_KEY)
python src/task8_pageindex_vectorless.py

# 2) Chạy test
python -m pytest tests/test_individual.py::TestTask8 -v   # kỳ vọng 2/2 PASSED

# 3) Check mọi kết quả gắn source='pageindex'
python -c "from src.task8_pageindex_vectorless import pageindex_search as s; r=s('cai nghiện ma túy', top_k=3); print(len(r), set(x['source'] for x in r))"
```
**Đạt khi:** trả list, `source` của mọi item == `'pageindex'`. (Có key thật → tự chuyển sang API.)

## 5. Ghi chú
- Đây là stand-in hợp lý cho PageIndex khi chưa đăng ký tài khoản pageindex.ai;
  vẫn đúng tinh thần "vectorless" (dựa cấu trúc + từ khoá, không vector store).
- Nếu sau này có key: chỉ cần điền `PAGEINDEX_API_KEY` vào `.env`, code tự chuyển sang API.
