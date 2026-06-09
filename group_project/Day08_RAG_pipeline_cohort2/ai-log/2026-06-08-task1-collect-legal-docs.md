# Log phiên làm việc — 2026-06-08

**Agent:** Claude Code (Opus 4.8)
**Branch:** `2A202600708`
**Phạm vi:** Task 1 — Thu thập văn bản pháp luật về ma tuý (Cá nhân)

---

## 1. Yêu cầu của người dùng
1. Tạo branch tên `2A202600708`.
2. Đọc `README.md`, tập trung hoàn thành **Task 1 — Thu Thập Văn Bản Pháp Luật**.
3. Tạo folder `ai-log/` ghi lại toàn bộ log làm việc với agent.

## 2. Các bước thực hiện

### 2.1. Tạo branch
- `git checkout -b 2A202600708` → đã tạo & chuyển nhánh thành công.
- Lưu ý: `git push` lên remote `looby239/Day08_RAG_pipeline_cohort2` bị **403** (tài khoản `Kaistory` không có quyền ghi). Chưa push được.

### 2.2. Phân tích yêu cầu Task 1
- Đọc `README.md` và `tests/test_individual.py::TestTask1`.
- Tiêu chí pass test:
  - Thư mục `data/landing/legal/` tồn tại.
  - Có **≥ 3 file** `.pdf/.docx/.doc`.
  - Mỗi file **> 1KB** (không rỗng/lỗi).

### 2.3. Tìm nguồn & tải tài liệu
- Dùng `WebSearch` + `WebFetch` để tìm link tải trực tiếp từ cổng `.gov.vn`.
- `vbpl.vn` chặn bot (404) → chuyển sang `datafiles.nghean.gov.vn` và `datafiles.chinhphu.vn`.
- Tải 3 văn bản (kèm User-Agent giả lập trình duyệt):

| File | Văn bản | Nguồn | Dung lượng |
|------|---------|-------|------------|
| `luat-phong-chong-ma-tuy-2021.pdf` | Luật Phòng, chống ma tuý 2021 (73/2021/QH14) | datafiles.nghean.gov.vn | 524.5 KB |
| `nghi-dinh-105-2021-huong-dan-luat-ma-tuy.pdf` | Nghị định 105/2021/NĐ-CP | datafiles.nghean.gov.vn | 3,720.8 KB |
| `bo-luat-hinh-su-2015-sua-doi-2017.pdf` | Bộ luật Hình sự 2015 (VBHN, Chương XX về ma tuý) | datafiles.chinhphu.vn | 2,596.4 KB |

- Kiểm tra header `%PDF` → cả 3 file đều là **PDF hợp lệ**.

### 2.4. Code hoá việc thu thập (reproducible)
- Cập nhật `src/task1_collect_legal_docs.py`:
  - Thêm danh sách `DOCUMENTS` (title + URL trực tiếp).
  - Hàm `download_file()` (bỏ qua nếu file đã có & >1KB) + `collect_all()`.
  - Fix encoding UTF-8 cho stdout để in tiếng Việt trên console Windows.

## 3. Kết quả
- `pytest tests/test_individual.py::TestTask1 -v` → **3/3 PASSED**.
- `python src/task1_collect_legal_docs.py` → chạy OK (skip vì file đã tồn tại).

## 4. Ghi chú
- `vbpl.vn` và `WebFetch` trả 404 cho bot → đã workaround bằng cổng dữ liệu tỉnh/chính phủ.
- Chưa push branch do thiếu quyền ghi remote — cần fork hoặc được thêm collaborator.
