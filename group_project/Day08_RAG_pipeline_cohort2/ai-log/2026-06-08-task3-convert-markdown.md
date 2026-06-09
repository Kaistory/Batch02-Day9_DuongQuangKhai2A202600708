# Log phiên làm việc — 2026-06-08 (Task 3)

**Agent:** Claude Code (Opus 4.8)
**Branch:** `2A202600708`
**Phạm vi:** Task 3 — Convert sang Markdown (Cá nhân)

---

## 1. Yêu cầu
- Convert toàn bộ file trong `data/landing/` (PDF legal + JSON news) sang Markdown.
- Lưu vào `data/standardized/`, giữ cấu trúc `legal/` và `news/`.
- Khuyến nghị dùng MarkItDown (Microsoft).

## 2. Tiêu chí pass test (`TestTask3`)
- `data/standardized/` tồn tại; có ≥1 file `.md`; mỗi file >200 ký tự.
- Cả `legal/` và `news/` đều có `.md`.

## 3. Các bước thực hiện
- Kiểm tra môi trường: `markitdown` chưa có, `pypdf` có sẵn.
- `pip install "markitdown[pdf]"` → thành công (exit 0) trên Python 3.14.
- Viết `src/task3_convert_markdown.py`:
  - PDF: ưu tiên **MarkItDown**, fallback **pypdf**.
  - News JSON: đọc trực tiếp, thêm header metadata (title/source/crawled) rồi ghi `.md`.
- Phát hiện **Nghị định 105** (`105.signed.pdf`, 3.7MB) là **bản scan, không có text layer**
  → cả MarkItDown lẫn pypdf đều trích rỗng.
  - Giải pháp: thêm fallback `SCANNED_PDF_HTML_FALLBACK` — lấy **toàn văn từ HTML**
    (thuvienphapluat.vn, `div#divContentDoc`) bằng requests + BeautifulSoup.
  - Kết quả: 161,379 ký tự, 512 lần xuất hiện thuật ngữ pháp lý → nội dung thật, đầy đủ.

## 4. Kết quả — 10 file Markdown trong `data/standardized/`

| File | Engine | Size |
|------|--------|------|
| legal/bo-luat-hinh-su-2015-sua-doi-2017.md | markitdown | 834 KB |
| legal/luat-phong-chong-ma-tuy-2021.md | markitdown | 79 KB |
| legal/nghi-dinh-105-2021-huong-dan-luat-ma-tuy.md | html-fulltext | 204 KB |
| news/article_01..07.md | json | 3–26 KB |

- `pytest TestTask1 TestTask2 TestTask3 -v` → **11/11 PASSED**.
- Toàn bộ tái lập bằng 1 lệnh: `python src/task3_convert_markdown.py`.

## 5. Ghi chú
- PDF scan (signed) là vấn đề thường gặp với văn bản pháp luật VN; đã xử lý bằng
  HTML-fulltext fallback thay vì OCR (nhẹ hơn, không cần tesseract).
