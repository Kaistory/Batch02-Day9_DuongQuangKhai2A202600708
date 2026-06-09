# Log phiên làm việc — 2026-06-08 (Task 2)

**Agent:** Claude Code (Opus 4.8)
**Branch:** `2A202600708`
**Phạm vi:** Task 2 — Crawl bài báo về nghệ sĩ liên quan ma tuý (Cá nhân)

---

## 1. Yêu cầu
- Crawl **≥ 5 bài báo** về nghệ sĩ Việt liên quan ma tuý → lưu `data/landing/news/`.
- Mỗi bài 1 file JSON có metadata (`url`, `title`, `date_crawled`, `content`).

## 2. Tiêu chí pass test (`TestTask2`)
- Thư mục `data/landing/news/` tồn tại.
- ≥ 5 file `.json/.html/.md/.txt`, mỗi file > 500 bytes.
- File JSON có trường `url`.

## 3. Các bước thực hiện
- `WebSearch` tìm bài báo trên VnExpress/VietnamNet/Tiền Phong/VOV/Báo Chính phủ/Dân Việt.
- Kiểm tra môi trường: **crawl4ai chưa cài**; `bs4` + `requests` có sẵn. Python 3.14.0.
- Thử `pip install crawl4ai` (chạy nền) — nhưng Python 3.14 rất mới, crawl4ai kéo
  theo Playwright/litellm nên nhiều khả năng không có wheel tương thích.
- Viết `src/task2_crawl_news.py`:
  - `crawl_article()` ưu tiên **Crawl4AI**, tự **fallback requests + BeautifulSoup**
    nếu không import được (giữ đúng tinh thần README mà vẫn chạy được ngay).
  - Trích `og:title`/`<h1>`/`<title>` + gom các `<p>` > 40 ký tự làm nội dung.
  - Bỏ qua bài có nội dung < 200 ký tự.

## 4. Kết quả
- Chạy `python src/task2_crawl_news.py` → **7/7 bài** lưu thành công:

| File | Tiêu đề | Size |
|------|---------|------|
| article_01.json | Những nghệ sĩ Việt 'ngã ngựa' vì ma túy | 7 KB |
| article_02.json | Sao Việt bị bắt, ngồi tù, mất danh tiếng vì chất cấm | 5 KB |
| article_03.json | 3 nữ nghệ sĩ Việt tự huỷ danh tiếng vì liên quan ma túy | 3 KB |
| article_04.json | Nghệ sĩ 'dính' ma túy: Khoảng trống sau những cú trượt ngã | 10 KB |
| article_05.json | Chưa đầy 1 tháng, 3 nghệ sĩ Việt bị khởi tố vì ma túy | 13 KB |
| article_06.json | Khởi tố ca sĩ Long Nhật, Sơn Ngọc Minh tổ chức sử dụng ma túy | 3 KB |
| article_07.json | Nhức nhối loạt nghệ sĩ vướng lao lý vì ma túy | 26 KB |

- `pytest tests/test_individual.py::TestTask2 -v` → **4/4 PASSED**.

## 5. Ghi chú
- Crawler hiện dùng `requests+bs4` (trường `"crawler"` trong JSON ghi rõ). Nếu sau này
  cài được Crawl4AI, script tự chuyển sang dùng Crawl4AI mà không cần sửa code.
