# ai-log/

Thư mục lưu lại **toàn bộ log quá trình làm việc với AI agent** (Claude Code) cho dự án RAG Pipeline v2.

## Mục đích
- Ghi lại từng phiên làm việc: yêu cầu của người dùng, các bước agent thực hiện, kết quả.
- Phục vụ truy vết (audit), tái lập (reproduce) và trình bày khi chấm điểm.

## Quy ước đặt tên
Mỗi phiên 1 file Markdown theo định dạng:

```
YYYY-MM-DD-<mo-ta-ngan>.md
```

Ví dụ: `2026-06-08-task1-collect-legal-docs.md`

## Cấu trúc 1 file log
- **Yêu cầu** — người dùng yêu cầu gì.
- **Các bước thực hiện** — agent đã làm gì (tìm nguồn, tải file, sửa code...).
- **Kết quả** — output, file tạo ra, test pass/fail.
- **Ghi chú** — vấn đề gặp phải và cách xử lý.
