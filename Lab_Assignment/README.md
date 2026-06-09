# Lab Assignment Day09 — Supervisor–Workers Multi-Agent

Cải tiến Agent (Day08) sang pattern **Supervisor–Workers** với **3 workers** chuyên môn.

> Đáp ứng yêu cầu trong `Lab-assignment-checklist.md` mục 2:
> *"Improve Agent Day08 sử dụng pattern Supervisor–Workers (ít nhất 2-3 workers)."*

## Kiến trúc

```
                ┌─────────────────────────────────────┐
                ▼                                     │ (lặp)
START ──► supervisor ──(định tuyến động)──► worker ───┘
                │
                └──(FINISH)──► finalize ──► END
```

- **1 Supervisor** — điều phối **ĐỘNG**: mỗi vòng nhìn `findings` đã tích lũy rồi
  quyết định gọi worker nào TIẾP THEO (structured output `Route`), hoặc `FINISH`
  khi đã đủ thông tin. Có `MAX_STEPS` chặn vòng lặp vô hạn.
- **3 Workers** — mỗi worker là **1 ReAct agent** (`create_react_agent`) với tool riêng:
  | Worker | Lĩnh vực | Tool |
  |---|---|---|
  | `legal_worker` | Hợp đồng, trách nhiệm dân sự (UCC) | `search_legal_db` |
  | `tax_worker` | Thuế, IRS, trốn thuế, FBAR/FATCA | `search_tax_db` |
  | `compliance_worker` | SEC, SOX, GDPR, FCPA | `search_compliance_db` |
- **finalize** — tổng hợp toàn bộ `findings` thành một báo cáo pháp lý cuối.

## Khác biệt với fan-out song song (Stage 4)

| | Stage 4 (router fan-out) | Supervisor–Workers (bài này) |
|---|---|---|
| Điều phối | Router chạy **1 lần**, bắn song song mọi worker cần (cố định) | **LẶP**: mỗi vòng chọn **1 worker**, xem kết quả rồi quyết tiếp |
| Linh hoạt | Quyết định 1 lần dựa trên câu hỏi | Quyết định **động** dựa trên kết quả đã có — bỏ qua/gọi thêm tùy ngữ cảnh |
| Đánh đổi | Nhanh hơn (song song) | Linh hoạt hơn, kiểm soát tốt hơn (nhưng tuần tự) |

## Chạy

```bash
# Cần GEMINI_API_KEY trong .env (xem .env.example ở thư mục gốc)
$env:PYTHONIOENCODING="utf-8"; uv run python Lab_Assignment/supervisor_workers.py
```

Câu hỏi demo (đa lĩnh vực — kích hoạt cả 3 worker):
> *"Một công ty vừa vi phạm hợp đồng cung cấp, vừa trốn thuế thu nhập ở nước ngoài,
> vừa làm rò rỉ dữ liệu khách hàng. Hãy phân tích toàn bộ hậu quả pháp lý."*

## Cơ chế chống vòng lặp

- `MAX_STEPS = 6`: supervisor chạy tối đa 6 vòng → buộc `FINISH`.
- System prompt yêu cầu **không gọi lại worker đã chạy** trừ khi thật cần, và
  `FINISH` khi đã đủ mọi khía cạnh liên quan.

## File

- [`supervisor_workers.py`](supervisor_workers.py) — toàn bộ hệ thống (state, supervisor, workers, finalize, graph).
