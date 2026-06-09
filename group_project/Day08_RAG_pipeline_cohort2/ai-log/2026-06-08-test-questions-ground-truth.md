# Log phiên làm việc — 2026-06-08 (Test Questions & Ground Truth)

**Agent:** Claude Code (Sonnet 4.6) · **Branch:** `2A202600708`
**Phạm vi:** Bài nhóm — Tạo bộ câu hỏi test và ground truth answers cho RAG Chatbot

---

## 1. Yêu cầu

- Đóng vai PO, tạo bộ **50 câu hỏi test** cho chatbot trả lời pháp luật ma tuý + tin tức.
- Tạo **ground truth answers** chuẩn cho 50 câu hỏi đó.
- Xuất ra file `test.md` và `ground_truth.md`.
- Commit và push lên remote.

---

## 2. Cấu trúc bộ câu hỏi (`test.md`)

Bộ 50 câu được chia thành 5 nhóm:

| Nhóm | Phạm vi | Số câu | Mục tiêu kiểm thử chính |
|------|---------|--------|------------------------|
| 1 | Kiến thức cơ bản pháp luật | 10 | Factual QA, metadata văn bản |
| 2 | Hình phạt & mức xử lý | 10 | Numeric QA, tra cứu điều luật |
| 3 | Cai nghiện & tái hòa nhập | 10 | Policy QA, thủ tục hành chính |
| 4 | Tin tức & tình huống thực tế | 10 | Retrieval QA từ corpus news |
| 5 | Câu hỏi phức tạp & edge cases | 10 | Reasoning, fallback, NLP |

---

## 3. Nguồn pháp lý dùng làm ground truth (`ground_truth.md`)

- **Luật Phòng, chống ma tuý số 73/2021/QH14** (hiệu lực 01/01/2022)
- **Bộ luật Hình sự 2015** (sửa đổi 2017) — Điều 248–256 (các tội ma túy)
- **Nghị định 144/2021/NĐ-CP** — xử phạt hành chính an ninh trật tự
- **Nghị định 116/2021/NĐ-CP** — cai nghiện ma túy
- **Nghị định 57/2022/NĐ-CP** — danh mục chất ma túy và tiền chất
- **Nghị định 90/2016/NĐ-CP** — điều trị nghiện bằng methadone
- **BLTTHS 2015** — Điều 99, 110–113 (chứng cứ điện tử, bắt người)
- **3 Công ước LHQ** về kiểm soát ma túy (1961, 1971, 1988)

---

## 4. Thiết kế ground truth theo loại câu hỏi

### 4.1 Câu hỏi pháp lý (Câu 1–30, 41–47)
- Trích dẫn **điều, khoản cụ thể** của văn bản pháp quy.
- Cung cấp **bảng so sánh** khi câu hỏi yêu cầu phân biệt (ví dụ: tàng trữ vs mua bán).
- Ghi rõ **ngưỡng định lượng** (gram, năm tù, mức phạt tiền).

### 4.2 Câu hỏi tin tức (Câu 31–40)
- Cung cấp **thông tin nền** (background knowledge).
- Ghi chú rõ chatbot cần **trích dẫn nguồn từ corpus** — không dựa vào training data.
- Đánh dấu `(nền)` để phân biệt với ground truth tuyệt đối.

### 4.3 Edge cases (Câu 48–50)
- Định nghĩa **kỳ vọng hành vi** của chatbot (behavioral spec), không phải nội dung pháp lý.
- Câu 48: graceful fallback — thừa nhận giới hạn, gợi ý nguồn chính thức.
- Câu 49: NLP robustness — xử lý lỗi chính tả tiếng Việt.
- Câu 50: anti-hallucination — không tóm tắt toàn bộ luật, hỏi lại để làm rõ.

---

## 5. Ma trận metrics RAGAS

| Metric | Câu áp dụng | Ghi chú |
|--------|------------|---------|
| Faithfulness | 1–47 | Đáp án phải dựa trên retrieved context |
| Answer Relevancy | 1–50 | Toàn bộ bộ câu hỏi |
| Context Recall | 1–40 | Corpus pháp lý + tin tức phải cover đủ |
| Hallucination Rate | 12, 43, 44 | Câu có số liệu cụ thể, dễ bịa đặt |
| Graceful Degradation | 48–50 | Kiểm thử fallback behavior |

---

## 6. Files tạo ra

```
test.md              — 50 câu hỏi, phân nhóm, có mục tiêu kiểm thử
ground_truth.md      — 50 đáp án chuẩn, có trích dẫn nguồn pháp lý
```

---

## 7. Commit

```
d106426  docs: thêm bộ 50 câu hỏi test và ground truth answers cho RAG chatbot
```

Đã push lên `myfork/2A202600708`.
