# Log phiên làm việc — 2026-06-08 (Task 4)

**Agent:** Claude Code (Opus 4.8) · **Branch:** `2A202600708`
**Phạm vi:** Task 4 — Chunking & Indexing (7 điểm)

---

## 1. Yêu cầu
- Đọc markdown trong `data/standardized/`, chọn 1 chunking strategy + 1 embedding model,
  index vào vector store. Ghi rõ chunk_size/overlap/dim và lý do.

## 2. Tiêu chí pass test (`TestTask4`)
- `CHUNK_SIZE > 0`, `CHUNK_OVERLAP > 0`, `OVERLAP < SIZE`.
- `load_documents()` trả list có `content`.
- `chunk_documents()` tạo ra chunks có `content`.
- Mỗi chunk ≤ `CHUNK_SIZE * 1.1`.

## 3. Quyết định & lý do
| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| Chunking | `RecursiveCharacterTextSplitter` | Tách theo ranh giới tự nhiên `\n\n→\n→". "→" "`, an toàn, tôn trọng size |
| `CHUNK_SIZE` | 1000 | Văn bản luật theo Điều/Khoản dài; 1000 ký tự đủ ngữ cảnh, không loãng |
| `CHUNK_OVERLAP` | 150 (15%) | Giữ liên tục ngữ cảnh, tránh cắt ngang câu/điều |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | Multilingual tốt tiếng Việt, nhẹ ~470MB, nhanh CPU (thay bge-m3 2.2GB) |
| Vector store | ChromaDB (cosine, persistent) | Local, không cần Docker (môi trường không có Docker) |

## 4. Triển khai (`src/task4_chunking_indexing.py`)
- Helper dùng chung cho Task 5/6/9:
  - `get_embedding_model()` — singleton (`lru_cache`).
  - `embed_texts()` — encode + `normalize_embeddings=True` (để cosine).
  - `get_chroma_client()` / `get_chroma_collection(create=)` — `hnsw:space=cosine`.
- Pipeline: `load_documents → chunk_documents → embed_chunks → index_to_vectorstore`.
- `index_to_vectorstore`: chèn Chroma theo lô 1000; **đồng thời** ghi `data/index/chunks.json`
  (chunk + metadata, không kèm embedding) để Task 6 BM25 dùng cùng tập chunk.
- Bỏ chunk < 20 ký tự (mảnh heading lẻ).

## 5. Kết quả
- Lệnh build: `python -m src.task4_chunking_indexing`
- **10 documents → 1103 chunks → embedded (dim=384) → ChromaDB + chunks.json** (~35s embed).
- Output: `data/index/chroma/` (ChromaDB) + `data/index/chunks.json`.
- Test `TestTask4`: **4/4 PASSED**.

## 6. Ghi chú
- Index là build 1 lần; chạy lại sẽ `delete_collection` rồi tạo mới (idempotent).
- `data/index/` nên cân nhắc thêm vào `.gitignore` nếu không muốn commit chroma (binary).
