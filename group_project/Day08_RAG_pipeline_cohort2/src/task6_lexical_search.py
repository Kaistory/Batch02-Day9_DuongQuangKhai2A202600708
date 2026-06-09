"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import os
import chromadb
from pathlib import Path

# Nạp corpus từ ChromaDB (lấy toàn bộ chunks đã xử lý)
CORPUS: list[dict] = []
try:
    db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("DrugLawDocs")
    data = collection.get()
    if data and data.get("documents"):
        for doc, meta in zip(data["documents"], data["metadatas"]):
            CORPUS.append({"content": doc, "metadata": meta})
except Exception as e:
    print(f"[WARNING] Failed to load CORPUS from ChromaDB: {e}")


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    if not corpus:
        return None
    # Tokenize - cho tiếng Việt dùng split() đơn giản (hoặc có thể tích hợp underthesea sau)
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# Khởi tạo global index
bm25_index = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not bm25_index or not CORPUS:
        return []

    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
