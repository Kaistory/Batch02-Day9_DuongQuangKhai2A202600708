"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import os
    import chromadb
    from sentence_transformers import SentenceTransformer
    from src.task4_chunking_indexing import EMBEDDING_MODEL

    # Embed query
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()

    # Query ChromaDB
    db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection("DrugLawDocs")
    except Exception:
        print("[ERROR] Collection 'DrugLawDocs' not found. Please run task 4 first.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    out = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for doc, dist, meta in zip(docs, distances, metadatas):
            # ChromaDB mặc định dùng L2 distance. Convert sang similarity score = 1 / (1 + distance)
            score = 1.0 / (1.0 + dist)
            out.append({
                "content": doc,
                "score": score,
                "metadata": meta
            })

    # Đảm bảo sort theo score giảm dần
    return sorted(out, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
