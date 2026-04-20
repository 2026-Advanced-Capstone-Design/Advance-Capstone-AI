from __future__ import annotations
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name="articles",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_and_store(article_id: int, text: str, metadata: dict) -> None:
    """기사 텍스트를 임베딩하여 ChromaDB에 저장합니다."""
    model = _get_model()
    collection = _get_collection()

    embedding = model.encode(text, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=[str(article_id)],
        embeddings=[embedding],
        documents=[text[:2000]],
        metadatas=[{
            "article_id": article_id,
            "topic": metadata.get("topic", ""),
            "keywords": ",".join(metadata.get("keywords", [])),
        }],
    )


def search_similar(text: str, top_k: int = 5, exclude_id: int | None = None) -> list[dict]:
    """
    의미 기반 유사 기사 top-k 검색 (코사인 유사도).
    반환: [{"article_id": int, "distance": float, "document": str}]
    """
    model = _get_model()
    collection = _get_collection()

    if collection.count() == 0:
        return []

    embedding = model.encode(text, normalize_embeddings=True).tolist()

    where = {"article_id": {"$ne": exclude_id}} if exclude_id is not None else None

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["distances", "documents", "metadatas"],
    )

    similar = []
    for i, doc_id in enumerate(results["ids"][0]):
        similar.append({
            "article_id": int(results["metadatas"][0][i]["article_id"]),
            "distance": results["distances"][0][i],
            "document": results["documents"][0][i],
        })
    return similar
