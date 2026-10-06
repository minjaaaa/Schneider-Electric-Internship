"""Indeksiranje chunk-ova u Qdrant i pretraga (dense vektori, BGE-M3)."""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient, models

from chunking import text_for_embedding
from config import EMBEDDING_MODEL, QDRANT_PATH



@lru_cache(maxsize=1) # Učitava model samo jednom, svaki sledeci get_model() vraća isti objekat
def get_model(name: str = EMBEDDING_MODEL):
    """Učitava embedding model samo jednom (prvi put ga i preuzima, ~2 GB)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def embed(texts: list[str], model=None) -> list[list[float]]:
    """Pretvara tekstove u normalizovane vektore."""
    model = model or get_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=16,
                           show_progress_bar=len(texts) > 16) # jedinicni vektori
    return vectors.tolist()


def get_client(in_memory: bool = False) -> QdrantClient:
    """Qdrant bez servera: na disku (qdrant_data/) ili samo u memoriji."""
    return QdrantClient(":memory:") if in_memory else QdrantClient(path=str(QDRANT_PATH))


def build_index(client: QdrantClient, collection: str, chunks: list[dict],
                with_path: bool = True, model=None) -> None:
    """Pravi (ili ponovo pravi) kolekciju i upisuje sve chunk-ove sa metapodacima."""
    vectors = embed([text_for_embedding(c, with_path) for c in chunks], model)

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    client.upsert(
        collection,
        points=[models.PointStruct(id=i, vector=v, payload=c)
                for i, (v, c) in enumerate(zip(vectors, chunks))],
    )


def search(client: QdrantClient, collection: str, query: str, k: int = 10,
           exclude_superseded: bool = False, model=None) -> list[dict]:
    """Vraća k najsličnijih chunk-ova (payload + score), od najboljeg ka najlošijem."""
    query_filter = None
    if exclude_superseded:
        query_filter = models.Filter(must_not=[
            models.FieldCondition(key="status", match=models.MatchValue(value="superseded"))
        ])
    result = client.query_points(collection, query=embed([query], model)[0], limit=k,
                                 query_filter=query_filter, with_payload=True)
    return [{**p.payload, "score": p.score} for p in result.points]