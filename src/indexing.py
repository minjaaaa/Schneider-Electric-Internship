"""Indeksiranje chunk-ova u Qdrant i pretraga (dense vektori, BGE-M3)."""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient, models

from chunking import text_for_embedding
from config import EMBEDDING_MODEL, QDRANT_PATH, HYBRID_PREFETCH

import hashlib
import json



@lru_cache(maxsize=1) # Učitava model samo jednom, svaki sledeci get_model() vraća isti objekat
def get_model(name: str = EMBEDDING_MODEL):
    """Učitava embedding model samo jednom (prvi put ga i preuzima, ~2 GB)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def embed(texts: list[str], model=None) -> list[list[float]]:
    """Pretvara tekstove u normalizovane vektore."""
    model = model or get_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=16, # obradjuje po 16 tekstova odjednom
                           show_progress_bar=len(texts) > 16) # jedinicni vektori
    return vectors.tolist() # encode vraca NumPy matricu (broj chunkova x 1024), a Qdrant ocekuje listu


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
    client.upsert( # update + insert, tj. upisuje sve chunk-ove u Qdrant
        collection,
        points=[models.PointStruct(id=i, vector=v, payload=c) # jedna tacka u bazi = jedinstven broj, vektor, metapodaci
                for i, (v, c) in enumerate(zip(vectors, chunks))],
    )

def superseded_filter(exclude: bool) -> models.Filter | None:
    """Filter koji iskljcuje chunk-ove sa statusom 'superseded'. None ako filter nije ukljucen."""
    if not exclude:
        return None
    return models.Filter(must_not=[
        models.FieldCondition(key="status", match=models.MatchValue(value="superseded"))
    ])
    

def search(client: QdrantClient, collection: str, query: str, k: int = 10,
           exclude_superseded: bool = False, model=None) -> list[dict]:
    """Vraća k najsličnijih chunkova (payload + score), od najboljeg ka najlošijem."""
    result = client.query_points(collection, query=embed([query], model)[0], limit=k,
                                 query_filter=superseded_filter(exclude_superseded),
                                 with_payload=True) # vraca i metaodatke chunka
    return [{**p.payload, "score": p.score} for p in result.points]
    # **p.payload raspakuje sva polja chunka: chunk_id, text, section, sections, section_title, heading_path, type, status, office, overriden_by

# ---------- Eksperimenti 4b, 7-9: BGE-M3 dense + sparse ----------

@lru_cache(maxsize=1)
def get_m3_model(name: str = EMBEDDING_MODEL):
    """BGE-M3 preko FlagEmbedding-a (daje i sparse težine). Isti fajlovi kao get_model()."""
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel(name, use_fp16=False) # fp16 - polovicna preciznost, brze racunanje na GPU

def to_sparse(weights: dict) -> models.SparseVector:
    """(id tokena : tezina) -> Qdrant SparseVector (indices, values)."""
    return models.SparseVector(indices = [int(t) for t in weights],
                               values = [float(w) for w in weights.values()])
def embed_m3(texts: list[str], model=None) -> tuple[list[list[float]], list[models.SparseVector]]:
    """Vraća (dense vektore, sparse vektore) za BGE-M3."""
    model = model or get_m3_model()
    out = model.encode(texts, batch_size=16, return_dense=True, return_sparse=True,
                       return_colbert_vecs=False) 
    dense =[list(map(float, v))for v in out["dense_vecs"]]
    sparse = [to_sparse(s) for s in out["lexical_weights"]]

    return dense, sparse

def build_hybrid_index(client: QdrantClient, collection: str, chunks: list[dict],
                       with_path: bool = True, model = None) -> None:
    """Kao build_index, ali svaka tačka ima dva imenovana vektora: "dense" i "sparse"."""
    dense, sparse = embed_m3([text_for_embedding(c, with_path) for c in chunks], model)

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(size=len(dense[0]), distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()}, # slicnost je skalarni proizvod
    )
    client.upsert(
        collection,
        points=[models.PointStruct(id=i, vector={"dense": d, "sparse": s}, payload=c)
                for i, (d, s, c) in enumerate(zip(dense, sparse, chunks))],
    )

# ---------- Čuvanje indeksa između pokretanja ----------

def index_fingerprint(chunks: list[dict], with_path: bool, kind: str) -> str:
    """Kratak heš svega od čega indeks zavisi: model, vrsta indeksa, putanja naslova, chunk-ovi."""
    h = hashlib.sha256()
    h.update(f"{EMBEDDING_MODEL}|{kind}|{with_path}".encode("utf-8"))
    for c in chunks:
        h.update(json.dumps(c, sort_keys=True, ensure_ascii=False).encode("utf-8")) # dumps pretvara ceo chunk u tekst sortiram da bi hes bio isti za iste podatke
    return h.hexdigest()[:10] # proveravam poklapanje prvih 10 znakova hesa, ne svih 64, dovoljno je za detekciju promena


def ensure_index(client: QdrantClient, chunks: list[dict], with_path: bool = True,
                 hybrid: bool = False) -> str:
    """Vraća ime kolekcije za ove chunk-ove; indeksira samo ako takva kolekcija još ne postoji."""
    kind = "m3" if hybrid else "dense"
    collection = f"{kind}_{index_fingerprint(chunks, with_path, kind)}"

    if client.collection_exists(collection) and client.count(collection).count == len(chunks):
        print(f"Koristim sačuvan indeks {collection} ({len(chunks)} chunk-ova)")
        return collection

    print(f"Indeksiram {collection} ({len(chunks)} chunk-ova)...")
    build = build_hybrid_index if hybrid else build_index
    build(client, collection, chunks, with_path=with_path)
    return collection

def hybrid_search(client: QdrantClient, collection: str, query: str, k: int = 10,
                  using: str = "hybrid", exclude_superseded: bool = False,
                  prefetch: int = HYBRID_PREFETCH, model=None) -> list[dict]:
    """using = "dense" (eksp. 4b) | "sparse" (eksp. 7) | "hybrid" = dense + sparse spojeni RRF-om (eksp. 8-9).
    Za "hybrid" je score RRF skor (Qdrant k=2): zavisi samo od pozicija, nije mera sličnosti."""

    query_filter = superseded_filter(exclude_superseded)
    dense, sparse = embed_m3([query], model)

    if using == "dense":
        result = client.query_points(collection, query=dense[0], using = "dense", limit=k,
                                     query_filter=query_filter, with_payload=True)
    elif using == "sparse":
        result = client.query_points(collection, query=sparse[0], using = "sparse", limit=k,
                                     query_filter=query_filter, with_payload=True)
    elif using == "hybrid":
        n = max(k, prefetch)
        result = client.query_points(collection, 
                                     prefetch = [
                                         models.Prefetch(query = dense[0], using = "dense", limit = n, filter=query_filter), 
                                         models.Prefetch(query = sparse[0], using = "sparse", limit = n, filter=query_filter),
                                     ],
                                     query = models.FusionQuery(fusion = models.Fusion.RRF),
                                     limit = k, query_filter = query_filter, with_payload = True,
                                     )
    else:
        raise ValueError(f"nepoznat nacin pretrage: {using!r}")
    return [{**p.payload, "score": p.score} for p in result.points]