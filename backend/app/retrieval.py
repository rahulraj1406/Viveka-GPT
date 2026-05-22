"""
Semantic retrieval over the Vivekananda corpus.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION = "vivekananda"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge retrieval works best with this instruction prefixed to queries
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(MODEL_NAME, device="mps")


@lru_cache(maxsize=1)
def _client():
    return QdrantClient(url="http://localhost:6333")


def search(query: str, top_k: int = 5):
    """Return top_k passages most relevant to the query."""
    vec = _model().encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    hits = _client().search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for h in hits:
        p = h.payload
        results.append({
            "text": p["text"],
            "volume": p["volume"],
            "chapter": p["chapter"],
            "source": p["source"],
            "score": round(h.score, 3),
        })
    return results


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "How do I overcome fear and self-doubt?"
    print(f"\nQuery: {q}\n" + "=" * 60)
    for i, r in enumerate(search(q), 1):
        print(f"\n[{i}] score={r['score']} | Vol {r['volume']} — {r['chapter']}")
        print(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))