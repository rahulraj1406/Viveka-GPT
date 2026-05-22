"""
Embed all chunks from chunks.jsonl and load them into Qdrant.
Uses bge-small-en-v1.5 (384-dim, fast, strong for its size).
"""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.jsonl"
COLLECTION = "vivekananda"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH = 64

# 1. Load chunks
chunks = [json.loads(line) for line in CHUNKS_PATH.open()]
print(f"Loaded {len(chunks)} chunks")

# 2. Load embedding model (downloads ~130MB first time, uses Metal on M4)
model = SentenceTransformer(MODEL_NAME, device="mps")
dim = model.get_sentence_embedding_dimension()
print(f"Model loaded, embedding dim = {dim}")

# 3. Connect to Qdrant and (re)create the collection
client = QdrantClient(url="http://localhost:6333")
client.delete_collection(COLLECTION)
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
)

# 4. Embed each batch and upload immediately (avoids building one giant list)
texts = [c["text"] for c in chunks]
total_uploaded = 0
for start in tqdm(range(0, len(texts), BATCH)):
    batch_chunks = chunks[start:start + BATCH]
    vectors = model.encode(
        [c["text"] for c in batch_chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    points = [
        PointStruct(
            id=start + i,
            vector=vec.tolist(),
            payload={
                "chunk_id": c["id"],
                "text": c["text"],
                "volume": c["volume"],
                "chapter": c["chapter"],
                "source": c["source"],
            },
        )
        for i, (c, vec) in enumerate(zip(batch_chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    total_uploaded += len(points)

print(f"Uploaded {total_uploaded} vectors to '{COLLECTION}'")

# Sanity check
info = client.get_collection(COLLECTION)
print(f"Collection now holds {info.points_count} points")