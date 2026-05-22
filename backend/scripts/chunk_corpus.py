"""
Convert raw chapter HTML into a single chunks.jsonl file.
Each line is one paragraph with metadata.
"""
import json
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "chunks.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 120  # skip very short paragraphs (headers, page numbers)

def parse_filename(path):
    """vol3_karma_yoga.html -> (3, 'Karma Yoga')"""
    stem = path.stem
    vol = int(stem.split("_")[0].replace("vol", ""))
    chapter = stem.split("_", 1)[1].replace("_", " ").title()
    return vol, chapter

chunks = []
chunk_id = 0
for path in tqdm(sorted(RAW_DIR.glob("vol*.html"))):
    vol, chapter = parse_filename(path)
    soup = BeautifulSoup(path.read_text(), "lxml")
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text) < MIN_CHARS:
            continue
        chunks.append({
            "id": f"cwsv_{chunk_id:05d}",
            "text": text,
            "volume": vol,
            "chapter": chapter,
            "source": "Complete Works of Swami Vivekananda",
        })
        chunk_id += 1

with OUT_PATH.open("w") as f:
    for c in chunks:
        f.write(json.dumps(c) + "\n")

print(f"Wrote {len(chunks)} chunks to {OUT_PATH}")