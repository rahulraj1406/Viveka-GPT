# VivekaGPT

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using the actual words of Swami Vivekananda, sourced from his *Complete Works* (9 volumes, ~4,900 passages).

Ask it anything — "How do I find inner strength?", "What is the purpose of life?", "How do I work without attachment?" — and it responds with grounded answers backed by real passages and inline citations.

---

## How it works

```
User question
     │
     ▼
[Embedding Model]  ←  bge-small-en-v1.5 (384-dim)
     │  encode query
     ▼
[Qdrant]  ←  cosine similarity search over 4,922 passage vectors
     │  top 6 passages
     ▼
[Groq API]  ←  llama-3.3-70b-versatile
     │  RAG prompt: system rules + passages + question
     ▼
[FastAPI]  →  Server-Sent Events stream
     │
     ▼
[Next.js Frontend]  ←  token-by-token streaming UI with source cards
```

---

## Stack

| Layer | Technology |
|---|---|
| Corpus | Wikisource — *Complete Works of Swami Vivekananda* |
| Embedding model | `BAAI/bge-small-en-v1.5` via `sentence-transformers` |
| Vector database | Qdrant (Docker, local) |
| LLM | Llama 3.3 70B via Groq API |
| Backend | FastAPI + Python 3.11 |
| Frontend | Next.js 16 + React 19 + Tailwind CSS v4 |

---

## Project structure

```
vivekagpt/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, /health and /ask endpoints
│   │   ├── rag.py           # RAG pipeline: retrieve → prompt → stream
│   │   ├── retrieval.py     # Qdrant semantic search
│   │   └── prompts.py       # System prompt + user prompt builder
│   ├── scripts/
│   │   ├── fetch_corpus.py  # Scrape Wikisource → raw HTML files
│   │   ├── chunk_corpus.py  # HTML → chunks.jsonl (one paragraph = one chunk)
│   │   ├── embed_corpus.py  # Embed chunks → load into Qdrant
│   │   └── test_groq.py     # Quick smoke-test for the Groq connection
│   ├── data/                # gitignored — generated locally
│   │   ├── raw/             # 411 HTML files (one per chapter)
│   │   └── chunks.jsonl     # 4,922 passage chunks with metadata
│   ├── frontend/
│   │   └── app/
│   │       ├── page.tsx     # Chat UI with streaming + source cards
│   │       └── layout.tsx   # Root layout
│   ├── requirements.txt
│   ├── .env                 # GROQ_API_KEY, GROQ_MODEL (gitignored)
│   └── .env.example
└── qdrant_storage/          # Qdrant persisted data (gitignored)
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for Qdrant)

### 1. Clone and create the virtual environment

```bash
git clone https://github.com/rahulraj1406/Viveka-GPT.git
cd vivekagpt
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Set up environment variables

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and add your GROQ_API_KEY
# GROQ_MODEL=llama-3.3-70b-versatile
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Start Qdrant

```bash
docker run -d \
  --name vivekagpt-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

### 4. Build the corpus (one-time)

```bash
cd backend

# Scrape chapters from Wikisource
python scripts/fetch_corpus.py               # downloads all 9 volumes
# or limit to specific volumes:
python scripts/fetch_corpus.py --volumes 1,2,3

# Chunk the HTML into paragraphs
python scripts/chunk_corpus.py

# Embed and index into Qdrant (~2 min on Apple Silicon)
python scripts/embed_corpus.py
```

### 5. Run the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 6. Run the frontend

```bash
cd backend/frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## API

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /ask`
Request body: `{"question": "How do I overcome fear?"}`

Returns a Server-Sent Events stream. Each event is a JSON line:

```
data: {"type": "sources", "sources": [{...}, ...]}
data: {"type": "token", "content": "Swami"}
data: {"type": "token", "content": " Vivekananda"}
...
data: {"type": "done"}
```

Source objects include `text`, `volume`, `chapter`, `source`, and `score`.

---

## Data pipeline in detail

**Fetch** (`fetch_corpus.py`) — Crawls the Wikisource index page for all 9 volumes, then each section page for individual chapter links. Uses the Wikisource REST API (`/api/rest_v1/page/html/`) for fully rendered HTML. Strips navigation headers, footnotes, and edit links. Saves one `.html` file per chapter.

**Chunk** (`chunk_corpus.py`) — Parses each HTML file with BeautifulSoup, extracts `<p>` tags, filters out very short paragraphs (< 120 chars), and writes one JSON line per paragraph to `chunks.jsonl` with `id`, `text`, `volume`, `chapter`, and `source` fields.

**Embed** (`embed_corpus.py`) — Loads `bge-small-en-v1.5`, encodes all paragraphs in batches of 64 on Apple MPS, and upserts into Qdrant with full payload metadata.

---

## Corpus stats

| Metric | Value |
|---|---|
| Volumes | 1–5 fully indexed (1–9 fetchable) |
| Chapter files | 411 HTML files |
| Passages (chunks) | 4,922 |
| Embedding dimensions | 384 |
| Vector distance | Cosine |
