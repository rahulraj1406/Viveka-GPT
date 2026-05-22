# VivekaGPT — Learning Guide

This document explains everything you built, how each piece works, why you made the choices you did, and what concepts you learned along the way. Read it top to bottom to fully understand the project.

---

## What is this project?

VivekaGPT is a chatbot that answers questions using the real, recorded words of Swami Vivekananda. When you ask "How do I find inner strength?", the app doesn't just make something up — it:

1. Searches through ~4,900 actual paragraphs from Vivekananda's books
2. Finds the most relevant ones
3. Feeds those paragraphs to an AI language model and says "answer the question using *only these passages*"
4. Streams the answer back word-by-word, with citations

This pattern — "search first, then generate" — is called **RAG (Retrieval-Augmented Generation)**. It's one of the most important and practical AI patterns in use today.

---

## The Big Picture: What is RAG?

Traditional AI (like plain ChatGPT) just generates answers from its training data. That has two problems:
- It can hallucinate (confidently say wrong things)
- It can't cite sources

RAG solves both. The flow is:

```
Question → Find relevant documents → Give those documents to the LLM → Get a grounded answer
```

Think of it like an open-book exam. Instead of the AI trying to remember everything, you hand it the relevant pages and say "answer from these."

---

## Day 1 — Building the Corpus

### What you did
You scraped 411 chapters from Wikisource (a free, legal source for public domain texts), cleaned them, and converted them into 4,922 individual paragraphs saved as a JSONL file.

### Tools used

**`httpx`** — A modern Python HTTP client. You used it to fetch web pages from Wikisource. Similar to `requests` but supports async. You learned that websites block you if you don't send a proper `User-Agent` header (the browser name string), and Wikisource returns `403 Forbidden` without one.

**`BeautifulSoup`** — An HTML parser. After fetching a page, HTML is just a big string of tags. BeautifulSoup lets you navigate it like a tree: "find the div with class `mw-parser-output`, then find all `<p>` tags inside it." You learned how Wikisource structures its pages — index → section → chapter — and wrote code to crawl each level.

**`tqdm`** — A progress bar library. Wrap any Python loop with `tqdm(...)` and it shows you a live progress bar in the terminal. Simple but very useful during long operations.

**Wikisource REST API** — Wikisource (like Wikipedia) has an API at `/api/rest_v1/page/html/<title>` that returns fully rendered HTML. You discovered this was necessary because some pages use a "transclusion" system (text stored in sub-pages) that doesn't appear in the regular HTML fetch. This was a real-world debugging lesson: the page looked empty in the browser scrape but had content via the API.

### Key concepts learned

**Web scraping ethics** — You added `time.sleep(1.0)` between each request and exponential backoff on `429 Too Many Requests` errors. Being polite to servers (not hammering them with requests) is both ethical and practical — you won't get IP-banned.

**URL encoding** — URLs can't contain special characters like `?`. They get encoded as `%3F`. You hit a bug where Python's `urllib.parse.quote()` was double-encoding already-encoded URLs (turning `%3F` into `%253F`). The fix was to `unquote()` first, then `quote()`. This is a very common real-world bug.

**JSONL format** — "JSON Lines" — one JSON object per line. Better than a single big JSON array for large datasets because you can process it line by line without loading everything into memory. `chunks.jsonl` has 4,922 lines, one passage each.

---

## Day 2 — Embeddings and Vector Search

This is the heart of the RAG system. Understanding this section means understanding modern AI search.

### What is an embedding?

A text embedding converts a sentence or paragraph into a list of numbers (a "vector"). The magic is: **sentences with similar meaning end up with similar numbers**.

For example:
- "Arise, awake, and stop not till the goal is reached" → `[0.12, -0.45, 0.89, ...]` (384 numbers)
- "Wake up and keep going until you succeed" → `[0.11, -0.43, 0.91, ...]` (very similar numbers)
- "The price of butter in 1920" → `[-0.67, 0.23, -0.12, ...]` (very different numbers)

You can then measure the "closeness" of two vectors mathematically. This is how semantic search works — instead of matching exact words, you match meaning.

### The embedding model: `bge-small-en-v1.5`

This model was made by BAAI (Beijing Academy of AI) and is one of the best small embedding models available. "Small" means it's fast (384 dimensions vs 1536 for OpenAI's models) and free to run locally. "v1.5" means version 1.5.

You ran it locally on Apple Silicon using **MPS** (Metal Performance Shaders) — Apple's GPU acceleration framework. This is why `device="mps"` appears in the code. On a Mac M-series chip, MPS gives you near-GPU speed without needing an Nvidia card.

**BGE query prefix** — BGE models are trained with a special instruction prefix for search queries: `"Represent this sentence for searching relevant passages: "`. This prefix is added to questions (not to the stored passages). It tells the model "this is a search query" vs "this is a document", which improves retrieval quality. This is a subtle but important detail about how these models work.

### What is Qdrant?

Qdrant is a **vector database** — a database specifically designed to store and search embeddings. Regular databases (like PostgreSQL) are great for finding exact matches: "give me all users where age = 25". Vector databases find approximate matches: "give me the 6 most similar vectors to this query vector".

You ran Qdrant in **Docker** — a containerization tool that runs software in an isolated environment. The command:

```bash
docker run -d \
  --name vivekagpt-qdrant \
  -p 6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

- `-d` = run in background (detached)
- `-p 6333:6333` = map port 6333 on your Mac to port 6333 inside the container
- `-v ...` = mount a folder so data persists even if the container stops

### Cosine similarity

This is the distance metric you used in Qdrant. It measures the angle between two vectors. Vectors pointing in the same direction = cosine similarity of 1.0 (identical). Vectors pointing opposite directions = -1.0. This is better than raw distance for embeddings because it doesn't care about the "magnitude" (how long the vector is), only the direction.

### What you built

`embed_corpus.py` does three things:
1. Loads all 4,922 passages
2. Encodes them in batches of 64 (batch processing is faster than one at a time)
3. Uploads each batch to Qdrant with the original text and metadata as "payload"

The metadata stored alongside each vector (`volume`, `chapter`, `text`, `source`) is called the **payload**. When Qdrant finds the most similar vectors, it returns this payload so you know which passages matched.

### Bug you fixed: Qdrant payload size limit

The first version tried to upload all 4,922 points in one API call — a 44 MB JSON payload. Qdrant's limit is 33 MB. The fix was to upload in the same batches of 64 used for embedding. Always a good practice: batch your writes, not just your reads.

---

## Day 3 — The RAG Pipeline and API

### `retrieval.py` — Semantic Search

```python
def search(query: str, top_k: int = 5):
    vec = _model().encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    hits = _client().search(collection_name=COLLECTION, query_vector=vec, limit=top_k)
    ...
```

When a question comes in:
1. Encode the question as a vector (with the BGE prefix)
2. Ask Qdrant for the `top_k` most similar vectors
3. Return the payloads (the actual passage text, volume, chapter, score)

`@lru_cache(maxsize=1)` — This decorator caches the result of the function. Since loading the embedding model takes ~2 seconds, you don't want to reload it on every request. `lru_cache` stores the result the first time and returns it instantly every time after.

### `prompts.py` — The System Prompt

The system prompt is the most important part of making the LLM behave correctly. It defines the "character" and "rules" of the AI:

- "Answer ONLY using the passages provided"
- "Never invent quotations"
- "You are NOT Swami Vivekananda"
- "Cite sources inline as [Vol X — Chapter]"

This is called **prompt engineering**. The quality of your system prompt directly determines the quality of your answers. Key rules you wrote:
- Grounded: only use what's in the context (prevents hallucination)
- Honest: say when the passages don't answer the question
- Attribution: always cite the source

### `rag.py` — Putting It Together

```python
def answer_stream(question, top_k=6):
    passages = search(question, top_k=top_k)  # step 1: retrieve
    yield ("sources", passages)               # immediately send sources to frontend
    user_prompt = build_user_prompt(question, passages)
    stream = groq_client.chat.completions.create(..., stream=True)  # step 2: generate
    for chunk in stream:
        yield ("token", chunk.choices[0].delta.content)  # step 3: stream tokens
```

This is a Python **generator** — a function that `yield`s values one at a time instead of returning everything at once. This is what makes the streaming work. The frontend receives tokens as they arrive from Groq, not after the whole answer is ready.

### Groq — The LLM Provider

Groq is a company that provides extremely fast LLM inference using custom hardware called LPUs (Language Processing Units). The model you used, **Llama 3.3 70B**, is Meta's open-source language model with 70 billion parameters. On Groq, it runs fast enough for smooth streaming.

`temperature=0.3` — A low temperature means the model is more "focused" and less random. A temperature of 0 would be completely deterministic (same answer every time). A temperature of 1.0 would be more creative but potentially less accurate. For RAG, low temperature is better because you want faithfulness to the source text.

### `main.py` — The FastAPI Server

**FastAPI** is a Python web framework for building APIs. It's fast, automatic, and modern. You used it to expose one main endpoint: `POST /ask`.

**Server-Sent Events (SSE)** — The streaming protocol used. Instead of waiting for the full answer, the server sends chunks of data as they become available. In the browser, you read these with a `ReadableStream`. Each event looks like:

```
data: {"type": "token", "content": "Arise"}\n\n
data: {"type": "token", "content": ", awake"}\n\n
```

The double newline `\n\n` is the SSE delimiter — it marks the end of one event.

**CORS (Cross-Origin Resource Sharing)** — A browser security rule that prevents a page on `localhost:3000` from calling an API on `localhost:8000` unless the API explicitly allows it. You added `CORSMiddleware` to FastAPI to allow the Next.js dev server to call the backend.

---

## Day 4 — The Frontend

### Technology choices

**Next.js 16** — A React framework. React is a JavaScript library for building user interfaces. Next.js adds file-based routing, server-side rendering, and a great developer experience on top of React.

**TypeScript** — JavaScript with types. Instead of `let x = "hello"`, you write `let x: string = "hello"`. TypeScript catches bugs before they happen. You defined types for `Source` and `Message` to make the chat state type-safe.

**Tailwind CSS v4** — A utility-first CSS framework. Instead of writing separate CSS files, you add class names directly in HTML: `className="bg-amber-600 text-white rounded-xl px-4 py-3"`. Fast to write, easy to read once you know the class names.

### The streaming frontend logic

This is the most technically complex part of the frontend:

```typescript
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const events = buffer.split("\n\n");
  buffer = events.pop() ?? "";  // keep incomplete event in buffer

  for (const evt of events) {
    const data = JSON.parse(evt.slice(5).trim());  // strip "data: " prefix
    // update state based on data.type
  }
}
```

Why the buffer? Network data arrives in arbitrary chunks. An SSE event (`data: {...}\n\n`) might arrive split across two network reads. The buffer accumulates bytes until a full `\n\n`-terminated event is available.

**React state management** — `useState` holds the chat messages. When a new token arrives, you update the last message's `content` by appending to it. This is why the text appears letter by letter — each `setMessages` call re-renders the last bubble with one more token.

### UI design decisions

- **Amber/stone color palette** — warm tones that feel appropriate for spiritual content
- **Source cards** — collapsed by default, expand on click. Shows the actual passage text, volume, and chapter. This is what makes the chatbot trustworthy — you can always verify what it said.
- **Suggested questions** — shown on the empty state to help users get started
- **Auto-scroll** — `useRef` + `scrollIntoView` keeps the bottom visible as the answer streams in

---

## The Full Data Flow (End to End)

Here is exactly what happens when you type "What is the purpose of life?" and hit Ask:

1. **Frontend**: `handleSend()` runs. It adds your message to state and opens a `fetch` to `POST /ask` with `stream: true` via `ReadableStream`.

2. **FastAPI** receives the request at `POST /ask`. It calls `answer_stream("What is the purpose of life?")`.

3. **RAG pipeline** (`rag.py`): calls `search(question)`.

4. **Retrieval** (`retrieval.py`): encodes the question with BGE (prepending the query prefix), gets a 384-dimensional vector, sends it to Qdrant via the REST API.

5. **Qdrant** does a cosine similarity search over 4,922 stored vectors, returns the top 6 most similar passages with their payloads.

6. **Back in `rag.py`**: yields `("sources", passages)` — the FastAPI generator immediately sends `data: {"type": "sources", ...}\n\n` to the frontend.

7. **Frontend**: receives the sources event, stores them in `message.sources` (visible when you click "Show X sources").

8. **RAG pipeline**: builds the user prompt by formatting the 6 passages + question into the template in `prompts.py`. Calls `groq_client.chat.completions.create(stream=True)`.

9. **Groq** starts generating tokens. Each token comes back in a streaming chunk. `rag.py` yields `("token", delta)` for each one.

10. **FastAPI** streams each token as `data: {"type": "token", "content": "..."}\n\n`.

11. **Frontend**: each token event triggers a `setMessages` call that appends the token to the last message's `content`. React re-renders the bubble, and you see the text appear word by word.

12. **FastAPI** sends `data: {"type": "done"}\n\n` and closes the stream.

13. **Frontend**: `setLoading(false)`, input re-enables.

---

## Concepts Summary

| Concept | What it means | Where you used it |
|---|---|---|
| RAG | Retrieve relevant docs, then generate with an LLM | The whole project |
| Embeddings | Text → numbers that capture meaning | `embed_corpus.py`, `retrieval.py` |
| Vector database | Database for similarity search | Qdrant |
| Cosine similarity | Measure of angle between vectors | Qdrant distance metric |
| Web scraping | Programmatically extracting content from websites | `fetch_corpus.py` |
| REST API | Web service that sends/receives JSON over HTTP | FastAPI backend, Groq, Qdrant |
| Server-Sent Events | One-way streaming from server to browser | `/ask` endpoint, frontend reader |
| Python generators | Functions that `yield` values one at a time | `answer_stream()` |
| Docker | Run software in isolated containers | Qdrant |
| Prompt engineering | Writing system prompts that shape LLM behavior | `prompts.py` |
| CORS | Browser security policy for cross-origin requests | FastAPI middleware |
| lru_cache | Cache expensive function results in memory | Model and client loading |
| TypeScript | Typed JavaScript | Frontend |
| React state | UI data that triggers re-renders when changed | `useState` in `page.tsx` |
| Tailwind CSS | Utility CSS classes in HTML | All frontend styling |

---

## Things That Went Wrong (And What You Learned)

**403 Forbidden from Wikisource** — Wikisource blocks requests without a `User-Agent` header. Fix: set a descriptive `User-Agent` string. Lesson: always identify your scraper politely.

**Empty page content** — Some Wikisource pages use transclusion (content stored in sub-pages). The regular HTML fetch returned a blank div. Fix: use the REST API endpoint which returns the fully rendered page. Lesson: understand the difference between raw HTML and rendered HTML.

**Double URL encoding** — The `?` in a chapter title like "What is Duty?" is stored as `%3F` in the href. Calling `quote()` on that turned `%3F` into `%253F`. Fix: `unquote()` before `quote()`. Lesson: always decode before re-encoding.

**429 Too Many Requests** — Wikisource rate-limits scrapers. Fix: exponential backoff — wait 5s, then 10s, then 20s on each retry. Lesson: external APIs have limits; retry with increasing delays.

**Qdrant 400 payload too large** — Trying to upsert 4,922 points at once sent a 44 MB payload over the limit. Fix: upsert in batches of 64. Lesson: batch your writes.

**`recreate_collection` deprecated** — The qdrant-client library deprecated `recreate_collection` in newer versions. Fix: call `delete_collection` then `create_collection`. Lesson: always check library changelogs when upgrading.

---

## What you could build next

- **Add more volumes**: run `python scripts/fetch_corpus.py --volumes 6,7,8,9` to index all 9 volumes
- **Better chunking**: instead of splitting by paragraph, try overlapping chunks of fixed token length — retrieval quality often improves
- **Hybrid search**: combine semantic (embedding) search with keyword (BM25) search — Qdrant supports this
- **Conversation history**: pass the last few messages to the LLM so it can answer follow-up questions
- **Reranking**: after retrieving top-20 passages, use a cross-encoder to pick the best 6 — more accurate than pure vector similarity
- **Evaluation**: build a set of test questions with known good answers and measure retrieval precision and answer quality
- **Deployment**: containerize the FastAPI backend with Docker, deploy Qdrant with persistence to a cloud VM, deploy the Next.js frontend to Vercel
