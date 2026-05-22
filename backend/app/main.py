"""
FastAPI app exposing the RAG pipeline over HTTP with streaming.
"""
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag import answer_stream

app = FastAPI(title="VivekaGPT API")

# Allow the Next.js dev server to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    """
    Server-Sent Events stream.
    Each line: data: {"type": "...", ...}\n\n
    """
    def event_generator():
        for event, data in answer_stream(req.question):
            if event == "sources":
                payload = {"type": "sources", "sources": data}
            else:
                payload = {"type": "token", "content": data}
            yield f"data: {json.dumps(payload)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )