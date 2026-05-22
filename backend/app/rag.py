"""
The core RAG pipeline: retrieve -> build prompt -> stream answer.
"""
import os
from dotenv import load_dotenv
from groq import Groq

from app.retrieval import search
from app.prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv()
_client = Groq(api_key=os.environ["GROQ_API_KEY"])
_model = os.environ["GROQ_MODEL"]


def answer_stream(question: str, top_k: int = 6):
    """
    Yields (event_type, data) tuples:
      ("sources", [passage dicts])  -- emitted once, first
      ("token", str)                -- emitted repeatedly as the answer streams
    """
    passages = search(question, top_k=top_k)
    yield ("sources", passages)

    user_prompt = build_user_prompt(question, passages)
    stream = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # low temp = faithful to source, less invention
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield ("token", delta)


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "How do I deal with failure?"
    print(f"Question: {q}\n" + "=" * 60)
    for event, data in answer_stream(q):
        if event == "sources":
            print(f"\nRetrieved {len(data)} passages:")
            for p in data:
                print(f"  - Vol {p['volume']} — {p['chapter']} (score {p['score']})")
            print("\nAnswer:\n" + "-" * 60)
        elif event == "token":
            print(data, end="", flush=True)
    print("\n")