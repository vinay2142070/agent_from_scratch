"""
BONUS: Semantic (vector) memory using OpenAI embeddings
========================================================
This shows how vector search actually works under the hood —
the same mechanism used by Pinecone, Weaviate, pgvector, etc.

Instead of keyword matching (what we used in agent.py),
semantic search embeds text into vectors and finds the
nearest neighbours using cosine similarity.

Drop-in replacement for LongTermMemory.search().
Requires: pip install openai python-dotenv
Same OPENAI_API_KEY in .env — no extra setup needed.
"""

import os
import json
import math
import sqlite3
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors.
    Returns 1.0 for identical direction, 0.0 for orthogonal.
    This is what every vector DB uses internally.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_embedding(text: str, client: OpenAI) -> list[float]:
    """
    Get a real text embedding from OpenAI.

    text-embedding-3-small returns a 1536-dim float vector.
    Semantically similar texts produce vectors with high cosine similarity,
    even if they share no exact words.

    Cost: ~$0.00002 per 1K tokens — effectively free for this use case.
    """
    resp = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return resp.data[0].embedding  # list of 1536 floats


class SemanticMemory:
    """
    Vector-based long-term memory backed by SQLite.

    Storage layout:
      facts table: id, key, value, embedding (JSON float array)

    On save:   embed the value → store vector alongside text
    On search: embed the query → cosine similarity against all stored vectors
               → return top-K results above threshold

    Real-world equivalent:
      - This class ≈ a Pinecone collection / pgvector table
      - cosine_similarity() ≈ pgvector's <=> operator
      - get_embedding() ≈ OpenAI text-embedding-3-small API call

    To scale beyond ~10k entries: swap SQLite for pgvector or Pinecone
    and replace the Python cosine loop with a native ANN index query.
    """

    SIMILARITY_THRESHOLD = 0.4  # lower than keyword search since vectors are denser

    def __init__(self, db_path: str = "agent_semantic_memory.db"):
        self.client = OpenAI()  # reads OPENAI_API_KEY from env
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()

    def save(self, key: str, value: str):
        text_to_embed = f"{key}: {value}"
        embedding = get_embedding(text_to_embed, self.client)
        emb_json = json.dumps(embedding)

        existing = self.conn.execute(
            "SELECT id FROM facts WHERE key = ?", (key,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE facts SET value = ?, embedding = ? WHERE key = ?",
                (value, emb_json, key)
            )
        else:
            self.conn.execute(
                "INSERT INTO facts (key, value, embedding) VALUES (?, ?, ?)",
                (key, value, emb_json)
            )
        self.conn.commit()
        print(f"  [semantic memory] saved '{key}' ({len(embedding)}-dim vector)")

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """
        1. Embed the query
        2. Load all stored embeddings from SQLite
        3. Compute cosine similarity for each
        4. Return top-K above threshold
        """
        query_embedding = get_embedding(query, self.client)
        rows = self.conn.execute(
            "SELECT key, value, embedding FROM facts"
        ).fetchall()

        scored = []
        for key, value, emb_json in rows:
            stored_emb = json.loads(emb_json)
            sim = cosine_similarity(query_embedding, stored_emb)
            if sim >= self.SIMILARITY_THRESHOLD:
                scored.append((sim, key, value))

        scored.sort(reverse=True)
        results = [(k, v) for _, k, v in scored[:limit]]
        print(f"  [semantic memory] '{query[:40]}' → {len(results)} results")
        return results


# ─────────────────────────────────────────────
# Demo: semantic search finds related facts
# even without exact keyword overlap
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Semantic memory demo")
    print("=" * 40)
    mem = SemanticMemory()

    print("\n[Saving facts...]")
    mem.save("user_name", "VK")
    mem.save("user_location", "Bengaluru, India")
    mem.save("user_preference", "prefers concise answers")
    mem.save("user_stack", "uses n8n, OpenAI API, SQLite for automation")
    mem.save("user_hobby", "building AI assistants on Telegram")

    print("\n[Searching: 'automation tools']")
    for k, v in mem.search("automation tools"):
        print(f"  [{k}] {v}")

    print("\n[Searching: 'where does the user live']")
    for k, v in mem.search("where does the user live"):
        print(f"  [{k}] {v}")

    print("\n[Searching: 'messaging app bot']")
    for k, v in mem.search("messaging app bot"):
        print(f"  [{k}] {v}")