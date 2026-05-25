# AI Agent from Scratch — Tutorial

Build a production-pattern AI agent with **zero frameworks**.
Covers the exact mechanisms used by LangChain, AWS Bedrock AgentCore, and Mem0.

## Files

| File | What it teaches |
|---|---|
| `agent.py` | Core agent: ReAct loop, tool routing, short + long-term memory |
| `semantic_memory.py` | Bonus: vector similarity search (how Pinecone/pgvector works) |

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
python agent.py
```

## Architecture walkthrough

### 1. ReAct loop (`Agent._react_loop`)

The loop that runs on every user turn:

```
User message
    ↓
LLM call (with tools + full message history)
    ↓
tool_use blocks? ──YES──→ execute tools → append results → loop back
    ↓ NO
text block = final answer → return to user
```

This is the **exact same pattern** as:
- LangGraph's `create_react_agent`
- AWS Bedrock's native agent runtime
- OpenAI's Assistants API

### 2. Tool system (`TOOLS` dict)

Each tool is:
```python
{
  "fn": python_function,       # actual executor
  "schema": {                  # what the LLM reads
      "name": "...",
      "description": "...",
      "input_schema": { ... }  # JSON Schema
  }
}
```

The LLM reads the schemas and decides when to call tools.
The router (`_execute_tool`) dispatches by name.

### 3. Short-term memory (`ShortTermMemory`)

In-memory message buffer for this session.
Compresses itself when it grows too large (summary → inject back).

```
Real-world equivalent:
  LangGraph Checkpointer (MemorySaver / PostgresSaver)
  AWS AgentCore AgentCoreMemorySaver
```

### 4. Long-term memory (`LongTermMemory`)

SQLite-backed key-value store + conversation summaries.
Persists across sessions. Injected into the system prompt at the start
of each turn (memory injection pattern).

```
Real-world equivalent:
  DynamoDB + LangChain DynamoDBChatMessageHistory
  AWS AgentCore AgentCoreMemoryStore
  Mem0's user-scope memory
```

### 5. Semantic memory (`semantic_memory.py`)

Embeds text to float vectors. Searches by cosine similarity.
Shows how Pinecone / pgvector / Weaviate work internally.

```
Real-world equivalent:
  pgvector's <=> operator
  Pinecone similarity search
  LangChain VectorStoreRetrieverMemory
```

## Message format (Anthropic API)

Understanding this is crucial. The conversation is a list of dicts:

```python
[
  {"role": "user",      "content": "What is 2^16?"},

  # LLM decided to use a tool:
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "tu_01", "name": "calculator",
     "input": {"expression": "2**16"}}
  ]},

  # Tool result goes back as a "user" message (Anthropic convention):
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "tu_01", "content": "65536"}
  ]},

  # LLM gives final answer:
  {"role": "assistant", "content": "2^16 is 65536."}
]
```

## Extending this agent

### Add a web search tool
```python
def tool_web_search(query: str) -> str:
    import requests
    resp = requests.get(
        "https://api.tavily.com/search",
        json={"query": query, "api_key": os.environ["TAVILY_API_KEY"]}
    )
    return resp.json()["results"][0]["content"]
```
Register it in `TOOLS` with a schema — done.

### Swap SQLite for PostgreSQL
```python
# In LongTermMemory.__init__:
import psycopg2
self.conn = psycopg2.connect(os.environ["DATABASE_URL"])
```

### Add vector search
Replace `LongTermMemory` with `SemanticMemory` from `semantic_memory.py`.
Use a real embedding model (OpenAI text-embedding-3-small, Cohere, etc.)
for actual semantic recall.

### Multi-agent setup
Run two `Agent` instances. Give one a `handoff_to_specialist` tool that
posts to a queue. The second agent polls the queue. That's a multi-agent
pipeline — same principles, just more instances.
