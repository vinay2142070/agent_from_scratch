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
# 1. Create and activate virtual env
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install openai python-dotenv

# 3. Create .env file in the project folder
echo "OPENAI_API_KEY=sk-...your-key-here" > .env

# 4. Run
python agent.py
```

Get your API key at https://platform.openai.com/api-keys

## Architecture walkthrough

### 1. ReAct loop (`Agent._react_loop`)

The loop that runs on every user turn:

```
User message
    ↓
LLM call (with tools + full message history)
    ↓
tool_calls in response? ──YES──→ execute tools → append results → loop back
    ↓ NO
msg.content = final answer → return to user
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
  "schema": {                  # what the LLM reads (OpenAI format)
      "type": "function",
      "function": {
          "name": "...",
          "description": "...",
          "parameters": { ... }   # JSON Schema
      }
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

## Message format (OpenAI API)

Understanding this is crucial. The conversation is a list of dicts:

```python
[
  # System prompt is always first
  {"role": "system", "content": "You are a helpful assistant..."},

  {"role": "user", "content": "What is 2^16?"},

  # LLM decided to use a tool — appended directly (NOT via add()):
  {
    "role": "assistant",
    "content": None,               # may be None when tool_calls present
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {"name": "calculator", "arguments": "{\"expression\": \"2**16\"}"}
      }
    ]
  },

  # Tool result goes back as role="tool" (OpenAI convention):
  {"role": "tool", "tool_call_id": "call_abc123", "content": "65536"},

  # LLM gives final answer:
  {"role": "assistant", "content": "2^16 is 65536."}
]
```

### Key gotcha: appending tool-call assistant messages

When the assistant response contains `tool_calls`, you must append
the message **directly** to `short_term.messages`, not via `add()`.

`add(role, content)` always builds `{"role": role, "content": content}`.
Passing a dict as `content` nests it incorrectly:

```python
# WRONG — nests the dict under "content":
self.short_term.add("assistant", {"role": "assistant", "tool_calls": [...]})
# produces: {"role": "assistant", "content": {"role": ..., "tool_calls": ...}}

# CORRECT — append the full dict flat:
self.short_term.messages.append({"role": "assistant", "content": None, "tool_calls": [...]})
```

## Extending this agent

### Add a web search tool
```python
def tool_web_search(query: str) -> str:
    import requests
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"query": query, "api_key": os.environ["TAVILY_API_KEY"]}
    )
    return resp.json()["results"][0]["content"]
```
Add to `TOOLS` with a schema in the same `"type": "function"` format — done.

### Swap SQLite for PostgreSQL
```python
# In LongTermMemory.__init__:
import psycopg2
self.conn = psycopg2.connect(os.environ["DATABASE_URL"])
```

### Add vector search
Replace `LongTermMemory` with `SemanticMemory` from `semantic_memory.py`.
Uses OpenAI `text-embedding-3-small` for real embeddings — just set
`OPENAI_API_KEY` in your `.env` (same key, no extra setup).

### Multi-agent setup
Run two `Agent` instances. Give one a `handoff_to_specialist` tool that
posts to a queue. The second agent polls the queue. That's a multi-agent
pipeline — same principles, just more instances.