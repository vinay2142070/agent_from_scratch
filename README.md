# AI Agent from Scratch — Tutorial

Build a production-pattern AI agent with **zero frameworks**.
Covers the exact mechanisms used by LangChain, AWS Bedrock AgentCore, and Mem0.

## Files

| File | What it teaches |
|---|---|
| `agent.py` | Core agent: ReAct loop, tool routing, short + long-term memory |
| `semantic_memory.py` | Vector memory: OpenAI embeddings + cosine similarity search |

## Setup

```bash
# 1. Create project folder and activate virtual env
mkdir agent-from-scratch && cd agent-from-scratch
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install openai python-dotenv

# 3. Create .env file
echo "OPENAI_API_KEY=sk-...your-key-here" > .env

# 4. Place both files in the folder
#    agent.py  +  semantic_memory.py

# 5. Run
python agent.py
```

Get your API key at https://platform.openai.com/api-keys

## How the two files connect

```
agent.py
  └── from semantic_memory import SemanticMemory
        ├── Agent.__init__()       → self.long_term = SemanticMemory()
        ├── tool_remember()        → _semantic_memory.save(key, value)
        ├── tool_recall()          → _semantic_memory.search(query)
        ├── _build_system_prompt() → _semantic_memory.search(user_input)  [memory injection]
        └── compress_if_needed()   → _semantic_memory.save_summary(text)
```

`agent.py` imports and owns one `SemanticMemory` instance.
All memory reads and writes go through it — keyword matching is gone entirely.

## Architecture walkthrough

### 1. ReAct loop (`Agent._react_loop`)

```
User message
    ↓
LLM call (with tools + full message history)
    ↓
tool_calls in response? ──YES──→ execute tools → append results → loop back
    ↓ NO
msg.content = final answer → return to user
```

Same pattern as LangGraph's `create_react_agent` and AWS Bedrock's agent runtime.

### 2. Tool system (`TOOLS` dict)

```python
{
  "fn": python_function,
  "schema": {
      "type": "function",
      "function": {
          "name": "...",
          "description": "...",
          "parameters": { ... }   # JSON Schema
      }
  }
}
```

### 3. Short-term memory (`ShortTermMemory`)

In-RAM message buffer. Compresses itself when it hits 20 messages —
summarises the oldest half and saves the summary to `SemanticMemory`.

```
Real-world equivalent:
  LangGraph MemorySaver / PostgresSaver
  AWS AgentCore AgentCoreMemorySaver
```

### 4. Long-term memory (`SemanticMemory` from semantic_memory.py)

SQLite-backed vector store. Every fact is embedded with
`text-embedding-3-small` (1536-dim) and stored as a JSON float array.
On retrieval, the query is embedded and compared via cosine similarity.

```
Real-world equivalent:
  pgvector table  (cosine_similarity ≈ pgvector's <=> operator)
  Pinecone collection
  LangChain VectorStoreRetrieverMemory
```

### 5. Memory injection (`Agent._build_system_prompt`)

Before every LLM call:
1. Search `SemanticMemory` for facts relevant to the current input
2. Prepend them to the system prompt

This is exactly what Mem0, AgentCore, and LangMem do on each invocation.

## Message format (OpenAI API)

```python
[
  {"role": "system",    "content": "You are a helpful assistant..."},
  {"role": "user",      "content": "What is 2^16?"},

  # Tool call — append directly to messages list, NOT via add():
  {
    "role": "assistant",
    "content": None,
    "tool_calls": [
      {"id": "call_abc", "type": "function",
       "function": {"name": "calculator", "arguments": "{\"expression\": \"2**16\"}"}}
    ]
  },

  # Tool result — role="tool" with matching tool_call_id:
  {"role": "tool", "tool_call_id": "call_abc", "content": "65536"},

  {"role": "assistant", "content": "2^16 is 65536."}
]
```

### Key gotcha: tool-call assistant messages

`add(role, content)` builds `{"role": role, "content": content}`.
Passing a dict as content nests it incorrectly — OpenAI returns 400.

```python
# WRONG:
self.short_term.add("assistant", {"tool_calls": [...]})

# CORRECT:
self.short_term.messages.append({"role": "assistant", "content": None, "tool_calls": [...]})
```

## Files generated at runtime

| File | Contents |
|---|---|
| `agent_semantic_memory.db` | Facts + summaries with embeddings (persists across runs) |

Type `memory` during interactive mode to inspect the current buffer and DB.

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
Add `pip install requests` and register in `TOOLS` with a schema.

### Swap SQLite for PostgreSQL + pgvector
```python
# In SemanticMemory.__init__:
import psycopg2
self.conn = psycopg2.connect(os.environ["DATABASE_URL"])
# Replace cosine_similarity() with pgvector's <=> operator for ANN at scale
```

### Multi-agent setup
Run two `Agent` instances. Give one a `handoff_to_specialist` tool that
posts to a queue. The second agent polls the queue. Same principles, more instances.