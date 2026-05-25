"""
==============================================================
  BUILD AN AI AGENT FROM SCRATCH — no frameworks
  Matches real-world agent patterns (ReAct loop, tool use,
  short-term + long-term memory, multi-turn conversation)
  
  ** OpenAI version (gpt-4o) **
==============================================================

ARCHITECTURE:
  User input
      ↓
  [Memory] inject context
      ↓
  LLM call  ← this is the "planner"
      ↓
  Parse response: TOOL_CALL or FINAL_ANSWER?
      ├─ TOOL_CALL → execute tool → append result → loop back
      └─ FINAL_ANSWER → save to memory → return to user

CONCEPTS COVERED:
  1. ReAct loop  (Reason → Act → Observe → Reason ...)
  2. Tool definition & routing
  3. Short-term memory  (conversation buffer in this session)
  4. Long-term memory   (SQLite — persists across runs)
  5. Semantic memory    (vector similarity search w/ numpy)
  6. Prompt engineering for tool use
"""

import os
import json
import math
import sqlite3
import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# SECTION 1: TOOL DEFINITIONS
# Tools are just Python functions + a JSON schema
# that the LLM reads to know what's available.
# ─────────────────────────────────────────────

def tool_calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def tool_get_datetime(timezone: str = "UTC") -> str:
    """Return current date and time."""
    now = datetime.datetime.utcnow()
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def tool_remember(key: str, value: str) -> str:
    """Save a fact to long-term memory (SQLite)."""
    db = LongTermMemory()
    db.save(key, value)
    return f"Saved: {key} = {value}"


def tool_recall(query: str) -> str:
    """Retrieve facts from long-term memory by fuzzy search."""
    db = LongTermMemory()
    results = db.search(query)
    if not results:
        return "Nothing found in long-term memory."
    return "\n".join(f"- [{k}] {v}" for k, v in results)


# OpenAI tool schema format uses "function" wrapper
# (different from Anthropic's flat schema format)
TOOLS = {
    "calculator": {
        "fn": tool_calculator,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression. Use Python math syntax.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "e.g. '2 ** 10' or 'sqrt(144)'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    },
    "get_datetime": {
        "fn": tool_get_datetime,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_datetime",
                "description": "Get the current date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "Timezone label (informational only)"}
                    },
                    "required": []
                }
            }
        }
    },
    "remember": {
        "fn": tool_remember,
        "schema": {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Save an important fact or user preference to long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Short label for the fact"},
                        "value": {"type": "string", "description": "The fact to remember"}
                    },
                    "required": ["key", "value"]
                }
            }
        }
    },
    "recall": {
        "fn": tool_recall,
        "schema": {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search long-term memory for facts or preferences.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"}
                    },
                    "required": ["query"]
                }
            }
        }
    },
}


# ─────────────────────────────────────────────
# SECTION 2: LONG-TERM MEMORY (SQLite)
# ─────────────────────────────────────────────

class LongTermMemory:
    def __init__(self, db_path: str = "agent_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()

    def save(self, key: str, value: str):
        existing = self.conn.execute(
            "SELECT id FROM facts WHERE key = ?", (key,)
        ).fetchone()
        if existing:
            self.conn.execute("UPDATE facts SET value = ? WHERE key = ?", (value, key))
        else:
            self.conn.execute("INSERT INTO facts (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def search(self, query: str, limit: int = 5):
        words = query.lower().split()
        results = self.conn.execute(
            "SELECT key, value FROM facts ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        scored = []
        for key, value in results:
            text = (key + " " + value).lower()
            score = sum(1 for w in words if w in text)
            if score > 0:
                scored.append((score, key, value))
        scored.sort(reverse=True)
        return [(k, v) for _, k, v in scored[:limit]]

    def save_summary(self, summary: str):
        self.conn.execute("INSERT INTO summaries (summary) VALUES (?)", (summary,))
        self.conn.commit()

    def get_recent_summaries(self, limit: int = 3):
        rows = self.conn.execute(
            "SELECT summary FROM summaries ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [r[0] for r in rows]


# ─────────────────────────────────────────────
# SECTION 3: SHORT-TERM MEMORY
# ─────────────────────────────────────────────

class ShortTermMemory:
    MAX_MESSAGES = 20

    def __init__(self):
        self.messages: list[dict] = []

    def add(self, role: str, content):
        self.messages.append({"role": role, "content": content})

    def get_all(self) -> list[dict]:
        return self.messages

    def compress_if_needed(self, llm_client, system_prompt: str):
        if len(self.messages) < self.MAX_MESSAGES:
            return

        half = len(self.messages) // 2
        old_msgs = self.messages[:half]
        self.messages = self.messages[half:]

        summary_prompt = (
            "Summarise this conversation history in 2-3 sentences, "
            "preserving key facts and decisions:\n\n" +
            "\n".join(
                f"{m['role'].upper()}: {m['content'] if isinstance(m['content'], str) else '[tool interaction]'}"
                for m in old_msgs
            )
        )

        # OpenAI call — system goes inside messages list
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are a helpful summariser."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        summary = resp.choices[0].message.content

        LongTermMemory().save_summary(summary)

        self.messages.insert(0, {
            "role": "user",
            "content": f"[Previous conversation summary: {summary}]"
        })
        self.messages.insert(1, {
            "role": "assistant",
            "content": "Understood. I have the context from our earlier conversation."
        })
        print(f"  [memory] Compressed {half} messages → summary")


# ─────────────────────────────────────────────
# SECTION 4: THE AGENT (ReAct loop)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools and memory.

When you need to perform a calculation, get the time, or remember/recall information,
use the appropriate tool. Always think step by step.

You have two types of memory:
- Short-term: the current conversation (already in your context)
- Long-term: facts saved across sessions (use `recall` to search, `remember` to save)

Be concise and direct. After using a tool, interpret the result naturally."""


class Agent:
    MAX_ITERATIONS = 10

    def __init__(self):
        self.client = OpenAI()  # reads OPENAI_API_KEY from env automatically
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.tool_schemas = [t["schema"] for t in TOOLS.values()]

    def _build_system_prompt(self, user_input: str) -> str:
        prompt = SYSTEM_PROMPT

        summaries = self.long_term.get_recent_summaries(limit=2)
        if summaries:
            prompt += "\n\nContext from previous sessions:\n"
            prompt += "\n".join(f"- {s}" for s in summaries)

        relevant_facts = self.long_term.search(user_input, limit=4)
        if relevant_facts:
            prompt += "\n\nRelevant facts from memory:\n"
            prompt += "\n".join(f"- [{k}] {v}" for k, v in relevant_facts)

        return prompt

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name not in TOOLS:
            return f"Unknown tool: {tool_name}"
        fn = TOOLS[tool_name]["fn"]
        try:
            return fn(**tool_input)
        except Exception as e:
            return f"Tool error: {e}"

    def _react_loop(self, system: str) -> str:
        """
        OpenAI ReAct loop.

        Key differences from Anthropic version:
        - System prompt goes as first message with role="system"
        - Tool calls are in response.choices[0].message.tool_calls
        - Tool results go back as role="tool" messages (not role="user")
        - assistant message must be appended as dict, not as object
        - tool_call_id links result back to the call

        Message flow:
          {role: system,     content: "You are..."}
          {role: user,       content: "What is 2^16?"}
          {role: assistant,  tool_calls: [{id, function: {name, arguments}}]}
          {role: tool,       tool_call_id: id, content: "65536"}
          {role: assistant,  content: "2^16 is 65536."}
        """
        for iteration in range(self.MAX_ITERATIONS):
            print(f"  [loop] iteration {iteration + 1}")

            # System prompt is prepended as a message each time
            messages = [{"role": "system", "content": system}] + self.short_term.get_all()

            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1024,
                tools=self.tool_schemas,
                messages=messages
            )

            msg = response.choices[0].message

            # Case 1: model wants to call tools
            if msg.tool_calls:
                # Append directly — NOT via add(), which would nest it under "content"
                self.short_term.messages.append({
                    "role": "assistant",
                    "content": msg.content,  # may be None — that's fine
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                # Execute each tool, append result as role="tool"
                for tc in msg.tool_calls:
                    tool_input = json.loads(tc.function.arguments)
                    print(f"  [tool] calling {tc.function.name}({tool_input})")
                    result = self._execute_tool(tc.function.name, tool_input)
                    print(f"  [tool] result: {result[:80]}")

                    # Each tool result is its own message in OpenAI format
                    self.short_term.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })
                continue

            # Case 2: final text answer
            if msg.content:
                self.short_term.add("assistant", msg.content)
                return msg.content

            return "Agent finished with no output."

        return "Reached max iterations without a final answer."

    def chat(self, user_input: str) -> str:
        system = self._build_system_prompt(user_input)
        self.short_term.add("user", user_input)
        answer = self._react_loop(system)
        self.short_term.compress_if_needed(self.client, system)
        return answer


# ─────────────────────────────────────────────
# SECTION 5: RUN IT
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI Agent (OpenAI gpt-4o) — ReAct + Memory demo")
    print("  Type 'quit' to exit, 'memory' to inspect state")
    print("=" * 60)

    agent = Agent()

    # demo_inputs = [
    #     "What is the square root of 1764?",
    #     "What time is it right now?",
    #     "Remember that my name is VK and I prefer concise answers.",
    #     "Calculate (2 ** 16) + (3 ** 8)",
    #     "What do you remember about me?",
    # ]

    # for user_input in demo_inputs:
    #     print(f"\nYou: {user_input}")
    #     response = agent.chat(user_input)
    #     print(f"Agent: {response}")
    #     print("-" * 40)

    print("\n[Demo done. Entering interactive mode]\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "memory":
            print("\n--- Short-term buffer ---")
            for m in agent.short_term.get_all():
                role = m["role"]
                content = m["content"]
                if isinstance(content, str):
                    print(f"  {role}: {content[:80]}")
                else:
                    print(f"  {role}: [structured]")
            print("\n--- Long-term facts ---")
            conn = sqlite3.connect("agent_memory.db")
            rows = conn.execute("SELECT key, value FROM facts ORDER BY created_at DESC LIMIT 10").fetchall()
            for k, v in rows:
                print(f"  [{k}] {v}")
            print()
            continue

        response = agent.chat(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()