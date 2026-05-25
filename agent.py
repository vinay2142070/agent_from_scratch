"""
==============================================================
  BUILD AN AI AGENT FROM SCRATCH — no frameworks
  Matches real-world agent patterns (ReAct loop, tool use,
  short-term + long-term memory, multi-turn conversation)

  ** OpenAI version (gpt-4o) + SemanticMemory **
==============================================================

ARCHITECTURE:
  User input
      ↓
  [SemanticMemory] inject relevant facts via vector search
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
  4. Long-term memory   (SemanticMemory — SQLite + vector search)
  5. Prompt engineering for tool use
"""

import os
import json
import math
import sqlite3
import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Import SemanticMemory — this replaces the old keyword-based LongTermMemory
from semantic_memory import SemanticMemory

load_dotenv()


# ─────────────────────────────────────────────
# SECTION 1: TOOL DEFINITIONS
# Tools are just Python functions + a JSON schema
# that the LLM reads to know what's available.
# ─────────────────────────────────────────────

# Module-level SemanticMemory instance shared by tool functions
# (initialised once in Agent.__init__, assigned here)
_semantic_memory: SemanticMemory = None


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
    """Save a fact to long-term semantic memory."""
    _semantic_memory.save(key, value)
    return f"Saved: {key} = {value}"


def tool_recall(query: str) -> str:
    """Retrieve semantically similar facts from long-term memory."""
    results = _semantic_memory.search(query)
    if not results:
        return "Nothing found in long-term memory."
    return "\n".join(f"- [{k}] {v}" for k, v in results)


# OpenAI tool schema format uses "function" wrapper
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
                "description": "Search long-term memory for facts or preferences using semantic similarity.",
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
# SECTION 2: SHORT-TERM MEMORY
# In-RAM message buffer for this session.
# Compresses itself when it grows too large.
# ─────────────────────────────────────────────

class ShortTermMemory:
    MAX_MESSAGES = 20

    def __init__(self):
        self.messages: list[dict] = []

    def add(self, role: str, content):
        self.messages.append({"role": role, "content": content})

    def get_all(self) -> list[dict]:
        return self.messages

    def compress_if_needed(self, llm_client: OpenAI, system_prompt: str):
        """
        When buffer grows large, summarise the oldest half and replace with
        a synthetic summary message. Also saves summary to SemanticMemory
        for cross-session recall.
        """
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

        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are a helpful summariser."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        summary = resp.choices[0].message.content

        # Save summary into SemanticMemory so it persists across sessions
        _semantic_memory.save_summary(summary)

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
# SECTION 3: THE AGENT (ReAct loop)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools and memory.

When you need to perform a calculation, get the time, or remember/recall information,
use the appropriate tool. Always think step by step.

You have two types of memory:
- Short-term: the current conversation (already in your context)
- Long-term: facts saved across sessions via vector search (use `recall` to search, `remember` to save)

Be concise and direct. After using a tool, interpret the result naturally."""


class Agent:
    MAX_ITERATIONS = 10

    def __init__(self):
        global _semantic_memory

        self.client = OpenAI()               # reads OPENAI_API_KEY from .env
        self.short_term = ShortTermMemory()
        self.long_term = SemanticMemory()    # vector-backed long-term memory
        _semantic_memory = self.long_term    # wire up tool functions
        self.tool_schemas = [t["schema"] for t in TOOLS.values()]

    def _build_system_prompt(self, user_input: str) -> str:
        """
        Memory injection: search SemanticMemory for facts relevant to the
        current input and prepend them to the system prompt.

        This is the exact mechanism used by Mem0, AgentCore, and LangMem
        before every LLM call — the difference is we use real vector
        similarity instead of keyword matching.
        """
        prompt = SYSTEM_PROMPT

        # Cross-session summaries
        summaries = self.long_term.get_recent_summaries(limit=2)
        if summaries:
            prompt += "\n\nContext from previous sessions:\n"
            prompt += "\n".join(f"- {s}" for s in summaries)

        # Semantically relevant facts
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

        Message flow:
          {role: system,     content: "You are..."}
          {role: user,       content: "What is 2^16?"}
          {role: assistant,  tool_calls: [{id, function: {name, arguments}}]}
          {role: tool,       tool_call_id: id, content: "65536"}
          {role: assistant,  content: "2^16 is 65536."}

        Key rule: assistant tool-call messages must be appended directly
        to short_term.messages (not via add()), otherwise the dict gets
        nested under "content" and OpenAI returns a 400 error.
        """
        for iteration in range(self.MAX_ITERATIONS):
            print(f"  [loop] iteration {iteration + 1}")

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
                # Append directly — NOT via add()
                self.short_term.messages.append({
                    "role": "assistant",
                    "content": msg.content,   # may be None — that's fine
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

                for tc in msg.tool_calls:
                    tool_input = json.loads(tc.function.arguments)
                    print(f"  [tool] calling {tc.function.name}({tool_input})")
                    result = self._execute_tool(tc.function.name, tool_input)
                    print(f"  [tool] result: {result[:80]}")

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
# SECTION 4: RUN IT
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI Agent (OpenAI gpt-4o) — ReAct + SemanticMemory")
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

    # Reset short-term buffer — demo leaves tool messages that would
    # cause OpenAI 400: "tool message with no preceding tool_calls"
    agent.short_term.messages = []
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
            print("\n--- Long-term facts (semantic DB) ---")
            conn = sqlite3.connect("agent_semantic_memory.db")
            rows = conn.execute("SELECT key, value FROM facts ORDER BY created_at DESC LIMIT 10").fetchall()
            for k, v in rows:
                print(f"  [{k}] {v}")
            print()
            continue

        response = agent.chat(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()