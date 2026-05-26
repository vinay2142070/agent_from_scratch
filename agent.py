"""
==============================================================
  AI AGENT FROM SCRATCH
  OpenAI gpt-4o | Streaming | Tools | Semantic Memory | Planning
==============================================================

WHAT'S IN THIS FILE:
  Section 1 — Tool functions + TOOLS registry
  Section 2 — ShortTermMemory  (in-RAM per-session buffer)
  Section 3 — Agent            (ReAct loop, streaming, memory injection)
  Section 4 — Planner          (breaks a goal into ordered steps)
  Section 5 — PlanExecutor     (runs each step, synthesises final answer)
  Section 6 — main()           (interactive CLI, mode selector)

HOW THE TWO MODES WORK:
  ┌─────────────────────────────────────────────────────────┐
  │ CHAT mode  (default)                                    │
  │   User message → memory inject → ReAct loop → reply    │
  │   Good for: single-turn questions, tool calls           │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │ PLAN mode  (prefix input with "plan:")                  │
  │   Goal → Planner (1 LLM call, JSON step list)          │
  │        → PlanExecutor (runs each step via Agent)        │
  │        → Synthesiser (1 final LLM call, clean answer)  │
  │   Good for: multi-step research, chained calculations   │
  └─────────────────────────────────────────────────────────┘

REAL-WORLD EQUIVALENTS:
  ReAct loop      ≈ LangGraph create_react_agent / Bedrock agent runtime
  ShortTermMemory ≈ LangGraph MemorySaver / Bedrock AgentCoreMemorySaver
  SemanticMemory  ≈ pgvector table / Pinecone collection / Mem0 user-scope
  Planner         ≈ LangGraph plan-and-execute first pass
  PlanExecutor    ≈ CrewAI sequential task runner / AutoGPT task queue

SETUP:
  pip install openai python-dotenv requests
  .env: OPENAI_API_KEY=sk-...   SERPER_API_KEY=...  (serper optional)

COMMANDS (interactive mode):
  plan: <goal>   — run multi-step planner
  memory         — inspect short-term buffer + semantic DB
  quit           — exit
"""

import os
import json
import math
import sqlite3
import datetime
import requests
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
from semantic_memory import SemanticMemory

load_dotenv()


# ══════════════════════════════════════════════════════════
# SECTION 1 — TOOL DEFINITIONS
#
# Each tool is:
#   "name": {
#       "fn":     the actual Python function to call
#       "schema": JSON schema the LLM reads to decide when/how to call it
#   }
#
# The LLM NEVER runs the function — it only outputs a tool_calls block.
# Your code reads that block, looks up the function by name, and runs it.
# This is true for every agent framework (LangChain, Bedrock, CrewAI, etc.)
# ══════════════════════════════════════════════════════════

# Module-level reference to the shared SemanticMemory instance.
# Assigned in Agent.__init__() so tool functions can access it
# without needing to construct their own DB connection.
_semantic_memory: SemanticMemory = None


def tool_calculator(expression: str) -> str:
    """
    Safely evaluate a math expression using Python's math module.
    Uses a whitelist of allowed names — no access to builtins.
    Example: sqrt(1764) → 42.0
    """
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def tool_get_datetime(timezone: str = "UTC") -> str:
    """Return the current UTC date and time as a formatted string."""
    now = datetime.datetime.utcnow()
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def tool_remember(key: str, value: str) -> str:
    """
    Save a fact to long-term SemanticMemory (SQLite + embedding).
    The value is embedded and stored so it can be retrieved by
    semantic similarity later — even without exact keyword match.
    """
    _semantic_memory.save(key, value)
    return f"Saved: {key} = {value}"


def tool_recall(query: str) -> str:
    """
    Search long-term SemanticMemory for facts similar to the query.
    Uses cosine similarity on embeddings — not keyword matching.
    Returns top matching key-value pairs.
    """
    results = _semantic_memory.search(query)
    if not results:
        return "Nothing found in long-term memory."
    return "\n".join(f"- [{k}] {v}" for k, v in results)


def tool_web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web via Serper (Google Search API).
    Returns titles, URLs, and snippets for the top results.
    Requires SERPER_API_KEY in .env. Get a free key at serper.dev
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not set in .env"
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": min(num_results, 10)},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("organic", [])[:num_results]
        if not items:
            return "No results found."
        lines = []
        for item in items:
            lines.append(f"Title: {item.get('title')}")
            lines.append(f"URL:   {item.get('link')}")
            lines.append(f"Snippet: {item.get('snippet', '')}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Search error: {e}"


def tool_http_get(url: str, headers: dict = None) -> str:
    """
    Fetch raw content from a URL via HTTP GET.
    Useful for reading a specific page after web_search returns its URL,
    or calling a public REST API. Response is capped at 4000 chars.
    """
    try:
        resp = requests.get(url, headers=headers or {}, timeout=10)
        resp.raise_for_status()
        text = resp.text[:4000]
        return text if text else "(empty response)"
    except Exception as e:
        return f"HTTP error: {e}"


# Tool registry — maps tool name → {fn, schema}
# schema is what gets sent to the LLM via the `tools` parameter.
# The LLM reads name + description + parameters to decide when to call.
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
                        "timezone": {
                            "type": "string",
                            "description": "Timezone label (informational only)"
                        }
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
    "web_search": {
        "fn": tool_web_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web using Google (via Serper API). "
                    "Returns titles, URLs, and snippets. "
                    "Use for current events, facts, or anything needing up-to-date information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default 5, max 10)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    },
    "http_get": {
        "fn": tool_http_get,
        "schema": {
            "type": "function",
            "function": {
                "name": "http_get",
                "description": (
                    "Fetch the raw content of any public URL via HTTP GET. "
                    "Useful for reading a specific web page or REST API response."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Full URL including https://"},
                        "headers": {
                            "type": "object",
                            "description": "Optional HTTP headers as key-value pairs"
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    },
}


# ══════════════════════════════════════════════════════════
# SECTION 2 — SHORT-TERM MEMORY
#
# Holds the raw message list for one session in RAM.
# Passed to the LLM on every call as the conversation history.
#
# When the buffer grows beyond MAX_MESSAGES, it summarises
# the oldest half and replaces them with a synthetic summary
# message — this is "sliding window + summary" compression,
# the same technique LangChain's ConversationSummaryMemory uses.
#
# The summary is also saved to SemanticMemory so it persists
# across sessions (cross-session context injection).
# ══════════════════════════════════════════════════════════

class ShortTermMemory:
    MAX_MESSAGES = 20  # compress when buffer exceeds this

    def __init__(self):
        self.messages: list[dict] = []

    def add(self, role: str, content):
        """
        Append a simple message.
        Use messages.append() directly for structured messages
        (tool_calls assistant messages) to avoid nesting under "content".
        """
        self.messages.append({"role": role, "content": content})

    def get_all(self) -> list[dict]:
        return self.messages

    def compress_if_needed(self, llm_client: OpenAI, system_prompt: str):
        """
        Compress old messages when buffer is full.

        Steps:
          1. Take the oldest half of messages
          2. Ask gpt-4o-mini to summarise them (cheap model, short task)
          3. Save summary to SemanticMemory (cross-session persistence)
          4. Replace the old messages with two synthetic messages:
               user:      "[Previous summary: ...]"
               assistant: "Understood."
          This keeps the buffer manageable while preserving context.
        """
        if len(self.messages) < self.MAX_MESSAGES:
            return

        half = len(self.messages) // 2
        old_msgs = self.messages[:half]
        self.messages = self.messages[half:]

        # Build a text representation of old messages for summarisation
        summary_prompt = (
            "Summarise this conversation history in 2-3 sentences, "
            "preserving key facts and decisions:\n\n" +
            "\n".join(
                f"{m['role'].upper()}: "
                f"{m['content'] if isinstance(m['content'], str) else '[tool interaction]'}"
                for m in old_msgs
            )
        )

        # Use the cheap mini model — summarisation doesn't need gpt-4o
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are a helpful summariser."},
                {"role": "user",   "content": summary_prompt}
            ]
        )
        summary = resp.choices[0].message.content

        # Persist summary to SemanticMemory for future sessions
        if _semantic_memory:
            _semantic_memory.save_summary(summary)

        # Inject synthetic summary at the start of the remaining buffer
        self.messages.insert(0, {
            "role": "user",
            "content": f"[Previous conversation summary: {summary}]"
        })
        self.messages.insert(1, {
            "role": "assistant",
            "content": "Understood. I have the context from our earlier conversation."
        })
        print(f"  [memory] Compressed {half} messages → summary")


# ══════════════════════════════════════════════════════════
# SECTION 3 — AGENT (ReAct loop with streaming)
#
# The core engine. One call to chat() = one user turn.
# Internally runs a loop until the LLM gives a final text answer.
#
# STREAMING:
#   stream=True returns chunks instead of a full response.
#   We accumulate:
#     - content      → streamed text, printed character by character
#     - tool_calls   → assembled from delta fragments (id + name + args)
#   After the stream ends, we check what we got and react.
#
# REACT LOOP (per iteration):
#   LLM call (streaming)
#     ↓
#   tool_calls present?
#     YES → execute each tool → append results → loop again
#     NO  → content is the final answer → return
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools and memory.

Available tools:
- calculator:   evaluate math expressions
- get_datetime: get current date and time
- remember:     save a fact to long-term memory
- recall:       search long-term memory by meaning
- web_search:   search Google for current information
- http_get:     fetch the full content of a URL

You have two types of memory:
- Short-term: the current conversation (already in your context)
- Long-term: facts saved across sessions via vector search (use recall/remember)

When using web_search, you can follow up with http_get on a specific URL to read full content.
Be concise and direct. After using a tool, interpret the result naturally."""


class Agent:
    MAX_ITERATIONS = 10  # safety cap on the ReAct loop

    def __init__(self):
        global _semantic_memory

        self.client = OpenAI()                # reads OPENAI_API_KEY from env
        self.short_term = ShortTermMemory()
        self.long_term = SemanticMemory()     # vector-backed SQLite store
        _semantic_memory = self.long_term     # wire up tool functions
        self.tool_schemas = [t["schema"] for t in TOOLS.values()]

    def _build_system_prompt(self, user_input: str) -> str:
        """
        Memory injection — runs before every LLM call.

        1. Fetch recent cross-session summaries from SemanticMemory
        2. Fetch facts semantically similar to the current user input
        3. Prepend both to the system prompt

        This is the exact mechanism used by Mem0, AgentCore, and LangMem.
        The LLM doesn't "remember" — we inject the memory into its context.
        """
        prompt = SYSTEM_PROMPT

        # Past session summaries (chronological context)
        summaries = self.long_term.get_recent_summaries(limit=2)
        if summaries:
            prompt += "\n\nContext from previous sessions:\n"
            prompt += "\n".join(f"- {s}" for s in summaries)

        # Facts semantically related to this specific query
        relevant_facts = self.long_term.search(user_input, limit=4)
        if relevant_facts:
            prompt += "\n\nRelevant facts from memory:\n"
            prompt += "\n".join(f"- [{k}] {v}" for k, v in relevant_facts)

        return prompt

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Look up the tool function by name and call it with the parsed args."""
        if tool_name not in TOOLS:
            return f"Unknown tool: {tool_name}"
        try:
            return TOOLS[tool_name]["fn"](**tool_input)
        except Exception as e:
            return f"Tool error: {e}"

    def _react_loop(self, system: str) -> str:
        """
        Streaming ReAct loop.

        STREAMING TOOL CALL ASSEMBLY:
          OpenAI streams tool calls as fragments across multiple chunks.
          Each chunk has delta.tool_calls[i] where i is the index.
          We accumulate fragments into tool_calls_map[index]:
            - id:        first chunk that has it
            - name:      concatenated (usually comes in one chunk)
            - arguments: concatenated across many chunks (JSON string)
          After the stream ends, tool_calls_map.values() has the full calls.

        IMPORTANT — why we use messages.append() for tool-call assistant messages:
          short_term.add(role, content) builds {"role": role, "content": content}.
          If content is a dict with tool_calls, it gets NESTED under "content"
          and OpenAI returns 400: "expected string, got object".
          So tool-call assistant messages go directly via messages.append().

        MESSAGE FLOW:
          {role: system,     content: "..."}
          {role: user,       content: "What is 2^16?"}
          {role: assistant,  content: None, tool_calls: [{id, function}]}  ← append directly
          {role: tool,       tool_call_id: id, content: "65536"}           ← append directly
          {role: assistant,  content: "2^16 is 65536."}                    ← via add()
        """
        for iteration in range(self.MAX_ITERATIONS):
            print(f"  [loop] iteration {iteration + 1}")

            # System prompt is prepended fresh each iteration
            messages = [{"role": "system", "content": system}] + self.short_term.get_all()

            # stream=True: response is a generator of chunks
            response = self.client.chat.completions.create(
                model="gpt-4o",
                stream=True,
                max_tokens=1024,
                tools=self.tool_schemas,
                messages=messages
            )

            # Accumulators for this stream
            content = ""
            tool_calls_map: dict[int, dict] = {}  # index → {id, name, arguments}
            finish_reason = None

            for chunk in response:
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta

                # Stream text content character by character
                if delta.content:
                    content += delta.content
                    print(delta.content, end="", flush=True)

                # Accumulate tool call fragments by index
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        entry = tool_calls_map.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            entry["id"] = tc.id
                        if tc.function.name:
                            entry["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["arguments"] += tc.function.arguments

            if content:
                print()  # newline after streamed output ends

            tool_calls = list(tool_calls_map.values())

            # ── Case 1: LLM wants to call one or more tools ──
            if tool_calls:
                # Append assistant message directly — NOT via add()
                # (see docstring above for why)
                self.short_term.messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"]
                            }
                        }
                        for tc in tool_calls
                    ]
                })

                # Execute each tool and append its result
                for tc in tool_calls:
                    tool_input = json.loads(tc["arguments"])
                    print(f"  [tool] calling {tc['name']}({tool_input})")
                    result = self._execute_tool(tc["name"], tool_input)
                    print(f"  [tool] result: {result[:120]}")

                    # Tool results use role="tool" (OpenAI convention)
                    # tool_call_id pairs this result with the call above
                    self.short_term.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result
                    })
                continue  # loop back to call LLM with the tool results

            # ── Case 2: LLM gave a final text answer ──
            if content:
                self.short_term.add("assistant", content)
                return content

            return "Agent finished with no output."

        return "Reached max iterations without a final answer."

    def chat(self, user_input: str) -> str:
        """
        Main entry point for a single user turn.
          1. Build system prompt with injected memory
          2. Add user message to short-term buffer
          3. Run the ReAct loop until final answer
          4. Compress buffer if it's grown too large
        """
        system = self._build_system_prompt(user_input)
        self.short_term.add("user", user_input)
        answer = self._react_loop(system)
        self.short_term.compress_if_needed(self.client, system)
        return answer


# ══════════════════════════════════════════════════════════
# SECTION 4 — PLANNER
#
# Makes ONE LLM call to decompose a goal into a list of Steps.
#
# What it sends:  goal + list of available tool names
# What it gets:   JSON with a steps array (id, description,
#                 tool_hint, depends_on)
#
# The depends_on field creates a dependency graph so the executor
# can skip steps whose prerequisites failed, and pass results
# from earlier steps as context to later ones.
#
# Real-world equivalent:
#   LangGraph plan-and-execute first pass
#   OpenAI o1's internal pre-reasoning
# ══════════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = """You are a planning assistant. Break the user's goal into
a clear sequence of steps that can each be handled by exactly one tool call or one
reasoning step.

Available tools: {tools}

Rules:
- Each step must be small, specific, and achievable with one tool call
- If a step needs the result of a prior step, list that step's id in depends_on
- tool_hint must be exactly one of the available tool names, or "none"
- Aim for 2-5 steps total. Don't over-split simple tasks.
- Return ONLY valid JSON — no explanation, no markdown fences.

Required JSON format:
{{
  "steps": [
    {{"id": 1, "description": "...", "tool_hint": "...", "depends_on": []}},
    {{"id": 2, "description": "...", "tool_hint": "...", "depends_on": [1]}}
  ]
}}"""


@dataclass
class Step:
    """
    One step in an execution plan.

    Lifecycle: pending → running → done | failed
    result is populated by PlanExecutor after the step runs.
    depends_on lists step ids that must be done before this step starts.
    """
    id: int
    description: str
    tool_hint: str           # which tool the planner expects this step to use
    depends_on: list         # ids of steps that must complete first
    status: str = "pending"  # pending | running | done | failed
    result: str = ""         # filled in by PlanExecutor


class Planner:
    def __init__(self, client: OpenAI):
        self.client = client

    def create_plan(self, goal: str) -> list[Step]:
        """
        Call the LLM once to decompose goal into a list of Steps.
        Falls back to a single-step plan if JSON parsing fails.
        """
        tool_names = ", ".join(TOOLS.keys())
        system = PLANNER_SYSTEM_PROMPT.format(tools=tool_names)

        print(f"\n  [planner] decomposing goal: '{goal[:60]}'")

        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=600,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Goal: {goal}"}
            ]
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if the LLM wraps output despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw.strip())
            steps = [Step(**s) for s in data["steps"]]
            print(f"  [planner] {len(steps)} steps:")
            for s in steps:
                deps = f" (needs steps {s.depends_on})" if s.depends_on else ""
                print(f"    step {s.id}: {s.description[:55]}{deps}  [tool: {s.tool_hint}]")
            return steps
        except Exception as e:
            # Graceful fallback — treat whole goal as one step
            print(f"  [planner] JSON parse failed ({e}), falling back to single step")
            return [Step(id=1, description=goal, tool_hint="none", depends_on=[])]


# ══════════════════════════════════════════════════════════
# SECTION 5 — PLAN EXECUTOR
#
# Runs each step of a plan in order, then synthesises a
# clean final answer from all step results.
#
# HOW EACH STEP RUNS:
#   - Creates a fresh Agent with a clean ShortTermMemory
#     (avoids tool message contamination between steps)
#   - Builds a focused task prompt:
#       "Your task: <description>
#        Results from prior steps: <context from depends_on>
#        Hint: use the '<tool_hint>' tool"
#   - Calls agent.chat(task) — full ReAct loop runs per step
#   - Stores result in step.result and in completed_results dict
#
# DEPENDENCY HANDLING:
#   - If any step in depends_on failed → skip this step
#   - completed_results[step_id] is passed as context to dependents
#
# SYNTHESIS:
#   - One final LLM call reads all step results and writes a clean answer
#   - This is the "reduce" phase after all the "map" steps are done
#
# Real-world equivalents:
#   CrewAI sequential task runner
#   LangGraph plan-and-execute executor node
#   AutoGPT task queue
# ══════════════════════════════════════════════════════════

class PlanExecutor:
    def __init__(self, client: OpenAI):
        self.client = client
        self.planner = Planner(client)

    def _run_step(self, step: Step, completed_results: dict[int, str]) -> str:
        """
        Run a single plan step as a focused mini-agent call.

        Uses a fresh Agent per step to avoid short-term buffer contamination.
        The task prompt is enriched with results from dependency steps.
        """
        # Start with the step description as the task
        task = step.description

        # Inject results from steps this one depends on
        if step.depends_on and completed_results:
            prior_context = "\n".join(
                f"- Step {dep}: {completed_results[dep]}"
                for dep in step.depends_on
                if dep in completed_results
            )
            if prior_context:
                task += f"\n\nResults from prior steps you can use:\n{prior_context}"

        # Give the agent a nudge toward the right tool
        if step.tool_hint and step.tool_hint != "none":
            task += f"\n\nHint: use the '{step.tool_hint}' tool for this step."

        # Fresh agent per step — clean short-term buffer, shared long-term memory
        mini_agent = Agent()
        mini_agent.short_term = ShortTermMemory()

        print(f"\n  [executor] step {step.id}: {step.description[:55]}...")
        result = mini_agent.chat(task)
        print(f"  [executor] step {step.id} done: {result[:100]}")
        return result

    def _synthesise(self, goal: str, steps: list[Step]) -> str:
        """
        Final LLM call: combine all step results into one clean answer.

        This is the "reduce" phase. The LLM sees:
          - Original goal
          - Each completed step's description + result
        And writes a direct, concise final answer.
        """
        completed = [s for s in steps if s.status == "done"]
        if not completed:
            return "No steps completed successfully."

        results_block = "\n".join(
            f"Step {s.id} ({s.description}):\n  {s.result}"
            for s in completed
        )

        resp = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": "Synthesise research results into a clear, concise final answer. Be direct."
                },
                {
                    "role": "user",
                    "content": f"Original goal: {goal}\n\nStep results:\n{results_block}\n\nWrite a clear final answer."
                }
            ]
        )
        return resp.choices[0].message.content

    def run(self, goal: str) -> str:
        """
        Full plan-and-execute flow:
          1. Planner decomposes goal → list of Steps
          2. Execute each step in order, respecting depends_on
          3. Synthesise a clean answer from all results
        """
        steps = self.planner.create_plan(goal)
        completed_results: dict[int, str] = {}

        print(f"\n  [executor] running {len(steps)} step(s)...")

        for step in steps:
            # Check if any dependency step failed — skip if so
            failed_deps = [
                dep for dep in step.depends_on
                if any(s.id == dep and s.status == "failed" for s in steps)
            ]
            if failed_deps:
                step.status = "failed"
                step.result = f"Skipped — dependency steps {failed_deps} failed"
                print(f"  [executor] step {step.id} SKIPPED (failed deps: {failed_deps})")
                continue

            step.status = "running"
            try:
                result = self._run_step(step, completed_results)
                step.status = "done"
                step.result = result
                completed_results[step.id] = result
            except Exception as e:
                step.status = "failed"
                step.result = f"Error: {e}"
                print(f"  [executor] step {step.id} FAILED: {e}")

        # Print execution summary
        print("\n  [executor] plan complete:")
        icons = {"done": "✓", "failed": "✗", "pending": "?", "running": "→"}
        for s in steps:
            print(f"    {icons.get(s.status,'?')} [{s.status:7}] step {s.id}: {s.description[:50]}")

        # Synthesise final answer from all completed step results
        print("\n  [executor] synthesising final answer...")
        return self._synthesise(goal, steps)


# ══════════════════════════════════════════════════════════
# SECTION 6 — MAIN (interactive CLI)
#
# Two modes:
#   Normal input  → agent.chat()         (single ReAct turn)
#   "plan: <goal>" → executor.run(goal)  (multi-step plan)
#
# Commands:
#   memory   — show short-term buffer + semantic DB contents
#   quit     — exit
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  AI Agent — streaming | tools | memory | planning")
    print("  Commands: 'plan: <goal>'  'memory'  'quit'")
    print("=" * 60)

    # Shared client and agent
    client = OpenAI()
    agent = Agent()
    executor = PlanExecutor(client)

    # Clear buffer on start (in case of leftover state)
    agent.short_term.messages = []

    print("\n[Ready. Type a message or 'plan: <your multi-step goal>']\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        # ── quit ──
        if user_input.lower() == "quit":
            break

        # ── memory inspector ──
        if user_input.lower() == "memory":
            print("\n--- Short-term buffer ---")
            for m in agent.short_term.get_all():
                content = m["content"]
                display = content[:80] if isinstance(content, str) else "[structured]"
                print(f"  {m['role']}: {display}")

            print("\n--- Long-term facts (semantic DB) ---")
            conn = sqlite3.connect("agent_semantic_memory.db")
            rows = conn.execute(
                "SELECT key, value FROM facts ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            for k, v in rows:
                print(f"  [{k}] {v}")
            conn.close()
            print()
            continue

        # ── plan mode ──
        if user_input.lower().startswith("plan:"):
            goal = user_input[5:].strip()
            if not goal:
                print("Usage: plan: <your multi-step goal>")
                continue
            print(f"\n[Plan mode] Goal: {goal}\n")
            answer = executor.run(goal)
            print(f"\n{'─'*60}")
            print(f"Final answer: {answer}")
            print('─'*60)
            continue

        # ── normal chat mode ──
        response = agent.chat(user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()