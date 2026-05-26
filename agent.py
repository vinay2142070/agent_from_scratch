"""
==============================================================
  AI AGENT FROM SCRATCH
  OpenAI gpt-4o | Streaming | MCP Tools | Semantic Memory | Planning
==============================================================

WHAT'S IN THIS FILE:
  Section 1 — MCPClient  (discovers + calls tools via MCP protocol)
  Section 2 — ShortTermMemory  (in-RAM per-session buffer)
  Section 3 — Agent            (ReAct loop, streaming, memory injection)
  Section 4 — Planner          (breaks a goal into ordered steps)
  Section 5 — PlanExecutor     (runs each step, synthesises final answer)
  Section 6 — main()           (interactive CLI, mode selector)

HOW TOOLS WORK NOW (MCP):
  Before: tool functions hardcoded in this file, called directly
  After:  tools live in mcp_server.py, called over JSON-RPC via subprocess

  On startup:
    Agent.__init__()
      → spawns mcp_server.py as subprocess
      → calls tools/list   → gets schemas → injects into LLM
  On each tool call:
    _execute_tool(name, args)
      → calls tools/call on the MCP server
      → gets result back as text
      → appends to message history as before

  Everything else (streaming, memory, planning) is identical.

HOW THE TWO MODES WORK:
  ┌─────────────────────────────────────────────────────────┐
  │ CHAT mode  (default)                                    │
  │   User message → memory inject → ReAct loop → reply    │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │ PLAN mode  (prefix input with "plan:")                  │
  │   Goal → Planner → PlanExecutor → Synthesiser          │
  └─────────────────────────────────────────────────────────┘

REAL-WORLD EQUIVALENTS:
  MCPClient       ≈ Claude Desktop MCP connector / LangChain MCP adapter
  ReAct loop      ≈ LangGraph create_react_agent / Bedrock agent runtime
  ShortTermMemory ≈ LangGraph MemorySaver / Bedrock AgentCoreMemorySaver
  SemanticMemory  ≈ pgvector table / Pinecone collection / Mem0 user-scope
  Planner         ≈ LangGraph plan-and-execute first pass
  PlanExecutor    ≈ CrewAI sequential task runner / AutoGPT task queue

SETUP:
  pip install openai python-dotenv requests
  .env: OPENAI_API_KEY=sk-...   SERPER_API_KEY=...  (serper optional)

FILES REQUIRED IN SAME FOLDER:
  mcp_server.py      — tool server (spawned automatically)
  semantic_memory.py — vector memory (used by mcp_server.py)

COMMANDS (interactive mode):
  plan: <goal>   — run multi-step planner
  memory         — inspect short-term buffer + semantic DB
  quit           — exit
"""

import sys
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
from semantic_memory import SemanticMemory
from mcp_client import MCPClientHTTP

load_dotenv()


# ══════════════════════════════════════════════════════════
# TOKEN TRACKER
#
# A single _tracker instance shared across the whole file.
# Every LLM call logs its usage via _tracker.log().
# A session summary prints when the agent exits.
#
# STREAMING CALLS (_react_loop):
#   Pass stream_options={"include_usage": True} to the API.
#   Usage only arrives on the LAST chunk — all other chunks
#   have chunk.usage = None. That final chunk also has an
#   empty choices list, so guard with: if not chunk.choices: continue
#
# NON-STREAMING CALLS (compress, planner, synthesise):
#   Usage is on response.usage directly — no special handling needed.
# ══════════════════════════════════════════════════════════

class TokenTracker:
    """Accumulates token usage across every LLM call in a session."""

    def __init__(self):
        self.input_total  = 0
        self.output_total = 0
        self.call_count   = 0

    def log(self, label: str, input_tokens: int, output_tokens: int):
        """Print per-call breakdown and update running session totals."""
        self.input_total  += input_tokens
        self.output_total += output_tokens
        self.call_count   += 1
        call_total    = input_tokens + output_tokens
        session_total = self.input_total + self.output_total
        print(
            f"  [tokens] {label:<22} "
            f"in: {input_tokens:>6,}  "
            f"out: {output_tokens:>5,}  "
            f"call: {call_total:>6,}  "
            f"│  session: {session_total:,}"
        )

    def summary(self):
        """Print cumulative totals for the entire session."""
        print(
            f"\n  [tokens] session summary ── "
            f"calls: {self.call_count}  "
            f"in: {self.input_total:,}  "
            f"out: {self.output_total:,}  "
            f"total: {self.input_total + self.output_total:,}"
        )


# Module-level singleton — imported nowhere, used everywhere in this file
_tracker = TokenTracker()


# ══════════════════════════════════════════════════════════
# SECTION 1 — MCP CLIENT
#
# Replaces the old hardcoded TOOLS dict and tool_* functions.
#
# What changed vs the old approach:
#   OLD: TOOLS = {"calculator": {"fn": tool_calculator, "schema": {...}}}
#        _execute_tool → looks up fn in TOOLS dict → calls it directly
#
#   NEW: MCPClient.list_tools() → asks server → returns schemas dynamically
#        MCPClient.call_tool()  → sends JSON-RPC to server → gets result
#
# The ReAct loop doesn't change at all.
# Only _execute_tool() changes: local fn call → mcp.call_tool()
#
# WHY THIS IS BETTER:
#   - Add a new tool: edit mcp_server.py only, agent.py untouched
#   - Use the same server from multiple agents simultaneously
#   - Server can be in any language (Node, Go, Rust) — protocol is JSON
#   - Remote tools: swap stdio transport for HTTP+SSE, nothing else changes
#
# MCP PROTOCOL (3 messages, that's the whole spec):
#   initialize   → handshake, exchange capabilities
#   tools/list   → server returns available tool schemas
#   tools/call   → agent calls a tool, server returns result
# ══════════════════════════════════════════════════════════

class MCPClient:
    """
    Manages a connection to one MCP server subprocess.

    Communication: JSON-RPC 2.0 over stdin/stdout (stdio transport).
    The server process is spawned on __init__ and terminated on close().

    To connect to multiple servers: create multiple MCPClient instances
    and merge their tool lists — each agent in production does this.
    """

    def __init__(self, server_script: str):
        # Spawn the server as a child process
        # line-buffered (bufsize=1) so each JSON line is flushed immediately
        self.process = subprocess.Popen(
            [sys.executable, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._next_id = 1
        self._handshake()

    def _send(self, method: str, params: dict = None) -> dict:
        """
        Send one JSON-RPC request, read and return one response.
        Each request gets a unique incrementing id so responses
        can be matched to requests (important when async).
        """
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params or {}
        }
        self._next_id += 1

        # Write to server stdin
        self.process.stdin.write(json.dumps(msg) + "\n")
        self.process.stdin.flush()

        # Read response from server stdout (blocking)
        line = self.process.stdout.readline()
        return json.loads(line.strip())

    def _handshake(self):
        """
        MCP initialize handshake — must happen before any other call.
        Client announces itself, server replies with its capabilities.
        Then client sends a notifications/initialized (no response needed).
        """
        resp = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "agent", "version": "1.0"},
            "capabilities": {}
        })
        # Notification — no id, no response expected
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        self.process.stdin.write(json.dumps(notif) + "\n")
        self.process.stdin.flush()
        server_name = resp["result"]["serverInfo"]["name"]
        print(f"  [mcp] connected to '{server_name}'")

    def list_tools(self) -> list[dict]:
        """
        Fetch available tools from the server.

        Returns MCP tool format:
          [{"name": "...", "description": "...", "inputSchema": {...}}, ...]

        We convert these to OpenAI format in to_openai_schemas() before
        passing to the LLM — inputSchema → parameters, wrapped in
        {"type": "function", "function": {...}}.
        """
        resp = self._send("tools/list")
        tools = resp["result"]["tools"]
        print(f"  [mcp] {len(tools)} tools: {[t['name'] for t in tools]}")
        return tools

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Execute a tool on the server and return the result as a string.

        The server runs the actual Python function and sends back:
          {"result": {"content": [{"type": "text", "text": "..."}]}}

        We extract just the text — same format as the old tool functions returned.
        This string gets appended to the message history as a tool result.
        """
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            return f"MCP error: {resp['error']['message']}"
        content = resp["result"]["content"]
        return content[0]["text"] if content else "(no result)"

    @staticmethod
    def to_openai_schemas(mcp_tools: list[dict]) -> list[dict]:
        """
        Convert MCP tool format → OpenAI tool format.

        MCP:    {"name": "...", "description": "...", "inputSchema": {...}}
        OpenAI: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

        The only difference is the key name: inputSchema → parameters.
        The JSON Schema object itself is identical.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"]   # rename only — content unchanged
                }
            }
            for t in mcp_tools
        ]

    def close(self):
        """Terminate the server subprocess cleanly."""
        self.process.terminate()


# ══════════════════════════════════════════════════════════
# SECTION 2 — SHORT-TERM MEMORY
#
# Holds the raw message list for one session in RAM.
# Passed to the LLM on every call as the conversation history.
#
# When the buffer grows beyond MAX_MESSAGES, it summarises
# the oldest half and replaces them with a synthetic summary
# message — "sliding window + summary" compression, same as
# LangChain's ConversationSummaryMemory.
#
# The summary is also saved to SemanticMemory so it persists
# across sessions.
# ══════════════════════════════════════════════════════════

class ShortTermMemory:
    MAX_MESSAGES = 20

    def __init__(self):
        self.messages: list[dict] = []

    def add(self, role: str, content):
        """
        Append a simple message.
        NOTE: for tool-call assistant messages, use messages.append()
        directly — passing a dict as content nests it under "content"
        and causes OpenAI 400 errors.
        """
        self.messages.append({"role": role, "content": content})

    def get_all(self) -> list[dict]:
        return self.messages

    def compress_if_needed(self, llm_client: OpenAI, system_prompt: str):
        """
        Compress old messages when buffer exceeds MAX_MESSAGES.
        Summarises the oldest half using gpt-4o-mini (cheap).
        Saves summary to SemanticMemory for cross-session recall.
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
                f"{m['role'].upper()}: "
                f"{m['content'] if isinstance(m['content'], str) else '[tool interaction]'}"
                for m in old_msgs
            )
        )
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are a helpful summariser."},
                {"role": "user",   "content": summary_prompt}
            ]
        )
        summary = resp.choices[0].message.content
        if resp.usage:
            _tracker.log("memory compress", resp.usage.prompt_tokens, resp.usage.completion_tokens)

        # Persist to SemanticMemory for future sessions
        if _long_term_memory:
            _long_term_memory.save_summary(summary)

        self.messages.insert(0, {
            "role": "user",
            "content": f"[Previous conversation summary: {summary}]"
        })
        self.messages.insert(1, {
            "role": "assistant",
            "content": "Understood. I have the context from our earlier conversation."
        })
        print(f"  [memory] Compressed {half} messages → summary")


# Module-level SemanticMemory reference for ShortTermMemory.compress_if_needed
# Assigned in Agent.__init__()
_long_term_memory: SemanticMemory = None


# ══════════════════════════════════════════════════════════
# SECTION 3 — AGENT (ReAct loop with streaming)
#
# Core change from the old version:
#   __init__:       spawns MCPClient, loads tool schemas from server
#   _execute_tool:  calls mcp.call_tool() instead of local fn lookup
#   Everything else is identical to before.
#
# STREAMING TOOL CALL ASSEMBLY:
#   OpenAI streams tool calls as fragments. We accumulate by index:
#     tool_calls_map[index] = {id, name, arguments (JSON string)}
#   After stream ends, tool_calls_map.values() has complete calls.
#
# MESSAGE FLOW (unchanged):
#   {role: system,     content: "..."}
#   {role: user,       content: "user message"}
#   {role: assistant,  content: None, tool_calls: [...]}  ← append directly
#   {role: tool,       tool_call_id: id, content: "result"} ← append directly
#   {role: assistant,  content: "final answer"}            ← via add()
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools and memory.

Available tools (loaded from MCP server):
- calculator:   evaluate math expressions
- get_datetime: get current date and time
- web_search:   search Google for current information
- http_get:     fetch the full content of a URL
- remember:     save a fact to long-term memory
- recall:       search long-term memory by meaning

You have two types of memory:
- Short-term: the current conversation (already in context)
- Long-term: facts saved across sessions via vector search

When using web_search, follow up with http_get on a specific URL to read full content.
Be concise and direct. After using a tool, interpret the result naturally."""


class Agent:
    MAX_ITERATIONS = 10

    def __init__(self):
        global _long_term_memory

        self.client = OpenAI()
        self.short_term = ShortTermMemory()
        self.long_term = SemanticMemory()
        _long_term_memory = self.long_term

        # ── MCP setup ──────────────────────────────────────
        # Spawn the MCP server subprocess
        # self.mcp = MCPClient("mcp_server.py")  # stdio
        self.mcp = MCPClientHTTP("http://localhost:8000")
        # Discover tools from server, convert to OpenAI format
        # This replaces the old hardcoded TOOLS dict entirely
        mcp_tools = self.mcp.list_tools()
        self.tool_schemas = MCPClient.to_openai_schemas(mcp_tools)
        # ───────────────────────────────────────────────────

    def _build_system_prompt(self, user_input: str) -> str:
        """
        Memory injection — runs before every LLM call.
        Fetches semantically relevant facts and prepends them to the
        system prompt. Same mechanism used by Mem0, AgentCore, LangMem.
        """
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
        """
        Execute a tool via MCP.

        OLD: fn = TOOLS[tool_name]["fn"]; return fn(**tool_input)
        NEW: return self.mcp.call_tool(tool_name, tool_input)

        That's the only change. The result is the same string either way.
        The ReAct loop above this call doesn't know or care about the difference.
        """
        print(f"  [mcp] → {tool_name}({tool_input})")
        result = self.mcp.call_tool(tool_name, tool_input)
        print(f"  [mcp] ← {result[:120]}")
        return result

    def _react_loop(self, system: str) -> str:
        """
        Streaming ReAct loop — unchanged from the non-MCP version.

        The only thing that changed in the entire loop is one line:
          self._execute_tool() now routes through MCP instead of TOOLS dict.
        """
        for iteration in range(self.MAX_ITERATIONS):
            print(f"  [loop] iteration {iteration + 1}")

            messages = [{"role": "system", "content": system}] + self.short_term.get_all()

            response = self.client.chat.completions.create(
                model="gpt-4o",
                stream=True,
                max_tokens=1024,
                tools=self.tool_schemas,
                messages=messages,
                # stream_options required to get usage on streaming calls.
                # Without this, chunk.usage is always None.
                stream_options={"include_usage": True}
            )

            content = ""
            tool_calls_map: dict[int, dict] = {}
            finish_reason = None
            usage = None  # populated by the final chunk only

            for chunk in response:
                # The last chunk carries usage and has an empty choices list.
                # Guard here prevents IndexError on chunk.choices[0].
                if chunk.usage is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta

                if delta.content:
                    content += delta.content
                    print(delta.content, end="", flush=True)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        entry = tool_calls_map.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:            entry["id"] = tc.id
                        if tc.function.name: entry["name"] += tc.function.name
                        if tc.function.arguments: entry["arguments"] += tc.function.arguments

            if content:
                print()

            # Log token usage for this ReAct iteration
            if usage:
                _tracker.log(
                    f"react iter {iteration + 1}",
                    usage.prompt_tokens,
                    usage.completion_tokens
                )

            tool_calls = list(tool_calls_map.values())

            # ── Case 1: LLM wants to call tools ──
            if tool_calls:
                # Append assistant message directly — NOT via add()
                # (add() nests content under "content" key → OpenAI 400)
                self.short_term.messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]}
                        }
                        for tc in tool_calls
                    ]
                })

                for tc in tool_calls:
                    tool_input = json.loads(tc["arguments"])
                    result = self._execute_tool(tc["name"], tool_input)

                    # Tool result: role="tool", linked by tool_call_id
                    self.short_term.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result
                    })
                continue

            # ── Case 2: Final text answer ──
            if content:
                self.short_term.add("assistant", content)
                return content

            return "Agent finished with no output."

        return "Reached max iterations without a final answer."

    def chat(self, user_input: str) -> str:
        system = self._build_system_prompt(user_input)
        self.short_term.add("user", user_input)
        answer = self._react_loop(system)
        self.short_term.compress_if_needed(self.client, system)
        return answer

    def close(self):
        """Terminate the MCP server subprocess."""
        self.mcp.close()


# ══════════════════════════════════════════════════════════
# SECTION 4 — PLANNER
#
# Unchanged — makes one LLM call to decompose a goal into Steps.
# Uses the same tool names (now coming from MCP) in the prompt.
# ══════════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = """You are a planning assistant. Break the user's goal into
a clear sequence of steps that can each be handled by exactly one tool call.

Available tools: {tools}

Rules:
- Each step must be small, specific, and achievable with one tool call
- If a step needs the result of a prior step, list that step's id in depends_on
- tool_hint must be exactly one of the available tool names, or "none"
- Aim for 2-5 steps total
- Return ONLY valid JSON — no explanation, no markdown fences

Required JSON format:
{{
  "steps": [
    {{"id": 1, "description": "...", "tool_hint": "...", "depends_on": []}},
    {{"id": 2, "description": "...", "tool_hint": "...", "depends_on": [1]}}
  ]
}}"""


@dataclass
class Step:
    id: int
    description: str
    tool_hint: str
    depends_on: list
    status: str = "pending"
    result: str = ""


class Planner:
    def __init__(self, client: OpenAI, tool_names: list[str]):
        self.client = client
        self.tool_names = tool_names  # from MCP discovery, not hardcoded

    def create_plan(self, goal: str) -> list[Step]:
        system = PLANNER_SYSTEM_PROMPT.format(tools=", ".join(self.tool_names))
        print(f"\n  [planner] decomposing: '{goal[:60]}'")

        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=600,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Goal: {goal}"}
            ]
        )

        raw = response.choices[0].message.content.strip()
        if response.usage:
            _tracker.log("planner", response.usage.prompt_tokens, response.usage.completion_tokens)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw.strip())
            steps = [Step(**s) for s in data["steps"]]
            print(f"  [planner] {len(steps)} steps:")
            for s in steps:
                deps = f" (needs {s.depends_on})" if s.depends_on else ""
                print(f"    step {s.id}: {s.description[:55]}{deps}  [tool: {s.tool_hint}]")
            return steps
        except Exception as e:
            print(f"  [planner] parse failed ({e}), falling back to single step")
            return [Step(id=1, description=goal, tool_hint="none", depends_on=[])]


# ══════════════════════════════════════════════════════════
# SECTION 5 — PLAN EXECUTOR
#
# Unchanged in logic — runs each step via a fresh Agent instance.
# The fresh Agent automatically connects to MCP and gets the same tools.
# ══════════════════════════════════════════════════════════

class PlanExecutor:
    def __init__(self, client: OpenAI, tool_names: list[str]):
        self.client = client
        self.planner = Planner(client, tool_names)

    def _run_step(self, step: Step, completed_results: dict[int, str]) -> str:
        task = step.description

        if step.depends_on and completed_results:
            prior_context = "\n".join(
                f"- Step {dep}: {completed_results[dep]}"
                for dep in step.depends_on if dep in completed_results
            )
            if prior_context:
                task += f"\n\nResults from prior steps:\n{prior_context}"

        if step.tool_hint and step.tool_hint != "none":
            task += f"\n\nHint: use the '{step.tool_hint}' tool for this step."

        # Fresh Agent per step — clean buffer, shared long-term memory
        # Each Agent spawns its own MCP connection to the server
        mini_agent = Agent()
        mini_agent.short_term = ShortTermMemory()

        print(f"\n  [executor] step {step.id}: {step.description[:55]}...")
        result = mini_agent.chat(task)
        mini_agent.close()
        print(f"  [executor] step {step.id} done: {result[:100]}")
        return result

    def _synthesise(self, goal: str, steps: list[Step]) -> str:
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
                {"role": "system", "content": "Synthesise results into a clear, concise answer."},
                {"role": "user",   "content": f"Goal: {goal}\n\nResults:\n{results_block}\n\nWrite a clear final answer."}
            ]
        )
        if resp.usage:
            _tracker.log("synthesise", resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content

    def run(self, goal: str) -> str:
        steps = self.planner.create_plan(goal)
        completed_results: dict[int, str] = {}

        print(f"\n  [executor] running {len(steps)} step(s)...")

        for step in steps:
            failed_deps = [
                dep for dep in step.depends_on
                if any(s.id == dep and s.status == "failed" for s in steps)
            ]
            if failed_deps:
                step.status = "failed"
                step.result = f"Skipped — deps {failed_deps} failed"
                print(f"  [executor] step {step.id} SKIPPED")
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

        print("\n  [executor] plan complete:")
        icons = {"done": "✓", "failed": "✗", "pending": "?", "running": "→"}
        for s in steps:
            print(f"    {icons.get(s.status,'?')} [{s.status:7}] step {s.id}: {s.description[:50]}")

        print("\n  [executor] synthesising...")
        return self._synthesise(goal, steps)


# ══════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  AI Agent — MCP tools | streaming | memory | planning")
    print("  Commands: 'plan: <goal>'  'memory'  'quit'")
    print("=" * 60)

    client = OpenAI()

    # Agent connects to MCP server on init
    agent = Agent()
    agent.short_term.messages = []

    # Pass discovered tool names to planner so it knows what to suggest
    tool_names = [t["function"]["name"] for t in agent.tool_schemas]
    executor = PlanExecutor(client, tool_names)

    print("\n[Ready. Type a message or 'plan: <your multi-step goal>']\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

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

            if user_input.lower().startswith("plan:"):
                goal = user_input[5:].strip()
                if not goal:
                    print("Usage: plan: <your multi-step goal>")
                    continue
                print(f"\n[Plan mode] Goal: {goal}\n")
                answer = executor.run(goal)
                print(f"\n{'─'*60}\nFinal answer: {answer}\n{'─'*60}")
                continue

            response = agent.chat(user_input)
            print(f"Agent: {response}\n")

    finally:
        # Always shut down the MCP server cleanly
        agent.close()
        # Print cumulative token usage for the whole session
        _tracker.summary()


if __name__ == "__main__":
    main()