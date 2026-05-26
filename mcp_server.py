"""
MCP SERVER — all tools live here
==================================
Handles: calculator, get_datetime, web_search, http_get, remember, recall

Speaks MCP protocol over stdio (JSON-RPC 2.0).
Spawned as a subprocess by agent.py on startup.

Transport: stdin → parse → handle → stdout
"""

import sys
import os
import json
import math
import sqlite3
import datetime
import requests
from semantic_memory import SemanticMemory

# SemanticMemory instance — shared across remember/recall tool calls
_mem = SemanticMemory()


def send(obj: dict):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# ── Tool implementations ───────────────────────────────────

def run_calculator(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


def run_get_datetime(**_) -> str:
    return f"UTC: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"


def run_web_search(query: str, num_results: int = 5) -> str:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not set"
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


def run_http_get(url: str, headers: dict = None) -> str:
    try:
        resp = requests.get(url, headers=headers or {}, timeout=10)
        resp.raise_for_status()
        return resp.text[:4000] or "(empty response)"
    except Exception as e:
        return f"HTTP error: {e}"


def run_remember(key: str, value: str) -> str:
    _mem.save(key, value)
    return f"Saved: {key} = {value}"


def run_recall(query: str) -> str:
    results = _mem.search(query)
    if not results:
        return "Nothing found in long-term memory."
    return "\n".join(f"- [{k}] {v}" for k, v in results)


# ── Tool registry ──────────────────────────────────────────
# name → (function, inputSchema)
TOOL_REGISTRY = {
    "calculator": (run_calculator, {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. 'sqrt(144)' or '2**10'"}
        },
        "required": ["expression"]
    }),
    "get_datetime": (run_get_datetime, {
        "type": "object",
        "properties": {},
        "required": []
    }),
    "web_search": (run_web_search, {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "description": "Number of results (default 5, max 10)"}
        },
        "required": ["query"]
    }),
    "http_get": (run_http_get, {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL including https://"},
            "headers": {"type": "object", "description": "Optional HTTP headers"}
        },
        "required": ["url"]
    }),
    "remember": (run_remember, {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Short label for the fact"},
            "value": {"type": "string", "description": "The fact to remember"}
        },
        "required": ["key", "value"]
    }),
    "recall": (run_recall, {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for in memory"}
        },
        "required": ["query"]
    }),
}


# ── JSON-RPC handler ──────────────────────────────────────

def handle(msg: dict) -> dict | None:
    method = msg.get("method", "")
    msg_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "agent-tools", "version": "1.0"},
                "capabilities": {"tools": {}}
            }
        }

    if method == "tools/list":
        # Build tool list from registry — name, description, inputSchema
        tools = [
            {
                "name": name,
                # Use function docstring as description, fallback to name
                "description": fn.__doc__.strip().split("\n")[0] if fn.__doc__ else name,
                "inputSchema": schema
            }
            for name, (fn, schema) in TOOL_REGISTRY.items()
        ]
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}}

    if method == "tools/call":
        name = msg.get("params", {}).get("name", "")
        args = msg.get("params", {}).get("arguments", {})

        if name not in TOOL_REGISTRY:
            return {
                "jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"}
            }

        fn, _ = TOOL_REGISTRY[name]
        try:
            result = fn(**args)
        except Exception as e:
            result = f"Tool error: {e}"

        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {"content": [{"type": "text", "text": result}]}
        }

    # Notifications (no id) — no response needed
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            response = handle(msg)
            if response is not None:
                send(response)
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None,
                  "error": {"code": -32700, "message": "Parse error"}})


if __name__ == "__main__":
    main()