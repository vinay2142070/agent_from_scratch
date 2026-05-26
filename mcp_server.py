"""
MCP SERVER — HTTP + SSE transport
===================================
Remote version of mcp_server.py. Runs as a standalone HTTP server
that any number of agents can connect to simultaneously.

TRANSPORT MECHANICS:
  GET  /sse         — client opens a persistent SSE connection
                      server immediately sends an "endpoint" event
                      telling the client which URL to POST requests to
  POST /message     — client sends JSON-RPC requests here
                      server responds HTTP 202 immediately
                      then pushes the actual result over the SSE stream
  GET  /health      — simple liveness check

WHY SSE INSTEAD OF WEBSOCKETS:
  SSE is one-directional (server → client) and works over plain HTTP/1.1.
  No upgrade handshake needed. Firewalls, proxies, and load balancers
  handle it without special config. The client sends requests via POST
  (normal HTTP), server pushes results via SSE. Clean separation.

SETUP:
  pip install flask

RUN:
  python mcp_server_http.py
  # Starts on http://localhost:8000

THEN IN agent.py:
  Change MCPClient("mcp_server.py")
  To    MCPClientHTTP("http://localhost:8000")
"""

import os
import json
import math
import queue
import uuid
import datetime
import threading
import requests as req_lib
from flask import Flask, Response, request, jsonify
from semantic_memory import SemanticMemory

app = Flask(__name__)

# One SemanticMemory instance shared across all connections
_mem = SemanticMemory()

# Per-session SSE queues
# When a client connects via GET /sse, a queue is created for its session_id.
# When a POST /message arrives with that session_id, the response is pushed
# into that queue, where the SSE generator picks it up and streams it.
_sessions: dict[str, queue.Queue] = {}
_sessions_lock = threading.Lock()


# ══════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# Identical to mcp_server.py — business logic doesn't change
# ══════════════════════════════════════════════════════════

def run_calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


def run_get_datetime(**_) -> str:
    """Return current UTC datetime."""
    return f"UTC: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"


def run_web_search(query: str, num_results: int = 5) -> str:
    """Search the web via Serper."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not set"
    try:
        resp = req_lib.post(
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
    """Fetch content from a URL."""
    try:
        resp = req_lib.get(url, headers=headers or {}, timeout=10)
        resp.raise_for_status()
        return resp.text[:4000] or "(empty response)"
    except Exception as e:
        return f"HTTP error: {e}"


def run_remember(key: str, value: str) -> str:
    """Save a fact to long-term memory."""
    _mem.save(key, value)
    return f"Saved: {key} = {value}"


def run_recall(query: str) -> str:
    """Search long-term memory."""
    results = _mem.search(query)
    if not results:
        return "Nothing found in long-term memory."
    return "\n".join(f"- [{k}] {v}" for k, v in results)


TOOL_REGISTRY = {
    "calculator":    (run_calculator,   {"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}),
    "get_datetime":  (run_get_datetime, {"type":"object","properties":{},"required":[]}),
    "web_search":    (run_web_search,   {"type":"object","properties":{"query":{"type":"string"},"num_results":{"type":"integer"}},"required":["query"]}),
    "http_get":      (run_http_get,     {"type":"object","properties":{"url":{"type":"string"},"headers":{"type":"object"}},"required":["url"]}),
    "remember":      (run_remember,     {"type":"object","properties":{"key":{"type":"string"},"value":{"type":"string"}},"required":["key","value"]}),
    "recall":        (run_recall,       {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
}


# ══════════════════════════════════════════════════════════
# JSON-RPC HANDLER
# Same logic as mcp_server.py — protocol doesn't change,
# only transport changes.
# ══════════════════════════════════════════════════════════

def handle_jsonrpc(msg: dict) -> dict | None:
    """Route a JSON-RPC message to the right handler."""
    method = msg.get("method", "")
    msg_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "agent-tools-http", "version": "1.0"},
                "capabilities": {"tools": {}}
            }
        }

    if method == "tools/list":
        tools = [
            {
                "name": name,
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
            return {"jsonrpc": "2.0", "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {name}"}}
        fn, _ = TOOL_REGISTRY[name]
        try:
            result = fn(**args)
        except Exception as e:
            result = f"Tool error: {e}"
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {"content": [{"type": "text", "text": result}]}
        }

    # notifications/initialized and other notifications — no response
    return None


# ══════════════════════════════════════════════════════════
# HTTP ROUTES
# ══════════════════════════════════════════════════════════

@app.route("/health")
def health():
    """Liveness check — useful for Docker/k8s health probes."""
    return jsonify({"status": "ok", "server": "agent-tools-http"})


@app.route("/sse")
def sse_endpoint():
    """
    SSE connection endpoint.

    The client connects here with Accept: text/event-stream.
    We:
      1. Create a unique session_id for this connection
      2. Create a Queue for this session
      3. Immediately send an "endpoint" event telling the client
         which URL to POST requests to (includes the session_id)
      4. Loop: block on queue.get(), format each item as an SSE event,
         yield it to the HTTP response stream

    The SSE format is just plain text:
      event: <event-name>\n
      data: <json-string>\n
      \n

    Flask's Response with generator + mimetype=text/event-stream
    keeps the connection open and flushes each yield immediately.
    """
    session_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()

    with _sessions_lock:
        _sessions[session_id] = q

    def generate():
        try:
            # First event: tell client the POST endpoint for this session
            # Client will use this URL for all subsequent requests
            yield f"event: endpoint\ndata: /message?session_id={session_id}\n\n"

            # Then stream results as they arrive in the queue
            while True:
                try:
                    # Block until a result is pushed (timeout allows periodic flush)
                    result = q.get(timeout=30)
                    if result is None:
                        # None = shutdown signal
                        break
                    # SSE event format
                    yield f"event: message\ndata: {json.dumps(result)}\n\n"
                except queue.Empty:
                    # Send a keep-alive comment to prevent proxy timeouts
                    yield ": keep-alive\n\n"
        finally:
            # Clean up session when client disconnects
            with _sessions_lock:
                _sessions.pop(session_id, None)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # disable nginx buffering
            "Connection": "keep-alive",
        }
    )


@app.route("/message", methods=["POST"])
def message_endpoint():
    """
    JSON-RPC request endpoint.

    Client POSTs JSON-RPC here (with ?session_id= from the endpoint event).
    We:
      1. Parse the JSON body
      2. Handle the JSON-RPC message (same logic as stdio version)
      3. Push the response into the session's SSE queue
      4. Return HTTP 202 immediately (result comes via SSE, not here)

    Why 202 and not 200 with the result?
      Because SSE is the response channel. Returning the result in the POST
      response would break the protocol — clients expect results on SSE.
      202 just means "received, processing".
    """
    session_id = request.args.get("session_id")

    with _sessions_lock:
        q = _sessions.get(session_id)

    if q is None:
        return jsonify({"error": "Unknown session_id"}), 400

    try:
        msg = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    # Handle in a thread so we don't block the POST response
    # (important for tools that do I/O like web_search)
    def handle_and_push():
        response = handle_jsonrpc(msg)
        if response is not None:
            q.put(response)

    threading.Thread(target=handle_and_push, daemon=True).start()

    # Return 202 immediately — result will arrive via SSE
    return "", 202


if __name__ == "__main__":
    print("MCP HTTP+SSE server starting on http://localhost:8000")
    print("Endpoints:")
    print("  GET  /sse      — SSE connection")
    print("  POST /message  — JSON-RPC requests")
    print("  GET  /health   — liveness check")
    app.run(host="0.0.0.0", port=8000, threaded=True)