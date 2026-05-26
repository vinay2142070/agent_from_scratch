"""
MCP CLIENT — HTTP + SSE transport
====================================
Drop-in replacement for the MCPClient class in agent.py.

Swap one line in Agent.__init__():
  FROM: self.mcp = MCPClient("mcp_server.py")
  TO:   self.mcp = MCPClientHTTP("http://localhost:8000")

Everything else in agent.py stays identical.

HOW IT WORKS:
  1. Connect to GET /sse → get session_id + endpoint URL
  2. Start background thread reading the SSE stream
  3. For each request: POST to /message?session_id=...
     then block on a threading.Event until SSE thread pushes the response
  4. Match responses to requests via JSON-RPC id field

WHY A BACKGROUND THREAD FOR SSE:
  SSE is a streaming response — it never "ends".
  We can't block the main thread reading it while also making POST requests.
  Solution: background thread reads SSE continuously, main thread POSTs.
  A dict of threading.Event objects (one per pending request id) lets the
  main thread block until its specific response arrives.

SETUP:
  pip install requests sseclient-py
"""

import json
import threading
import requests
import sseclient


class MCPClientHTTP:
    """
    MCP client over HTTP + SSE transport.
    Public interface is identical to MCPClient (stdio version):
      list_tools() → list[dict]
      call_tool(name, arguments) → str
      to_openai_schemas(mcp_tools) → list[dict]
      close()
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._next_id = 1
        self._lock = threading.Lock()

        # Map of request_id → {"event": threading.Event, "result": dict}
        # Main thread registers an entry before POSTing, SSE thread fills result
        self._pending: dict[int, dict] = {}

        # SSE connection state
        self._session_id: str = None
        self._post_url: str = None
        self._sse_thread: threading.Thread = None
        self._connected = threading.Event()
        self._running = True

        self._connect()
        self._handshake()

    def _connect(self):
        """
        Open the SSE connection in a background thread.
        Blocks until the server sends the "endpoint" event with the session_id.
        """
        ready = threading.Event()

        def sse_reader():
            """
            Background thread: reads SSE stream continuously.

            SSE events arrive as:
              event: endpoint
              data: /message?session_id=abc123

              event: message
              data: {"jsonrpc":"2.0","id":1,"result":{...}}

            For "endpoint" events: store the POST URL, signal _connected.
            For "message" events: parse JSON, find pending request by id,
              store result, signal its threading.Event.
            """
            resp = requests.get(
                f"{self.base_url}/sse",
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=None   # SSE connection lives forever
            )
            client = sseclient.SSEClient(resp)

            for event in client.events():
                if not self._running:
                    break

                if event.event == "endpoint":
                    # Server told us the POST URL for this session
                    path = event.data.strip()
                    self._post_url = f"{self.base_url}{path}"
                    # Extract session_id from URL param
                    self._session_id = path.split("session_id=")[-1]
                    self._connected.set()
                    print(f"  [mcp-http] connected, session={self._session_id[:8]}...")

                elif event.event == "message":
                    # A JSON-RPC response arrived
                    try:
                        msg = json.loads(event.data)
                        msg_id = msg.get("id")
                        if msg_id in self._pending:
                            self._pending[msg_id]["result"] = msg
                            self._pending[msg_id]["event"].set()
                    except json.JSONDecodeError:
                        pass

        self._sse_thread = threading.Thread(target=sse_reader, daemon=True)
        self._sse_thread.start()

        # Block until server sends the endpoint event (max 10s)
        if not self._connected.wait(timeout=10):
            raise ConnectionError(f"Could not connect to MCP server at {self.base_url}")

    def _send(self, method: str, params: dict = None) -> dict:
        """
        Send one JSON-RPC request via POST, wait for response via SSE.

        Steps:
          1. Assign a unique id to this request
          2. Register a threading.Event for this id in _pending
          3. POST the JSON-RPC message to /message?session_id=...
          4. Block on the Event until SSE thread signals it
          5. Return the result
        """
        with self._lock:
            msg_id = self._next_id
            self._next_id += 1

        # Register before posting — avoid race where response arrives before registration
        pending_entry = {"event": threading.Event(), "result": None}
        self._pending[msg_id] = pending_entry

        msg = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params or {}
        }

        # POST the request — server returns 202 immediately
        resp = requests.post(
            self._post_url,
            json=msg,
            timeout=10
        )
        if resp.status_code not in (200, 202):
            del self._pending[msg_id]
            raise RuntimeError(f"POST failed: {resp.status_code} {resp.text}")

        # Wait for SSE thread to push the response (max 30s for slow tools)
        if not pending_entry["event"].wait(timeout=30):
            del self._pending[msg_id]
            raise TimeoutError(f"Timeout waiting for response to request {msg_id}")

        result = pending_entry["result"]
        del self._pending[msg_id]
        return result

    def _handshake(self):
        """MCP initialize handshake — identical to stdio version."""
        resp = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "agent-http", "version": "1.0"},
            "capabilities": {}
        })
        # Send initialized notification (no response expected)
        requests.post(self._post_url, json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }, timeout=5)
        name = resp["result"]["serverInfo"]["name"]
        print(f"  [mcp-http] handshake done with '{name}'")

    def list_tools(self) -> list[dict]:
        """Fetch available tools from the server."""
        resp = self._send("tools/list")
        tools = resp["result"]["tools"]
        print(f"  [mcp-http] {len(tools)} tools: {[t['name'] for t in tools]}")
        return tools

    def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool, return result as string."""
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            return f"MCP error: {resp['error']['message']}"
        content = resp["result"]["content"]
        return content[0]["text"] if content else "(no result)"

    @staticmethod
    def to_openai_schemas(mcp_tools: list[dict]) -> list[dict]:
        """Convert MCP format → OpenAI format. Identical to stdio version."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"]
                }
            }
            for t in mcp_tools
        ]

    def close(self):
        """Stop the SSE reader thread."""
        self._running = False


# ══════════════════════════════════════════════════════════
# HOW TO USE IN agent.py
# ══════════════════════════════════════════════════════════
#
# In agent.py, Section 3 (Agent class), change __init__:
#
#   from mcp_client_http import MCPClientHTTP   # add this import
#
#   def __init__(self):
#       ...
#       # OLD (stdio):
#       self.mcp = MCPClient("mcp_server.py")
#
#       # NEW (HTTP):
#       self.mcp = MCPClientHTTP("http://localhost:8000")
#
#       # Everything below stays the same:
#       mcp_tools = self.mcp.list_tools()
#       self.tool_schemas = MCPClientHTTP.to_openai_schemas(mcp_tools)
#
# That's the only change needed in agent.py.
# ══════════════════════════════════════════════════════════