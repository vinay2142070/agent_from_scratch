"""
MCP CLIENT — replaces hardcoded TOOLS with dynamic discovery
=============================================================
This shows how your agent.py would change if it used MCP
instead of hardcoded tool functions.

The key difference:
  BEFORE: tools defined in agent.py, functions called directly
  AFTER:  tools discovered from MCP server, called via JSON-RPC

Everything else — the ReAct loop, memory, streaming — stays identical.
MCP only changes how tools are registered and executed.

HOW IT WORKS:
  1. Spawn mcp_server.py as a subprocess
  2. Send initialize → get capabilities handshake
  3. Send tools/list → get tool schemas (inject into LLM)
  4. LLM responds with tool_calls
  5. Send tools/call → get result → append to messages → loop

RUN:
  python mcp_client.py
  (automatically spawns mcp_server.py)
"""

import json
import subprocess
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    """
    Manages the connection to one MCP server (subprocess).

    In production: you'd connect to multiple servers simultaneously.
    Each server is a separate process (or HTTP endpoint).
    Your agent merges all their tool lists into one TOOLS registry.
    """

    def __init__(self, server_script: str):
        # Spawn the server as a child process
        # stdin/stdout are the communication channels (stdio transport)
        self.process = subprocess.Popen(
            [sys.executable, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1  # line-buffered
        )
        self._next_id = 1
        self._handshake()

    def _send(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request, return the parsed response."""
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params or {}
        }
        self._next_id += 1

        # Write request to server's stdin
        self.process.stdin.write(json.dumps(msg) + "\n")
        self.process.stdin.flush()

        # Read response from server's stdout
        line = self.process.stdout.readline()
        return json.loads(line.strip())

    def _handshake(self):
        """
        MCP initialize handshake.
        Client sends its capabilities, server replies with its own.
        Must happen before any other calls.
        """
        resp = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "demo-agent", "version": "1.0"},
            "capabilities": {}
        })
        # After initialize, send the initialized notification (no response expected)
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        self.process.stdin.write(json.dumps(notif) + "\n")
        self.process.stdin.flush()
        print(f"  [mcp] connected to server: {resp['result']['serverInfo']['name']}")

    def list_tools(self) -> list[dict]:
        """
        Fetch available tools from the server.

        Returns a list of tool dicts with name, description, inputSchema.
        We convert these to OpenAI's format (inputSchema → parameters)
        so they can be passed directly to chat.completions.create().
        """
        resp = self._send("tools/list")
        tools = resp["result"]["tools"]
        print(f"  [mcp] discovered {len(tools)} tools: {[t['name'] for t in tools]}")
        return tools

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        Execute a tool on the server.

        The server runs the actual function and returns the result.
        We get back a content array — extract the text from the first block.
        """
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            return f"MCP error: {resp['error']['message']}"
        content = resp["result"]["content"]
        return content[0]["text"] if content else "(no result)"

    def to_openai_schemas(self, mcp_tools: list[dict]) -> list[dict]:
        """
        Convert MCP tool schemas → OpenAI tool schemas.

        MCP uses "inputSchema", OpenAI uses "parameters" — same JSON Schema,
        different key name. This is the only translation needed.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"]  # rename inputSchema → parameters
                }
            }
            for t in mcp_tools
        ]

    def close(self):
        self.process.terminate()


def run_agent_with_mcp():
    """
    A minimal ReAct loop using MCP tools instead of hardcoded TOOLS.

    Notice how similar this is to your existing _react_loop():
    - Same message structure
    - Same tool_calls handling
    - Same loop logic
    Only _execute_tool() changes: it calls mcp.call_tool() instead
    of looking up a local Python function.
    """
    client = OpenAI()
    mcp = MCPClient("mcp_server.py")

    # Step 1: discover tools from server
    mcp_tools = mcp.list_tools()
    # Step 2: convert to OpenAI format for the LLM
    tool_schemas = mcp.to_openai_schemas(mcp_tools)

    messages = []
    system = (
        "You are a helpful assistant with access to tools. "
        "Use them when appropriate. Be concise."
    )

    print("\n[MCP Agent ready. Type 'quit' to exit]\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        # ReAct loop — identical structure to your agent.py
        for iteration in range(10):
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=512,
                tools=tool_schemas,
                messages=[{"role": "system", "content": system}] + messages
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                # Append assistant tool-call message directly
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
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

                # Execute each tool via MCP (instead of local function call)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    print(f"  [mcp] calling {tc.function.name}({args})")
                    result = mcp.call_tool(tc.function.name, args)
                    print(f"  [mcp] result: {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })
                continue

            # Final answer
            if msg.content:
                messages.append({"role": "assistant", "content": msg.content})
                print(f"Agent: {msg.content}\n")
                break

    mcp.close()


if __name__ == "__main__":
    run_agent_with_mcp()