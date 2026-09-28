"""Drive the memlab MCP server over stdio like a client would, print answers and timings.

  python scripts/mcp_smoke.py <python> <config> <root> "<code query>" "<decision query>"
"""
import json
import subprocess
import sys
import time

py, config, root, code_q, dec_q = sys.argv[1:6]
p = subprocess.Popen([py, "-m", "memlab", "serve", "--config", config, "--root", root],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                     cwd=__file__.rsplit("scripts", 1)[0])


def call(i, method, params=None):
    p.stdin.write((json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}) + "\n").encode())
    p.stdin.flush()
    return json.loads(p.stdout.readline())


t = time.time()
init = call(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "smoke"}})
p.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'); p.stdin.flush()
tools = call(2, "tools/list")
print(f"initialize+list: {time.time() - t:.2f}s, server={init['result']['serverInfo']}, "
      f"tools={[x['name'] for x in tools['result']['tools']]}")
for i, (name, args) in enumerate([("search_code", {"query": code_q, "budget_tokens": 1500}),
                                  ("search_decisions", {"query": dec_q, "k": 3}),
                                  ("search_code", {"query": code_q, "budget_tokens": 1500})], start=3):
    t = time.time()
    r = call(i, "tools/call", {"name": name, "arguments": args})
    text = r["result"]["content"][0]["text"]
    heads = [ln for ln in text.splitlines() if ln.startswith("### ") or ln.startswith("search_")]
    print(f"\n{name} ({time.time() - t:.1f}s, isError={r['result']['isError']}):")
    print("\n".join(heads[:12]))
p.stdin.close()
p.wait(timeout=30)
