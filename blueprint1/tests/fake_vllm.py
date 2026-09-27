#!/usr/bin/env python3
"""Minimal OpenAI/vLLM-compatible fake server for testing the router and benches
without a GPU. Streams N tokens, exposes /health and a few vllm:* metrics, and
records the requests it received at GET /_seen."""
import asyncio
import json
import sys
import time

from aiohttp import web

PORT = int(sys.argv[1])
TOKEN_DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
SEEN: list[dict] = []
COUNTERS = {"vllm:prefix_cache_queries_total": 0.0, "vllm:prefix_cache_hits_total": 0.0,
            "vllm:num_preemptions_total": 0.0, "vllm:generation_tokens_total": 0.0}
PREFIXES: set[str] = set()


async def gen(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    SEEN.append({"path": request.path, "max_tokens": body.get("max_tokens"), "stream": bool(body.get("stream"))})
    text = json.dumps(body.get("messages") or body.get("prompt"))
    COUNTERS["vllm:prefix_cache_queries_total"] += len(text)
    if text[:200] in PREFIXES:
        COUNTERS["vllm:prefix_cache_hits_total"] += len(text) * 0.8
    PREFIXES.add(text[:200])
    n = int(body.get("max_tokens") or 16)
    COUNTERS["vllm:generation_tokens_total"] += n
    chat = request.path.endswith("chat/completions")
    await asyncio.sleep(0.01)  # "prefill"
    if not body.get("stream"):
        choice = {"index": 0, "message": {"role": "assistant", "content": "tok " * n}} if chat \
            else {"index": 0, "text": "tok " * n}
        return web.json_response({"id": "x", "choices": [choice],
                                  "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": n}})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(n):
        await asyncio.sleep(TOKEN_DELAY)
        choice = {"index": 0, "delta": {"content": f"t{i} "}} if chat else {"index": 0, "text": f"t{i} "}
        await resp.write(f"data: {json.dumps({'id': 'x', 'choices': [choice]})}\n\n".encode())
    usage = {"prompt_tokens": len(text) // 4, "completion_tokens": n, "total_tokens": len(text) // 4 + n}
    await resp.write(f"data: {json.dumps({'id': 'x', 'choices': [], 'usage': usage})}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def metrics(_):
    lines = [f'{k}{{model_name="qwen7b"}} {v}' for k, v in COUNTERS.items()]
    return web.Response(text="\n".join(lines) + "\n")


app = web.Application()
app.router.add_post("/v1/completions", gen)
app.router.add_post("/v1/chat/completions", gen)
app.router.add_get("/health", lambda _: web.Response(text="ok"))
app.router.add_get("/metrics", metrics)
app.router.add_get("/v1/models", lambda _: web.json_response({"data": [{"id": "qwen7b"}]}))
app.router.add_get("/_seen", lambda _: web.json_response(SEEN))
web.run_app(app, port=PORT, print=lambda *_: None)
