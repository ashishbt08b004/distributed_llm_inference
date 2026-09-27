#!/usr/bin/env python3
"""Multi-turn conversation benchmark (pattern 4 of the grid).

`vllm bench serve --dataset-name sharegpt` sends independent single-turn
prompts, so it never exercises cross-turn KV reuse. This script does:

  * N sessions start as a Poisson process (--session-rate per second)
  * every session shares one long system prompt (cross-session prefix reuse)
  * turn 1 carries a per-session document (~1k tokens); later turns append the
    previous answer + a short follow-up, so context grows every turn
    (cross-turn prefix reuse: the case LMCache/Mooncake + sticky routing target)
  * each turn streams /v1/chat/completions and records TTFT, ITL and E2E

The output JSON uses the same key names as `vllm bench serve` (p95_ttft_ms,
p95_itl_ms, output_throughput, ...) plus a per-turn TTFT breakdown, so
analyze.py can treat all runs the same way.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import datetime

import aiohttp

WORDS = ("latency throughput cache prefix decode prefill tensor kernel memory bandwidth token batch "
         "scheduler router gateway replica shard pipeline attention matrix vector cluster node "
         "network storage tier eviction policy request session context window model weight "
         "quantization precision queue budget metric dashboard alert capacity headroom").split()


def words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def one_turn(session, url, model, messages, max_tokens, api_key):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
            "ignore_eos": True, "min_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.perf_counter()
    ttft, last, itls, text, usage = None, None, [], [], {}
    async with session.post(f"{url}/v1/chat/completions", json=body, headers=headers) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
        buf = b""
        async for chunk in r.content.iter_any():
            buf += chunk
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                line = event.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                msg = json.loads(data)
                if msg.get("usage"):
                    usage = msg["usage"]
                for ch in msg.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        else:
                            itls.append(now - last)
                        last = now
                        text.append(delta)
    e2e = time.perf_counter() - t0
    return {"ttft": ttft if ttft is not None else e2e, "itls": itls, "e2e": e2e, "text": "".join(text),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", len(itls) + 1)}


async def run_session(sid, args, http, system_prompt, results, start_delay):
    await asyncio.sleep(start_delay)
    rng = random.Random(args.seed * 1000 + sid)
    doc = words(rng, args.doc_words)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is document #{sid}:\n{doc}\n\nSummarise its main theme."}]
    for turn in range(args.turns):
        try:
            res = await one_turn(http, args.base_url, args.model, messages, args.max_tokens, args.api_key)
        except Exception as e:  # noqa: BLE001
            results.append({"session": sid, "turn": turn, "error": str(e)})
            return
        res.update(session=sid, turn=turn)
        results.append(res)
        messages.append({"role": "assistant", "content": res.pop("text")})
        messages.append({"role": "user", "content": f"Follow-up {turn + 1}: {words(rng, args.followup_words)}?"})
        if args.think_time:
            await asyncio.sleep(rng.expovariate(1 / args.think_time))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--sessions", type=int, default=50)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--session-rate", type=float, default=2.0, help="new sessions / s (Poisson)")
    ap.add_argument("--think-time", type=float, default=0.0, help="mean seconds between turns")
    ap.add_argument("--system-words", type=int, default=600)
    ap.add_argument("--doc-words", type=int, default=700)
    ap.add_argument("--followup-words", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--result-file", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    system_prompt = "You are a careful infrastructure analyst. Reference handbook: " + words(rng, args.system_words)
    t, delays = 0.0, []
    for _ in range(args.sessions):
        delays.append(t)
        t += rng.expovariate(args.session_rate)

    results: list[dict] = []
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as http:
        t0 = time.perf_counter()
        await asyncio.gather(*(run_session(i, args, http, system_prompt, results, d)
                               for i, d in enumerate(delays)))
        duration = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    ttft = [r["ttft"] * 1000 for r in ok]
    itl = [x * 1000 for r in ok for x in r["itls"]]
    e2e = [r["e2e"] * 1000 for r in ok]
    tpot = [(r["e2e"] - r["ttft"]) * 1000 / max(r["output_tokens"] - 1, 1) for r in ok]
    out_tok = sum(r["output_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] for r in ok)

    summary = {
        "backend": "multiturn-chat", "label": args.label, "base_url": args.base_url,
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "sessions": args.sessions, "turns": args.turns, "session_rate": args.session_rate,
        "completed": len(ok), "failed": len(errors), "duration": duration,
        "total_input_tokens": in_tok, "total_output_tokens": out_tok,
        "request_throughput": len(ok) / duration, "output_throughput": out_tok / duration,
        "total_token_throughput": (in_tok + out_tok) / duration,
    }
    for name, vals in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2el", e2e)):
        summary[f"mean_{name}_ms"] = statistics.fmean(vals) if vals else float("nan")
        summary[f"median_{name}_ms"] = pct(vals, 50)
        for p in (95, 99):
            summary[f"p{p}_{name}_ms"] = pct(vals, p)
    by_turn = {}
    for turn in range(args.turns):
        tt = [r for r in ok if r["turn"] == turn]
        by_turn[turn + 1] = {
            "n": len(tt),
            "mean_prompt_tokens": statistics.fmean([r["prompt_tokens"] for r in tt]) if tt else 0,
            "mean_ttft_ms": statistics.fmean([r["ttft"] * 1000 for r in tt]) if tt else float("nan"),
            "p95_ttft_ms": pct([r["ttft"] * 1000 for r in tt], 95),
        }
    summary["per_turn"] = by_turn
    summary["errors"] = [e["error"] for e in errors[:10]]

    with open(args.result_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"multi-turn: {len(ok)} ok / {len(errors)} failed in {duration:.1f}s | "
          f"TTFT p95 {summary['p95_ttft_ms']:.0f} ms | ITL p95 {summary['p95_itl_ms']:.1f} ms | "
          f"{summary['output_throughput']:.0f} out tok/s")
    for turn, row in by_turn.items():
        print(f"  turn {turn}: prompt ~{row['mean_prompt_tokens']:.0f} tok, mean TTFT {row['mean_ttft_ms']:.0f} ms")
    if errors:
        print("first errors:", summary["errors"][:3])


if __name__ == "__main__":
    asyncio.run(main())
