#!/usr/bin/env python3
"""Run the analyst crew against the inference cluster.

  python run.py ask "Which region had the most customers?"   # free-form question
  python run.py ask --qid q06 -v                             # one graded question, verbose agent trace
  python run.py eval                                         # all 12 questions, 4 at a time
  python run.py eval --qids q02,q09 --repeat 3 --concurrency 8 --prometheus http://localhost:9090
  python run.py eval --label "C no-kv-tier" --prometheus http://localhost:9090   # label runs for the report
  python run.py aggregate                                    # rebuild the report CSVs from all runs

`eval` runs every question in its own subprocess (so the per-run counters stay
separate), grades each against data/questions.json, and writes
../bench-results/agent/<timestamp>/{results.jsonl,summary.md}, next to the
cluster benchmarks that `blueprint1/scripts/laptop.sh fetch` brings back.
BENCH_RESULTS_DIR moves the whole tree. Each run folder also keeps meta.json
(settings, label, cluster metrics) and a <qid>_r<n>.trace.jsonl per question (every
LLM call: agent, latency, tokens, what the model returned). After every eval,
agent_runs.csv and agent_questions.csv next to the run folders are rebuilt from
all runs (see report.py).
Config comes from the environment or a .env file next to this script (see .env.example).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from report import FAILURE_HELP, FAILURES, aggregate, classify

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "data" / "questions.json"


def load_env() -> None:
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                if v[:1] in ('"', "'") and v[0] in v[1:]:
                    v = v[1:v.index(v[0], 1)]          # quoted: keep everything inside the quotes
                else:
                    v = re.split(r"\s+#", v, maxsplit=1)[0].strip()   # unquoted: drop "  # comment"
                os.environ.setdefault(k.strip(), v)
    # No phoning home from a benchmark client.
    for k, v in {"CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true",
                 "CREWAI_TRACING_ENABLED": "false", "CREWAI_DISABLE_TRACKING": "true"}.items():
        os.environ.setdefault(k, v)


def load_questions() -> dict[str, dict]:
    if not QUESTIONS.exists():
        sys.exit("data/questions.json missing: run `python make_data.py` first")
    return {q["id"]: q for q in json.loads(QUESTIONS.read_text())}


# ----------------------------------------------------------------------------- grading
def extract_final(text: str) -> str | None:
    hits = re.findall(r"FINAL_ANSWER:\s*(.+)", text or "")
    return hits[-1].strip().strip("*`\"' .") if hits else None


def grade(q: dict, final: str | None) -> bool:
    if final is None:
        return False
    if q["type"] == "text":
        return str(q["answer"]).lower() in final.lower()
    m = re.search(r"-?\d[\d,]*\.?\d*", final)
    if not m:
        return False
    got, want = float(m.group().replace(",", "")), float(q["answer"])
    return abs(got - want) <= max(abs(want) * q["tolerance"], 0.005)


# ----------------------------------------------------------------------------- one crew run
def _clip(value, n: int = 1500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= n else text[:n] + f"... [{len(text)} chars]"


def run_one(question: str, verbose: bool, trace: list | None = None) -> dict:
    """Kick off the crew in THIS process and return the answer plus call statistics.
    If `trace` is a list, one record per LLM call is appended to it."""
    from crewai.events import BaseEventListener
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallFailedEvent, LLMCallStartedEvent

    import crew
    import tools

    stats = {"llm_calls": 0, "llm_failures": 0, "prompt_tokens": 0, "completion_tokens": 0, "llm_seconds": []}
    started: dict[str, tuple[float, int]] = {}   # call_id -> (start time, messages sent)
    t0 = time.perf_counter()

    class Counter(BaseEventListener):
        def setup_listeners(self, bus):
            @bus.on(LLMCallStartedEvent)
            def _start(_src, ev):
                started[ev.call_id] = (time.perf_counter(), len(ev.messages or []))

            @bus.on(LLMCallCompletedEvent)
            def _done(_src, ev):
                stats["llm_calls"] += 1
                t_start, n_msgs = started.pop(ev.call_id, (None, None))
                secs = time.perf_counter() - t_start if t_start else None
                if secs is not None:
                    stats["llm_seconds"].append(secs)
                usage = ev.usage or {}
                stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                if trace is not None:
                    trace.append({
                        "n": stats["llm_calls"], "agent": getattr(ev, "agent_role", None),
                        "t_s": round(t_start - t0, 2) if t_start else None,
                        "seconds": round(secs, 2) if secs is not None else None,
                        "messages_in": n_msgs, "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "finish_reason": ev.finish_reason, "response": _clip(ev.response)})

            @bus.on(LLMCallFailedEvent)
            def _fail(_src, ev):
                stats["llm_failures"] += 1
                started.pop(ev.call_id, None)
                if trace is not None:
                    trace.append({"n": None, "agent": getattr(ev, "agent_role", None),
                                  "error": _clip(getattr(ev, "error", ""), 500)})

    listener = Counter()  # noqa: F841 — registering is a side effect of construction
    error = None
    try:
        output = crew.build_crew(question, verbose=verbose).kickoff()
        text = output.raw
    except Exception as e:  # noqa: BLE001 — report failed runs instead of losing them
        text, error = "", f"{type(e).__name__}: {e}"
    lat = stats.pop("llm_seconds")
    return {
        "answer_text": text, "final": extract_final(text), "error": error,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "tool_calls": sum(tools.CALLS.values()), "tool_calls_by_name": dict(tools.CALLS),
        "guardrail_failures": dict(crew.GUARDRAIL_FAILURES),
        **stats,
        "llm_seconds_p50": round(statistics.median(lat), 2) if lat else None,
        "llm_seconds_max": round(max(lat), 2) if lat else None,
    }


# ----------------------------------------------------------------------------- cluster metrics
PROM_QUERIES = {
    "router_queue_depth_max": "max_over_time(sum(kv_router_queue_depth)[{w}s:5s])",
    "router_rejected": "sum(increase(kv_router_rejected_total[{w}s]))",
    "router_queue_wait_avg_s": "sum(increase(kv_router_queue_wait_seconds_sum[{w}s])) / "
                               "clamp_min(sum(increase(kv_router_queue_wait_seconds_count[{w}s])),1)",
    "vllm_prefix_hit_rate": "sum(increase(vllm:prefix_cache_hits_total[{w}s])) / "
                            "clamp_min(sum(increase(vllm:prefix_cache_queries_total[{w}s])),1)",
    "vllm_ttft_avg_s": "sum(increase(vllm:time_to_first_token_seconds_sum[{w}s])) / "
                       "clamp_min(sum(increase(vllm:time_to_first_token_seconds_count[{w}s])),1)",
    "vllm_waiting_max": "max_over_time(sum(vllm:num_requests_waiting)[{w}s:5s])",
    "vllm_preemptions": "sum(increase(vllm:num_preemptions_total[{w}s]))",
}


def prometheus_report(url: str, start: float, end: float) -> dict:
    w = max(30, int(end - start) + 15)
    out = {}
    for name, q in PROM_QUERIES.items():
        qs = urllib.parse.urlencode({"query": q.format(w=w), "time": end})
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/api/v1/query?{qs}", timeout=10) as r:
                res = json.load(r)["data"]["result"]
            out[name] = round(float(res[0]["value"][1]), 4) if res else None
        except Exception as e:  # noqa: BLE001
            out[name] = f"error: {e}"
    return out


# ----------------------------------------------------------------------------- commands
def cmd_ask(args) -> None:
    qs = load_questions()
    q = qs[args.qid] if args.qid else None
    question = q["question"] if q else args.question
    if not question:
        sys.exit("give a question or --qid")
    trace: list = []
    res = run_one(question, args.verbose, trace)
    if q:
        res.update(qid=q["id"], difficulty=q["difficulty"], expected=q["answer"], correct=grade(q, res["final"]))
        res["failure"] = classify(res)
    if args.trace_out:
        Path(args.trace_out).write_text("".join(json.dumps(t, default=str) + "\n" for t in trace))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res))
        return
    print("\n" + "=" * 70)
    print(res["answer_text"] or f"(no answer) {res['error']}")
    print("=" * 70)
    if q:
        print(f"expected: {q['answer']}   got: {res['final']}   correct: {res['correct']}"
              + (f"   failure: {res['failure']} ({FAILURE_HELP[res['failure']]})" if res["failure"] else ""))
    print(f"tool calls: {res['tool_calls']} {res['tool_calls_by_name']}")
    print(f"LLM calls: {res['llm_calls']} (failed {res['llm_failures']})  tokens in/out: "
          f"{res['prompt_tokens']}/{res['completion_tokens']}  wall: {res['wall_seconds']} s")
    if res["guardrail_failures"]:
        print(f"guardrail rejections (agent retried): {res['guardrail_failures']}")


def cmd_eval(args) -> None:
    qs = load_questions()
    ids = args.qids.split(",") if args.qids else list(qs)
    unknown = [i for i in ids if i not in qs]
    if unknown:
        sys.exit(f"unknown question ids: {unknown}")
    jobs = [(qid, r) for r in range(args.repeat) for qid in ids]
    started_at = datetime.now()
    out_dir = agent_root() / started_at.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    print(f"{len(jobs)} runs, {args.concurrency} at a time -> {out_dir}  "
          f"(model {os.environ.get('LLM_MODEL', 'hosted_vllm/qwen7b')} @ "
          f"{os.environ.get('LLM_BASE_URL', 'http://localhost:4000/v1')})")

    def one(job):
        qid, rep = job
        path = out_dir / f"{qid}_r{rep}.json"
        with open(out_dir / f"{qid}_r{rep}.log", "w") as log:
            proc = subprocess.run([sys.executable, str(HERE / "run.py"), "ask", "--qid", qid, "--json-out", str(path),
                                   "--trace-out", str(out_dir / f"{qid}_r{rep}.trace.jsonl")],
                                  stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        if proc.returncode != 0 or not path.exists():
            return {"qid": qid, "repeat": rep, "difficulty": qs[qid]["difficulty"], "expected": qs[qid]["answer"],
                    "correct": False, "failure": "error", "error": f"worker exit {proc.returncode}",
                    "tool_calls": 0, "llm_calls": 0, "wall_seconds": None}
        res = json.loads(path.read_text()) | {"repeat": rep}
        mark = "ok " if res["correct"] else "BAD"
        print(f"  {mark} {qid} r{rep}: got {res['final']!r} (want {res['expected']!r})  "
              f"tools={res['tool_calls']} llm={res['llm_calls']} {res['wall_seconds']}s"
              + (f"  [{res['failure']}]" if res.get("failure") else "")
              + (f"  error={res['error']}" if res.get("error") else ""), flush=True)
        return res

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(one, jobs))
    t1 = time.time()
    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    prom = prometheus_report(args.prometheus, t0, t1) if args.prometheus else None
    from crew import sampling
    meta = {"run_id": out_dir.name, "started_at": started_at.isoformat(timespec="seconds"),
            "label": args.label or "", "model": os.environ.get("LLM_MODEL", "hosted_vllm/qwen7b"),
            "base_url": os.environ.get("LLM_BASE_URL", "http://localhost:4000/v1"), **sampling(),
            "max_iter": int(os.environ.get("AGENT_MAX_ITER", "15")),
            "guardrail_retries": int(os.environ.get("GUARDRAIL_RETRIES", "2")),
            "concurrency": args.concurrency, "repeat": args.repeat, "qids": ids,
            "elapsed_s": round(t1 - t0, 1), "prometheus": prom}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    summary = summarize(results, t1 - t0, args, prom, meta)
    (out_dir / "summary.md").write_text(summary)
    print("\n" + summary)
    runs_csv, q_csv, n = aggregate(agent_root())
    print(f"report CSVs rebuilt from {n} run(s): {runs_csv}\n                                  {q_csv}")


def cmd_aggregate(_args) -> None:
    runs_csv, q_csv, n = aggregate(agent_root())
    print(f"{n} run(s) -> {runs_csv}\n          {q_csv}")


def agent_root() -> Path:
    return Path(os.environ.get("BENCH_RESULTS_DIR") or HERE.parent / "bench-results") / "agent"


def summarize(results: list[dict], elapsed: float, args, prom: dict | None, meta: dict) -> str:
    ok = [r for r in results if r.get("wall_seconds") is not None]
    tool = [r["tool_calls"] for r in ok]
    llm = [r["llm_calls"] for r in ok]
    wall = sorted(r["wall_seconds"] for r in ok)
    tokens_in = sum(r.get("prompt_tokens", 0) for r in ok)
    tokens_out = sum(r.get("completion_tokens", 0) for r in ok)
    p = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None  # noqa: E731
    lines = [
        f"# Analyst crew run — {datetime.now():%Y-%m-%d %H:%M}" + (f" — {meta['label']}" if meta["label"] else ""),
        "",
        f"- endpoint: `{os.environ.get('LLM_MODEL', 'hosted_vllm/qwen7b')}` @ "
        f"`{os.environ.get('LLM_BASE_URL', 'http://localhost:4000/v1')}`",
        f"- runs: {len(results)} (concurrency {args.concurrency}, repeat {args.repeat}), elapsed {elapsed:.0f} s",
        f"- **accuracy: {sum(r['correct'] for r in results)}/{len(results)}**",
        f"- tool calls: total {sum(tool)}, per run mean {statistics.fmean(tool):.1f} / max {max(tool)}" if tool else "",
        f"- LLM calls: total {sum(llm)}, per run mean {statistics.fmean(llm):.1f}" if llm else "",
        f"- tokens in/out: {tokens_in} / {tokens_out}"
        + (f"  ({tokens_in / elapsed:.0f} prompt tok/s offered)" if elapsed else ""),
        f"- wall per run: p50 {p(wall, .5)} s, p95 {p(wall, .95)} s" if wall else "",
        f"- sampling: temperature {meta['temperature']}, top_p {meta['top_p']}, top_k {meta['top_k']}, "
        f"repetition_penalty {meta['repetition_penalty']}; max_iter {meta['max_iter']}, "
        f"guardrail retries {meta['guardrail_retries']}",
        "",
        "**Why runs failed:** " + (", ".join(
            f"{f} {n} ({FAILURE_HELP[f]})" for f in FAILURES
            if (n := sum(r.get("failure") == f for r in results))) or "none"),
        "",
        "| run | difficulty | correct | expected | got | failure | tool calls | LLM calls | guardrail retries | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r["qid"], r["repeat"])):
        lines.append(f"| {r['qid']} r{r['repeat']} | {r.get('difficulty', '')} | {'✅' if r['correct'] else '❌'} | "
                     f"{r.get('expected', '')} | {r.get('final') or r.get('error') or ''} | {r.get('failure', '')} | "
                     f"{r['tool_calls']} | {r['llm_calls']} | {sum((r.get('guardrail_failures') or {}).values())} | "
                     f"{r['wall_seconds']} |")
    if prom:
        lines += ["", "## Cluster metrics over the run (Prometheus)", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in prom.items()]
    return "\n".join(line for line in lines if line is not None) + "\n"


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="run the crew on one question")
    a.add_argument("question", nargs="?")
    a.add_argument("--qid", help="a question id from data/questions.json (graded)")
    a.add_argument("-v", "--verbose", action="store_true", help="print the full agent trace")
    a.add_argument("--json-out", help=argparse.SUPPRESS)
    a.add_argument("--trace-out", help="write one JSON line per LLM call (agent, latency, tokens, response)")
    e = sub.add_parser("eval", help="run many graded questions concurrently")
    e.add_argument("--qids", help="comma-separated ids (default: all)")
    e.add_argument("--repeat", type=int, default=1)
    e.add_argument("--concurrency", type=int, default=4)
    e.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    e.add_argument("--prometheus", help="e.g. http://localhost:9090 (from `laptop.sh tunnel`)")
    e.add_argument("--label", help='what this run is, for the report, e.g. "C no-kv-tier" or "B sglang"')
    sub.add_parser("aggregate", help="rebuild agent_runs.csv / agent_questions.csv from every run folder")
    args = ap.parse_args()
    {"ask": cmd_ask, "eval": cmd_eval, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
