#!/usr/bin/env python3
"""Build the ablation results table from a results directory.

Expects, per config X and pattern P (rate1, rate8, rateinf, multiturn, sharegpt, or any
rate<N> such as the quick profile's rate4):
  results_X_P.json            vllm bench serve / multiturn_bench.py output
  metrics_X_P_before.txt      scrape_metrics.py snapshot before the run
  metrics_X_P_after.txt       ... and after

Writes summary.md and summary.csv next to them and prints the markdown.
Standard library only, so it runs anywhere (node, laptop, bench pod).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

PATTERNS = ["rate1", "rate8", "rateinf", "multiturn", "sharegpt"]
LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)$")


def parse_metrics(path: Path) -> dict[str, dict[str, float]]:
    """{target: {metric_name: summed value over label sets}}"""
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    if not path.exists():
        return {}
    target = "?"
    for raw in path.read_text().splitlines():
        if raw.startswith("# TARGET "):
            target = raw[9:].strip()
            continue
        if not raw or raw.startswith("#"):
            continue
        m = LINE.match(raw.strip())
        if m:
            try:
                out[target][m.group(1)] += float(m.group(3))
            except ValueError:
                pass
    return out


def deltas(before: dict, after: dict) -> dict[str, dict[str, float]]:
    d = {}
    for target, metrics in after.items():
        b = before.get(target, {})
        d[target] = {k: v - b.get(k, 0.0) for k, v in metrics.items()}
    return d


def sum_where(d: dict, pred) -> float | None:
    vals = [v for metrics in d.values() for k, v in metrics.items() if pred(k)]
    return sum(vals) if vals else None


def server_stats(d: dict) -> dict[str, float | None]:
    vllm = {t: m for t, m in d.items() if not t.endswith(":9003")}
    moon = {t: m for t, m in d.items() if t.endswith(":9003")}

    def is_(suffix, external):
        return lambda k: k.endswith(suffix) and ("external" in k) == external and k.startswith("vllm:")

    hits = sum_where(vllm, is_("prefix_cache_hits_total", False))
    queries = sum_where(vllm, is_("prefix_cache_queries_total", False))
    ext_hits = sum_where(vllm, is_("prefix_cache_hits_total", True))
    ext_queries = sum_where(vllm, is_("prefix_cache_queries_total", True))
    preempt = sum_where(vllm, lambda k: k == "vllm:num_preemptions_total")
    moon_ops = sum_where(moon, lambda k: re.search(r"(put|get)", k, re.I) is not None
                         and re.search(r"(total|count)$", k) is not None and not k.endswith("_bucket"))
    return {
        "prefix_hit_pct": 100 * hits / queries if hits is not None and queries else None,
        "external_hit_pct": 100 * ext_hits / ext_queries if ext_hits is not None and ext_queries else None,
        "preemptions": preempt,
        "mooncake_ops": moon_ops,
    }


def fmt(v, digits=0, suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:,.{digits}f}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="/bench/results")
    args = ap.parse_args()
    rdir = Path(args.results_dir)

    rows = []
    for f in sorted(rdir.glob("results_*_*.json")):
        m = re.match(r"results_([A-Z])_(\w+)\.json$", f.name)
        if not m:
            continue
        cfg, pattern = m.groups()
        r = json.loads(f.read_text())
        d = deltas(parse_metrics(rdir / f"metrics_{cfg}_{pattern}_before.txt"),
                   parse_metrics(rdir / f"metrics_{cfg}_{pattern}_after.txt"))
        rows.append({
            "config": cfg, "pattern": pattern,
            "completed": r.get("completed"), "failed": r.get("failed", 0),
            "ttft_p50_ms": r.get("median_ttft_ms"), "ttft_p95_ms": r.get("p95_ttft_ms"),
            "itl_p95_ms": r.get("p95_itl_ms"), "tpot_mean_ms": r.get("mean_tpot_ms"),
            "e2el_p95_ms": r.get("p95_e2el_ms"),
            "out_tok_s": r.get("output_throughput"), "req_s": r.get("request_throughput"),
            **server_stats(d),
            "per_turn": r.get("per_turn"),
        })
    if not rows:
        raise SystemExit(f"no results_*.json in {rdir}")
    def pattern_key(p: str):
        # Known patterns in their usual order; other rate<N> by N; anything else last.
        if p in PATTERNS:
            return (PATTERNS.index(p), 0.0, p)
        m = re.fullmatch(r"rate(\d+(?:\.\d+)?)", p)
        return (0.5, float(m.group(1)), p) if m else (99, 0.0, p)

    rows.sort(key=lambda x: (pattern_key(x["pattern"]), x["config"]))
    present = sorted({x["pattern"] for x in rows}, key=pattern_key)

    md = ["# Blueprint 1 — ablation results", "",
          "| Pattern | Config | OK | TTFT p50 (ms) | TTFT p95 (ms) | ITL p95 (ms) | E2E p95 (ms) | Out tok/s | "
          "Prefix hit (HBM) | Connector hit (LMCache) | Preemptions | Mooncake ops |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['pattern']} | {x['config']} | {x['completed']}"
                  f"{'/' + str(x['failed']) + ' failed' if x['failed'] else ''} | "
                  f"{fmt(x['ttft_p50_ms'])} | {fmt(x['ttft_p95_ms'])} | {fmt(x['itl_p95_ms'], 1)} | "
                  f"{fmt(x['e2el_p95_ms'])} | {fmt(x['out_tok_s'])} | {fmt(x['prefix_hit_pct'], 1, '%')} | "
                  f"{fmt(x['external_hit_pct'], 1, '%')} | {fmt(x['preemptions'])} | {fmt(x['mooncake_ops'])} |")

    # Relative change between neighbouring configs, per pattern.
    by = {(x["config"], x["pattern"]): x for x in rows}
    md += ["", "## Layer contribution (relative change)", "",
           "| Pattern | Step | TTFT p95 | ITL p95 | Out tok/s |", "|---|---|---|---|---|"]
    for p in present:
        for a, b in (("A", "B"), ("B", "C")):
            if (a, p) in by and (b, p) in by:
                xa, xb = by[(a, p)], by[(b, p)]

                def rel(k):
                    va, vb = xa.get(k), xb.get(k)
                    return fmt(100 * (vb - va) / va, 0, "%") if va and vb is not None else "n/a"
                md.append(f"| {p} | {a} → {b} | {rel('ttft_p95_ms')} | {rel('itl_p95_ms')} | {rel('out_tok_s')} |")

    turns = [x for x in rows if x["per_turn"]]
    if turns:
        md += ["", "## Multi-turn: mean TTFT by turn (ms)", "",
               "| Config | " + " | ".join(f"turn {t}" for t in turns[0]["per_turn"]) + " |",
               "|---|" + "---|" * len(turns[0]["per_turn"])]
        for x in turns:
            md.append(f"| {x['config']} | " + " | ".join(
                f"{fmt(v['mean_ttft_ms'])} (~{fmt(v['mean_prompt_tokens'])} tok)" for v in x["per_turn"].values()) + " |")

    text = "\n".join(md) + "\n"
    (rdir / "summary.md").write_text(text)
    keys = [k for k in rows[0] if k != "per_turn"]
    with open(rdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(text)
    print(f"wrote {rdir / 'summary.md'} and {rdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
