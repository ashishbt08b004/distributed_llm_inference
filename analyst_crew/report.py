"""Failure classification and CSV aggregation for eval runs. Stdlib only.

Every `run.py eval` writes one folder, bench-results/agent/<run_id>/, with
meta.json (settings, label, cluster metrics) and results.jsonl (one record per
question run). `aggregate()` rebuilds two report-ready CSVs from ALL folders:

  bench-results/agent/agent_runs.csv       one row per eval run: settings, accuracy
                                           (overall and by difficulty), failure counts,
                                           call/token/latency totals, cluster metrics
  bench-results/agent/agent_questions.csv  one row per question run: the same settings
                                           plus outcome, failure reason, per-tool counts

Rebuilding from the folders (instead of appending) keeps the CSVs consistent:
delete a bad run folder and re-run `python run.py aggregate`.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

TOOL_NAMES = ["calculator", "column_stats", "describe_table", "distinct_values", "list_tables",
              "read_notes", "run_sql", "sample_rows", "save_note"]
# A tool call written as text: Qwen's tags, a leaked chat-template token, or JSON naming a tool.
TEXT_TOOL_CALL = re.compile(
    r"<tool_call>|<\|im_start\|>|<\|im_end\|>|\"name\"\s*:\s*\"(" + "|".join(TOOL_NAMES) + r")\"")

FAILURES = ["error", "text_tool_call", "no_final_answer", "no_sql", "wrong_value"]
FAILURE_HELP = {
    "error": "the run crashed or an LLM call failed",
    "text_tool_call": "the final text is a tool call written as text/JSON, not an answer",
    "no_final_answer": "no FINAL_ANSWER line",
    "no_sql": "answered without running a single query (made-up numbers)",
    "wrong_value": "ran queries and answered, but the value is wrong",
}


def classify(res: dict) -> str:
    """Why a question run failed ('' if it was correct). First matching reason wins."""
    if res.get("correct"):
        return ""
    if res.get("error"):
        return "error"
    if TEXT_TOOL_CALL.search(res.get("answer_text") or ""):
        return "text_tool_call"
    if res.get("final") is None:
        return "no_final_answer"
    by = res.get("tool_calls_by_name") or {}
    if by.get("run_sql", 0) + by.get("column_stats", 0) == 0:
        return "no_sql"
    return "wrong_value"


# ----------------------------------------------------------------------------- aggregation
SETTINGS = ["label", "model", "base_url", "temperature", "top_p", "top_k", "repetition_penalty",
            "max_tokens", "max_iter", "guardrail_retries", "concurrency", "repeat"]


def _pct(xs: list, q: float):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def _rate(rows: list[dict]) -> float | None:
    return round(sum(bool(r.get("correct")) for r in rows) / len(rows), 4) if rows else None


def load_run(folder: Path) -> tuple[dict, list[dict]] | None:
    res_file = folder / "results.jsonl"
    if not res_file.exists():
        return None
    meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
    meta.setdefault("run_id", folder.name)
    results = [json.loads(line) for line in res_file.read_text().splitlines() if line.strip()]
    for r in results:
        r.setdefault("failure", classify(r))
    return meta, results


def run_row(meta: dict, results: list[dict]) -> dict:
    done = [r for r in results if r.get("wall_seconds") is not None]
    elapsed = meta.get("elapsed_s")
    tokens_in = sum(r.get("prompt_tokens") or 0 for r in done)
    row = {"run_id": meta["run_id"], "started_at": meta.get("started_at", ""),
           **{k: meta.get(k, "") for k in SETTINGS},
           "n_runs": len(results), "correct": sum(bool(r.get("correct")) for r in results),
           "accuracy": _rate(results)}
    for d in ("easy", "medium", "hard"):
        row[f"accuracy_{d}"] = _rate([r for r in results if r.get("difficulty") == d])
    for f in FAILURES:
        row[f"fail_{f}"] = sum(r.get("failure") == f for r in results)
    tc = [r.get("tool_calls") or 0 for r in done]
    lc = [r.get("llm_calls") or 0 for r in done]
    row.update({
        "tool_calls_total": sum(tc), "tool_calls_mean": round(statistics.fmean(tc), 2) if tc else None,
        "llm_calls_total": sum(lc), "llm_calls_mean": round(statistics.fmean(lc), 2) if lc else None,
        "llm_failures": sum(r.get("llm_failures") or 0 for r in done),
        "guardrail_retries": sum(sum((r.get("guardrail_failures") or {}).values()) for r in done),
        "prompt_tokens": tokens_in, "completion_tokens": sum(r.get("completion_tokens") or 0 for r in done),
        "elapsed_s": elapsed,
        "wall_p50_s": _pct([r["wall_seconds"] for r in done], .5),
        "wall_p95_s": _pct([r["wall_seconds"] for r in done], .95),
        "runs_per_min": round(60 * len(results) / elapsed, 2) if elapsed else None,
        "prompt_tok_per_s": round(tokens_in / elapsed) if elapsed else None,
    })
    for k, v in (meta.get("prometheus") or {}).items():
        row[f"prom_{k}"] = v
    return row


def question_rows(meta: dict, results: list[dict]) -> list[dict]:
    rows = []
    for r in sorted(results, key=lambda x: (x.get("qid", ""), x.get("repeat", 0))):
        by = r.get("tool_calls_by_name") or {}
        rows.append({
            "run_id": meta["run_id"], **{k: meta.get(k, "") for k in SETTINGS},
            "qid": r.get("qid"), "difficulty": r.get("difficulty"), "repeat": r.get("repeat"),
            "correct": int(bool(r.get("correct"))), "failure": r.get("failure", ""),
            "expected": r.get("expected"), "got": r.get("final"),
            "tool_calls": r.get("tool_calls"), **{f"tool_{t}": by.get(t, 0) for t in TOOL_NAMES},
            "llm_calls": r.get("llm_calls"), "llm_failures": r.get("llm_failures"),
            "guardrail_retries": sum((r.get("guardrail_failures") or {}).values()),
            "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"),
            "wall_s": r.get("wall_seconds"), "llm_p50_s": r.get("llm_seconds_p50"),
            "llm_max_s": r.get("llm_seconds_max"), "error": r.get("error") or "",
        })
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    cols: list[str] = []
    for r in rows:                      # union of columns, in first-seen order
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate(agent_root: Path) -> tuple[Path, Path, int]:
    runs, questions = [], []
    for folder in sorted(p for p in agent_root.iterdir() if p.is_dir()):
        loaded = load_run(folder)
        if loaded:
            meta, results = loaded
            runs.append(run_row(meta, results))
            questions += question_rows(meta, results)
    runs_csv, q_csv = agent_root / "agent_runs.csv", agent_root / "agent_questions.csv"
    _write(runs_csv, runs)
    _write(q_csv, questions)
    return runs_csv, q_csv, len(runs)
