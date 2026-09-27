#!/usr/bin/env python3
"""Scripted OpenAI-compatible chat server for testing the crew without a model.

Answers question q01 ("customers in the West region") by replaying a fixed
tool-call script per agent (recognised from the system prompt), so the test
exercises real tool execution, CrewAI's tool loop, counting and grading.
Stdlib only.   python tests/fake_llm.py 18999 [--misbehave]

--misbehave: the SQL Analyst first answers with a tool call written as JSON text
(what Qwen2.5-7B did on the cluster) until the guardrail's feedback shows up in
its prompt; then it behaves. Tests that the guardrail makes the agent retry.
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEST = "SELECT COUNT(*) AS n FROM customers WHERE region = 'West'"
SCRIPT = {
    "Data Planner": [("list_tables", {}), ("describe_table", {"table_name": "customers"}),
                     ("distinct_values", {"table_name": "customers", "column_name": "region"})],
    "SQL Analyst": [("run_sql", {"query": WEST}), ("save_note", {"key": "answer", "value": "see last query"})],
    "Reviewer": [("read_notes", {}), ("run_sql", {"query": WEST}), ("calculator", {"expression": "100 + 3"})],
}
FINAL = {
    "Data Planner": "1. customers table, filter region = 'West'. 2. COUNT(*).",
    "SQL Analyst": "Counted customers where region = 'West'.",
}


MISBEHAVE = "--misbehave" in sys.argv


def reply(body: dict) -> dict:
    msgs = body["messages"]
    role = next((r for r in SCRIPT if msgs[0]["content"].startswith(f"You are {r}")), None)
    done = sum(1 for m in msgs if m["role"] == "tool")
    corrected = any("written as text" in str(m.get("content") or "") for m in msgs)
    if MISBEHAVE and role == "SQL Analyst" and not corrected:
        text = '```json\n{"name": "run_sql", "arguments": {"query": "SELECT 1"}}\n```'
        return {"id": "fake", "object": "chat.completion", "created": 0, "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
    steps = SCRIPT.get(role, [])
    if done < len(steps) and body.get("tools"):
        name, args = steps[done]
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_{done}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}
        finish = "tool_calls"
    else:
        if role == "Reviewer":
            counts = [re.search(r"^n\n(\d+)$", m["content"] or "", re.M) for m in msgs if m["role"] == "tool"]
            n = next((c.group(1) for c in reversed(counts) if c), "unknown")
            text = f"Both queries agree.\nFINAL_ANSWER: {n}"
        else:
            text = FINAL.get(role, "ok")
        msg, finish = {"role": "assistant", "content": text}, "stop"
    prompt = sum(len(str(m.get("content") or "")) for m in msgs) // 4
    return {"id": "fake", "object": "chat.completion", "created": 0, "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": 20, "total_tokens": prompt + 20}}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        data = json.dumps(reply(body)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
