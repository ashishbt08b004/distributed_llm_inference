"""The data analyst crew: planner -> analyst -> reviewer, run sequentially.

Every agent talks to the same OpenAI-compatible endpoint (the Blueprint 1
LiteLLM gateway by default) and re-sends a long, growing prompt on each step,
so a run is a burst of prefix-sharing requests: exactly what the cluster's
prefix cache, kv_router affinity and admission queue are built for.

Lessons from the first cluster run (1/12 correct) are built in:
  * Greedy decoding (temperature 0) sent Qwen2.5-7B into repetition loops
    ('<|im_start|><|im_start|>...'). Sampling now follows Qwen's recommended
    settings and generation stops at '<|im_start|>'.
  * The model sometimes wrote a tool call as text/JSON instead of calling the
    tool; CrewAI took that text as the final answer. Guardrails now reject it.
  * Agents answered without running any SQL (invented columns and numbers).
    Guardrails now require real queries from the analyst AND the reviewer.
  * Second run (3/12): Qwen batched 5-14 tool calls into one reply; one malformed
    or cut-off call makes vLLM's hermes parser return the whole batch as text. So
    the Planner never saw the schema and everyone guessed table/column names.
    Generation now stops at '</tool_call>' (vLLM still parses the unclosed call),
    so each reply carries exactly one tool call. The Planner must actually look at
    the schema, every agent can, and SQL errors list the real table names.
A failed guardrail sends the feedback back to the agent, which retries the task.
"""
from __future__ import annotations

import os
import re
from collections import Counter

from crewai import LLM, Agent, Crew, Process, Task

import tools
from report import TEXT_TOOL_CALL

FINAL_MARKER = "FINAL_ANSWER:"

GUARDRAIL_FAILURES: Counter[str] = Counter()   # "<task>:<reason>" -> rejections (reported per run)
_STOP = ["<|im_start|>", "<|endoftext|>"]
# Server-side only: end every reply after its first tool call. vLLM's hermes parser
# accepts the unclosed '<tool_call>{...}' at the end of the output.
_STOP_SERVER = _STOP + ["</tool_call>"]
# Final values that mean "I could not work it out".
_NON_ANSWER = re.compile(r"^(null|none|n/?a|nan|unknown|-|\?|not (available|applicable|found))$", re.I)
_UNGROUNDED = re.compile(r"hypothetical|let'?s assume|assuming the query|schema (is|isn'?t) (not )?available", re.I)


def sampling() -> dict:
    """Sampling settings (also recorded with every eval run)."""
    return {
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.3")),
        "top_p": float(os.environ.get("LLM_TOP_P", "0.8")),
        "top_k": int(os.environ.get("LLM_TOP_K", "20")),
        "repetition_penalty": float(os.environ.get("LLM_REPETITION_PENALTY", "1.05")),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1024")),
    }


def make_llm() -> LLM:
    s = sampling()
    return LLM(
        model=os.environ.get("LLM_MODEL", "hosted_vllm/qwen7b"),
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:4000/v1"),
        api_key=os.environ.get("LLM_API_KEY") or "dummy",
        temperature=s["temperature"], top_p=s["top_p"], max_tokens=s["max_tokens"],
        timeout=float(os.environ.get("LLM_TIMEOUT", "300")),
        stop=_STOP,
        # vLLM-specific sampling rides in the request body (LiteLLM passes it through).
        # `stop` is repeated here because CrewAI applies `stop` only client-side.
        additional_params={"extra_body": {"top_k": s["top_k"], "repetition_penalty": s["repetition_penalty"],
                                          "stop": _STOP_SERVER}},
    )


# ----------------------------------------------------------------------------- guardrails
def _text_tool_call(text: str) -> str | None:
    if TEXT_TOOL_CALL.search(text or ""):
        return ("Your answer contains a tool call written as text (JSON or <tool_call> tags). Text is not "
                "executed. Call the tool through the tool-calling interface, wait for its result, and only "
                "then write your answer in plain prose.")
    return None


def _check_plan(output):  # -> (ok, output or feedback)
    text = output.raw or ""
    if problem := _text_tool_call(text):
        GUARDRAIL_FAILURES["plan:text_tool_call"] += 1
        return False, problem
    if tools.CALLS["describe_table"] == 0:
        GUARDRAIL_FAILURES["plan:no_schema"] += 1
        return False, ("You have not looked at the schema. Call list_tables, then describe_table for each "
                       "table the question needs, one tool call at a time, and base the plan on what they show.")
    return True, text


def _check_analysis(output):  # -> (ok, output or feedback); CrewAI rejects a string annotation
    text = output.raw or ""
    if problem := _text_tool_call(text):
        GUARDRAIL_FAILURES["analysis:text_tool_call"] += 1
        return False, problem
    if tools.sql_calls() == 0:
        GUARDRAIL_FAILURES["analysis:no_sql"] += 1
        return False, ("You have not run any query. Use run_sql (or column_stats) on the real tables and base "
                       "the answer only on the rows it returns. Never invent table names, columns or numbers.")
    tools.mark_analysis_done()
    return True, text


def _check_review(output):  # -> (ok, output or feedback)
    text = output.raw or ""
    if problem := _text_tool_call(text):
        GUARDRAIL_FAILURES["review:text_tool_call"] += 1
        return False, problem
    if tools.sql_calls_since_analysis() == 0:
        GUARDRAIL_FAILURES["review:no_sql"] += 1
        return False, ("You have not verified anything yourself. Re-run the computation with run_sql or "
                       "column_stats (a differently written query), compare, then answer.")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines or not re.fullmatch(rf"{FINAL_MARKER}\s*\S.*", lines[-1].strip("*` ")):
        GUARDRAIL_FAILURES["review:no_final_answer"] += 1
        return False, (f"The last line must be exactly '{FINAL_MARKER} <value>': a plain number, or the name "
                       "if the question asks 'which'.")
    value = lines[-1].strip("*` ")[len(FINAL_MARKER):].strip()
    if _NON_ANSWER.match(value) or _UNGROUNDED.search(text):
        GUARDRAIL_FAILURES["review:ungrounded"] += 1
        return False, ("That is not an answer from the data. The data exists: call list_tables and "
                       "describe_table to find the real table and column names, run the query, and answer "
                       "with the value it returns. Never assume or invent results.")
    return True, text


def build_crew(question: str, verbose: bool = False) -> Crew:
    llm = make_llm()
    max_iter = int(os.environ.get("AGENT_MAX_ITER", "15"))
    retries = int(os.environ.get("GUARDRAIL_RETRIES", "2"))
    common = dict(llm=llm, allow_delegation=False, verbose=verbose, max_iter=max_iter)
    tool_rule = ("Call exactly one tool per reply and wait for its result before the next. Always use the "
                 "tool-calling interface; never write a tool call as text or JSON. Use only table and column "
                 "names that list_tables / describe_table have shown you; if a query fails with 'no such "
                 "table/column', look the name up with describe_table and fix the query.")
    literal = ("Answer the question exactly as worded, with the most direct query on the table that holds that "
               "entity (for example 'customers in a region' means rows of the customers table). If two of your "
               "queries disagree, the one that matches the literal wording and the data dictionary wins.")

    planner = Agent(
        role="Data Planner",
        goal="Understand exactly which tables, columns, filter values and metric definitions a question needs",
        backstory="You never guess a schema. You look at the tables, read the data dictionary and the "
                  "official metric definitions, and check real column values before anyone writes a query. "
                  + tool_rule,
        tools=tools.EXPLORE, **common)
    analyst = Agent(
        role="SQL Analyst",
        goal="Answer the question with correct SQLite queries, checking intermediate results",
        backstory="You write small queries first to check row counts and filters, then build up the final "
                  "query. When a query fails you read the error and fix it. You save what worked as notes. "
                  + tool_rule,
        tools=tools.ANALYZE, **common)
    reviewer = Agent(
        role="Reviewer",
        goal="Independently verify the analyst's answer and state the final answer",
        backstory="You distrust numbers until you have reproduced them yourself with a query. You check that "
                  "the metric definition, date range and status filter match the question exactly. "
                  + tool_rule,
        tools=tools.REVIEW, **common)

    plan = Task(
        description=(
            f"Question: {question}\n\n"
            "Explore the shop database before any analysis: call list_tables, describe every table the "
            "question could need, describe '_metrics' for the official metric definitions, and use "
            "distinct_values on every text column you will filter on. Save the key facts with save_note.\n"
            "Do NOT compute the answer."),
        expected_output="A numbered query plan: tables, join keys, exact filter values, date range, "
                        "and the metric formula to use.",
        agent=planner, guardrail=_check_plan, guardrail_max_retries=retries)
    analyze = Task(
        description=(
            f"Question: {question}\n\n"
            "Follow the plan. Check filters and row counts with small queries first, then compute the answer "
            "with run_sql (or column_stats for medians / percentiles, calculator for arithmetic). "
            "Every number you report must come from a tool result. " + literal + " "
            "Save the final number and the query that produced it with save_note."),
        expected_output="The answer, the exact SQL that produced it, and the intermediate checks you ran.",
        agent=analyst, context=[plan], guardrail=_check_analysis, guardrail_max_retries=retries)
    review = Task(
        description=(
            f"Question: {question}\n\n"
            "Read the notes, then verify the analyst's answer yourself: run at least one independent query "
            "with run_sql or column_stats (for example a differently written query), and compare. If they "
            "disagree, find out which is right. " + literal + " Round as the question asks.\n"
            f"The LAST line of your answer must be exactly: {FINAL_MARKER} <value>\n"
            "where <value> is a plain number (no units, %, $ or thousands separators), or, when the question "
            "asks 'which ...', the name itself (not a number)."),
        expected_output=f"Short verification summary, then a last line '{FINAL_MARKER} <value>'.",
        agent=reviewer, context=[plan, analyze], guardrail=_check_review, guardrail_max_retries=retries)

    return Crew(agents=[planner, analyst, reviewer], tasks=[plan, analyze, review],
                process=Process.sequential, verbose=verbose)
