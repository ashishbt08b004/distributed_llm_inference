"""Tools the crew can call. All local: SQLite (read-only), a safe calculator, a scratchpad.

Every tool returns a short string and never raises. Errors come back as text so
the agent can read them and retry, which is part of what drives the tool-call
count up. Outputs are truncated to keep the growing agent context inside the
model's window.
"""
from __future__ import annotations

import ast
import functools
import math
import operator
import re
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

from crewai.tools import tool

DB_PATH = Path(__file__).resolve().parent / "data" / "shop.db"
MAX_ROWS = 40
MAX_CHARS = 2000

CALLS: Counter[str] = Counter()     # tool name -> calls in this process
NOTES: dict[str, str] = {}          # shared scratchpad for the crew
_SQL_AT_ANALYSIS: int | None = None  # SQL calls made by the time the analysis passed its guardrail


def sql_calls() -> int:
    """Queries actually run against the data (the guardrails require some)."""
    return CALLS["run_sql"] + CALLS["column_stats"]


def mark_analysis_done() -> None:
    global _SQL_AT_ANALYSIS
    _SQL_AT_ANALYSIS = sql_calls()


def sql_calls_since_analysis() -> int:
    return sql_calls() - (_SQL_AT_ANALYSIS or 0)


def _counted(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        CALLS[fn.__name__] += 1
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — the agent should see the error, not crash
            return f"ERROR: {type(e).__name__}: {e}"
    return wrapper


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} missing: run `python make_data.py` first")
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + f"\n... [truncated, {len(text)} chars total]"


def _table(cur: sqlite3.Cursor, limit: int = MAX_ROWS) -> str:
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(limit + 1)
    more = len(rows) > limit
    lines = [" | ".join(cols)] + [" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:limit]]
    if more:
        lines.append(f"... more rows not shown (limit {limit}); aggregate or add LIMIT/WHERE")
    return _clip("\n".join(lines)) if rows else " | ".join(cols) + "\n(0 rows)"


def _schema_hint(con: sqlite3.Connection, err: Exception) -> str:
    """SQL error + the real table names, so a guessed name gets corrected instead of abandoned."""
    msg = f"ERROR: {err}"
    if "no such" in str(err):
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'data_dictionary' ORDER BY name")]
        msg += (f". The real tables are: {', '.join(tables)}. Call describe_table(<table>) for its columns "
                "and fix the query; the data you need is in these tables.")
    return msg


_READ_ONLY = re.compile(r"^\s*(select|with)\b", re.I)
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I)


def _check_sql(query: str) -> str:
    q = query.strip().rstrip(";")
    if not _READ_ONLY.match(q) or _FORBIDDEN.search(q) or ";" in q:
        raise ValueError("only a single read-only SELECT / WITH statement is allowed")
    return q


# ----------------------------------------------------------------------------- schema exploration
@tool("list_tables")
@_counted
def list_tables() -> str:
    """List every table in the shop database with its row count."""
    with _connect() as con:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return "\n".join(f"{n}: {con.execute(f'SELECT COUNT(*) FROM {n}').fetchone()[0]} rows" for n in names)


@tool("describe_table")
@_counted
def describe_table(table_name: str) -> str:
    """Show the columns of one table, their types, and their business meaning from the data dictionary.
    Use table_name='_metrics' to read the official metric definitions (net revenue, margin, AOV)."""
    with _connect() as con:
        docs = dict(con.execute("SELECT column_name, description FROM data_dictionary WHERE table_name=?",
                                (table_name,)).fetchall())
        if table_name == "_metrics":
            return "\n".join(f"{k}: {v}" for k, v in docs.items())
        cols = con.execute(f"PRAGMA table_info({table_name})").fetchall()
        if not cols:
            return f"ERROR: no table named {table_name!r}. Call list_tables."
        return "\n".join(f"{c[1]} {c[2]}" + (f"  -- {docs[c[1]]}" if c[1] in docs else "") for c in cols)


@tool("sample_rows")
@_counted
def sample_rows(table_name: str, n: int = 5) -> str:
    """Return the first n rows (max 10) of a table, to see what real values look like."""
    if not re.fullmatch(r"[A-Za-z_]+", table_name):
        return "ERROR: invalid table name"
    with _connect() as con:
        return _table(con.execute(f"SELECT * FROM {table_name} LIMIT ?", (min(int(n), 10),)))


@tool("distinct_values")
@_counted
def distinct_values(table_name: str, column_name: str) -> str:
    """List the distinct values of a column with their counts (top 25). Useful before filtering on text columns."""
    if not re.fullmatch(r"[A-Za-z_]+", table_name) or not re.fullmatch(r"[A-Za-z_]+", column_name):
        return "ERROR: invalid table or column name"
    with _connect() as con:
        return _table(con.execute(f"SELECT {column_name}, COUNT(*) AS n FROM {table_name} "
                                  f"GROUP BY 1 ORDER BY n DESC LIMIT 25"))


# ----------------------------------------------------------------------------- analysis
@tool("run_sql")
@_counted
def run_sql(query: str) -> str:
    """Run ONE read-only SQLite SELECT (or WITH ... SELECT) and return at most 40 rows.
    SQLite dialect: dates are ISO text, so use BETWEEN '2025-01-01' AND '2025-03-31', LIKE '2024-%',
    strftime(), julianday(). Aggregate in SQL instead of pulling raw rows."""
    with _connect() as con:
        try:
            return _table(con.execute(_check_sql(query)))
        except sqlite3.OperationalError as e:
            return _schema_hint(con, e)


@tool("column_stats")
@_counted
def column_stats(query: str) -> str:
    """Run a read-only SELECT that returns ONE numeric column, and get count, mean, median, min, max,
    p90 and stdev of it. Use this for medians and percentiles, which SQLite cannot compute."""
    with _connect() as con:
        try:
            vals = [r[0] for r in con.execute(_check_sql(query)).fetchall() if r and r[0] is not None]
        except sqlite3.OperationalError as e:
            return _schema_hint(con, e)
    if not vals:
        return "no non-NULL values"
    vals = [float(v) for v in vals]
    s = sorted(vals)
    p90 = s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (f"count={len(vals)} mean={statistics.fmean(vals):.4f} median={statistics.median(vals):.4f} "
            f"min={s[0]:.4f} max={s[-1]:.4f} p90={p90:.4f} stdev={sd:.4f}")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("only numbers, + - * / ** %, and round/abs/min/max/sqrt/log are allowed")


@tool("calculator")
@_counted
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(1520.5 - 1300) / 1300 * 100' or 'round(2/3, 2)'."""
    return str(_eval(ast.parse(expression, mode="eval")))


# ----------------------------------------------------------------------------- scratchpad
@tool("save_note")
@_counted
def save_note(key: str, value: str) -> str:
    """Save an intermediate finding (a number, a query that worked, a definition) under a short key,
    so other crew members can read it with read_notes."""
    NOTES[key] = value[:500]
    return f"saved {key!r} ({len(NOTES)} notes)"


@tool("read_notes")
@_counted
def read_notes() -> str:
    """Read every note saved so far by the crew."""
    return "\n".join(f"{k}: {v}" for k, v in NOTES.items()) or "(no notes yet)"


EXPLORE = [list_tables, describe_table, sample_rows, distinct_values, save_note]
ANALYZE = [list_tables, describe_table, run_sql, column_stats, calculator, save_note, read_notes]
REVIEW = [list_tables, describe_table, run_sql, column_stats, calculator, read_notes]
TOOL_NAMES = sorted({t.name for t in EXPLORE + ANALYZE + REVIEW})
