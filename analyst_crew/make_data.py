#!/usr/bin/env python3
"""Build the synthetic "shop" SQLite database and the question set with ground truth.

    python make_data.py            # -> data/shop.db, data/questions.json

Deterministic (seeded), stdlib only. The data has a few traps on purpose so the
agents have to explore before they query: revenue only counts *completed*
orders and must apply the per-order discount, cost lives on `products`, dates
are ISO text, and `data_dictionary` documents all of it.
"""
from __future__ import annotations

import json
import random
import sqlite3
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DB_PATH = DATA / "shop.db"
QUESTIONS_PATH = DATA / "questions.json"

REGIONS = ["North", "South", "East", "West", "Central"]
SEGMENTS = [("consumer", 0.70), ("small_business", 0.22), ("enterprise", 0.08)]
CATEGORIES = {
    "Electronics": (80, 900, 0.62), "Home": (15, 250, 0.48), "Sports": (20, 300, 0.52),
    "Books": (8, 60, 0.40), "Beauty": (6, 90, 0.35), "Toys": (10, 120, 0.45),
}
CHANNELS = [("web", 0.5), ("mobile", 0.35), ("store", 0.15)]
TICKET_CATS = ["shipping", "billing", "product", "account"]

SCHEMA = """
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY, name TEXT, region TEXT, segment TEXT, signup_date TEXT);
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, list_price REAL, unit_cost REAL);
CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers, order_date TEXT,
  status TEXT, channel TEXT, discount_pct REAL);
CREATE TABLE order_items (
  order_id INTEGER REFERENCES orders, line_no INTEGER, product_id INTEGER REFERENCES products,
  quantity INTEGER, unit_price REAL, PRIMARY KEY (order_id, line_no));
CREATE TABLE support_tickets (
  ticket_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers, category TEXT,
  priority TEXT, opened_at TEXT, resolved_at TEXT, satisfaction INTEGER);
CREATE TABLE data_dictionary (table_name TEXT, column_name TEXT, description TEXT);
"""

DICTIONARY = [
    ("customers", "customer_id", "Primary key"),
    ("customers", "region", "Sales region: North, South, East, West, Central"),
    ("customers", "segment", "consumer | small_business | enterprise"),
    ("customers", "signup_date", "ISO date (YYYY-MM-DD) the account was created"),
    ("products", "category", "Electronics, Home, Sports, Books, Beauty, Toys"),
    ("products", "list_price", "Current catalogue price. NOT the price paid; see order_items.unit_price"),
    ("products", "unit_cost", "Cost of goods per unit, used for margin"),
    ("orders", "order_date", "ISO date (YYYY-MM-DD) the order was placed"),
    ("orders", "status", "completed | cancelled | returned. Only 'completed' orders count as revenue"),
    ("orders", "channel", "web | mobile | store"),
    ("orders", "discount_pct", "Whole-order discount in percent (0-30), applies to every line"),
    ("order_items", "unit_price", "Price actually charged per unit, before the order discount"),
    ("order_items", "quantity", "Units on this line"),
    ("support_tickets", "opened_at", "ISO datetime (YYYY-MM-DD HH:MM:SS)"),
    ("support_tickets", "resolved_at", "ISO datetime, NULL while the ticket is open"),
    ("support_tickets", "satisfaction", "1-5 survey score, NULL if the customer did not answer"),
    ("_metrics", "net_revenue", "SUM(quantity * unit_price * (1 - discount_pct/100)) over COMPLETED orders"),
    ("_metrics", "gross_margin_pct", "(net_revenue - SUM(quantity * unit_cost)) / net_revenue * 100, completed orders"),
    ("_metrics", "average_order_value", "net_revenue / number of completed orders"),
]


def _pick(rng: random.Random, weighted: list[tuple[str, float]]) -> str:
    return rng.choices([w[0] for w in weighted], weights=[w[1] for w in weighted])[0]


def _day(start: date, end: date, rng: random.Random) -> date:
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def build(db_path: Path = DB_PATH, seed: int = 42) -> sqlite3.Connection:
    rng = random.Random(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO data_dictionary VALUES (?,?,?)", DICTIONARY)

    first = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Ravi", "Ana", "Kenji", "Omar", "Ivy", "Leo", "Sara"]
    last = ["Patel", "Smith", "Garcia", "Chen", "Okafor", "Silva", "Kim", "Novak", "Haddad", "Brown"]
    customers = []
    for cid in range(1, 501):
        region = rng.choices(REGIONS, weights=[22, 18, 24, 20, 16])[0]
        signup = _day(date(2023, 1, 1), date(2025, 3, 31), rng)
        customers.append((cid, f"{rng.choice(first)} {rng.choice(last)}", region, _pick(rng, SEGMENTS),
                          signup.isoformat()))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    products, pid = [], 0
    for cat, (lo, hi, margin) in CATEGORIES.items():
        for i in range(10):
            pid += 1
            price = round(rng.uniform(lo, hi), 2)
            cost = round(price * (1 - margin) * rng.uniform(0.85, 1.15), 2)
            products.append((pid, f"{cat[:4].upper()}-{i + 1:02d}", cat, price, cost))
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    # Regional growth differs on purpose so "which region grew most" has a clear answer.
    region_boost = {"North": 1.0, "South": 1.35, "East": 0.9, "West": 1.1, "Central": 0.8}
    orders, items, oid = [], [], 0
    for cid, _, region, segment, signup in customers:
        base = {"consumer": 7, "small_business": 12, "enterprise": 22}[segment]
        n = max(0, int(rng.gauss(base, base / 3)))
        signup_d = date.fromisoformat(signup)
        for _ in range(n):
            d = _day(max(signup_d, date(2024, 1, 1)), date(2025, 6, 30), rng)
            if d.year == 2025 and rng.random() > region_boost[region] * 0.75:
                continue
            oid += 1
            channel = _pick(rng, CHANNELS)
            r = rng.random()
            ret_rate = 0.12 if channel == "mobile" else 0.07
            status = "returned" if r < ret_rate else "cancelled" if r < ret_rate + 0.06 else "completed"
            disc = rng.choice([0, 0, 0, 5, 10, 15, 20, 30]) if segment != "enterprise" else rng.choice([10, 15, 20])
            orders.append((oid, cid, d.isoformat(), status, channel, float(disc)))
            for line in range(1, rng.choice([1, 1, 2, 2, 3, 4]) + 1):
                p = rng.choice(products)
                # BEAU-03 is a known-bad product: returns cluster on it.
                if p[1] == "BEAU-03" and status == "completed" and rng.random() < 0.3:
                    orders[-1] = (oid, cid, d.isoformat(), "returned", channel, float(disc))
                    status = "returned"
                qty = rng.choice([1, 1, 1, 2, 2, 3, 5]) * (3 if segment == "enterprise" else 1)
                price = round(p[3] * rng.uniform(0.9, 1.05), 2)
                items.append((oid, line, p[0], qty, price))
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)

    tickets = []
    returned_customers = {o[1] for o in orders if o[3] == "returned"}
    for tid in range(1, 1501):
        cid = rng.randrange(1, 501)
        cat = rng.choice(TICKET_CATS)
        opened = datetime(2024, 1, 1) + timedelta(minutes=rng.randrange(546 * 24 * 60))
        hours = rng.expovariate(1 / {"shipping": 30, "billing": 14, "product": 48, "account": 8}[cat])
        resolved = None if rng.random() < 0.1 else opened + timedelta(hours=hours)
        sat = None
        if resolved and rng.random() < 0.7:
            mu = 3.9 - (0.9 if cid in returned_customers else 0) - min(hours, 96) / 96
            sat = min(5, max(1, round(rng.gauss(mu, 0.8))))
        tickets.append((tid, cid, cat, rng.choice(["low", "medium", "high"]), opened.strftime("%Y-%m-%d %H:%M:%S"),
                        resolved.strftime("%Y-%m-%d %H:%M:%S") if resolved else None, sat))
    con.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?)", tickets)
    con.commit()
    return con


NET = "oi.quantity * oi.unit_price * (1 - o.discount_pct / 100.0)"


def _one(con: sqlite3.Connection, sql: str):
    return con.execute(sql).fetchone()[0]


def questions(con: sqlite3.Connection) -> list[dict]:
    """Each question: id, difficulty, text, answer, type (number|text), tolerance (relative)."""
    q = []

    def add(qid, difficulty, text, answer, kind="number", tol=0.01):
        q.append({"id": qid, "difficulty": difficulty, "question": text, "answer": answer, "type": kind,
                  "tolerance": tol})

    add("q01", "easy", "How many customers are in the West region?",
        _one(con, "SELECT COUNT(*) FROM customers WHERE region='West'"), tol=0)
    add("q02", "medium", "What was total net revenue in Q1 2025 (January-March)? Round to 2 decimals.",
        round(_one(con, f"""SELECT SUM({NET}) FROM orders o JOIN order_items oi USING(order_id)
            WHERE o.status='completed' AND o.order_date BETWEEN '2025-01-01' AND '2025-03-31'"""), 2))
    add("q03", "medium", "Which product category had the highest net revenue in 2024?",
        _one(con, f"""SELECT p.category FROM orders o JOIN order_items oi USING(order_id)
            JOIN products p USING(product_id) WHERE o.status='completed' AND o.order_date LIKE '2024-%'
            GROUP BY p.category ORDER BY SUM({NET}) DESC LIMIT 1"""), "text")
    add("q04", "medium", "Which region had the highest average order value for completed orders in the first half of 2025?",
        _one(con, f"""SELECT c.region FROM orders o JOIN order_items oi USING(order_id)
            JOIN customers c USING(customer_id) WHERE o.status='completed'
            AND o.order_date BETWEEN '2025-01-01' AND '2025-06-30'
            GROUP BY c.region ORDER BY SUM({NET}) / COUNT(DISTINCT o.order_id) DESC LIMIT 1"""), "text")
    add("q05", "medium", "What percentage of orders placed through the mobile channel in 2024 were returned? One decimal.",
        round(_one(con, """SELECT 100.0 * SUM(status='returned') / COUNT(*) FROM orders
            WHERE channel='mobile' AND order_date LIKE '2024-%'"""), 1), tol=0.02)
    add("q06", "hard", "Among products with at least 50 order lines, which product (by name) has the highest "
        "return rate, measured as order lines on returned orders divided by all its order lines?",
        _one(con, """SELECT p.name FROM order_items oi JOIN orders o USING(order_id) JOIN products p USING(product_id)
            GROUP BY p.product_id HAVING COUNT(*) >= 50
            ORDER BY 1.0 * SUM(o.status='returned') / COUNT(*) DESC LIMIT 1"""), "text")
    add("q07", "hard", "What is the average number of completed orders per enterprise customer, counting "
        "enterprise customers with zero completed orders too? Two decimals.",
        round(_one(con, """SELECT AVG(n) FROM (SELECT c.customer_id,
            (SELECT COUNT(*) FROM orders o WHERE o.customer_id=c.customer_id AND o.status='completed') AS n
            FROM customers c WHERE c.segment='enterprise')"""), 2))
    hours = [(datetime.fromisoformat(r) - datetime.fromisoformat(o)).total_seconds() / 3600 for o, r in con.execute(
        "SELECT opened_at, resolved_at FROM support_tickets WHERE category='billing' AND resolved_at IS NOT NULL")]
    add("q08", "hard", "What is the median resolution time, in hours, of resolved billing support tickets? One decimal.",
        round(statistics.median(hours), 1), tol=0.03)
    growth = con.execute(f"""SELECT c.region,
            SUM(CASE WHEN o.order_date BETWEEN '2024-10-01' AND '2024-12-31' THEN {NET} END) AS q4,
            SUM(CASE WHEN o.order_date BETWEEN '2025-01-01' AND '2025-03-31' THEN {NET} END) AS q1
            FROM orders o JOIN order_items oi USING(order_id) JOIN customers c USING(customer_id)
            WHERE o.status='completed' GROUP BY c.region""").fetchall()
    add("q09", "hard", "Which region's net revenue grew the most, in percentage terms, from Q4 2024 to Q1 2025?",
        max(growth, key=lambda r: (r[2] - r[1]) / r[1])[0], "text")
    add("q10", "hard", "What was the gross margin percentage for the Electronics category in 2024? One decimal.",
        round(_one(con, f"""SELECT 100.0 * (SUM({NET}) - SUM(oi.quantity * p.unit_cost)) / SUM({NET})
            FROM orders o JOIN order_items oi USING(order_id) JOIN products p USING(product_id)
            WHERE o.status='completed' AND p.category='Electronics' AND o.order_date LIKE '2024-%'"""), 1), tol=0.02)
    add("q11", "hard", "How many customers placed their first completed order within 30 days of signing up?",
        _one(con, """SELECT COUNT(*) FROM (SELECT c.customer_id FROM customers c JOIN orders o USING(customer_id)
            WHERE o.status='completed' GROUP BY c.customer_id
            HAVING julianday(MIN(o.order_date)) - julianday(c.signup_date) <= 30)"""), tol=0)
    add("q12", "hard", "What is the average support satisfaction score of customers who have at least one "
        "returned order? Average over their answered tickets, two decimals.",
        round(_one(con, """SELECT AVG(satisfaction) FROM support_tickets WHERE satisfaction IS NOT NULL
            AND customer_id IN (SELECT customer_id FROM orders WHERE status='returned')"""), 2), tol=0.01)
    return q


def main() -> None:
    con = build()
    qs = questions(con)
    QUESTIONS_PATH.write_text(json.dumps(qs, indent=2) + "\n")
    counts = {t: _one(con, f"SELECT COUNT(*) FROM {t}")
              for t in ("customers", "products", "orders", "order_items", "support_tickets")}
    con.close()
    print(f"wrote {DB_PATH.relative_to(HERE)}  {counts}")
    print(f"wrote {QUESTIONS_PATH.relative_to(HERE)}  ({len(qs)} questions)")


if __name__ == "__main__":
    main()
