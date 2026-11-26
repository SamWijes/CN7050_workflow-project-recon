"""
Business calendar: the 'today' of the simulated world. Every PO, inbound, collection and install uses it,
so stock only exists once its inbound batch has actually arrived. If never set, it falls back to the
real date (so the deterministic main.py still works).
"""
import sqlite3
from datetime import datetime, timedelta

ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"


def _conn():
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS business_calendar (id INTEGER PRIMARY KEY CHECK (id = 1), today TEXT)")
    return conn


def today() -> str:
    conn = _conn()
    row = conn.execute("SELECT today FROM business_calendar WHERE id = 1").fetchone()
    conn.close()
    return row[0] if row else datetime.now().strftime("%Y-%m-%d")


def set_today(date_str: str) -> str:
    conn = _conn()
    conn.execute("INSERT INTO business_calendar (id, today) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET today = excluded.today",
                 (date_str,))
    conn.commit()
    conn.close()
    return date_str


def shift(date_str: str, days: int) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def days_between(a: str, b: str) -> int:
    return (datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days


# The warehouse needs material the day before the subcon collects, and the subcon collects
# the day before installation.
def collection_date(install_date: str) -> str:
    return shift(install_date, -1)


def need_by_date(install_date: str) -> str:
    return shift(install_date, -2)