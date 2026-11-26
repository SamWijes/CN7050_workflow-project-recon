"""Persistent agent activity log. The UI reads this table to show what each agent did."""
import json
import sqlite3
from datetime import datetime

ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"


def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            agent TEXT NOT NULL,
            action TEXT NOT NULL,
            work_order_id TEXT,
            detail TEXT
        )
    """)


def log_event(agent: str, action: str, detail=None, work_order_id: str = None) -> dict:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = detail if isinstance(detail, str) else json.dumps(detail, default=str)
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    _ensure_table(conn)
    conn.execute("INSERT INTO agent_events (ts, agent, action, work_order_id, detail) VALUES (?, ?, ?, ?, ?)",
                 (ts, agent, action, work_order_id, text))
    conn.commit()
    conn.close()
    print(f"[{ts}] {agent:<22} {action:<28} {work_order_id or ''}")
    return {"ts": ts, "agent": agent, "action": action, "work_order_id": work_order_id}