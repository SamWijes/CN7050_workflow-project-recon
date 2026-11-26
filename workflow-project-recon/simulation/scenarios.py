"""
Scenario definitions. A scenario changes the WORLD around the agents (what the Field App reports,
how people reply). Agents never read this module or the truth store; they must discover problems
through reconciliation, just as they would in real life.
"""
import json
import sqlite3

SIM_DB = "databases/simulation.db"

SCENARIOS = {
    "happy_path": {},
    # Engineer reports fewer bulk items installed than were issued; subcon has the balance
    "qty_mismatch": {"bulk_sku": "UPVC-20", "missing_pct": 0.15, "reply": "returns_balance"},
    # As above, but the subcon only returns part of it (offcuts wasted) -> agent must escalate
    "qty_mismatch_wastage": {"bulk_sku": "PWR-CBL-2C-20", "missing_pct": 0.20, "reply": "wastage"},
        # Subcon replies vaguely ("a few left over") the first time -> Field agent should ask the KAM / chase again
    "qty_mismatch_vague": {"bulk_sku": "UPVC-20", "missing_pct": 0.15, "reply": "vague_then_exact"},
    # A unit fails, a replacement is issued, but the faulty unit never reaches quarantine
    "faulty_unaccounted": {"fault_sku": "PSU-01", "reply": "returned_late"},
    # As above, but the engineer cannot find the faulty unit -> agent must escalate
    "faulty_lost": {"fault_sku": "PSU-01", "reply": "lost"},
}

# Default demo assignment (site_id -> scenario). Sites not listed run happy_path.
DEMO_ASSIGNMENT = {
    "SITE-002": "qty_mismatch",
    "SITE-003": "faulty_unaccounted",
    "SITE-005": "faulty_lost",
}


def _conn():
    conn = sqlite3.connect(SIM_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS sim_truth (
                        work_order_id TEXT NOT NULL, scenario TEXT NOT NULL, facts_json TEXT NOT NULL)""")
    return conn


def reset_truth():
    conn = _conn()
    conn.execute("DELETE FROM sim_truth")
    conn.commit()
    conn.close()


def record_truth(work_order_id: str, scenario: str, facts: dict):
    conn = _conn()
    conn.execute("INSERT INTO sim_truth VALUES (?, ?, ?)", (work_order_id, scenario, json.dumps(facts)))
    conn.commit()
    conn.close()


def get_truth(work_order_id: str) -> list:
    conn = _conn()
    rows = conn.execute("SELECT scenario, facts_json FROM sim_truth WHERE work_order_id = ?", (work_order_id,)).fetchall()
    conn.close()
    return [{"scenario": s, **json.loads(f)} for s, f in rows]


def parse_assignment(text: str) -> dict:
    """'SITE-002=qty_mismatch,SITE-003=faulty_unaccounted' -> dict"""
    out = {}
    for pair in filter(None, (p.strip() for p in text.split(","))):
        site, scen = pair.split("=")
        if scen not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scen}. Choose from {list(SCENARIOS)}")
        out[site.strip()] = scen.strip()
    return out