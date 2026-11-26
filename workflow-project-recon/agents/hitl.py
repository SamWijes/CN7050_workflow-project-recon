"""
Human in the loop store, shared by every agent and the UI:
  project_settings  - reconciliation policy per project (and where it came from)
  human_questions   - questions any agent asks a human, and the answers
  recon_requests    - reconciliation requested by a human at any time
  billing_batches   - consolidated invoices (one per reconciliation batch)
"""
import json
import sqlite3
from datetime import datetime

ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"
POLICIES = ("end_of_project", "every_n_sites", "on_request", "unspecified")
MAX_QUESTIONS_PER_SITE = 2


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _conn():
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS project_settings (
            project_name TEXT PRIMARY KEY,
            recon_policy TEXT NOT NULL DEFAULT 'unspecified',
            recon_every_n INTEGER NOT NULL DEFAULT 0,
            policy_source TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS human_questions (
            question_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            work_order_id TEXT,
            agent TEXT NOT NULL,
            question TEXT NOT NULL,
            options_json TEXT,
            prev_status TEXT,
            status TEXT NOT NULL DEFAULT 'OPEN',      -- OPEN, ANSWERED, CONSUMED
            answer TEXT,
            asked_at TEXT NOT NULL,
            answered_at TEXT
        );
        CREATE TABLE IF NOT EXISTS recon_requests (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            requested_by TEXT NOT NULL,
            scope_json TEXT,                          -- null = every installed site
            status TEXT NOT NULL DEFAULT 'OPEN',      -- OPEN, DONE
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS billing_batches (
            batch_id TEXT PRIMARY KEY,
            project_name TEXT NOT NULL,
            work_orders_json TEXT NOT NULL,
            invoice_ids_json TEXT NOT NULL,
            grand_total_gbp REAL NOT NULL,
            has_anomalies INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING_APPROVAL',
            file_path TEXT,
            created_at TEXT NOT NULL,
            decided_by TEXT,
            decided_at TEXT
        );
    """)
    return conn


# ---------------------------------------------------------------------------
# Reconciliation policy
# ---------------------------------------------------------------------------

def get_policy(project_name: str) -> dict:
    conn = _conn()
    row = conn.execute("SELECT recon_policy, recon_every_n, policy_source FROM project_settings WHERE project_name = ?",
                       (project_name,)).fetchone()
    conn.close()
    if not row:
        return {"policy": "unspecified", "every_n": 0, "source": None}
    return {"policy": row[0], "every_n": row[1], "source": row[2]}


def set_policy(project_name: str, policy: str, every_n: int = 0, source: str = "") -> dict:
    if policy not in POLICIES:
        return {"error": f"policy must be one of {POLICIES}"}
    if policy == "every_n_sites" and every_n < 1:
        return {"error": "every_n_sites needs every_n of at least 1"}
    conn = _conn()
    conn.execute("""INSERT INTO project_settings (project_name, recon_policy, recon_every_n, policy_source, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(project_name) DO UPDATE SET recon_policy = excluded.recon_policy,
                        recon_every_n = excluded.recon_every_n, policy_source = excluded.policy_source,
                        updated_at = excluded.updated_at""",
                 (project_name, policy, every_n if policy == "every_n_sites" else 0, source, _now()))
    conn.commit()
    conn.close()
    return {"status": "SET", **get_policy(project_name)}


# ---------------------------------------------------------------------------
# Questions from any agent to a human
# ---------------------------------------------------------------------------

def ask(project_name: str, agent: str, question: str, options: list = None, work_order_id: str = None) -> dict:
    conn = _conn()
    # One open question at a time per site (or per agent for project level questions)
    if work_order_id:
        dup = conn.execute("SELECT question_id FROM human_questions WHERE work_order_id = ? AND status = 'OPEN'",
                           (work_order_id,)).fetchone()
        asked = conn.execute("SELECT COUNT(*) FROM human_questions WHERE work_order_id = ?", (work_order_id,)).fetchone()[0]
    else:
        dup = conn.execute("SELECT question_id FROM human_questions WHERE project_name = ? AND work_order_id IS NULL "
                           "AND agent = ? AND status = 'OPEN'", (project_name, agent)).fetchone()
        asked = 0
    if dup:
        conn.close()
        return {"status": "ALREADY_ASKED", "question_id": dup[0]}
    if work_order_id and asked >= MAX_QUESTIONS_PER_SITE:
        conn.close()
        return {"status": "LIMIT_REACHED",
                "reason": f"Already asked {asked} questions about {work_order_id}. Decide from the policy, or escalate."}

    prev = None
    if work_order_id:
        row = conn.execute("SELECT status FROM work_orders WHERE work_order_id = ?", (work_order_id,)).fetchone()
        prev = row[0] if row else None
        conn.execute("UPDATE work_orders SET status = 'AWAITING_HUMAN_ANSWER' WHERE work_order_id = ?", (work_order_id,))
    cur = conn.execute("""INSERT INTO human_questions (project_name, work_order_id, agent, question, options_json,
                                                       prev_status, asked_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                       (project_name, work_order_id, agent, question, json.dumps(options or []), prev, _now()))
    conn.commit()
    qid = cur.lastrowid
    conn.close()
    return {"status": "ASKED", "question_id": qid}


def open_questions(project_name: str) -> list:
    conn = _conn()
    rows = conn.execute("""SELECT question_id, work_order_id, agent, question, options_json, asked_at
                           FROM human_questions WHERE project_name = ? AND status = 'OPEN' ORDER BY question_id""",
                        (project_name,)).fetchall()
    conn.close()
    return [{"question_id": r[0], "work_order_id": r[1], "agent": r[2], "question": r[3],
             "options": json.loads(r[4] or "[]"), "asked_at": r[5]} for r in rows]


def answer(question_id: int, text: str) -> dict:
    """A human answers. The site goes back to the stage it was in, and the asking agent picks it up."""
    conn = _conn()
    row = conn.execute("SELECT work_order_id, prev_status FROM human_questions WHERE question_id = ? AND status = 'OPEN'",
                       (question_id,)).fetchone()
    if not row:
        conn.close()
        return {"status": "FAILED", "reason": f"Question {question_id} is not open."}
    wo, prev = row
    conn.execute("UPDATE human_questions SET status = 'ANSWERED', answer = ?, answered_at = ? WHERE question_id = ?",
                 (text, _now(), question_id))
    if wo and prev:
        conn.execute("UPDATE work_orders SET status = ? WHERE work_order_id = ? AND status = 'AWAITING_HUMAN_ANSWER'",
                     (prev, wo))
    conn.commit()
    conn.close()
    return {"status": "ANSWERED", "question_id": question_id, "work_order_id": wo}


def take_answers(agent: str, project_name: str, work_order_id: str = None) -> list:
    """Answered questions this agent has not yet acted on (for one site, or project level). Marks them used."""
    conn = _conn()
    if work_order_id:
        rows = conn.execute("""SELECT question_id, question, answer FROM human_questions
                               WHERE work_order_id = ? AND agent = ? AND status = 'ANSWERED'""",
                            (work_order_id, agent)).fetchall()
    else:
        rows = conn.execute("""SELECT question_id, question, answer FROM human_questions
                               WHERE project_name = ? AND work_order_id IS NULL AND agent = ? AND status = 'ANSWERED'""",
                            (project_name, agent)).fetchall()
    conn.executemany("UPDATE human_questions SET status = 'CONSUMED' WHERE question_id = ?", [(r[0],) for r in rows])
    conn.commit()
    conn.close()
    return [{"question": r[1], "answer": r[2]} for r in rows]


def has_open_project_question(project_name: str, agent: str) -> bool:
    return any(q["work_order_id"] is None and q["agent"] == agent for q in open_questions(project_name))


# ---------------------------------------------------------------------------
# Reconciliation requests (a human can ask at any time, from the CLI or the UI)
# ---------------------------------------------------------------------------

def request_reconciliation(project_name: str, requested_by: str = "KAM", work_order_ids: list = None) -> dict:
    conn = _conn()
    cur = conn.execute("INSERT INTO recon_requests (project_name, requested_by, scope_json, created_at) VALUES (?, ?, ?, ?)",
                       (project_name, requested_by, json.dumps(work_order_ids) if work_order_ids else None, _now()))
    conn.commit()
    conn.close()
    return {"status": "REQUESTED", "request_id": cur.lastrowid}


def open_recon_requests(project_name: str) -> list:
    conn = _conn()
    rows = conn.execute("SELECT request_id, requested_by, scope_json FROM recon_requests "
                        "WHERE project_name = ? AND status = 'OPEN'", (project_name,)).fetchall()
    conn.close()
    return [{"request_id": r[0], "requested_by": r[1], "scope": json.loads(r[2]) if r[2] else None} for r in rows]


def close_recon_requests(request_ids: list):
    conn = _conn()
    conn.executemany("UPDATE recon_requests SET status = 'DONE' WHERE request_id = ?", [(i,) for i in request_ids])
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Billing batches (consolidated invoice per reconciliation batch)
# ---------------------------------------------------------------------------

def save_batch(batch_id, project_name, work_orders, invoice_ids, total, has_anomalies, file_path) -> None:
    conn = _conn()
    conn.execute("""INSERT OR REPLACE INTO billing_batches (batch_id, project_name, work_orders_json, invoice_ids_json,
                        grand_total_gbp, has_anomalies, file_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                 (batch_id, project_name, json.dumps(work_orders), json.dumps(invoice_ids), total,
                  int(has_anomalies), file_path, _now()))
    conn.commit()
    conn.close()


def pending_batches(project_name: str) -> list:
    conn = _conn()
    rows = conn.execute("""SELECT batch_id, work_orders_json, invoice_ids_json, grand_total_gbp, has_anomalies, file_path
                           FROM billing_batches WHERE project_name = ? AND status = 'PENDING_APPROVAL'""",
                        (project_name,)).fetchall()
    conn.close()
    return [{"batch_id": r[0], "work_orders": json.loads(r[1]), "invoice_ids": json.loads(r[2]),
             "total_gbp": r[3], "has_anomalies": bool(r[4]), "file": r[5]} for r in rows]


def decide_batch(batch_id: str, approver: str, approved: bool) -> dict:
    from tools.finance_tools import record_invoice_decision
    conn = _conn()
    row = conn.execute("SELECT invoice_ids_json FROM billing_batches WHERE batch_id = ? AND status = 'PENDING_APPROVAL'",
                       (batch_id,)).fetchone()
    if not row:
        conn.close()
        return {"status": "FAILED", "reason": f"Batch {batch_id} is not pending."}
    status = "APPROVED" if approved else "REJECTED"
    conn.execute("UPDATE billing_batches SET status = ?, decided_by = ?, decided_at = ? WHERE batch_id = ?",
                 (status, approver, _now(), batch_id))
    conn.commit()
    conn.close()
    for inv in json.loads(row[0]):
        record_invoice_decision(inv, approver, approved)
    return {"status": status, "batch_id": batch_id}