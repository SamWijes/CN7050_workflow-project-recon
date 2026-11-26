"""
Mailbox: every email the agents send and every reply from engineers / subcons.
The UI shows one mailbox per person from this table.
"""
import sqlite3
from datetime import datetime

ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"


def _conn():
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mailbox (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            work_order_id TEXT,
            sender TEXT NOT NULL,
            recipient TEXT NOT NULL,
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('OUT', 'IN')),
            in_reply_to INTEGER,
            status TEXT NOT NULL CHECK(status IN ('SENT', 'AWAITING_REPLY', 'REPLIED', 'RECEIVED', 'PROCESSED')),
            sent_at TEXT NOT NULL
        )
    """)
    return conn


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def send_mail(work_order_id: str, sender: str, recipient: str, subject: str, body: str,
              expects_reply: bool = False, thread_id: str = None) -> dict:
    conn = _conn()
    cur = conn.execute("""
        INSERT INTO mailbox (thread_id, work_order_id, sender, recipient, subject, body, direction, status, sent_at)
        VALUES (?, ?, ?, ?, ?, ?, 'OUT', ?, ?)
    """, (thread_id or "pending", work_order_id, sender, recipient, subject, body,
          "AWAITING_REPLY" if expects_reply else "SENT", _now()))
    msg_id = cur.lastrowid
    thread_id = thread_id or f"T{msg_id:05d}"
    conn.execute("UPDATE mailbox SET thread_id = ? WHERE message_id = ?", (thread_id, msg_id))
    conn.commit()
    conn.close()
    return {"message_id": msg_id, "thread_id": thread_id}


def deliver_reply(message_id: int, body: str) -> dict:
    """A human (simulated or typed in the UI) replies to an agent email."""
    conn = _conn()
    row = conn.execute("SELECT thread_id, work_order_id, sender, recipient, subject FROM mailbox "
                       "WHERE message_id = ? AND status = 'AWAITING_REPLY'", (message_id,)).fetchone()
    if not row:
        conn.close()
        return {"status": "FAILED", "reason": f"Message {message_id} is not awaiting a reply."}
    thread_id, wo, agent_addr, person_addr, subject = row
    cur = conn.execute("""
        INSERT INTO mailbox (thread_id, work_order_id, sender, recipient, subject, body, direction,
                             in_reply_to, status, sent_at)
        VALUES (?, ?, ?, ?, ?, ?, 'IN', ?, 'RECEIVED', ?)
    """, (thread_id, wo, person_addr, agent_addr, f"RE: {subject}", body, message_id, _now()))
    conn.execute("UPDATE mailbox SET status = 'REPLIED' WHERE message_id = ?", (message_id,))
    conn.execute("UPDATE work_orders SET status = 'REPLY_RECEIVED' WHERE work_order_id = ? AND status = 'AWAITING_REPLY'",
                 (wo,))
    conn.commit()
    conn.close()
    return {"status": "DELIVERED", "reply_id": cur.lastrowid, "work_order_id": wo}


def awaiting_replies(project_name: str = None) -> list:
    conn = _conn()
    q = """SELECT m.message_id, m.work_order_id, m.recipient, m.subject, m.body, m.sent_at FROM mailbox m
           JOIN work_orders w ON w.work_order_id = m.work_order_id
           WHERE m.status = 'AWAITING_REPLY'"""
    args = []
    if project_name:
        q += " AND w.project_name = ?"
        args.append(project_name)
    rows = conn.execute(q + " ORDER BY m.message_id", args).fetchall()
    conn.close()
    keys = ["message_id", "work_order_id", "recipient", "subject", "body", "sent_at"]
    return [dict(zip(keys, r)) for r in rows]


def take_new_replies(work_order_id: str) -> list:
    """Unread replies for a work order; marks them PROCESSED."""
    conn = _conn()
    rows = conn.execute("""SELECT message_id, sender, subject, body, sent_at FROM mailbox
                           WHERE work_order_id = ? AND direction = 'IN' AND status = 'RECEIVED'
                           ORDER BY message_id""", (work_order_id,)).fetchall()
    conn.executemany("UPDATE mailbox SET status = 'PROCESSED' WHERE message_id = ?", [(r[0],) for r in rows])
    conn.commit()
    conn.close()
    return [dict(zip(["message_id", "from", "subject", "body", "sent_at"], r)) for r in rows]


def get_mailbox(address: str) -> list:
    """Every message to or from one address, newest thread first (for the UI)."""
    conn = _conn()
    rows = conn.execute("""SELECT message_id, thread_id, work_order_id, sender, recipient, subject, body,
                                  direction, status, sent_at
                           FROM mailbox WHERE sender = ? OR recipient = ?
                           ORDER BY thread_id DESC, message_id""", (address, address)).fetchall()
    conn.close()
    keys = ["message_id", "thread_id", "work_order_id", "sender", "recipient", "subject", "body",
            "direction", "status", "sent_at"]
    return [dict(zip(keys, r)) for r in rows]