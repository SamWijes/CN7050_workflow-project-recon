import os
import json
import random
import sqlite3
from datetime import datetime, timedelta

ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"
WMS_DB = "databases/wms_inventory.db"
BOM_FILE = "mock_apis/bom_templates.json"
OUTBOX_DIR = "outbox"
os.makedirs(OUTBOX_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# BOM generation
# ---------------------------------------------------------------------------

def _serialized_map() -> dict:
    conn = sqlite3.connect(WMS_DB)
    rows = conn.execute("SELECT sku, serialized FROM sku_master").fetchall()
    conn.close()
    return {sku: bool(s) for sku, s in rows}


def generate_site_bom(site_category: str, site_id: str, seed: int = 42) -> list:
    """
    Expands a site category into a concrete BOM.
    Kits and fixed items are copied as is. Ranged items get a random qty.
    Linked items share one random factor so uPVC conduit and power cable stay proportional.
    The RNG is seeded per site, so the same seed always gives the same plan.
    """
    with open(BOM_FILE, "r", encoding="utf-8") as f:
        templates = json.load(f)
    if site_category not in templates or site_category == "kits":
        raise ValueError(f"Unknown site_category '{site_category}'. Use: full_site, partial_site, upgrade")

    spec = templates[site_category]
    rng = random.Random(f"{seed}-{site_id}")
    lines = {}

    def add(sku, qty):
        lines[sku] = lines.get(sku, 0) + int(qty)

    for kit in spec.get("kits", []):
        for item in templates["kits"][kit]:
            add(item["sku"], item["qty"])
    for item in spec.get("fixed", []):
        add(item["sku"], item["qty"])
    for item in spec.get("ranged", []):
        add(item["sku"], rng.randint(item["min"], item["max"]))
    linked = spec.get("linked")
    if linked:
        t = rng.random()  # 0 = shortest cable run, 1 = longest
        for item in linked["items"]:
            add(item["sku"], round(item["min"] + t * (item["max"] - item["min"])))

    serialized = _serialized_map()
    return [{"sku": sku, "qty": qty, "serialized": serialized.get(sku, False)} for sku, qty in lines.items()]


def get_work_order_bom(work_order_id: str) -> list:
    """Returns the BOM that was actually planned for this work order (from the ledger)."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    rows = conn.execute(
        "SELECT sku, qty_planned, serialized FROM reconciliation_ledger WHERE work_order_id = ? ORDER BY id",
        (work_order_id,),
    ).fetchall()
    conn.close()
    return [{"sku": s, "qty": q, "serialized": bool(ser)} for s, q, ser in rows]


def record_ledger_quantity(work_order_id: str, items: list, column: str) -> None:
    """Adds item quantities to one ledger column (qty_dispatched, qty_installed, qty_quarantined, qty_returned)."""
    if column not in {"qty_dispatched", "qty_installed", "qty_quarantined", "qty_returned"}:
        raise ValueError(f"Invalid ledger column {column}")
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    for item in items:
        conn.execute(
            f"UPDATE reconciliation_ledger SET {column} = {column} + ? WHERE work_order_id = ? AND sku = ?",
            (item["qty"], work_order_id, item["sku"]),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Team scheduling & notifications
# ---------------------------------------------------------------------------

def get_available_team(scheduled_date: str) -> dict:
    """Finds the first Subcontractor and Internal Engineer not booked on scheduled_date."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()
    subcon = cur.execute("""
        SELECT resource_id, email FROM resources
        WHERE role = 'SUBCONTRACTOR'
          AND resource_id NOT IN (SELECT subcon_id FROM work_orders WHERE scheduled_date = ?)
        ORDER BY resource_id LIMIT 1
    """, (scheduled_date,)).fetchone()
    engineer = cur.execute("""
        SELECT resource_id, email FROM resources
        WHERE role = 'ENGINEER'
          AND resource_id NOT IN (SELECT engineer_id FROM work_orders WHERE scheduled_date = ?)
        ORDER BY resource_id LIMIT 1
    """, (scheduled_date,)).fetchone()
    conn.close()

    if not subcon or not engineer:
        return {"available": False, "error": f"No free Subcon + Engineer pair available on {scheduled_date}."}
    return {"available": True, "subcon_id": subcon[0], "subcon_email": subcon[1],
            "engineer_id": engineer[0], "engineer_email": engineer[1]}


def send_notification_email(work_order_id: str, recipient_email: str, recipient_role: str, subject: str, body: str) -> dict:
    """Logs the email to email_outbox and saves a readable .txt copy in outbox/."""
    sent_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO email_outbox (work_order_id, recipient_email, recipient_role, subject, body, sent_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (work_order_id, recipient_email, recipient_role, subject, body, sent_at))
    email_id = cur.lastrowid
    conn.commit()
    conn.close()

    filename = os.path.join(OUTBOX_DIR, f"email_{email_id}_{work_order_id}_{recipient_role}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"To: {recipient_email} ({recipient_role})\nDate: {sent_at}\nSubject: {subject}\n")
        f.write("-" * 50 + "\n" + body + "\n")
    return {"email_id": email_id, "to": recipient_email, "file": filename}


def create_work_order_and_notify(project_name: str, site_id: str, site_category: str,
                                 scheduled_date: str, seed: int = 42) -> dict:
    """Books a team, generates the site BOM, seeds the ledger and emails Subcon, Engineer and Warehouse."""
    team = get_available_team(scheduled_date)
    if not team["available"]:
        return {"status": "FAILED", "reason": team["error"]}

    bom = generate_site_bom(site_category, site_id, seed)
    work_order_id = f"WO-{site_id}-{scheduled_date.replace('-', '')}"

    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO work_orders (work_order_id, project_name, site_id, site_category, scheduled_date,
                                     subcon_id, engineer_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PLANNED')
        """, (work_order_id, project_name, site_id, site_category, scheduled_date,
              team["subcon_id"], team["engineer_id"]))
        for item in bom:
            cur.execute("""
                INSERT INTO reconciliation_ledger (work_order_id, site_id, sku, serialized, qty_planned)
                VALUES (?, ?, ?, ?, ?)
            """, (work_order_id, site_id, item["sku"], int(item["serialized"]), item["qty"]))
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.close()
        return {"status": "FAILED", "reason": f"Database constraint error: {e}"}
    conn.close()

    bom_summary = "\n".join(f"  - {i['sku']}: {i['qty']}" for i in bom)
    e1 = send_notification_email(
        work_order_id, team["subcon_email"], "SUBCONTRACTOR",
        f"Work Order {work_order_id} Assigned ({site_id})",
        f"Hello {team['subcon_id']},\n\nYou are assigned to {site_id} ({site_category}) on {scheduled_date}.\n"
        f"On site Engineer: {team['engineer_id']}.\nPlease collect the allocated materials from Warehouse:\n{bom_summary}")
    e2 = send_notification_email(
        work_order_id, team["engineer_email"], "ENGINEER",
        f"Site Supervision & App Confirmation: {work_order_id} ({site_id})",
        f"Hello {team['engineer_id']},\n\nYou are assigned to supervise {site_id} on {scheduled_date} with "
        f"{team['subcon_id']}.\nPlease verify material receipt, log any faulty units, and submit installation "
        f"confirmation via the Field App.")
    e3 = send_notification_email(
        work_order_id, "warehouse@intellicommstech.local", "WAREHOUSE",
        f"Advance Material Prep Required: {work_order_id} ({site_id})",
        f"Hello Warehouse Team,\n\nPlease prepare the following BOM for {site_id} "
        f"(Date: {scheduled_date}, Collector: {team['subcon_id']}):\n{bom_summary}")

    return {"status": "PLANNED", "work_order_id": work_order_id, "site_id": site_id,
            "scheduled_date": scheduled_date, "subcon_id": team["subcon_id"],
            "engineer_id": team["engineer_id"], "bom": bom,
            "emails_sent": [e1["file"], e2["file"], e3["file"]]}


def plan_multi_site_rollout(project_name: str, full_sites: int = 0, upgrades: int = 0,
                            partial_sites: int = 0, start_date: str = "2026-12-01", seed: int = 42) -> dict:
    """Schedules sites across days (max 5 per day with 5 teams) and returns the aggregated project BOM."""
    site_queue = (["full_site"] * full_sites + ["upgrade"] * upgrades + ["partial_site"] * partial_sites)
    if not site_queue:
        return {"status": "FAILED", "reason": "No sites requested."}

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    aggregated, work_orders, failures = {}, [], []

    for idx, category in enumerate(site_queue, start=1):
        site_id = f"SITE-{idx:03d}"
        while not get_available_team(current_date.strftime("%Y-%m-%d"))["available"]:
            current_date += timedelta(days=1)
        date_str = current_date.strftime("%Y-%m-%d")

        res = create_work_order_and_notify(project_name, site_id, category, date_str, seed)
        if res["status"] != "PLANNED":
            failures.append({"site_id": site_id, "reason": res["reason"]})
            continue
        work_orders.append({"work_order_id": res["work_order_id"], "site_id": site_id,
                            "category": category, "date": date_str,
                            "team": f"{res['subcon_id']} + {res['engineer_id']}"})
        for item in res["bom"]:
            agg = aggregated.setdefault(item["sku"], {"sku": item["sku"], "qty": 0, "serialized": item["serialized"]})
            agg["qty"] += item["qty"]

    return {"status": "ROLLOUT_PLANNED" if not failures else "PARTIALLY_PLANNED",
            "project_name": project_name, "total_sites_scheduled": len(work_orders),
            "start_date": start_date, "end_date": current_date.strftime("%Y-%m-%d"),
            "aggregated_bom": list(aggregated.values()), "work_orders": work_orders, "failures": failures}



# ---------------------------------------------------------------------------
# Work order lookups (used by the LangGraph agents)
# ---------------------------------------------------------------------------

def get_work_order(work_order_id: str) -> dict:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    row = conn.execute("""
        SELECT work_order_id, project_name, site_id, site_category, scheduled_date, subcon_id, engineer_id, status
        FROM work_orders WHERE work_order_id = ?
    """, (work_order_id,)).fetchone()
    conn.close()
    if not row:
        return None
    keys = ["work_order_id", "project_name", "site_id", "site_category", "scheduled_date",
            "subcon_id", "engineer_id", "status"]
    return dict(zip(keys, row))


def set_work_order_status(work_order_id: str, status: str) -> None:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    conn.execute("UPDATE work_orders SET status = ? WHERE work_order_id = ?", (status, work_order_id))
    conn.commit()
    conn.close()