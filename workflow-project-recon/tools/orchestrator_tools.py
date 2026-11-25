import os
import json
import sqlite3
from datetime import datetime

ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"
BOM_FILE = "mock_apis/bom_templates.json"
OUTBOX_DIR = "outbox"
os.makedirs(OUTBOX_DIR, exist_ok=True)


def get_bom_for_category(site_category: str) -> list:
    """Returns the standard BOM list for full_site, partial_site, or upgrade."""
    with open(BOM_FILE, "r", encoding="utf-8") as f:
        templates = json.load(f)
    if site_category not in templates:
        raise ValueError(f"Unknown site_category '{site_category}'. Use: {list(templates.keys())}")
    return templates[site_category]


def get_available_team(scheduled_date: str) -> dict:
    """Finds the first available Subcontractor and Internal Engineer not booked on scheduled_date."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()

    subcon = cur.execute("""
        SELECT resource_id, email FROM resources
        WHERE role = 'SUBCONTRACTOR'
          AND resource_id NOT IN (
              SELECT subcon_id FROM work_orders WHERE scheduled_date = ?
          )
        ORDER BY resource_id ASC LIMIT 1
    """, (scheduled_date,)).fetchone()

    engineer = cur.execute("""
        SELECT resource_id, email FROM resources
        WHERE role = 'ENGINEER'
          AND resource_id NOT IN (
              SELECT engineer_id FROM work_orders WHERE scheduled_date = ?
          )
        ORDER BY resource_id ASC LIMIT 1
    """, (scheduled_date,)).fetchone()

    conn.close()

    if not subcon or not engineer:
        return {"available": False, "error": f"No free Subcon + Engineer pair available on {scheduled_date}."}

    return {
        "available": True,
        "subcon_id": subcon[0],
        "subcon_email": subcon[1],
        "engineer_id": engineer[0],
        "engineer_email": engineer[1],
    }


def send_notification_email(work_order_id: str, recipient_email: str, recipient_role: str, subject: str, body: str) -> dict:
    """Logs the email to email_outbox in SQLite and saves a readable .txt file in outbox/."""
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
        f.write(f"To: {recipient_email} ({recipient_role})\n")
        f.write(f"Date: {sent_at}\n")
        f.write(f"Subject: {subject}\n")
        f.write("-" * 50 + "\n")
        f.write(body + "\n")

    return {"email_id": email_id, "to": recipient_email, "file": filename}


def create_work_order_and_notify(project_name: str, site_id: str, site_category: str, scheduled_date: str) -> dict:
    """
    Plans a site: checks team availability, creates the work order, seeds the reconciliation
    ledger with the BOM, and emails Subcon, Engineer, and Warehouse.
    """
    team = get_available_team(scheduled_date)
    if not team["available"]:
        return {"status": "FAILED", "reason": team["error"]}

    bom = get_bom_for_category(site_category)
    work_order_id = f"WO-{site_id}-{scheduled_date.replace('-', '')}"

    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO work_orders (work_order_id, project_name, site_id, site_category, scheduled_date, subcon_id, engineer_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PLANNED')
        """, (work_order_id, project_name, site_id, site_category, scheduled_date, team["subcon_id"], team["engineer_id"]))

        for item in bom:
            cur.execute("""
                INSERT INTO reconciliation_ledger (work_order_id, site_id, sku, qty_planned)
                VALUES (?, ?, ?, ?)
            """, (work_order_id, site_id, item["sku"], item["qty"]))

        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.close()
        return {"status": "FAILED", "reason": f"Database constraint error: {e}"}
    conn.close()

    bom_summary = "\n".join([f"  - {i['sku']}: {i['qty']} units" for i in bom])

    # 1. Email Subcontractor
    e1 = send_notification_email(
        work_order_id, team["subcon_email"], "SUBCONTRACTOR",
        f"Work Order {work_order_id} Assigned ({site_id})",
        f"Hello {team['subcon_id']},\n\nYou are assigned to {site_id} ({site_category}) on {scheduled_date}.\n"
        f"On-site Engineer: {team['engineer_id']}.\nPlease collect the allocated materials from Warehouse:\n{bom_summary}"
    )

    # 2. Email Internal Engineer
    e2 = send_notification_email(
        work_order_id, team["engineer_email"], "ENGINEER",
        f"Site Supervision & App Confirmation: {work_order_id} ({site_id})",
        f"Hello {team['engineer_id']},\n\nYou are assigned to supervise {site_id} on {scheduled_date} with {team['subcon_id']}.\n"
        f"Please verify material receipt, log any faulty units, and submit installation confirmation via the Field App."
    )

    # 3. Email Warehouse (Advance Material Prep)
    e3 = send_notification_email(
        work_order_id, "warehouse@intellicommstech.local", "WAREHOUSE",
        f"Advance Material Prep Required: {work_order_id} ({site_id})",
        f"Hello Warehouse Team,\n\nPlease check stock and prepare the following BOM for {site_id} (Date: {scheduled_date}, Collector: {team['subcon_id']}):\n{bom_summary}"
    )

    return {
        "status": "PLANNED",
        "work_order_id": work_order_id,
        "site_id": site_id,
        "scheduled_date": scheduled_date,
        "subcon_id": team["subcon_id"],
        "engineer_id": team["engineer_id"],
        "bom": bom,
        "emails_sent": [e1["file"], e2["file"], e3["file"]],
    }
# Adding multi agent workflow to orchastrator
def plan_multi_site_rollout(
    project_name: str,
    full_sites: int = 0,
    upgrades: int = 0,
    partial_sites: int = 0,
    start_date: str = "2026-12-01",
) -> dict:
    """
    Schedules a multi-site rollout across available days (max 5 sites/day with 5 Subcons & 5 Engineers),
    aggregates total BOM demand, and returns a rollout summary.
    """
    site_queue = (
        [("full_site", i + 1) for i in range(full_sites)]
        + [("upgrade", i + 1) for i in range(upgrades)]
        + [("partial_site", i + 1) for i in range(partial_sites)]
    )

    if not site_queue:
        return {"status": "FAILED", "reason": "No sites requested."}

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    scheduled_count = 0
    aggregated_bom = {}
    sample_work_orders = []

    for idx, (category, _) in enumerate(site_queue, start=1):
        site_id = f"SITE-{idx:03d}"

        while not get_available_team(current_date.strftime("%Y-%m-%d"))["available"]:
            current_date += timedelta(days=1)

        date_str = current_date.strftime("%Y-%m-%d")
        res = create_work_order_and_notify(project_name, site_id, category, date_str)
        if res["status"] == "PLANNED":
            scheduled_count += 1
            if len(sample_work_orders) < 5:
                sample_work_orders.append({
                    "work_order_id": res["work_order_id"],
                    "site_id": site_id,
                    "category": category,
                    "date": date_str,
                    "team": f"{res['subcon_id']} + {res['engineer_id']}",
                })
            for item in res["bom"]:
                sku = item["sku"]
                if sku not in aggregated_bom:
                    aggregated_bom[sku] = {"sku": sku, "qty": 0, "serialized": item["serialized"]}
                aggregated_bom[sku]["qty"] += item["qty"]

    return {
        "status": "ROLLOUT_PLANNED",
        "project_name": project_name,
        "total_sites_scheduled": scheduled_count,
        "start_date": start_date,
        "end_date": current_date.strftime("%Y-%m-%d"),
        "aggregated_bom": list(aggregated_bom.values()),
        "sample_work_orders": sample_work_orders,
    }

if __name__ == "__main__":
    # Quick test: schedule two sites on the same date to verify conflict-free team assignment
    res1 = create_work_order_and_notify("5G-Rollout-Essex", "SITE-104", "full_site", "2026-12-01")
    print("Site 104:", json.dumps(res1, indent=2))

    res2 = create_work_order_and_notify("5G-Rollout-Essex", "SITE-105", "partial_site", "2026-12-01")
    print("\nSite 105:", json.dumps(res2, indent=2))