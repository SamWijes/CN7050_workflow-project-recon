import json
import random
import sqlite3
from datetime import datetime
from rag.rag_query import query_telecom_rag

WMS_DB = "databases/wms_inventory.db"
ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"


def simulate_field_app_report(
    work_order_id: str,
    site_id: str,
    engineer_id: str,
    dispatched_items: list,
    mode: str = "demo",
) -> dict:
    """
    Simulates the Internal Engineer submitting installation & fault logs via the Field App.
    mode='demo'   -> forces the first SKU-RU-5G unit to fail (DOA) for repeatable demos.
    mode='random' -> 20% fault probability on each serialized unit.
    """
    installed = []
    faults = []
    demo_fault_triggered = False

    for item in dispatched_items:
        sku = item["sku"]
        sn = item.get("serial_number")
        qty = item["qty"]

        is_faulty = False
        if sn:
            if mode == "demo" and sku == "SKU-RU-5G" and not demo_fault_triggered:
                is_faulty = True
                demo_fault_triggered = True
            elif mode == "random" and random.random() < 0.20:
                is_faulty = True

        if is_faulty:
            faults.append({
                "sku": sku,
                "serial_number": sn,
                "qty": 1,
                "fault_log": f"{site_id}: 5G Radio Unit {sn} failed to power on after tower mounting.",
            })
        else:
            installed.append({"sku": sku, "serial_number": sn, "qty": qty})

    return {
        "work_order_id": work_order_id,
        "site_id": site_id,
        "engineer_id": engineer_id,
        "installed_items": installed,
        "faulty_items": faults,
    }


def quarantine_faulty_unit(sku: str, serial_number: str, engineer_id: str, site_id: str) -> dict:
    """Logs the defective serial number into wms_ledger as QUARANTINE."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(WMS_DB)
    cur = conn.cursor()

    desc_row = cur.execute(
        "SELECT item_description FROM wms_ledger WHERE sku = ? LIMIT 1", (sku,)
    ).fetchone()
    item_desc = desc_row[0] if desc_row else sku

    cur.execute("""
        INSERT INTO wms_ledger
        (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
        VALUES (?, ?, ?, 1, ?, ?, 'QUARANTINE', ?)
    """, (sku, item_desc, serial_number, today, engineer_id, site_id))

    conn.commit()
    conn.close()
    return {"status": "QUARANTINED", "sku": sku, "serial_number": serial_number, "bay": "BAY-Q1"}


def process_field_report(report: dict) -> dict:
    """
    Processes the engineer's Field App report:
    - Updates installed & quarantined counts in orchestrator_ledger.db
    - Runs RAG fault triage on any defective units
    - Quarantines faulty serials and returns replenishment requests
    """
    work_order_id = report["work_order_id"]
    site_id = report["site_id"]
    engineer_id = report["engineer_id"]

    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()

    # Record installed quantities
    for item in report["installed_items"]:
        cur.execute("""
            UPDATE reconciliation_ledger
            SET qty_installed = qty_installed + ?
            WHERE work_order_id = ? AND sku = ?
        """, (item["qty"], work_order_id, item["sku"]))

    replenishment_requests = []
    triage_results = []

    # Triage & quarantine faults via local RAG + Qwen2.5
    for fault in report["faulty_items"]:
        rag_res = query_telecom_rag(fault["fault_log"], source_filter="telecom_fault_manual.txt")
        decision = rag_res["agent_decision"]
        triage_results.append({"serial_number": fault["serial_number"], "decision": decision})

        quarantine_faulty_unit(fault["sku"], fault["serial_number"], engineer_id, site_id)

        cur.execute("""
            UPDATE reconciliation_ledger
            SET qty_quarantined = qty_quarantined + 1
            WHERE work_order_id = ? AND sku = ?
        """, (work_order_id, fault["sku"]))

        if decision.get("trigger_escalation_or_replenishment"):
            replenishment_requests.append({
                "sku": fault["sku"],
                "qty": 1,
                "serialized": True,
                "reason": decision.get("policy_code", "FAULT"),
                "failed_serial": fault["serial_number"],
                "collector": engineer_id,
            })

    conn.commit()
    conn.close()

    return {
        "work_order_id": work_order_id,
        "site_id": site_id,
        "triage_results": triage_results,
        "replenishment_requests": replenishment_requests,
    }


if __name__ == "__main__":
    # Quick standalone test of the Field App + RAG Fault Triage loop
    sample_dispatched = [
        {"sku": "SKU-RU-5G", "serial_number": "SN-RU-1001", "qty": 1},
        {"sku": "SKU-RU-5G", "serial_number": "SN-RU-1002", "qty": 1},
        {"sku": "SKU-BBU-01", "serial_number": "SN-BBU-2001", "qty": 1},
    ]

    report = simulate_field_app_report(
        work_order_id="WO-SITE-104-20261201",
        site_id="SITE-104",
        engineer_id="Engineer-A",
        dispatched_items=sample_dispatched,
        mode="demo",
    )
    result = process_field_report(report)
    print(json.dumps(result, indent=2))