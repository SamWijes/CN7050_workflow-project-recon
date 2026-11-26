import json
import random
import sqlite3
from datetime import datetime
from rag.rag_query import query_telecom_rag
from tools.inventory_tools import get_sku_description
from tools.orchestrator_tools import record_ledger_quantity

WMS_DB = "databases/wms_inventory.db"

# Free text an engineer would type into the Field App, per part type.
# The RAG agent must work out the FAULT code from this text; the code itself is never given.
FAULT_NOTES = {
    "CAB-OUT-01":  "Cabinet {sn} arrived with a dented door and broken seal, water ingress risk.",
    "BAT-48V-01":  "Battery {sn} reads 41V at commissioning and the casing is swollen.",
    "PSU-01":      "PSU {sn} shows no output and the status LED stays off after power on.",
    "BBU-01":      "BBU {sn} optical port 2 shows LOS, link will not come up after cleaning the connector.",
    "CARD-CC-01":  "CC card {sn} stuck in boot loop, never reaches operational state.",
    "CARD-BBN-01": "BBN card {sn} not detected by the BBU after reseating twice.",
    "RECT-01":     "Rectifier shelf {sn} raises a mains input alarm with mains present.",
    "SMR-01":      "SMR module {sn} red fault LED, zero current share.",
}

DEMO_FAULT_SKU = "BBU-01"


def simulate_field_app_report(work_order_id: str, site_id: str, engineer_id: str,
                              dispatched_items: list, mode: str = "demo", seed: int = None) -> dict:
    """
    Simulates the engineer's Field App submission.
    mode='demo'   -> the first BBU on the site fails, for repeatable demos.
    mode='random' -> 10% fault chance per serialized unit.
    mode='clean'  -> no faults.
    """
    rng = random.Random(seed)
    installed, faults = [], []
    demo_done = False

    for item in dispatched_items:
        sku, sn = item["sku"], item.get("serial_number")
        faulty = False
        if sn:
            if mode == "demo" and sku == DEMO_FAULT_SKU and not demo_done:
                faulty, demo_done = True, True
            elif mode == "random" and rng.random() < 0.10:
                faulty = True

        if faulty:
            note = FAULT_NOTES.get(sku, "{sn} failed during installation.").format(sn=sn)
            faults.append({"sku": sku, "serial_number": sn, "qty": 1, "fault_log": f"{site_id}: {note}"})
        else:
            installed.append({"sku": sku, "serial_number": sn, "qty": item["qty"]})

    return {"work_order_id": work_order_id, "site_id": site_id, "engineer_id": engineer_id,
            "installed_items": installed, "faulty_items": faults}


def quarantine_faulty_unit(sku: str, serial_number: str, engineer_id: str, work_order_id: str,
                           txn_date: str = None) -> dict:
    """Books the returned faulty serial into quarantine. It can never be picked for dispatch again."""
    txn_date = txn_date or datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(WMS_DB)
    conn.execute("""
        INSERT INTO wms_ledger
        (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
        VALUES (?, ?, ?, 1, ?, ?, 'QUARANTINE', ?)
    """, (sku, get_sku_description(sku), serial_number, txn_date, engineer_id, work_order_id))
    conn.commit()
    conn.close()
    return {"status": "QUARANTINED", "sku": sku, "serial_number": serial_number, "bay": "BAY-Q1"}


def process_field_report(report: dict, txn_date: str = None) -> dict:
    """Records installs, triages faults with RAG, quarantines faulty serials, returns replenishment requests."""
    wo = report["work_order_id"]
    record_ledger_quantity(wo, report["installed_items"], "qty_installed")

    replenishment_requests, triage_results = [], []
    for fault in report["faulty_items"]:
        rag = query_telecom_rag(fault["fault_log"], source_filter="telecom_fault_manual.txt")
        decision = rag["agent_decision"]
        triage_results.append({"serial_number": fault["serial_number"], "sku": fault["sku"],
                               "decision": decision, "retrieved_context": rag["retrieved_context"]})

        quarantine_faulty_unit(fault["sku"], fault["serial_number"], report["engineer_id"], wo, txn_date)
        record_ledger_quantity(wo, [fault], "qty_quarantined")

        if decision.get("trigger_escalation_or_replenishment"):
            replenishment_requests.append({"sku": fault["sku"], "qty": 1, "serialized": True,
                                           "reason": decision.get("policy_code", "FAULT"),
                                           "failed_serial": fault["serial_number"],
                                           "collector": report["engineer_id"]})

    return {"work_order_id": wo, "site_id": report["site_id"],
            "triage_results": triage_results, "replenishment_requests": replenishment_requests}
