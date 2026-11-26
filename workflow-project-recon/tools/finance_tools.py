import os
import json
import sqlite3
from datetime import datetime
from tools.logistics_tools import get_delivered_quantities

ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"
INVOICES_DIR = "invoices"
os.makedirs(INVOICES_DIR, exist_ok=True)


def run_three_way_reconciliation(work_order_id: str) -> dict:
    """
    Per SKU on a work order:
      1. WMS vs carrier:  qty_dispatched == qty delivered on signed waybills
      2. Accountability:  qty_dispatched == installed + quarantined + returned
      3. Completion:      serialized items only, installed == planned
         (bulk quantities are estimates, so unused bulk is returned, not treated as missing)
    """
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    rows = conn.execute("""
        SELECT id, site_id, sku, serialized, qty_planned, qty_dispatched, qty_installed, qty_quarantined, qty_returned
        FROM reconciliation_ledger WHERE work_order_id = ?
    """, (work_order_id,)).fetchall()
    if not rows:
        conn.close()
        return {"status": "FAILED", "reason": f"No ledger rows found for {work_order_id}."}

    delivered = get_delivered_quantities(work_order_id)
    line_items, discrepancies = [], []

    for row_id, site_id, sku, serialized, planned, dispatched, installed, quarantined, returned in rows:
        issues = []
        if delivered.get(sku, 0) != dispatched:
            issues.append(f"carrier delivered {delivered.get(sku, 0)} but WMS dispatched {dispatched}")
        unaccounted = dispatched - (installed + quarantined + returned)
        if unaccounted != 0:
            issues.append(f"{unaccounted} unaccounted (dispatched {dispatched}, installed {installed}, "
                          f"quarantined {quarantined}, returned {returned})")
        if serialized and installed != planned:
            issues.append(f"installed {installed} of {planned} planned")

        status = "RECONCILED" if not issues else "DISCREPANCY"
        conn.execute("UPDATE reconciliation_ledger SET reconciliation_status = ? WHERE id = ?", (status, row_id))
        entry = {"sku": sku, "serialized": bool(serialized), "planned": planned, "dispatched": dispatched,
                 "delivered": delivered.get(sku, 0), "installed": installed, "quarantined": quarantined,
                 "returned": returned, "unaccounted": unaccounted, "status": status, "issues": issues}
        line_items.append(entry)
        if issues:
            discrepancies.append(entry)

    reconciled = not discrepancies
    conn.execute("UPDATE work_orders SET status = ? WHERE work_order_id = ?",
                 ("RECONCILED" if reconciled else "RECON_DISCREPANCY", work_order_id))
    conn.commit()
    conn.close()
    return {"work_order_id": work_order_id, "site_id": rows[0][1], "reconciled": reconciled,
            "line_items": line_items, "discrepancies": discrepancies}


def validate_billing(lines: list) -> list:
    """Flags any billed rate outside the contracted variance (SLA-BILLING-02). The ML check plugs in here later."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    contract = {c: (r, v) for c, r, v in conn.execute(
        "SELECT code, unit_rate_gbp, allowed_variance_pct FROM contract_rates")}
    conn.close()
    anomalies = []
    for line in lines:
        rate, var = contract[line["code"]]
        deviation_pct = abs(line["unit_price_gbp"] - rate) / rate * 100
        if deviation_pct > var:
            anomalies.append({"code": line["code"], "billed": line["unit_price_gbp"], "contract": rate,
                              "deviation_pct": round(deviation_pct, 2), "allowed_pct": var})
    return anomalies


def generate_client_invoice(work_order_id: str) -> dict:
    """Builds the invoice at contract rates once reconciliation passes, then waits for human sign off."""
    recon = run_three_way_reconciliation(work_order_id)
    if not recon.get("reconciled"):
        return {"status": "BLOCKED",
                "reason": "Cannot invoice until all reconciliation discrepancies are resolved.",
                "discrepancies": recon.get("discrepancies", [])}

    conn = sqlite3.connect(ORCHESTRATOR_DB)
    rates = {c: (d, r, b) for c, d, r, b in conn.execute(
        "SELECT code, description, unit_rate_gbp, billable FROM contract_rates WHERE rate_type = 'MATERIAL'")}
    category, sched = conn.execute("SELECT site_category, scheduled_date FROM work_orders WHERE work_order_id = ?",
                                   (work_order_id,)).fetchone()
    labour = conn.execute("SELECT code, description, unit_rate_gbp FROM contract_rates "
                          "WHERE rate_type = 'LABOUR' AND site_category = ?", (category,)).fetchone()
    conn.close()

    lines = []
    for item in recon["line_items"]:
        desc, rate, billable = rates[item["sku"]]
        if not billable or item["installed"] == 0:
            continue  # consumables are covered by the contract allowance
        lines.append({"code": item["sku"], "description": desc, "qty": item["installed"],
                      "unit_price_gbp": rate, "line_total_gbp": round(rate * item["installed"], 2)})
    lines.append({"code": labour[0], "description": labour[1], "qty": 1,
                  "unit_price_gbp": labour[2], "line_total_gbp": labour[2]})

    anomalies = validate_billing(lines)
    subtotal = round(sum(l["line_total_gbp"] for l in lines), 2)
    vat = round(subtotal * 0.20, 2)
    invoice_id = f"INV-{work_order_id}"
    status = "HELD_ANOMALY" if anomalies else "PENDING_APPROVAL"

    invoice = {"invoice_id": invoice_id, "work_order_id": work_order_id, "site_id": recon["site_id"],
               "site_category": category, "work_completed_date": sched,
               "created_date": datetime.now().strftime("%Y-%m-%d"), "currency": "GBP", "lines": lines,
               "subtotal_gbp": subtotal, "vat_20pct_gbp": vat, "grand_total_gbp": round(subtotal + vat, 2),
               "billing_anomalies": anomalies, "status": status}
    filepath = os.path.join(INVOICES_DIR, f"{invoice_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(invoice, f, indent=2)

    conn = sqlite3.connect(ORCHESTRATOR_DB)
    conn.execute("INSERT OR REPLACE INTO invoices (invoice_id, work_order_id, grand_total_gbp, status, "
                 "anomalies_json, created_date) VALUES (?, ?, ?, ?, ?, ?)",
                 (invoice_id, work_order_id, invoice["grand_total_gbp"], status, json.dumps(anomalies),
                  invoice["created_date"]))
    conn.execute("UPDATE work_orders SET status = 'AWAITING_SIGNOFF' WHERE work_order_id = ?", (work_order_id,))
    conn.commit()
    conn.close()
    return {"status": status, "invoice_id": invoice_id, "invoice_file": filepath,
            "grand_total_gbp": invoice["grand_total_gbp"], "billing_anomalies": anomalies}


def record_invoice_decision(invoice_id: str, approver: str, approved: bool) -> dict:
    """Human sign off (KAM / client). Only an approval closes the work order."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    row = conn.execute("SELECT work_order_id FROM invoices WHERE invoice_id = ?", (invoice_id,)).fetchone()
    if not row:
        conn.close()
        return {"status": "FAILED", "reason": f"Invoice {invoice_id} not found."}
    new_status = "APPROVED" if approved else "REJECTED"
    conn.execute("UPDATE invoices SET status = ?, approved_by = ?, approved_at = ? WHERE invoice_id = ?",
                 (new_status, approver, now, invoice_id))
    conn.execute("UPDATE work_orders SET status = ? WHERE work_order_id = ?",
                 ("CLOSED" if approved else "INVOICE_REJECTED", row[0]))
    conn.commit()
    conn.close()
    return {"status": new_status, "invoice_id": invoice_id, "approved_by": approver}
