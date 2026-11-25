import os
import json
import sqlite3
from datetime import datetime

ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"
SUPPLIERS_DB = "databases/suppliers.db"
INVOICES_DIR = "invoices"
os.makedirs(INVOICES_DIR, exist_ok=True)


def run_three_way_reconciliation(work_order_id: str) -> dict:
    """
    Verifies for every SKU on a work order:
      1. qty_dispatched == qty_installed + qty_quarantined  (material accountability)
      2. qty_installed  == qty_planned                      (site completion)
    """
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT id, site_id, sku, qty_planned, qty_dispatched, qty_installed, qty_quarantined
        FROM reconciliation_ledger
        WHERE work_order_id = ?
    """, (work_order_id,)).fetchall()

    if not rows:
        conn.close()
        return {"status": "FAILED", "reason": f"No ledger rows found for {work_order_id}."}

    discrepancies = []
    line_items = []
    site_id = rows[0][1]

    for row_id, _, sku, planned, dispatched, installed, quarantined in rows:
        balanced = (dispatched == installed + quarantined) and (installed == planned)
        status = "RECONCILED" if balanced else "DISCREPANCY"

        cur.execute(
            "UPDATE reconciliation_ledger SET reconciliation_status = ? WHERE id = ?",
            (status, row_id),
        )

        entry = {
            "sku": sku,
            "planned": planned,
            "dispatched": dispatched,
            "installed": installed,
            "quarantined": quarantined,
            "status": status,
        }
        line_items.append(entry)
        if not balanced:
            discrepancies.append(entry)

    all_reconciled = len(discrepancies) == 0
    if all_reconciled:
        cur.execute("UPDATE work_orders SET status = 'RECONCILED' WHERE work_order_id = ?", (work_order_id,))

    conn.commit()
    conn.close()

    return {
        "work_order_id": work_order_id,
        "site_id": site_id,
        "reconciled": all_reconciled,
        "line_items": line_items,
        "discrepancies": discrepancies,
    }


def generate_client_invoice(work_order_id: str, markup_pct: float = 0.15) -> dict:
    """Generates a client invoice JSON only after 3-way reconciliation passes, then closes the WO."""
    recon = run_three_way_reconciliation(work_order_id)
    if not recon.get("reconciled"):
        return {
            "status": "BLOCKED",
            "reason": "Cannot invoice until all 3-way reconciliation discrepancies are resolved.",
            "discrepancies": recon.get("discrepancies", []),
        }

    # Pull baseline unit costs from suppliers.db
    sup_conn = sqlite3.connect(SUPPLIERS_DB)
    cost_map = {
        r[0]: r[1]
        for r in sup_conn.execute("SELECT sku, MIN(unit_price_gbp) FROM supplier_catalog GROUP BY sku")
    }
    sup_conn.close()

    invoice_lines = []
    subtotal = 0.0
    for item in recon["line_items"]:
        unit_cost = cost_map.get(item["sku"], 0.0)
        billable_unit = round(unit_cost * (1 + markup_pct), 2)
        line_total = round(billable_unit * item["installed"], 2)
        subtotal += line_total
        invoice_lines.append({
            "sku": item["sku"],
            "qty_installed": item["installed"],
            "unit_price_gbp": billable_unit,
            "line_total_gbp": line_total,
        })

    vat = round(subtotal * 0.20, 2)
    grand_total = round(subtotal + vat, 2)
    invoice_id = f"INV-{work_order_id}"
    invoice_data = {
        "invoice_id": invoice_id,
        "work_order_id": work_order_id,
        "site_id": recon["site_id"],
        "issued_date": datetime.now().strftime("%Y-%m-%d"),
        "currency": "GBP",
        "lines": invoice_lines,
        "subtotal_gbp": round(subtotal, 2),
        "vat_20pct_gbp": vat,
        "grand_total_gbp": grand_total,
        "status": "ISSUED",
    }

    filepath = os.path.join(INVOICES_DIR, f"{invoice_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(invoice_data, f, indent=2)

    # Close the work order
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    conn.execute("UPDATE work_orders SET status = 'CLOSED' WHERE work_order_id = ?", (work_order_id,))
    conn.commit()
    conn.close()

    return {"status": "INVOICED_AND_CLOSED", "invoice_file": filepath, "grand_total_gbp": grand_total}