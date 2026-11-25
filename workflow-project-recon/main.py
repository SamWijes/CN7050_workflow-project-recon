import sys
import json
import sqlite3
from database.setup_databases import init_all_databases, ORCHESTRATOR_DB
from tools.orchestrator_tools import create_work_order_and_notify
from tools.inventory_tools import (
    check_bom_stock,
    query_suppliers,
    create_purchase_order,
    verify_and_inbound_packing_list,
    dispatch_bom_from_wms,
)
from tools.logistics_tools import schedule_inbound_shipment, create_outbound_waybill
from tools.field_tools import simulate_field_app_report, process_field_report
from tools.finance_tools import generate_client_invoice


def record_dispatched_in_ledger(work_order_id: str, dispatched_items: list):
    """Updates qty_dispatched in orchestrator_ledger.db after WMS dispatch."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()
    for item in dispatched_items:
        cur.execute("""
            UPDATE reconciliation_ledger
            SET qty_dispatched = qty_dispatched + ?
            WHERE work_order_id = ? AND sku = ?
        """, (item["qty"], work_order_id, item["sku"]))
    conn.commit()
    conn.close()


def fulfill_shortages(shortages: list, inbound_date: str):
    """Inventory & Logistics agents source, clear, and inbound any missing SKUs."""
    for s in shortages:
        suppliers = query_suppliers(s["sku"], s["shortage_qty"])
        chosen = suppliers[0]
        po = create_purchase_order(chosen["supplier_id"], s["sku"], s["shortage_qty"])
        print(f"  [Inventory] PO {po['po_id']} -> {chosen['supplier_name']} ({chosen['supplier_type']}) for {s['shortage_qty']}x {s['sku']}")

        ship = schedule_inbound_shipment(po["po_id"], pallets_required=1, inbound_date=inbound_date, serialized=s["serialized"])
        print(f"  [Logistics] Cleared {ship['shipment_id']} -> Bay {ship['assigned_bay']}")

        inbound = verify_and_inbound_packing_list(ship["shipment_id"], ship["packing_list"])
        print(f"  [Inventory] Inbound status: {inbound['status']} ({inbound['total_qty']} units)")


def run_project_workflow(project_name: str, site_id: str, site_category: str, scheduled_date: str, field_mode: str = "demo"):
    print(f"\n{'='*65}\nSTARTING 5-AGENT WORKFLOW: {site_id} ({site_category}) on {scheduled_date}\n{'='*65}")

    # 1. Main Orchestrator (Planning & Notifications)
    plan = create_work_order_and_notify(project_name, site_id, site_category, scheduled_date)
    if plan["status"] != "PLANNED":
        print("Planning failed:", plan)
        return
    wo_id = plan["work_order_id"]
    print(f"[Orchestrator] Planned {wo_id} | Subcon: {plan['subcon_id']} | Engineer: {plan['engineer_id']} | Emails: 3 sent")

    # 2. Inventory & Sourcing + Logistics (Check Stock & Source Shortages)
    stock = check_bom_stock(plan["bom"])
    if not stock["all_in_stock"]:
        print(f"[Inventory] Shortages detected: {[s['sku'] for s in stock['shortages']]}")
        fulfill_shortages(stock["shortages"], inbound_date=scheduled_date)

    # 3. Dispatch BOM from WMS & Create Outbound Waybill
    dispatch = dispatch_bom_from_wms(wo_id, site_id, plan["bom"], collector=plan["subcon_id"])
    record_dispatched_in_ledger(wo_id, dispatch["items"])
    waybill = create_outbound_waybill(wo_id, site_id, dispatch["items"], scheduled_date)
    print(f"[Logistics] Dispatched {len(dispatch['items'])} lines | Waybill: {waybill['waybill_id']}")

    # 4. Field Execution Agent (App Report + RAG Fault Triage)
    field_report = simulate_field_app_report(wo_id, site_id, plan["engineer_id"], dispatch["items"], mode=field_mode)
    triage = process_field_report(field_report)
    print(f"[Field Agent] Installed: {len(field_report['installed_items'])} lines | Faults: {len(field_report['faulty_items'])}")

    # 5. Automated Fault Replenishment Loop (if RAG triggered replenishment)
    if triage["replenishment_requests"]:
        print("[Replenishment Loop] Triggered by RAG policy:")
        for req in triage["replenishment_requests"]:
            print(f"  -> {req['reason']}: Replacing {req['failed_serial']} ({req['sku']}) via {req['collector']}")
            rep_bom = [{"sku": req["sku"], "qty": req["qty"], "serialized": req["serialized"]}]
            rep_stock = check_bom_stock(rep_bom)
            if not rep_stock["all_in_stock"]:
                fulfill_shortages(rep_stock["shortages"], inbound_date=scheduled_date)

            rep_dispatch = dispatch_bom_from_wms(wo_id, site_id, rep_bom, collector=req["collector"])
            record_dispatched_in_ledger(wo_id, rep_dispatch["items"])

            # Engineer installs the replacement unit cleanly
            rep_install = {
                "work_order_id": wo_id,
                "site_id": site_id,
                "engineer_id": req["collector"],
                "installed_items": rep_dispatch["items"],
                "faulty_items": [],
            }
            process_field_report(rep_install)
            print(f"  -> Replacement installed: {rep_dispatch['items'][0]['serial_number']}")

    # 6. Invoicing & Financial Reconciliation Agent
    invoice = generate_client_invoice(wo_id)
    print(f"[Finance Agent] {invoice['status']} | File: {invoice.get('invoice_file')} | Total: £{invoice.get('grand_total_gbp')}")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    init_all_databases(force_reset=reset)
    run_project_workflow("5G-Rollout-Essex", "SITE-104", "full_site", "2026-12-01", field_mode="demo")