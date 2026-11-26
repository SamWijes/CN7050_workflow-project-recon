"""
Project level operations used by the agents.
Each function works site by site and never aborts the whole project because one site fails.
"""
import sqlite3
from datetime import datetime, timedelta
from tools.orchestrator_tools import ORCHESTRATOR_DB, get_work_order_bom, record_ledger_quantity
from tools.inventory_tools import (check_bom_stock, query_suppliers, create_purchase_order,
                                   verify_and_inbound_packing_list, dispatch_bom_from_wms)
from tools.logistics_tools import schedule_inbound_shipment, create_outbound_waybill, confirm_site_delivery
from tools.field_tools import simulate_field_app_report, process_field_report
from tools.finance_tools import generate_client_invoice


def get_project_work_orders(project_name: str) -> list:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    rows = conn.execute("""
        SELECT work_order_id, site_id, site_category, scheduled_date, subcon_id, engineer_id, status
        FROM work_orders WHERE project_name = ? ORDER BY scheduled_date, site_id
    """, (project_name,)).fetchall()
    conn.close()
    keys = ["work_order_id", "site_id", "site_category", "scheduled_date", "subcon_id", "engineer_id", "status"]
    return [dict(zip(keys, r)) for r in rows]


def choose_supplier(suppliers: list, need_by: str, order_date: str) -> dict:
    """Cheapest supplier that can deliver in time; if none can, the fastest one."""
    days = (datetime.strptime(need_by, "%Y-%m-%d") - datetime.strptime(order_date, "%Y-%m-%d")).days
    on_time = [s for s in suppliers if s["lead_time_days"] <= days]
    if on_time:
        return min(on_time, key=lambda s: s["unit_price_gbp"])
    return min(suppliers, key=lambda s: s["lead_time_days"])

def source_shortage(sku: str, qty: int, serialized: bool, need_by: str, order_date: str = None,
                    supplier_id: str = None) -> dict:
    """PO -> inbound shipment -> packing list check -> stock booked in. supplier_id lets an agent pick."""
    order_date = order_date or datetime.now().strftime("%Y-%m-%d")
    suppliers = query_suppliers(sku, qty)
    if not suppliers:
        return {"sku": sku, "status": "FAILED", "reason": f"No supplier can supply {qty} of {sku}"}
    chosen = next((s for s in suppliers if s["supplier_id"] == supplier_id), None) if supplier_id else None
    chosen = chosen or choose_supplier(suppliers, need_by, order_date)
    po = create_purchase_order(chosen["supplier_id"], sku, qty)
    if po["status"] != "ORDERED":
        return {"sku": sku, "status": "FAILED", "reason": po["reason"]}

    arrival = (datetime.strptime(order_date, "%Y-%m-%d") + timedelta(days=chosen["lead_time_days"])).strftime("%Y-%m-%d")
    # Pallet space scales with quantity (10 serialized units or 500 bulk units per pallet)
    pallets = max(0.05, round(qty / (10 if serialized else 500), 2))
    ship = schedule_inbound_shipment(po["po_id"], pallets, arrival, serialized)
    if ship["status"] != "CLEARED_AND_SCHEDULED":
        return {"sku": sku, "po_id": po["po_id"], "status": ship["status"], "reason": ship.get("reason")}
    inbound = verify_and_inbound_packing_list(ship["shipment_id"], ship["packing_list"], txn_date=arrival)
    return {"sku": sku, "qty": qty, "po_id": po["po_id"], "supplier": chosen["supplier_name"],
            "unit_price_gbp": chosen["unit_price_gbp"], "arrival_date": arrival,
            "late": arrival > need_by, "status": inbound["status"]}


def source_and_inbound_project_shortages(project_name: str) -> dict:
    """Aggregates the planned BOM of every site in the project and sources the shortfall."""
    wos = get_project_work_orders(project_name)
    if not wos:
        return {"status": "FAILED", "reason": f"No work orders for {project_name}"}
    need_by = min(wo["scheduled_date"] for wo in wos)

    agg = {}
    for wo in wos:
        for item in get_work_order_bom(wo["work_order_id"]):
            a = agg.setdefault(item["sku"], {"sku": item["sku"], "qty": 0, "serialized": item["serialized"]})
            a["qty"] += item["qty"]

    stock = check_bom_stock(list(agg.values()))
    if stock["all_in_stock"]:
        return {"status": "ALL_IN_STOCK", "shortages_resolved": []}

    resolved = [source_shortage(s["sku"], s["shortage_qty"], s["serialized"], need_by) for s in stock["shortages"]]
    failed = [r for r in resolved if r["status"] != "RECEIVED_CONFIRMED"]
    return {"status": "SHORTAGES_RESOLVED" if not failed else "SHORTAGES_PARTIAL",
            "shortages_resolved": resolved, "failed": failed}


def dispatch_and_deliver(wo: dict, bom: list, collector: str) -> dict:
    """WMS dispatch -> ledger -> waybill -> proof of delivery on site."""
    dispatch = dispatch_bom_from_wms(wo["work_order_id"], wo["site_id"], bom, collector, txn_date=wo["scheduled_date"])
    if dispatch["status"] != "DISPATCHED":
        return dispatch
    record_ledger_quantity(wo["work_order_id"], dispatch["items"], "qty_dispatched")
    waybill = create_outbound_waybill(wo["work_order_id"], wo["site_id"], dispatch["items"], wo["scheduled_date"])
    confirm_site_delivery(waybill["waybill_id"], wo["engineer_id"], wo["scheduled_date"])
    dispatch["waybill_id"] = waybill["waybill_id"]
    return dispatch


def execute_site(wo: dict, field_mode: str = "demo") -> dict:
    """Dispatch, Field App report, RAG triage and the replacement loop for one site."""
    bom = get_work_order_bom(wo["work_order_id"])

    # Earlier fault replacements may have used stock planned for this site, so top up first
    topups = [source_shortage(s["sku"], s["shortage_qty"], s["serialized"], wo["scheduled_date"])
              for s in check_bom_stock(bom)["shortages"]]
    if any(t["status"] != "RECEIVED_CONFIRMED" for t in topups):
        return {"site_id": wo["site_id"], "status": "SOURCING_FAILED", "reason": topups}

    dispatch = dispatch_and_deliver(wo, bom, wo["subcon_id"])
    if dispatch["status"] != "DISPATCHED":
        return {"site_id": wo["site_id"], "status": "DISPATCH_FAILED", "reason": dispatch}

    report = simulate_field_app_report(wo["work_order_id"], wo["site_id"], wo["engineer_id"],
                                       dispatch["items"], mode=field_mode)
    triage = process_field_report(report, txn_date=wo["scheduled_date"])

    replacements = []
    for req in triage["replenishment_requests"]:
        rep_bom = [{"sku": req["sku"], "qty": req["qty"], "serialized": True}]
        if not check_bom_stock(rep_bom)["all_in_stock"]:
            src = source_shortage(req["sku"], req["qty"], True, wo["scheduled_date"])
            if src["status"] != "RECEIVED_CONFIRMED":
                replacements.append({"failed_serial": req["failed_serial"], "status": "SOURCING_FAILED", "detail": src})
                continue
        rep = dispatch_and_deliver(wo, rep_bom, req["collector"])
        if rep["status"] != "DISPATCHED":
            replacements.append({"failed_serial": req["failed_serial"], "status": "DISPATCH_FAILED", "detail": rep})
            continue
        process_field_report({"work_order_id": wo["work_order_id"], "site_id": wo["site_id"],
                              "engineer_id": req["collector"], "installed_items": rep["items"],
                              "faulty_items": []}, txn_date=wo["scheduled_date"])
        replacements.append({"failed_serial": req["failed_serial"],
                             "replacement_serial": rep["items"][0]["serial_number"], "status": "INSTALLED"})

    return {"site_id": wo["site_id"], "status": "FIELD_COMPLETE",
            "faults": [{"serial": t["serial_number"], "sku": t["sku"],
                        "policy": t["decision"].get("policy_code")} for t in triage["triage_results"]],
            "replacements": replacements}


def execute_project_dispatch_and_field(project_name: str, field_mode: str = "demo") -> dict:
    results = [execute_site(wo, field_mode) for wo in get_project_work_orders(project_name)]
    return {"status": "FIELD_EXECUTION_COMPLETE" if all(r["status"] == "FIELD_COMPLETE" for r in results)
            else "FIELD_EXECUTION_PARTIAL", "sites": results}


def reconcile_and_invoice_project(project_name: str) -> dict:
    invoiced, blocked = [], []
    for wo in get_project_work_orders(project_name):
        res = generate_client_invoice(wo["work_order_id"])
        if res["status"] in ("PENDING_APPROVAL", "HELD_ANOMALY"):
            invoiced.append({"site_id": wo["site_id"], "invoice_id": res["invoice_id"],
                             "status": res["status"], "total_gbp": res["grand_total_gbp"]})
        else:
            blocked.append({"site_id": wo["site_id"],
                            "issues": [i for d in res["discrepancies"] for i in d["issues"]]})
    return {"status": "FINANCE_COMPLETE", "invoices_awaiting_signoff": invoiced, "sites_blocked": blocked,
            "total_pending_gbp": round(sum(i["total_gbp"] for i in invoiced), 2)}
