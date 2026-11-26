"""
LangChain tools for each agent. They wrap the Phase 1 functions and keep their output short,
because a 7B model works best with small, clear tool results.
"""
import os
import json
import sqlite3
import threading
from typing import Optional
from langchain_core.tools import tool

from agents.events import log_event
from agents.sim_config import SIM
from rag.rag_query import retrieve_policies
from tools.orchestrator_tools import (ORCHESTRATOR_DB, WMS_DB, get_work_order, get_work_order_bom,
                                      set_work_order_status, record_ledger_quantity)
from tools.inventory_tools import check_bom_stock, query_suppliers
from tools.logistics_tools import get_delivered_quantities
from tools.field_tools import process_field_report, quarantine_faulty_unit
from tools.inventory_tools import record_return, get_outbound_history, get_sku_description
from tools.mailbox_tools import send_mail, take_new_replies
from simulation.field_app import simulate_site_report
from agents import hitl
from tools.finance_tools import run_three_way_reconciliation, generate_client_invoice
from tools.project_tools import get_project_work_orders, source_shortage, dispatch_and_deliver



def _wo_or_error(work_order_id: str):
    wo = get_work_order(work_order_id)
    return wo, (None if wo else {"error": f"Unknown work order {work_order_id}"})


def _claim(work_order_id: str, from_status: str, to_status: str) -> bool:
    """Atomically moves a work order between stages. Only one caller can win, so duplicate or
    parallel tool calls from the LLM can never process the same site twice."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.execute("UPDATE work_orders SET status = ? WHERE work_order_id = ? AND status = ?",
                       (to_status, work_order_id, from_status))
    conn.commit()
    conn.close()
    return cur.rowcount == 1


_replacement_lock = threading.Lock()


def _is_serialized(sku: str) -> bool:
    conn = sqlite3.connect(WMS_DB)
    row = conn.execute("SELECT serialized FROM sku_master WHERE sku = ?", (sku,)).fetchone()
    conn.close()
    return bool(row and row[0])


# ---------------------------------------------------------------------------
# Shared tools (every agent)
# ---------------------------------------------------------------------------

@tool
def search_knowledge(query: str, source: str) -> list:
    """Search company policy documents. source must be 'fault_manual' (fault codes and field actions)
    or 'contract' (contract rates, reconciliation and billing rules). Returns the matching policy text."""
    return retrieve_policies(query, source)


@tool
def get_site_status(work_order_id: str) -> dict:
    """Current stage, team, BOM ledger and invoice status for one work order."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    lines = conn.execute("""
        SELECT sku, qty_planned, qty_dispatched, qty_installed, qty_quarantined, qty_returned, reconciliation_status
        FROM reconciliation_ledger WHERE work_order_id = ? ORDER BY id
    """, (work_order_id,)).fetchall()
    inv = conn.execute("SELECT invoice_id, status, grand_total_gbp FROM invoices WHERE work_order_id = ?",
                       (work_order_id,)).fetchone()
    conn.close()
    wo["ledger"] = [dict(zip(["sku", "planned", "dispatched", "installed", "quarantined", "returned", "status"], l))
                    for l in lines]
    wo["invoice"] = dict(zip(["invoice_id", "status", "total_gbp"], inv)) if inv else None
    return wo


@tool
def list_project_sites(project_name: str) -> list:
    """All work orders in a project with their current stage."""
    return [{k: wo[k] for k in ("work_order_id", "site_category", "scheduled_date", "status")}
            for wo in get_project_work_orders(project_name)]


# ---------------------------------------------------------------------------
# Inventory & Sourcing Agent
# ---------------------------------------------------------------------------

@tool
def get_stock_level(sku: str) -> dict:
    """Good units available in the warehouse for one SKU (quarantined units excluded)."""
    res = check_bom_stock([{"sku": sku, "qty": 0, "serialized": _is_serialized(sku)}])
    return {"sku": sku, "available": res["stock_check"][0]["on_hand"]}


def _reserved(exclude_wo: str) -> dict:
    """Stock already promised to other sites that are STOCK_READY but not yet dispatched."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    rows = conn.execute("""
        SELECT l.sku, SUM(l.qty_planned - l.qty_dispatched) FROM reconciliation_ledger l
        JOIN work_orders w ON w.work_order_id = l.work_order_id
        WHERE w.status IN ('STOCK_READY', 'DISPATCHING') AND w.work_order_id != ?
        GROUP BY l.sku
    """, (exclude_wo,)).fetchall()
    conn.close()
    return {sku: q for sku, q in rows if q > 0}


def _site_shortages(work_order_id: str) -> list:
    """Shortages for this site after allowing for stock reserved by other ready sites."""
    reserved = _reserved(work_order_id)
    bom = [dict(i, qty=i["qty"] + reserved.get(i["sku"], 0)) for i in get_work_order_bom(work_order_id)]
    return [{"sku": s["sku"], "shortage_qty": s["shortage_qty"]} for s in check_bom_stock(bom)["shortages"]]


@tool
def check_site_stock(work_order_id: str) -> dict:
    """Compares the site's planned BOM with warehouse stock and lists any shortages."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    shortages = _site_shortages(work_order_id)
    return {"work_order_id": work_order_id, "need_by": wo["scheduled_date"],
            "all_in_stock": not shortages, "shortages": shortages}


@tool
def compare_suppliers(sku: str, qty: int) -> list:
    """Suppliers able to supply qty of sku, with unit price (GBP) and lead time (days)."""
    return [{k: s[k] for k in ("supplier_id", "supplier_name", "supplier_type", "unit_price_gbp", "lead_time_days")}
            for s in query_suppliers(sku, qty)]


@tool
def order_from_supplier(work_order_id: str, sku: str, qty: int, supplier_id: str) -> dict:
    """Raises a PO with the chosen supplier, receives the shipment and books the stock into the warehouse."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    res = source_shortage(sku, qty, _is_serialized(sku), wo["scheduled_date"], supplier_id=supplier_id)
    log_event("Inventory Agent", "PURCHASE_ORDER", res, work_order_id)
    return {k: res.get(k) for k in ("sku", "qty", "po_id", "supplier", "unit_price_gbp", "arrival_date", "late", "status")}

@tool
def order_site_shortages(work_order_id: str, strategy: str = "cheapest_on_time") -> dict:
    """Orders every shortage for the site in one step. strategy: 'cheapest_on_time' (cheapest supplier
    that still meets the need_by date) or 'fastest' (shortest lead time, use when the date is at risk)."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    orders = []
    for sh in _site_shortages(work_order_id):
        suppliers = query_suppliers(sh["sku"], sh["shortage_qty"])
        pick = None
        if strategy == "fastest" and suppliers:
            pick = min(suppliers, key=lambda x: x["lead_time_days"])["supplier_id"]
        res = source_shortage(sh["sku"], sh["shortage_qty"], _is_serialized(sh["sku"]), wo["scheduled_date"],
                              supplier_id=pick)
        log_event("Inventory Agent", "PURCHASE_ORDER", res, work_order_id)
        orders.append({k: res.get(k) for k in ("sku", "qty", "supplier", "arrival_date", "late", "status")})
    return {"work_order_id": work_order_id, "strategy": strategy, "orders": orders,
            "next_step": "call confirm_site_stock_ready"}

@tool
def confirm_site_stock_ready(work_order_id: str) -> dict:
    """Confirms every BOM line is in stock (allowing for other sites' reservations) and marks the
    site STOCK_READY for Logistics."""
    with _replacement_lock:  # serialize reservations so two sites cannot claim the same unit
        shortages = _site_shortages(work_order_id)
        if shortages:
            return {"status": "NOT_READY", "work_order_id": work_order_id, "shortages": shortages}
        if not _claim(work_order_id, "PLANNED", "STOCK_READY"):
            return {"status": "UNCHANGED", "reason": "Site is not in PLANNED stage."}
    log_event("Inventory Agent", "STOCK_READY", None, work_order_id)
    return {"status": "STOCK_READY", "work_order_id": work_order_id}

# ---------------------------------------------------------------------------
# Inventory & Sourcing Agent: project level procurement (one PO per part for all sites)
# ---------------------------------------------------------------------------

def _planned_sites(project_name: str) -> list:
    return [wo for wo in get_project_work_orders(project_name) if wo["status"] == "PLANNED"]


def _project_shortages(project_name: str) -> tuple:
    """Total shortage for every PLANNED site, after stock already reserved by ready sites."""
    planned = _planned_sites(project_name)
    agg = {}
    for wo in planned:
        for item in get_work_order_bom(wo["work_order_id"]):
            a = agg.setdefault(item["sku"], {"sku": item["sku"], "qty": 0, "serialized": item["serialized"]})
            a["qty"] += item["qty"]
    reserved = _reserved("")
    bom = [dict(a, qty=a["qty"] + reserved.get(a["sku"], 0)) for a in agg.values()]
    shortages = check_bom_stock(bom)["shortages"]
    need_by = min((wo["scheduled_date"] for wo in planned), default=None)
    return planned, shortages, need_by


@tool
def check_project_stock(project_name: str) -> dict:
    """Combined shortage across every PLANNED site in the project (what one project order must cover)."""
    planned, shortages, need_by = _project_shortages(project_name)
    return {"project_name": project_name, "planned_sites": [w["work_order_id"] for w in planned], "need_by": need_by,
            "shortages": [{"sku": s["sku"], "shortage_qty": s["shortage_qty"]} for s in shortages]}


@tool
def plan_project_procurement(project_name: str, strategy: str = "cheapest_on_time") -> dict:
    """Raises ONE purchase order per part covering every PLANNED site, then marks each site STOCK_READY
    (earliest date first). strategy: 'cheapest_on_time' or 'fastest'. Reports any order that will be late."""
    orders, ready, not_ready = [], [], []
    with _replacement_lock:  # a duplicate call waits, then finds nothing left to order
        planned, shortages, need_by = _project_shortages(project_name)
        if not planned:
            return {"status": "NOTHING_PLANNED", "project_name": project_name}
        for sh in shortages:
            suppliers = query_suppliers(sh["sku"], sh["shortage_qty"])
            pick = min(suppliers, key=lambda x: x["lead_time_days"])["supplier_id"] \
                if strategy == "fastest" and suppliers else None
            res = source_shortage(sh["sku"], sh["shortage_qty"], sh["serialized"], need_by, supplier_id=pick)
            log_event("Inventory Agent", "PROJECT_PO", res)
            orders.append({k: res.get(k) for k in ("sku", "qty", "supplier", "arrival_date", "late", "status", "reason")})
        for wo in sorted(planned, key=lambda w: w["scheduled_date"]):
            if not _site_shortages(wo["work_order_id"]) and _claim(wo["work_order_id"], "PLANNED", "STOCK_READY"):
                ready.append(wo["work_order_id"])
                log_event("Inventory Agent", "STOCK_READY", None, wo["work_order_id"])
            else:
                not_ready.append(wo["work_order_id"])
    late = [o for o in orders if o.get("late") or o.get("status") != "RECEIVED_CONFIRMED"]
    return {"project_name": project_name, "purchase_orders": len(orders), "orders": orders,
            "sites_ready": ready, "sites_not_ready": not_ready,
            "conflicts": late, "advice": "If any order is late or failed, use ask_human." if late else "All on time."}



# ---------------------------------------------------------------------------
# Logistics & Transport Agent
# ---------------------------------------------------------------------------

@tool
def dispatch_site(work_order_id: str) -> dict:
    """Picks the BOM from the warehouse, creates the waybill and records proof of delivery on site."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    if not _claim(work_order_id, "STOCK_READY", "DISPATCHING"):
        return {"error": f"{work_order_id} is {wo['status']}, it must be STOCK_READY before dispatch."}
    res = dispatch_and_deliver(wo, get_work_order_bom(work_order_id), wo["subcon_id"])
    if res["status"] != "DISPATCHED":
        # Stock was used elsewhere (e.g. by a fault replacement): hand the site back to Inventory
        set_work_order_status(work_order_id, "PLANNED")
        log_event("Logistics Agent", "HANDBACK -> Inventory", res.get("shortages") or res.get("reason"), work_order_id)
        return {"status": "RETURNED_TO_INVENTORY", "reason": "Insufficient stock at dispatch time.",
                "shortages": res.get("shortages")}
    set_work_order_status(work_order_id, "DELIVERED")
    log_event("Logistics Agent", "DELIVERED", {"waybill": res["waybill_id"], "lines": len(res["items"])}, work_order_id)
    return {"status": "DELIVERED", "waybill_id": res["waybill_id"], "received_by": wo["engineer_id"],
            "lines_dispatched": len(res["items"])}


@tool
def get_delivery_record(work_order_id: str) -> dict:
    """Quantities delivered to site according to signed waybills."""
    return {"work_order_id": work_order_id, "delivered": get_delivered_quantities(work_order_id)}


# ---------------------------------------------------------------------------
# Field Execution Agent
# ---------------------------------------------------------------------------

def _pending_serialized(work_order_id: str) -> int:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    n = conn.execute("""SELECT COALESCE(SUM(qty_planned - qty_installed), 0) FROM reconciliation_ledger
                        WHERE work_order_id = ? AND serialized = 1""", (work_order_id,)).fetchone()[0]
    conn.close()
    return n


def _outstanding_faults(work_order_id: str) -> list:
    """Quarantined serials on this site whose SKU is still short of its planned install count."""
    conn = sqlite3.connect(WMS_DB)
    quarantined = conn.execute("SELECT sku, serial_number FROM wms_ledger WHERE transaction_type = 'QUARANTINE' "
                               "AND reference = ? ORDER BY id", (work_order_id,)).fetchall()
    conn.close()
    out, used = [], {}
    for sku, sn in quarantined:
        pending = _pending_serialized_sku(work_order_id, sku) - used.get(sku, 0)
        if pending > 0:
            out.append({"sku": sku, "failed_serial": sn, "replacement_needed": True})
            used[sku] = used.get(sku, 0) + 1
    return out


@tool
def run_site_installation(work_order_id: str) -> dict:
    """Reads the engineer's Field App report, records installs, quarantines faulty serials and
    classifies each fault against the fault manual. Returns faults that need a replacement.
    For a site already in REPLACEMENT_PENDING it returns the faults still waiting, so you can retry."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    if wo["status"] == "REPLACEMENT_PENDING":
        # Report already processed: return the faults still waiting for a replacement so they can be retried
        return {"work_order_id": work_order_id, "status": "REPLACEMENT_PENDING",
                "faults": _outstanding_faults(work_order_id), "collector": wo["engineer_id"]}
    if not _claim(work_order_id, "DELIVERED", "INSTALLING"):
        return {"error": f"{work_order_id} is {wo['status']}, materials must be DELIVERED first."}

    # Field App sees exactly what the carrier delivered
    conn = sqlite3.connect(WMS_DB)
    items = [{"sku": s, "serial_number": sn, "qty": -q} for s, sn, q in conn.execute(
        "SELECT sku, serial_number, qty FROM wms_ledger WHERE transaction_type = 'OUTBOUND' AND reference = ?",
        (work_order_id,))]
    conn.close()

    scenario = SIM["site_scenarios"].get(wo["site_id"], "happy_path")
    report = simulate_site_report(wo, items, SIM["field_mode"], scenario)
    triage = process_field_report(report, txn_date=wo["scheduled_date"])
    faults = [{"sku": t["sku"], "failed_serial": t["serial_number"],
               "policy_code": t["decision"].get("policy_code"),
               "replacement_needed": bool(t["decision"].get("trigger_escalation_or_replenishment"))}
              for t in triage["triage_results"]]

    status = "REPLACEMENT_PENDING" if _pending_serialized(work_order_id) else "INSTALLED"
    set_work_order_status(work_order_id, status)
    log_event("Field Agent", "FIELD_REPORT", {"installed_lines": len(report["installed_items"]), "faults": faults},
              work_order_id)
    return {"work_order_id": work_order_id, "status": status, "faults": faults, "collector": wo["engineer_id"]}


@tool
def request_replacement(work_order_id: str, sku: str, failed_serial: str) -> dict:
    """Direct replenishment loop with Inventory: sources a replacement if stock is short, dispatches it
    to the engineer and records the install. Use only for faults where the manual requires replacement."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    with _replacement_lock:
        if _pending_serialized_sku(work_order_id, sku) <= 0:
            return {"status": "NOT_NEEDED", "reason": f"No outstanding {sku} on {work_order_id}; already replaced."}
        return _do_replacement(wo, sku, failed_serial)


def _pending_serialized_sku(work_order_id: str, sku: str) -> int:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    row = conn.execute("SELECT qty_planned - qty_installed FROM reconciliation_ledger WHERE work_order_id = ? AND sku = ?",
                       (work_order_id, sku)).fetchone()
    conn.close()
    return row[0] if row else 0


def _do_replacement(wo: dict, sku: str, failed_serial: str) -> dict:
    work_order_id = wo["work_order_id"]
    rep_bom = [{"sku": sku, "qty": 1, "serialized": True}]
    log_event("Field Agent", "REPLACEMENT_REQUEST -> Inventory", {"sku": sku, "failed_serial": failed_serial},
              work_order_id)

    if not check_bom_stock(rep_bom)["all_in_stock"]:
        src = source_shortage(sku, 1, True, wo["scheduled_date"])
        log_event("Inventory Agent", "PURCHASE_ORDER (replacement)", src, work_order_id)
        if src["status"] != "RECEIVED_CONFIRMED":
            return {"status": "SOURCING_FAILED", "detail": src}

    rep = dispatch_and_deliver(wo, rep_bom, wo["engineer_id"])
    if rep["status"] != "DISPATCHED":
        return {"status": "DISPATCH_FAILED", "detail": rep.get("reason")}
    record_ledger_quantity(work_order_id, rep["items"], "qty_installed")

    new_serial = rep["items"][0]["serial_number"]
    if not _pending_serialized(work_order_id):
        set_work_order_status(work_order_id, "INSTALLED")
    log_event("Field Agent", "REPLACEMENT_INSTALLED", {"failed": failed_serial, "new": new_serial}, work_order_id)
    return {"status": "INSTALLED", "failed_serial": failed_serial, "replacement_serial": new_serial,
            "site_status": get_work_order(work_order_id)["status"]}

# ---------------------------------------------------------------------------
# Field Execution Agent: reconciliation investigations (email the person who holds the items)
# ---------------------------------------------------------------------------

def _unaccounted(work_order_id: str) -> list:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    rows = conn.execute("""
        SELECT sku, serialized, qty_dispatched - qty_installed - qty_quarantined - qty_returned
        FROM reconciliation_ledger WHERE work_order_id = ?
    """, (work_order_id,)).fetchall()
    conn.close()
    return [{"sku": s, "serialized": bool(ser), "unaccounted": u} for s, ser, u in rows if u != 0]


def _resource(resource_id: str):
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    row = conn.execute("SELECT resource_id, role, email FROM resources WHERE resource_id = ?", (resource_id,)).fetchone()
    conn.close()
    return dict(zip(["resource_id", "role", "email"], row)) if row else None


def _query_facts(work_order_id: str) -> list:
    """Per unaccounted SKU: part number, description, qty, serials, outbound date, who collected it."""
    facts = []
    for u in _unaccounted(work_order_id):
        history = get_outbound_history(work_order_id, u["sku"])
        if u["serialized"]:
            conn = sqlite3.connect(WMS_DB)
            q = {r[0] for r in conn.execute("SELECT serial_number FROM wms_ledger WHERE transaction_type = 'QUARANTINE' "
                                            "AND reference = ?", (work_order_id,))}
            conn.close()
            # Serials issued to someone other than the original subcon are the replacement chain
            candidates = [h for h in history if h["serial_number"] not in q]
            collector = history[-1]["collected_by"] if history else None
        else:
            candidates = history
            collector = history[0]["collected_by"] if history else None
        facts.append({"sku": u["sku"], "description": get_sku_description(u["sku"]),
                      "unaccounted_qty": u["unaccounted"], "serialized": u["serialized"],
                      "issued": [{"serial": h["serial_number"], "qty": h["qty"], "outbound_date": h["outbound_date"],
                                  "collected_by": h["collected_by"]} for h in candidates],
                      "contact": collector})
    return facts


@tool
def investigate_discrepancy(work_order_id: str) -> dict:
    """For a site that failed reconciliation: which items are unaccounted for, their part numbers,
    serials, outbound dates and who collected them (the person to contact)."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    facts = _query_facts(work_order_id)
    log_event("Field Agent", "INVESTIGATING", facts, work_order_id)
    return {"work_order_id": work_order_id, "site_id": wo["site_id"], "unaccounted_items": facts}


@tool
def send_email(work_order_id: str, recipient_id: str, subject: str, body: str) -> dict:
    """Email an engineer or subcontractor (recipient_id such as 'Subcon-B' or 'Engineer-C') and wait
    for their reply. The site pauses until they answer. Write the body yourself: be specific and polite."""
    person = _resource(recipient_id)
    if not person:
        return {"error": f"Unknown recipient {recipient_id}"}
    if not _claim(work_order_id, "RECON_DISCREPANCY", "AWAITING_REPLY") and \
       not _claim(work_order_id, "REPLY_RECEIVED", "AWAITING_REPLY"):
        return {"error": "Site is not under investigation, or an email is already awaiting a reply."}

    # Always attach the hard facts so the email is correct even if the model paraphrases
    ref = "\n".join(f"  - {f['sku']} {f['description']}: {f['unaccounted_qty']} unaccounted; issued "
                    + "; ".join(f"{i['serial'] or str(i['qty']) + ' units'} on {i['outbound_date']} to {i['collected_by']}"
                                for i in f["issued"])
                    for f in _query_facts(work_order_id))
    full_body = f"{body}\n\n--- Reconciliation reference ({work_order_id}) ---\n{ref}"
    msg = send_mail(work_order_id, "field.agent@intellicommstech.local", person["email"], subject, full_body,
                    expects_reply=True)
    log_event("Field Agent", f"EMAIL -> {recipient_id}", {"subject": subject}, work_order_id)
    return {"status": "AWAITING_REPLY", "message_id": msg["message_id"], "to": person["email"]}


@tool
def read_replies(work_order_id: str) -> dict:
    """New email replies received for this work order."""
    replies = take_new_replies(work_order_id)
    return {"work_order_id": work_order_id, "replies": replies,
            "still_unaccounted": [{"sku": f["sku"], "qty": f["unaccounted_qty"]} for f in _query_facts(work_order_id)]}


def _after_resolution(work_order_id: str) -> str:
    """If every item is now accounted for, send the site back to Finance to reconcile again."""
    if not _unaccounted(work_order_id) and not _pending_serialized(work_order_id):
        set_work_order_status(work_order_id, "INSTALLED")
        return "INSTALLED (ready to reconcile again)"
    return get_work_order(work_order_id)["status"]


@tool
def record_site_return(work_order_id: str, sku: str, qty: int, returned_by: str) -> dict:
    """Book unused bulk material that a subcontractor says they are returning to the warehouse."""
    open_qty = next((u["unaccounted"] for u in _unaccounted(work_order_id) if u["sku"] == sku), 0)
    if qty <= 0 or qty > open_qty:
        return {"error": f"{qty} is not valid: only {open_qty} of {sku} is unaccounted on {work_order_id}."}
    record_return(work_order_id, sku, qty, returned_by, txn_date=get_work_order(work_order_id)["scheduled_date"])
    log_event("Inventory Agent", "RETURN_BOOKED", {"sku": sku, "qty": qty, "by": returned_by}, work_order_id)
    return {"status": "RETURN_BOOKED", "sku": sku, "qty": qty, "site_status": _after_resolution(work_order_id)}


@tool
def book_late_quarantine(work_order_id: str, sku: str, serial_number: str, returned_by: str) -> dict:
    """Book a faulty serialized unit into quarantine when the engineer confirms it is back at the warehouse."""
    issued = {h["serial_number"] for h in get_outbound_history(work_order_id, sku)}
    if serial_number not in issued:
        return {"error": f"{serial_number} was never issued to {work_order_id}."}
    if not any(u["sku"] == sku and u["unaccounted"] > 0 for u in _unaccounted(work_order_id)):
        return {"error": f"Nothing unaccounted for {sku} on {work_order_id}."}
    quarantine_faulty_unit(sku, serial_number, returned_by, work_order_id,
                           txn_date=get_work_order(work_order_id)["scheduled_date"])
    record_ledger_quantity(work_order_id, [{"sku": sku, "qty": 1}], "qty_quarantined")
    log_event("Inventory Agent", "LATE_QUARANTINE", {"sku": sku, "serial": serial_number}, work_order_id)
    return {"status": "QUARANTINED", "serial": serial_number, "site_status": _after_resolution(work_order_id)}


@tool
def escalate_to_kam(work_order_id: str, reason: str) -> dict:
    """Hand the site to the Key Accounts Manager when items cannot be recovered (lost, wasted, disputed)."""
    set_work_order_status(work_order_id, "NEEDS_ATTENTION")
    log_event("Field Agent", "ESCALATED_TO_KAM", reason, work_order_id)
    return {"status": "NEEDS_ATTENTION", "reason": reason}

# ---------------------------------------------------------------------------
# Invoicing & Financial Agent
# ---------------------------------------------------------------------------

@tool
def reconcile_site(work_order_id: str) -> dict:
    """Three way reconciliation: WMS dispatch vs carrier delivery vs field install, quarantine and returns."""
    res = run_three_way_reconciliation(work_order_id)
    out = {"work_order_id": work_order_id, "reconciled": res.get("reconciled"),
           "issues": [{"sku": d["sku"], "issues": d["issues"]} for d in res.get("discrepancies", [])]}
    log_event("Finance Agent", "RECONCILED" if out["reconciled"] else "RECON_DISCREPANCY", out["issues"], work_order_id)
    return out


@tool
def create_invoice(work_order_id: str) -> dict:
    """Builds the invoice at contract rates for a reconciled site and sends it for human sign off."""
    res = generate_client_invoice(work_order_id)
    log_event("Finance Agent", f"INVOICE_{res['status']}",
              {k: res.get(k) for k in ("invoice_id", "grand_total_gbp", "billing_anomalies")}, work_order_id)
    if res["status"] == "BLOCKED":
        return {"status": "BLOCKED", "issues": [i for d in res["discrepancies"] for i in d["issues"]]}
    return {k: res.get(k) for k in ("status", "invoice_id", "grand_total_gbp", "billing_anomalies")}

# ---------------------------------------------------------------------------
# Invoicing & Financial Agent: reconciliation policy and batch reconciliation
# ---------------------------------------------------------------------------

@tool
def get_reconciliation_policy(project_name: str) -> dict:
    """When this project should be reconciled: end_of_project, every_n_sites, on_request, or unspecified."""
    return {"project_name": project_name, **hitl.get_policy(project_name)}


@tool
def set_reconciliation_policy(project_name: str, policy: str, every_n: int = 0, source: str = "") -> dict:
    """Store the reconciliation policy. policy: 'end_of_project', 'every_n_sites' (give every_n) or 'on_request'.
    source: where it came from, e.g. 'KAM answer' or 'contract clause'."""
    res = hitl.set_policy(project_name, policy, every_n, source)
    if "error" not in res:
        log_event("Finance Agent", "RECON_POLICY_SET", res)
    return res


@tool
def reconcile_batch(project_name: str, work_order_ids: list[str]) -> dict:
    """Three way reconciliation for a batch of installed sites. Clean sites become RECONCILED; sites with
    discrepancies go to the Field agent for investigation."""
    clean, issues = [], []
    for wo in work_order_ids:
        w = get_work_order(wo)
        if not w or w["status"] not in ("INSTALLED", "RECONCILED"):
            continue
        res = run_three_way_reconciliation(wo)
        if res.get("reconciled"):
            clean.append(wo)
        else:
            issues.append({"work_order_id": wo, "issues": [i for d in res.get("discrepancies", []) for i in d["issues"]]})
    log_event("Finance Agent", "BATCH_RECONCILED", {"clean": clean, "discrepancies": [i["work_order_id"] for i in issues]})
    return {"reconciled": clean, "discrepancies": issues,
            "next_step": "create_batch_invoice for the reconciled sites" if clean else "nothing to invoice"}


@tool
def create_batch_invoice(project_name: str, work_order_ids: list[str]) -> dict:
    """One consolidated invoice (contract rates, a section per site) for the RECONCILED sites in this batch.
    It is held for human sign off."""
    with _replacement_lock:
        wos = [wo for wo in work_order_ids if (get_work_order(wo) or {}).get("status") == "RECONCILED"]
        if not wos:
            return {"error": "No RECONCILED sites in this batch (already invoiced, or reconcile_batch first)."}
        sites, invoice_ids, anomalies = [], [], False
        for wo in wos:
            inv = generate_client_invoice(wo)
            if inv["status"] == "BLOCKED":
                continue
            invoice_ids.append(inv["invoice_id"])
            anomalies = anomalies or bool(inv["billing_anomalies"])
            with open(inv["invoice_file"], encoding="utf-8") as f:
                sites.append(json.load(f))
        if not invoice_ids:
            return {"error": "Every site in the batch failed final reconciliation."}
        n = len([1 for b in os.listdir("invoices") if b.startswith(f"BATCH-{project_name}-")]) + 1
        batch_id = f"BATCH-{project_name}-{n:02d}"
        subtotal = round(sum(s["subtotal_gbp"] for s in sites), 2)
        doc = {"batch_id": batch_id, "project_name": project_name, "currency": "GBP",
               "sites": [{"site_id": s["site_id"], "work_order_id": s["work_order_id"], "lines": s["lines"],
                          "subtotal_gbp": s["subtotal_gbp"]} for s in sites],
               "subtotal_gbp": subtotal, "vat_20pct_gbp": round(subtotal * 0.2, 2),
               "grand_total_gbp": round(subtotal * 1.2, 2), "billing_anomalies": anomalies,
               "status": "PENDING_APPROVAL"}
        path = os.path.join("invoices", f"{batch_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        hitl.save_batch(batch_id, project_name, wos, invoice_ids, doc["grand_total_gbp"], anomalies, path)
    log_event("Finance Agent", "BATCH_INVOICE", {"batch": batch_id, "sites": wos, "total": doc["grand_total_gbp"]})
    return {"status": "PENDING_APPROVAL", "batch_id": batch_id, "sites": wos, "grand_total_gbp": doc["grand_total_gbp"],
            "billing_anomalies": anomalies}


# ---------------------------------------------------------------------------
# ask_human: every agent gets its own copy, so the question records who asked
# ---------------------------------------------------------------------------

def make_ask_human(agent_display: str):
    @tool
    def ask_human(question: str, options: Optional[list[str]] = None, work_order_id: str = "",
                  project_name: str = "") -> dict:
        """Ask the Key Accounts Manager when you face a real conflict that your tools and the policy documents
        cannot settle. Give a clear question and 2 to 4 short options. Pass work_order_id for a site question
        (that site pauses until answered) or project_name for a project question. Other sites carry on."""
        wo = work_order_id or None
        project = project_name or ((get_work_order(wo) or {}).get("project_name") if wo else None)
        if not project:
            return {"error": "Give a valid work_order_id or project_name."}
        if isinstance(options, str):
            options = [o.strip() for o in options.split(",") if o.strip()]
        res = hitl.ask(project, agent_display, question, options, wo)
        if res["status"] == "ASKED":
            log_event(agent_display, "QUESTION -> KAM", {"question": question, "options": options}, wo)
        return {**res, "note": "Stop here. You will get the answer in a later turn."
                if res["status"] in ("ASKED", "ALREADY_ASKED") else res.get("reason")}
    return ask_human

SHARED_TOOLS = [search_knowledge, get_site_status, list_project_sites]
INVENTORY_TOOLS = [check_site_stock, order_site_shortages, confirm_site_stock_ready, compare_suppliers, get_stock_level]
LOGISTICS_TOOLS = [dispatch_site, get_delivery_record]
FIELD_TOOLS = [run_site_installation, request_replacement]
FINANCE_TOOLS = [reconcile_site, create_invoice]