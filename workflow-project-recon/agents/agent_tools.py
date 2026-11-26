"""
LangChain tools for each agent. They wrap the Phase 1 functions and keep their output short,
because a 7B model works best with small, clear tool results.

Warehouse model: every purchase (imports and local purchases) arrives at the warehouse in dated inbound
batches. Stock exists only once Logistics receives a batch. Subcontractors collect their kit from the
warehouse (engineers collect replacements) and sign a collection note. Nothing is delivered to site.
"""
import os
import re
import json
import sqlite3
import threading
from typing import Optional
from langchain_core.tools import tool

from agents.events import log_event
from agents.sim_config import SIM
from agents import hitl
from rag.rag_query import retrieve_policies
from tools import clock
from tools.orchestrator_tools import (ORCHESTRATOR_DB, WMS_DB, get_work_order, get_work_order_bom,
                                      set_work_order_status, record_ledger_quantity)
from tools.inventory_tools import (check_bom_stock, query_suppliers, verify_and_inbound_packing_list,
                                   record_return, get_outbound_history, get_sku_description)
from tools.logistics_tools import (LOGISTICS_DB, in_transit_quantities, due_shipments, get_collected_quantities)
from tools.field_tools import process_field_report, quarantine_faulty_unit
from tools.finance_tools import run_three_way_reconciliation, generate_client_invoice, validate_billing
from tools.mailbox_tools import send_mail, take_new_replies
from tools.project_tools import get_project_work_orders, source_shortage, issue_from_warehouse
from simulation.field_app import simulate_site_report

_lock = threading.Lock()          # serialises every check-then-write, so duplicate tool calls are harmless
KAM_EMAIL = "kam@intellicommstech.local"
WAREHOUSE_EMAIL = "warehouse@intellicommstech.local"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wo_or_error(work_order_id: str):
    wo = get_work_order(work_order_id)
    return wo, (None if wo else {"error": f"Unknown work order {work_order_id}"})


def _claim(work_order_id: str, from_status: str, to_status: str) -> bool:
    """Atomically moves a work order between stages. Only one caller can win."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.execute("UPDATE work_orders SET status = ? WHERE work_order_id = ? AND status = ?",
                       (to_status, work_order_id, from_status))
    conn.commit()
    conn.close()
    return cur.rowcount == 1


def _is_serialized(sku: str) -> bool:
    conn = sqlite3.connect(WMS_DB)
    row = conn.execute("SELECT serialized FROM sku_master WHERE sku = ?", (sku,)).fetchone()
    conn.close()
    return bool(row and row[0])


def _all_skus() -> list:
    conn = sqlite3.connect(WMS_DB)
    rows = conn.execute("SELECT sku, serialized FROM sku_master").fetchall()
    conn.close()
    return [{"sku": s, "qty": 0, "serialized": bool(ser)} for s, ser in rows]


def _on_hand() -> dict:
    """Good units physically in the warehouse, per SKU."""
    return {c["sku"]: c["on_hand"] for c in check_bom_stock(_all_skus())["stock_check"]}


def _reserved(statuses: tuple, exclude_wo: str = "") -> dict:
    """Material promised to sites in the given stages and not yet collected (plus outstanding replacements)."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    marks = ",".join("?" * len(statuses))
    rows = conn.execute(f"""
        SELECT l.sku, SUM(l.qty_planned - l.qty_dispatched) FROM reconciliation_ledger l
        JOIN work_orders w ON w.work_order_id = l.work_order_id
        WHERE w.status IN ({marks}) AND w.work_order_id != ? GROUP BY l.sku""", (*statuses, exclude_wo)).fetchall()
    rep = conn.execute("""
        SELECT l.sku, SUM(l.qty_planned - l.qty_installed) FROM reconciliation_ledger l
        JOIN work_orders w ON w.work_order_id = l.work_order_id
        WHERE w.status = 'REPLACEMENT_PENDING' AND l.serialized = 1 AND w.work_order_id != ? GROUP BY l.sku""",
                       (exclude_wo,)).fetchall()
    conn.close()
    out = {}
    for sku, q in rows + rep:
        if q and q > 0:
            out[sku] = out.get(sku, 0) + q
    return out


ON_HAND_HOLDERS = ("STOCK_READY", "KIT_READY", "COLLECTING")
PROJECTED_HOLDERS = ("ON_ORDER",) + ON_HAND_HOLDERS


def _free_on_hand(exclude_wo: str = "") -> dict:
    on_hand, held = _on_hand(), _reserved(ON_HAND_HOLDERS, exclude_wo)
    return {sku: on_hand.get(sku, 0) - held.get(sku, 0) for sku in on_hand}


def _projected_free() -> dict:
    """On hand + in transit, minus everything already promised to sites."""
    on_hand, transit, held = _on_hand(), in_transit_quantities(), _reserved(PROJECTED_HOLDERS)
    skus = set(on_hand) | set(transit)
    return {sku: on_hand.get(sku, 0) + transit.get(sku, 0) - held.get(sku, 0) for sku in skus}


def _refresh_stock_status(project_name: str) -> list:
    """ON_ORDER sites whose material is now physically in the warehouse become STOCK_READY (earliest first)."""
    free = _free_on_hand()
    promoted = []
    for wo in get_project_work_orders(project_name):
        if wo["status"] != "ON_ORDER":
            continue
        bom = get_work_order_bom(wo["work_order_id"])
        if all(free.get(i["sku"], 0) >= i["qty"] for i in bom):
            for i in bom:
                free[i["sku"]] -= i["qty"]
            if _claim(wo["work_order_id"], "ON_ORDER", "STOCK_READY"):
                promoted.append(wo["work_order_id"])
                log_event("Inventory Agent", "STOCK_READY", "all material in the warehouse", wo["work_order_id"])
    return promoted


def _replies(work_order_id: str) -> list:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    rows = conn.execute("SELECT body FROM mailbox WHERE work_order_id = ? AND direction = 'IN'", (work_order_id,)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (text or "").lower())).strip()


def _grounded(work_order_id: str, quote: str, intent: str, intent_label: str):
    """Evidence check: the quote must really appear in a reply for this site and must say what is claimed."""
    if len(_norm(quote).split()) < 3:
        return f"Give evidence_quote: the exact sentence from the reply (at least 3 words)."
    if not any(_norm(quote) in _norm(body) for body in _replies(work_order_id)):
        return "evidence_quote was not found in any reply for this site. Quote the reply exactly."
    if not re.search(intent, quote, re.I):
        return f"The quoted sentence does not say the items are {intent_label}. Use a different tool or ask_human."
    return None


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
    """Current stage, team, dates, BOM ledger and invoice status for one work order."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    lines = conn.execute("""
        SELECT sku, qty_planned, qty_dispatched, qty_installed, qty_quarantined, qty_returned, qty_written_off,
               reconciliation_status FROM reconciliation_ledger WHERE work_order_id = ? ORDER BY id
    """, (work_order_id,)).fetchall()
    inv = conn.execute("SELECT invoice_id, status, grand_total_gbp FROM invoices WHERE work_order_id = ?",
                       (work_order_id,)).fetchone()
    conn.close()
    wo["business_date"] = clock.today()
    wo["collection_date"] = clock.collection_date(wo["scheduled_date"])
    wo["ledger"] = [dict(zip(["sku", "planned", "issued", "installed", "quarantined", "returned", "written_off",
                              "status"], l)) for l in lines]
    wo["invoice"] = dict(zip(["invoice_id", "status", "total_gbp"], inv)) if inv else None
    return wo


@tool
def list_project_sites(project_name: str) -> list:
    """All work orders in a project with their current stage."""
    return [{k: wo[k] for k in ("work_order_id", "site_category", "scheduled_date", "status")}
            for wo in get_project_work_orders(project_name)]


# ---------------------------------------------------------------------------
# Inventory & Sourcing Agent: project procurement in dated inbound batches
# ---------------------------------------------------------------------------

@tool
def get_stock_level(sku: str) -> dict:
    """Units of one SKU on hand in the warehouse and on their way (in transit)."""
    return {"sku": sku, "on_hand": _on_hand().get(sku, 0), "in_transit": in_transit_quantities().get(sku, 0),
            "business_date": clock.today()}


@tool
def compare_suppliers(sku: str, qty: int) -> list:
    """Suppliers able to supply qty of sku, with unit price (GBP) and lead time (days)."""
    return [{k: s[k] for k in ("supplier_id", "supplier_name", "supplier_type", "unit_price_gbp", "lead_time_days")}
            for s in query_suppliers(sku, qty)]


def _planned_groups(project_name: str) -> list:
    """PLANNED sites grouped by install date (one inbound batch per install date)."""
    groups = {}
    for wo in get_project_work_orders(project_name):
        if wo["status"] == "PLANNED":
            groups.setdefault(wo["scheduled_date"], []).append(wo)
    return sorted(groups.items())


def _group_need(sites: list) -> dict:
    need = {}
    for wo in sites:
        for i in get_work_order_bom(wo["work_order_id"]):
            need.setdefault(i["sku"], {"qty": 0, "serialized": i["serialized"]})["qty"] += i["qty"]
    return need


def _next_batch_no(project_name: str) -> int:
    conn = sqlite3.connect(LOGISTICS_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS shipment_batches (shipment_id TEXT PRIMARY KEY, batch_ref TEXT, project_name TEXT)")
    refs = [r[0] for r in conn.execute("SELECT DISTINCT batch_ref FROM shipment_batches WHERE project_name = ?",
                                       (project_name,))]
    conn.close()
    nums = [int(r.split("-")[1]) for r in refs if re.fullmatch(r"(IMP|LP)-\d+", r)]
    return max(nums, default=0) + 1


@tool
def check_project_stock(project_name: str) -> dict:
    """For every PLANNED site, grouped by install date: what is needed, and what is short after counting
    stock on hand and already on its way."""
    free = _projected_free()
    out = []
    for date, sites in _planned_groups(project_name):
        short = {}
        for sku, n in _group_need(sites).items():
            have = max(free.get(sku, 0), 0)
            if have < n["qty"]:
                short[sku] = n["qty"] - have
            free[sku] = have - n["qty"] if have >= n["qty"] else 0
        out.append({"install_date": date, "material_needed_by": clock.need_by_date(date),
                    "sites": [s["work_order_id"] for s in sites], "shortages": short})
    return {"project_name": project_name, "business_date": clock.today(), "groups": out}


@tool
def plan_project_procurement(project_name: str, strategy: str = "cheapest_on_time") -> dict:
    """Raises the project's purchase orders as dated inbound batches to the WAREHOUSE: one batch per install
    date (IMP-xx for imports, LP-xx for local purchases), each due before that day's collection. Emails the
    material plan to the KAM and the warehouse.
    strategy: 'cheapest_on_time' (default; a line that cannot arrive in time is NOT ordered and is reported as a
    conflict for the KAM), 'fastest' (pay for the quickest supplier) or 'accept_delay' (cheapest, even if late)."""
    batches, conflicts = [], []
    with _lock:   # a duplicate call waits, then finds every site already on order
        groups = _planned_groups(project_name)
        if not groups:
            return {"status": "NOTHING_PLANNED", "project_name": project_name}
        free = _projected_free()
        k = _next_batch_no(project_name)
        for date, sites in groups:
            need_by = clock.need_by_date(date)
            lines, from_stock = [], []
            for sku, n in _group_need(sites).items():
                have = max(free.get(sku, 0), 0)
                if have >= n["qty"]:
                    free[sku] = have - n["qty"]
                    from_stock.append(sku)
                    continue
                free[sku] = 0
                qty = n["qty"] - have
                suppliers = query_suppliers(sku, qty)
                days = clock.days_between(clock.today(), need_by)
                on_time = [x for x in suppliers if x["lead_time_days"] <= days]
                if not suppliers or (strategy == "cheapest_on_time" and not on_time):
                    fastest = min(suppliers, key=lambda x: x["lead_time_days"]) if suppliers else None
                    conflicts.append({"sku": sku, "qty": qty, "needed_by": need_by, "install_date": date,
                                      "problem": "no supplier can supply this quantity" if not suppliers else
                                      f"fastest supplier {fastest['supplier_name']} arrives "
                                      f"{clock.shift(clock.today(), fastest['lead_time_days'])}, after {need_by}"})
                    continue   # held: not ordered until the KAM decides
                pick = None
                if strategy == "fastest":
                    pick = min(suppliers, key=lambda x: x["lead_time_days"])["supplier_id"]
                elif strategy == "accept_delay" and not on_time:
                    pick = min(suppliers, key=lambda x: x["unit_price_gbp"])["supplier_id"]
                res = source_shortage(sku, qty, n["serialized"], need_by, supplier_id=pick,
                                      receive_now=False, batch_no=k, project_name=project_name)
                lines.append({k2: res.get(k2) for k2 in ("sku", "qty", "kind", "supplier", "batch_ref",
                                                          "arrival_date", "late", "status", "reason")})
            held = [c for c in conflicts if c.get("install_date") == date]
            if not held:   # only a fully covered install date moves on; held lines keep its sites PLANNED
                for wo in sites:
                    _claim(wo["work_order_id"], "PLANNED", "ON_ORDER")
            batches.append({"batch_no": k, "install_date": date, "material_needed_by": need_by,
                            "sites": [s["work_order_id"] for s in sites], "orders": lines,
                            "covered_from_stock": from_stock})
            k += 1
        ready_now = _refresh_stock_status(project_name)

    # Material plan: what arrives at the warehouse, when, and for which sites
    text = [f"Material plan for {project_name} (business date {clock.today()})", ""]
    for b in batches:
        text.append(f"Install date {b['install_date']}  ({len(b['sites'])} sites: {', '.join(b['sites'])})")
        refs = {}
        for o in b["orders"]:
            refs.setdefault((o["batch_ref"], o["arrival_date"], o["supplier"]), []).append(f"{o['qty']} x {o['sku']}")
        for (ref, eta, sup), items in sorted(refs.items(), key=lambda x: str(x[0][0])):
            text.append(f"  {ref}  {sup}  ETA warehouse {eta}: " + ", ".join(items))
        if b["covered_from_stock"]:
            text.append(f"  From existing stock: {', '.join(b['covered_from_stock'])}")
        text.append("")
    text.append("All material is delivered to the warehouse. Subcontractors collect kits the day before installation.")
    body = "\n".join(text)
    for to in (KAM_EMAIL, WAREHOUSE_EMAIL):
        send_mail(None, "inventory.agent@intellicommstech.local", to, f"Material plan: {project_name}", body)
    log_event("Inventory Agent", "MATERIAL_PLAN", body)
    if conflicts:
        log_event("Inventory Agent", "PROCUREMENT_CONFLICT", conflicts)
    return {"project_name": project_name, "batches": batches, "sites_stock_ready_now": ready_now,
            "conflicts": conflicts,
            "advice": ("Some lines cannot arrive in time and were NOT ordered: ask_human whether to pay for the "
                       "fastest supplier or accept the delay, then call plan_project_procurement again with "
                       "strategy 'fastest' or 'accept_delay'.") if conflicts else "All batches arrive in time."}


# ---------------------------------------------------------------------------
# Logistics & Warehouse Agent: receive inbound batches, prepare kits, issue kits to subcontractors
# ---------------------------------------------------------------------------

@tool
def check_inbound_due(project_name: str) -> dict:
    """Inbound batches due at the warehouse today or earlier and not yet received."""
    due = due_shipments(clock.today())
    return {"business_date": clock.today(),
            "due": [{"shipment_id": d["shipment_id"], "batch_ref": d["batch_ref"], "supplier": d["supplier"],
                     "eta": d["eta"], "lines": len(d["packing_list"])} for d in due]}


@tool
def receive_inbound(project_name: str) -> dict:
    """Receives every inbound shipment due today: checks each packing list against its PO and books the stock
    into the warehouse. Sites whose material is now complete become STOCK_READY."""
    received, problems = {}, []
    with _lock:
        for d in due_shipments(clock.today()):
            res = verify_and_inbound_packing_list(d["shipment_id"], d["packing_list"], txn_date=clock.today())
            if res["status"] == "RECEIVED_CONFIRMED":
                received.setdefault(d["batch_ref"] or "unbatched", []).append(
                    f"{res['total_qty']} x {d['packing_list'][0]['sku']}")
            else:
                problems.append({"shipment_id": d["shipment_id"], "batch_ref": d["batch_ref"], **res})
        ready = _refresh_stock_status(project_name)
    for ref, items in received.items():
        log_event("Logistics Agent", f"INBOUND_RECEIVED {ref}", items)
    return {"business_date": clock.today(), "received": received, "problems": problems,
            "sites_now_stock_ready": ready,
            "advice": "Packing list problems: ask_human." if problems else "All received."}


@tool
def prepare_kits(project_name: str) -> dict:
    """Picks and stages a kit for every STOCK_READY site (one kit per work order) for subcontractor collection."""
    staged = []
    with _lock:
        for wo in get_project_work_orders(project_name):
            if wo["status"] == "STOCK_READY" and _claim(wo["work_order_id"], "STOCK_READY", "KIT_READY"):
                staged.append({"work_order_id": wo["work_order_id"], "collector": wo["subcon_id"],
                               "collection_date": clock.collection_date(wo["scheduled_date"])})
                log_event("Logistics Agent", "KIT_READY", f"collect by {wo['subcon_id']}", wo["work_order_id"])
    return {"kits_prepared": staged}


@tool
def issue_kits(project_name: str) -> dict:
    """Issues every staged kit whose collection date has arrived to its subcontractor: WMS outbound and a signed
    collection note. Any kit that is no longer complete goes back to Inventory."""
    issued, handed_back = [], []
    with _lock:
        for wo in get_project_work_orders(project_name):
            if wo["status"] != "KIT_READY" or clock.collection_date(wo["scheduled_date"]) > clock.today():
                continue
            if not _claim(wo["work_order_id"], "KIT_READY", "COLLECTING"):
                continue
            res = issue_from_warehouse(wo, get_work_order_bom(wo["work_order_id"]), wo["subcon_id"], clock.today())
            if res["status"] == "DISPATCHED":
                set_work_order_status(wo["work_order_id"], "COLLECTED")
                issued.append({"work_order_id": wo["work_order_id"], "collected_by": wo["subcon_id"],
                               "lines": len(res["items"])})
                log_event("Logistics Agent", f"COLLECTED by {wo['subcon_id']}",
                          {"lines": len(res["items"]), "collection_note": True}, wo["work_order_id"])
            else:
                set_work_order_status(wo["work_order_id"], "PLANNED")
                handed_back.append(wo["work_order_id"])
                log_event("Logistics Agent", "HANDBACK -> Inventory", res.get("reason"), wo["work_order_id"])
    return {"business_date": clock.today(), "issued": issued, "returned_to_inventory": handed_back}


# ---------------------------------------------------------------------------
# Field Execution Agent: installation and replacements
# ---------------------------------------------------------------------------

def _pending_serialized(work_order_id: str, sku: str = None) -> int:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    q = "SELECT COALESCE(SUM(qty_planned - qty_installed), 0) FROM reconciliation_ledger WHERE work_order_id = ? AND serialized = 1"
    args = [work_order_id]
    if sku:
        q += " AND sku = ?"
        args.append(sku)
    n = conn.execute(q, args).fetchone()[0]
    conn.close()
    return n


def replacement_in_transit(work_order_id: str, sku: str = None):
    """ETA of a replacement already ordered for this site (None if nothing is on its way)."""
    conn = sqlite3.connect(LOGISTICS_DB)
    try:
        rows = conn.execute("""SELECT s.inbound_date, s.packing_list_json FROM inbound_shipments s
                               JOIN shipment_batches b ON b.shipment_id = s.shipment_id
                               WHERE b.batch_ref = ? AND s.status = 'IN_TRANSIT'""", (f"REP-{work_order_id}",)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    etas = [eta for eta, pl in rows if sku is None or any(i["sku"] == sku for i in json.loads(pl))]
    return min(etas) if etas else None


def _outstanding_faults(work_order_id: str) -> list:
    """Every serialized part still short of its planned install count, with the failed serial where known
    (a faulty unit the engineer kept is not in quarantine, so it shows as 'not handed in')."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    pending = conn.execute("""SELECT sku, qty_planned - qty_installed FROM reconciliation_ledger
                              WHERE work_order_id = ? AND serialized = 1 AND qty_planned > qty_installed""",
                           (work_order_id,)).fetchall()
    conn.close()
    conn = sqlite3.connect(WMS_DB)
    quarantined = {}
    for sku, sn in conn.execute("SELECT sku, serial_number FROM wms_ledger WHERE transaction_type = 'QUARANTINE' "
                                "AND reference = ? ORDER BY id", (work_order_id,)):
        quarantined.setdefault(sku, []).append(sn)
    conn.close()
    out = []
    for sku, n in pending:
        serials = quarantined.get(sku, [])
        for i in range(n):
            out.append({"sku": sku, "failed_serial": serials[i] if i < len(serials) else "not handed in",
                        "replacement_needed": True, "replacement_eta": replacement_in_transit(work_order_id, sku)})
    return out


@tool
def run_site_installation(work_order_id: str) -> dict:
    """On the install date: reads the engineer's Field App report, records installs, quarantines faulty serials
    and classifies each fault against the fault manual. For a site in REPLACEMENT_PENDING it lists the faults
    still waiting, so you can request their replacement."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    if wo["status"] == "REPLACEMENT_PENDING":
        return {"work_order_id": work_order_id, "status": "REPLACEMENT_PENDING",
                "faults": _outstanding_faults(work_order_id), "collector": wo["engineer_id"]}
    if wo["scheduled_date"] > clock.today():
        return {"error": f"Installation is scheduled for {wo['scheduled_date']}; today is {clock.today()}."}
    if not _claim(work_order_id, "COLLECTED", "INSTALLING"):
        return {"error": f"{work_order_id} is {wo['status']}; the subcontractor must have COLLECTED the kit first."}

    conn = sqlite3.connect(WMS_DB)
    items = [{"sku": s, "serial_number": sn, "qty": -q} for s, sn, q in conn.execute(
        "SELECT sku, serial_number, qty FROM wms_ledger WHERE transaction_type = 'OUTBOUND' AND reference = ?",
        (work_order_id,))]
    conn.close()
    scenario = SIM["site_scenarios"].get(wo["site_id"], "happy_path")
    report = simulate_site_report(wo, items, SIM["field_mode"], scenario)
    triage = process_field_report(report, txn_date=clock.today())
    faults = [{"sku": t["sku"], "failed_serial": t["serial_number"], "policy_code": t["decision"].get("policy_code"),
               "replacement_needed": bool(t["decision"].get("trigger_escalation_or_replenishment"))}
              for t in triage["triage_results"]]
    status = "REPLACEMENT_PENDING" if _pending_serialized(work_order_id) else "INSTALLED"
    set_work_order_status(work_order_id, status)
    log_event("Field Agent", "FIELD_REPORT", {"installed_lines": len(report["installed_items"]), "faults": faults},
              work_order_id)
    return {"work_order_id": work_order_id, "status": status, "faults": faults, "collector": wo["engineer_id"]}


@tool
def request_replacement(work_order_id: str, sku: str, failed_serial: str) -> dict:
    """Replacement loop with Inventory: if a good unit is in the warehouse the engineer collects it and installs
    it; otherwise a local purchase is ordered to the warehouse and the site waits for it."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    with _lock:
        if _pending_serialized(work_order_id, sku) <= 0:
            return {"status": "NOT_NEEDED", "reason": f"No outstanding {sku} on {work_order_id}."}
        if _free_on_hand(exclude_wo=work_order_id).get(sku, 0) >= 1:
            log_event("Field Agent", "REPLACEMENT_REQUEST -> Inventory", {"sku": sku, "failed": failed_serial},
                      work_order_id)
            rep = issue_from_warehouse(wo, [{"sku": sku, "qty": 1, "serialized": True}], wo["engineer_id"], clock.today())
            if rep["status"] != "DISPATCHED":
                return {"status": "ISSUE_FAILED", "detail": rep.get("reason")}
            record_ledger_quantity(work_order_id, rep["items"], "qty_installed")
            if not _pending_serialized(work_order_id):
                set_work_order_status(work_order_id, "INSTALLED")
            log_event("Field Agent", f"REPLACEMENT_INSTALLED (collected by {wo['engineer_id']})",
                      {"failed": failed_serial, "new": rep["items"][0]["serial_number"]}, work_order_id)
            return {"status": "INSTALLED", "replacement_serial": rep["items"][0]["serial_number"],
                    "site_status": get_work_order(work_order_id)["status"]}
        eta = replacement_in_transit(work_order_id, sku)
        if eta:
            return {"status": "ON_ORDER", "eta": eta, "note": "Already ordered; the site waits for the inbound."}
        src = source_shortage(sku, 1, True, clock.today(), receive_now=False, batch_no=f"REP-{work_order_id}",
                              project_name=wo["project_name"])
        log_event("Inventory Agent", "REPLACEMENT_PO", src, work_order_id)
        if src.get("status") != "IN_TRANSIT":
            return {"status": "SOURCING_FAILED", "detail": src}
        return {"status": "ON_ORDER", "eta": src["arrival_date"], "supplier": src["supplier"]}


# ---------------------------------------------------------------------------
# Field Execution Agent: reconciliation investigations (email the person who holds the items)
# ---------------------------------------------------------------------------

def _unaccounted(work_order_id: str) -> list:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    rows = conn.execute("""
        SELECT sku, serialized, qty_dispatched - qty_installed - qty_quarantined - qty_returned - qty_written_off
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
    """Per unaccounted SKU: part number, description, qty, serials, collection date and who collected it."""
    facts = []
    for u in _unaccounted(work_order_id):
        history = get_outbound_history(work_order_id, u["sku"])
        if u["serialized"]:
            conn = sqlite3.connect(WMS_DB)
            q = {r[0] for r in conn.execute("SELECT serial_number FROM wms_ledger WHERE transaction_type = 'QUARANTINE' "
                                            "AND reference = ?", (work_order_id,))}
            conn.close()
            candidates = [h for h in history if h["serial_number"] not in q]
            collector = history[-1]["collected_by"] if history else None
        else:
            candidates = history
            collector = history[0]["collected_by"] if history else None
        facts.append({"sku": u["sku"], "description": get_sku_description(u["sku"]),
                      "unaccounted_qty": u["unaccounted"], "serialized": u["serialized"],
                      "issued": [{"serial": h["serial_number"], "qty": h["qty"], "collected_on": h["outbound_date"],
                                  "collected_by": h["collected_by"]} for h in candidates],
                      "contact": collector})
    return facts


@tool
def investigate_discrepancy(work_order_id: str) -> dict:
    """For a site that failed reconciliation: which items are unaccounted for, part numbers, serials,
    collection dates and who collected them (the person to contact)."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    facts = _query_facts(work_order_id)
    log_event("Field Agent", "INVESTIGATING", facts, work_order_id)
    return {"work_order_id": work_order_id, "site_id": wo["site_id"], "unaccounted_items": facts}


@tool
def send_email(work_order_id: str, recipient_id: str, subject: str, body: str) -> dict:
    """Email an engineer or subcontractor (recipient_id such as 'Subcon-B' or 'Engineer-C') and wait for their
    reply. The site pauses until they answer. Write the body yourself: be specific and polite."""
    person = _resource(recipient_id)
    if not person:
        return {"error": f"Unknown recipient {recipient_id}"}
    if not _claim(work_order_id, "RECON_DISCREPANCY", "AWAITING_REPLY") and \
       not _claim(work_order_id, "REPLY_RECEIVED", "AWAITING_REPLY"):
        return {"error": "Site is not under investigation, or an email is already awaiting a reply."}
    ref = "\n".join(f"  - {f['sku']} {f['description']}: {f['unaccounted_qty']} unaccounted; collected "
                    + "; ".join(f"{i['serial'] or str(i['qty']) + ' units'} on {i['collected_on']} by {i['collected_by']}"
                                for i in f["issued"])
                    for f in _query_facts(work_order_id))
    msg = send_mail(work_order_id, "field.agent@intellicommstech.local", person["email"], subject,
                    f"{body}\n\n--- Reconciliation reference ({work_order_id}) ---\n{ref}", expects_reply=True)
    log_event("Field Agent", f"EMAIL -> {recipient_id}", {"subject": subject}, work_order_id)
    return {"status": "AWAITING_REPLY", "message_id": msg["message_id"], "to": person["email"]}


@tool
def read_replies(work_order_id: str) -> dict:
    """New email replies received for this work order, and what is still unaccounted."""
    with _lock:
        replies = take_new_replies(work_order_id)
    return {"work_order_id": work_order_id, "replies": replies,
            "still_unaccounted": [{"sku": f["sku"], "qty": f["unaccounted_qty"], "serialized": f["serialized"]}
                                  for f in _query_facts(work_order_id)]}


def _after_resolution(work_order_id: str) -> str:
    """If every item is now accounted for, send the site back to Finance to reconcile again."""
    if not _unaccounted(work_order_id) and not _pending_serialized(work_order_id):
        set_work_order_status(work_order_id, "INSTALLED")
        return "INSTALLED (ready to reconcile again)"
    return get_work_order(work_order_id)["status"]


RETURN_INTENT = r"\b(return|returning|returned|drop|dropping|dropped|bring|bringing|brought|send|sending|sent|back to)\b"
INSTALL_INTENT = r"\b(install|installed|used|fitted|put in)\b"
HANDIN_INTENT = r"\b(drop|dropped|return|returned|hand|handed|quarantine|brought|back|left)\b"


@tool
def record_site_return(work_order_id: str, sku: str, qty: int, returned_by: str, evidence_quote: str) -> dict:
    """Book unused BULK material that the reply says is being returned to the warehouse. evidence_quote must be
    the exact sentence from the reply saying so, including the quantity."""
    if _is_serialized(sku):
        return {"error": f"{sku} is serialized. Use book_late_quarantine, or ask_human."}
    problem = _grounded(work_order_id, evidence_quote, RETURN_INTENT, "being returned")
    if problem:
        return {"error": problem}
    if str(qty) not in evidence_quote:
        return {"error": f"The quote does not state {qty}. If the reply gives no exact number, ask_human."}
    with _lock:
        open_qty = next((u["unaccounted"] for u in _unaccounted(work_order_id) if u["sku"] == sku), 0)
        if qty <= 0 or qty > open_qty:
            return {"error": f"{qty} is not valid: only {open_qty} of {sku} is unaccounted on {work_order_id}."}
        conn = sqlite3.connect(WMS_DB)
        dup = conn.execute("SELECT 1 FROM wms_ledger WHERE transaction_type = 'RETURN' AND reference = ? AND sku = ? "
                           "AND qty = ? AND handled_by = ?", (work_order_id, sku, qty, returned_by)).fetchone()
        conn.close()
        if dup:
            return {"error": f"This return ({qty} x {sku} from {returned_by}) is already booked."}
        record_return(work_order_id, sku, qty, returned_by, txn_date=clock.today())
    log_event("Inventory Agent", "RETURN_BOOKED", {"sku": sku, "qty": qty, "by": returned_by,
                                                   "evidence": evidence_quote}, work_order_id)
    return {"status": "RETURN_BOOKED", "sku": sku, "qty": qty, "site_status": _after_resolution(work_order_id)}


@tool
def report_installed_claim(work_order_id: str, sku: str, qty: int, evidence_quote: str) -> dict:
    """Use when a reply says the missing BULK items were actually installed (the Field App under reported).
    This changes what the client is billed, so it goes to the KAM to accept or reject. evidence_quote must be the
    exact sentence from the reply."""
    if _is_serialized(sku):
        return {"error": f"{sku} is serialized: ask_human instead."}
    problem = _grounded(work_order_id, evidence_quote, INSTALL_INTENT, "installed")
    if problem:
        return {"error": problem}
    open_qty = next((u["unaccounted"] for u in _unaccounted(work_order_id) if u["sku"] == sku), 0)
    if qty <= 0 or qty > open_qty:
        return {"error": f"Only {open_qty} of {sku} is unaccounted on {work_order_id}."}
    res = hitl.ask(get_work_order(work_order_id)["project_name"], "Field Agent",
                   f"{work_order_id}: the Field App recorded {qty} x {sku} as not installed, but the reply says: "
                   f"\"{evidence_quote}\". Accepting adds {qty} x {sku} to the installed (billed) quantity.",
                   ["Accept the claim and correct the installed quantity",
                    "Reject the claim and treat the items as missing"], work_order_id)
    if res["status"] == "ASKED":
        log_event("Field Agent", "QUESTION -> KAM", {"installed_claim": f"{qty} x {sku}", "evidence": evidence_quote},
                  work_order_id)
    return {**res, "note": "Stop here. You will get the KAM's decision in a later turn."}


@tool
def correct_installed_quantity(work_order_id: str, sku: str, qty: int) -> dict:
    """Only after the KAM ACCEPTED an installed claim: adds qty to the installed quantity for this site."""
    answer = hitl.last_answer(get_work_order(work_order_id)["project_name"], "Field Agent", work_order_id) or ""
    if "accept" not in answer.lower():
        return {"error": "The KAM has not accepted an installed claim for this site."}
    with _lock:
        open_qty = next((u["unaccounted"] for u in _unaccounted(work_order_id) if u["sku"] == sku), 0)
        if qty <= 0 or qty > open_qty:
            return {"error": f"Only {open_qty} of {sku} is unaccounted."}
        record_ledger_quantity(work_order_id, [{"sku": sku, "qty": qty}], "qty_installed")
    log_event("Field Agent", "INSTALLED_QTY_CORRECTED", {"sku": sku, "qty": qty, "approved": answer}, work_order_id)
    return {"status": "CORRECTED", "site_status": _after_resolution(work_order_id)}


@tool
def book_late_quarantine(work_order_id: str, sku: str, serial_number: str, returned_by: str, evidence_quote: str) -> dict:
    """Book a faulty serialized unit into quarantine when the reply says it was handed back to the warehouse.
    evidence_quote must be the exact sentence from the reply."""
    problem = _grounded(work_order_id, evidence_quote, HANDIN_INTENT, "handed back to the warehouse")
    if problem:
        return {"error": problem}
    with _lock:
        issued = {h["serial_number"] for h in get_outbound_history(work_order_id, sku)}
        if serial_number not in issued:
            return {"error": f"{serial_number} was never issued to {work_order_id}."}
        conn = sqlite3.connect(WMS_DB)
        already = conn.execute("SELECT 1 FROM wms_ledger WHERE transaction_type = 'QUARANTINE' AND serial_number = ? "
                               "AND sku = ?", (serial_number, sku)).fetchone()
        conn.close()
        if already:
            return {"error": f"{serial_number} is already in quarantine."}
        if not any(u["sku"] == sku and u["unaccounted"] > 0 for u in _unaccounted(work_order_id)):
            return {"error": f"Nothing unaccounted for {sku} on {work_order_id}."}
        quarantine_faulty_unit(sku, serial_number, returned_by, work_order_id, txn_date=clock.today())
        record_ledger_quantity(work_order_id, [{"sku": sku, "qty": 1}], "qty_quarantined")
    log_event("Inventory Agent", "LATE_QUARANTINE", {"sku": sku, "serial": serial_number}, work_order_id)
    return {"status": "QUARANTINED", "serial": serial_number, "site_status": _after_resolution(work_order_id)}


@tool
def escalate_to_kam(work_order_id: str, reason: str) -> dict:
    """Hand the site to the KAM when items cannot be recovered (lost, wasted, disputed). Give a clear reason."""
    wo, err = _wo_or_error(work_order_id)
    if err:
        return err
    res = hitl.raise_escalation(wo["project_name"], work_order_id, "Field Agent", reason, _query_facts(work_order_id))
    log_event("Field Agent", "ESCALATED_TO_KAM", reason, work_order_id)
    return res


# ---------------------------------------------------------------------------
# Invoicing & Financial Agent
# ---------------------------------------------------------------------------

@tool
def reconcile_site(work_order_id: str) -> dict:
    """Three way reconciliation for one site: WMS issue vs signed collection notes vs installed, quarantined,
    returned and written off."""
    res = run_three_way_reconciliation(work_order_id)
    return {"work_order_id": work_order_id, "reconciled": res.get("reconciled"),
            "issues": [{"sku": d["sku"], "issues": d["issues"]} for d in res.get("discrepancies", [])]}


@tool
def get_reconciliation_policy(project_name: str) -> dict:
    """When this project should be reconciled: end_of_project, every_n_sites, on_request, or unspecified."""
    return {"project_name": project_name, **hitl.get_policy(project_name)}


@tool
def set_reconciliation_policy(project_name: str, policy: str, every_n: int = 0, source: str = "") -> dict:
    """Store the reconciliation policy. policy: 'end_of_project', 'every_n_sites' (give every_n) or 'on_request'.
    source: where it came from, e.g. 'KAM answer' or 'contract clause'."""
    answer = hitl.last_answer(project_name, "Finance Agent")
    parsed = hitl.parse_policy_text(answer) if answer else None
    if parsed and (parsed[0] != policy or (policy == "every_n_sites" and parsed[1] != every_n)):
        return {"error": f"The KAM answered '{answer}', which means policy='{parsed[0]}' every_n={parsed[1]}. "
                         f"Call again with those values."}
    res = hitl.set_policy(project_name, policy, every_n, source)
    if "error" not in res:
        log_event("Finance Agent", "RECON_POLICY_SET", {**res, "kam_answer": answer})
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
    with _lock:
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
        n = len([1 for b in os.listdir("invoices") if re.match(rf"BATCH-{re.escape(project_name)}-\d+\.json$", b)]) + 1
        batch_id = f"BATCH-{project_name}-{n:02d}"
        subtotal = round(sum(s["subtotal_gbp"] for s in sites), 2)
        doc = {"batch_id": batch_id, "project_name": project_name, "currency": "GBP", "invoice_date": clock.today(),
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


@tool
def get_rejected_batches(project_name: str) -> dict:
    """Consolidated invoices the KAM rejected, with the KAM's reason."""
    return {"rejected": [{k: b[k] for k in ("batch_id", "work_orders", "total_gbp", "kam_reason")}
                         for b in hitl.rejected_batches(project_name)]}


@tool
def explain_batch(batch_id: str) -> dict:
    """Every billed line of a consolidated invoice next to its contract rate, with any variance breaches."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    row = conn.execute("SELECT file_path FROM billing_batches WHERE batch_id = ?", (batch_id,)).fetchone()
    conn.close()
    if not row:
        return {"error": f"Unknown batch {batch_id}"}
    with open(row[0], encoding="utf-8") as f:
        doc = json.load(f)
    lines = [dict(l, site=s["site_id"]) for s in doc["sites"] for l in s["lines"]]
    return {"batch_id": batch_id, "lines": [{k: l[k] for k in ("site", "code", "qty", "unit_price_gbp")} for l in lines],
            "contract_breaches": validate_billing(lines), "grand_total_gbp": doc["grand_total_gbp"]}


@tool
def resubmit_batch(batch_id: str, explanation_to_kam: str) -> dict:
    """Send a rejected invoice back to the KAM with your explanation (e.g. every line matches the contract,
    citing the clause). Use only when the rejection reason does not need a change."""
    if len(explanation_to_kam.strip()) < 15:
        return {"error": "Write a proper explanation for the KAM."}
    res = hitl.resubmit_batch(batch_id, explanation_to_kam)
    if res["status"] == "RESUBMITTED":
        log_event("Finance Agent", "BATCH_RESUBMITTED", {**res, "explanation": explanation_to_kam})
    return res


@tool
def cancel_batch(batch_id: str, reason: str) -> dict:
    """Withdraw a rejected invoice for good (the sites are not billed)."""
    res = hitl.cancel_batch(batch_id, reason)
    if res["status"] == "CANCELLED":
        log_event("Finance Agent", "BATCH_CANCELLED", {**res, "reason": reason})
    return res


# ---------------------------------------------------------------------------
# ask_human: every agent gets its own copy, so the question records who asked
# ---------------------------------------------------------------------------

def make_ask_human(agent_display: str):
    @tool
    def ask_human(question: str, options: Optional[list[str]] = None, work_order_id: str = "",
                  project_name: str = "") -> dict:
        """Ask the Key Accounts Manager when you face a real conflict that your tools and the policy documents
        cannot settle. ALWAYS give 2 to 4 short options. Pass work_order_id for a site question (that site
        pauses until answered) or project_name for a project question. Other sites carry on."""
        wo = work_order_id or None
        project = project_name or ((get_work_order(wo) or {}).get("project_name") if wo else None)
        if not project:
            return {"error": "Give a valid work_order_id or project_name."}
        if isinstance(options, str):
            options = [o.strip() for o in options.split(",") if o.strip()]
        res = hitl.ask(project, agent_display, question, options, wo)
        if res["status"] == "LIMIT_REACHED" and wo:
            esc = hitl.raise_escalation(project, wo, agent_display,
                                        f"Still unresolved after {hitl.MAX_QUESTIONS_PER_SITE} questions: {question}",
                                        _query_facts(wo))
            log_event(agent_display, "ESCALATED_TO_KAM", "question limit reached", wo)
            return {**esc, "note": "Question limit reached, so the site was escalated to the KAM. Stop here."}
        if res["status"] == "ASKED":
            log_event(agent_display, "QUESTION -> KAM", {"question": question, "options": options}, wo)
        return {**res, "note": "Stop here. You will get the answer in a later turn."
                if res["status"] in ("ASKED", "ALREADY_ASKED") else res.get("reason")}
    return ask_human


SHARED_TOOLS = [search_knowledge, get_site_status, list_project_sites]
INVENTORY_TOOLS = [check_project_stock, plan_project_procurement, compare_suppliers, get_stock_level]
LOGISTICS_TOOLS = [check_inbound_due, receive_inbound, prepare_kits, issue_kits]
FIELD_TOOLS = [run_site_installation, request_replacement, investigate_discrepancy, send_email, read_replies,
               record_site_return, report_installed_claim, correct_installed_quantity, book_late_quarantine,
               escalate_to_kam]
FINANCE_TOOLS = [reconcile_site, get_reconciliation_policy, set_reconciliation_policy, reconcile_batch,
                 create_batch_invoice, get_rejected_batches, explain_batch, resubmit_batch, cancel_batch]