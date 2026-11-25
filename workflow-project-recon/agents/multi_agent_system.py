import json
import sqlite3
from database.setup_databases import init_all_databases, ORCHESTRATOR_DB
from rag.rag_query import query_telecom_rag
from agents.base_agent import BaseAgent
from tools.orchestrator_tools import plan_multi_site_rollout, get_bom_for_category
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


# ---------------------------------------------------------------------------
# Batch Helper Tools (keep multi-site execution fast & within LLM context)
# ---------------------------------------------------------------------------

def get_project_work_orders(project_name: str) -> list:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    rows = conn.execute(
        "SELECT work_order_id, site_id, site_category, scheduled_date, subcon_id, engineer_id "
        "FROM work_orders WHERE project_name = ?",
        (project_name,),
    ).fetchall()
    conn.close()
    return [
        {
            "work_order_id": r[0],
            "site_id": r[1],
            "site_category": r[2],
            "scheduled_date": r[3],
            "subcon_id": r[4],
            "engineer_id": r[5],
        }
        for r in rows
    ]


def source_and_inbound_project_shortages(project_name: str, inbound_date: str = "2026-11-30") -> dict:
    """Aggregates BOM across all planned sites in a project, sources shortages, clears & inbounds."""
    wos = get_project_work_orders(project_name)
    agg = {}
    for wo in wos:
        for item in get_bom_for_category(wo["site_category"]):
            sku = item["sku"]
            if sku not in agg:
                agg[sku] = {"sku": sku, "qty": 0, "serialized": item["serialized"]}
            agg[sku]["qty"] += item["qty"]

    stock = check_bom_stock(list(agg.values()))
    if stock["all_in_stock"]:
        return {"status": "ALL_IN_STOCK", "shortages_resolved": 0}

    resolved = []
    for s in stock["shortages"]:
        suppliers = query_suppliers(s["sku"], s["shortage_qty"])
        if not suppliers:
            return {"status": "FAILED", "reason": f"No supplier has {s['shortage_qty']} of {s['sku']}"}
        # Prefer IntelliCommmsTech (INTERNAL) for bulk orders, else fastest supplier
        chosen = next((sup for sup in suppliers if sup["supplier_type"] == "INTERNAL"), suppliers[0])
        po = create_purchase_order(chosen["supplier_id"], s["sku"], s["shortage_qty"])
        ship = schedule_inbound_shipment(po["po_id"], pallets_required=1, inbound_date=inbound_date, serialized=s["serialized"])
        inbound = verify_and_inbound_packing_list(ship["shipment_id"], ship["packing_list"])
        resolved.append({"sku": s["sku"], "po_id": po["po_id"], "supplier": chosen["supplier_name"], "status": inbound["status"]})

    return {"status": "SHORTAGES_RESOLVED", "details": resolved}


def execute_project_dispatch_and_field(project_name: str, field_mode: str = "random") -> dict:
    """Dispatches BOMs, creates waybills, runs Field App + RAG fault triage & replenishment for all sites."""
    wos = get_project_work_orders(project_name)
    total_faults = 0
    replacements_completed = 0

    for wo in wos:
        bom = get_bom_for_category(wo["site_category"])
        dispatch = dispatch_bom_from_wms(wo["work_order_id"], wo["site_id"], bom, collector=wo["subcon_id"])
        if dispatch["status"] != "DISPATCHED":
            return {"status": "FAILED", "site_id": wo["site_id"], "reason": dispatch}

        # Record dispatched quantities in orchestrator_ledger
        conn = sqlite3.connect(ORCHESTRATOR_DB)
        for item in dispatch["items"]:
            conn.execute(
                "UPDATE reconciliation_ledger SET qty_dispatched = qty_dispatched + ? "
                "WHERE work_order_id = ? AND sku = ?",
                (item["qty"], wo["work_order_id"], item["sku"]),
            )
        conn.commit()
        conn.close()

        create_outbound_waybill(wo["work_order_id"], wo["site_id"], dispatch["items"], wo["scheduled_date"])

        # Field App + RAG Fault Triage
        report = simulate_field_app_report(wo["work_order_id"], wo["site_id"], wo["engineer_id"], dispatch["items"], mode=field_mode)
        triage = process_field_report(report)
        total_faults += len(report["faulty_items"])

        # Automated replacement for RAG-flagged faults
        for req in triage["replenishment_requests"]:
            rep_bom = [{"sku": req["sku"], "qty": req["qty"], "serialized": req["serialized"]}]
            rep_stock = check_bom_stock(rep_bom)
            if not rep_stock["all_in_stock"]:
                s = rep_stock["shortages"][0]
                sup = query_suppliers(s["sku"], s["shortage_qty"])[0]
                po = create_purchase_order(sup["supplier_id"], s["sku"], s["shortage_qty"])
                ship = schedule_inbound_shipment(po["po_id"], 1, wo["scheduled_date"], True)
                verify_and_inbound_packing_list(ship["shipment_id"], ship["packing_list"])

            rep_dispatch = dispatch_bom_from_wms(wo["work_order_id"], wo["site_id"], rep_bom, collector=req["collector"])
            conn = sqlite3.connect(ORCHESTRATOR_DB)
            conn.execute(
                "UPDATE reconciliation_ledger SET qty_dispatched = qty_dispatched + 1 "
                "WHERE work_order_id = ? AND sku = ?",
                (wo["work_order_id"], req["sku"]),
            )
            conn.commit()
            conn.close()

            process_field_report({
                "work_order_id": wo["work_order_id"],
                "site_id": wo["site_id"],
                "engineer_id": req["collector"],
                "installed_items": rep_dispatch["items"],
                "faulty_items": [],
            })
            replacements_completed += 1

    return {
        "status": "FIELD_EXECUTION_COMPLETE",
        "sites_executed": len(wos),
        "faults_triaged_via_rag": total_faults,
        "replacements_installed": replacements_completed,
    }


def reconcile_and_invoice_project(project_name: str) -> dict:
    """Runs 3-way reconciliation and generates invoices for all sites in the project."""
    wos = get_project_work_orders(project_name)
    closed = 0
    blocked = []
    total_revenue = 0.0

    for wo in wos:
        res = generate_client_invoice(wo["work_order_id"])
        if res["status"] == "INVOICED_AND_CLOSED":
            closed += 1
            total_revenue += res["grand_total_gbp"]
        else:
            blocked.append({"work_order_id": wo["work_order_id"], "reason": res})

    return {
        "status": "FINANCE_COMPLETE",
        "sites_invoiced_and_closed": closed,
        "sites_blocked": len(blocked),
        "total_project_revenue_gbp": round(total_revenue, 2),
    }


# ---------------------------------------------------------------------------
# Instantiate Specialist Agents
# ---------------------------------------------------------------------------

inventory_logistics_agent = BaseAgent(
    name="Inventory & Logistics Agent",
    system_prompt="You manage warehouse stock, IntelliCommmsTech/3rd-party sourcing, customs clearance, and inbound verification.",
    tools=[{
        "type": "function",
        "function": {
            "name": "source_and_inbound_project_shortages",
            "description": "Check aggregated project BOM, order shortages, clear customs, and inbound into WMS.",
            "parameters": {
                "type": "object",
                "properties": {"project_name": {"type": "string"}},
                "required": ["project_name"],
            },
        },
    }],
    tool_map={"source_and_inbound_project_shortages": source_and_inbound_project_shortages},
)

field_agent = BaseAgent(
    name="Field Execution Agent",
    system_prompt="You coordinate site dispatch, engineer Field App confirmations, RAG fault triage, and faulty unit replacements.",
    tools=[{
        "type": "function",
        "function": {
            "name": "execute_project_dispatch_and_field",
            "description": "Dispatch materials, run Field App simulation, RAG fault triage, and replacements for a project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string"},
                    "field_mode": {"type": "string", "enum": ["demo", "random"]},
                },
                "required": ["project_name"],
            },
        },
    }],
    tool_map={"execute_project_dispatch_and_field": execute_project_dispatch_and_field},
)

finance_agent = BaseAgent(
    name="Finance & Reconciliation Agent",
    system_prompt="You perform 3-way reconciliation (Dispatched = Installed + Quarantined) and issue client invoices.",
    tools=[{
        "type": "function",
        "function": {
            "name": "reconcile_and_invoice_project",
            "description": "Reconcile and invoice all work orders for a project.",
            "parameters": {
                "type": "object",
                "properties": {"project_name": {"type": "string"}},
                "required": ["project_name"],
            },
        },
    }],
    tool_map={"reconcile_and_invoice_project": reconcile_and_invoice_project},
)


# ---------------------------------------------------------------------------
# Main Orchestrator (Supervisor) Agent
# ---------------------------------------------------------------------------

orchestrator_agent = BaseAgent(
    name="Main Orchestrator Agent",
    system_prompt=(
        "You are the Main Orchestrator for IntelliCommmsTech. "
        "When a Key Account Manager requests a multi-site rollout, execute these steps in order: "
        "1) Call plan_multi_site_rollout to schedule teams and generate work orders. "
        "2) Call delegate_to_inventory_logistics to source and inbound any stock shortages. "
        "3) Call delegate_to_field_execution to dispatch materials, run field installs, and handle RAG fault replacements. "
        "4) Call delegate_to_finance to reconcile all sites and generate client invoices. "
        "Present a concise executive summary of the completed rollout."
    ),
    tools=[
        {
            "type": "function",
            "function": {
                "name": "plan_multi_site_rollout",
                "description": "Schedule a multi-site rollout across available Subcons and Engineers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string"},
                        "full_sites": {"type": "integer"},
                        "upgrades": {"type": "integer"},
                        "partial_sites": {"type": "integer"},
                        "start_date": {"type": "string"},
                    },
                    "required": ["project_name", "full_sites", "upgrades", "partial_sites"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delegate_to_inventory_logistics",
                "description": "Delegate stock checking, sourcing, and inbound clearance to the Inventory & Logistics Agent.",
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delegate_to_field_execution",
                "description": "Delegate site dispatch, field app installs, and RAG fault triage to the Field Execution Agent.",
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                    "required": ["task"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delegate_to_finance",
                "description": "Delegate 3-way reconciliation and invoicing to the Finance Agent.",
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                    "required": ["task"],
                },
            },
        },
    ],
    tool_map={
        "plan_multi_site_rollout": plan_multi_site_rollout,
        "delegate_to_inventory_logistics": lambda task: {"agent_reply": inventory_logistics_agent.run(task)},
        "delegate_to_field_execution": lambda task: {"agent_reply": field_agent.run(task)},
        "delegate_to_finance": lambda task: {"agent_reply": finance_agent.run(task)},
    },
    max_steps=8,
)


if __name__ == "__main__":
    init_all_databases(force_reset=True)
    # Test with 6 sites first (2 days across 5 teams); scale to 120/30/20 once supplier stock is bumped
    kam_prompt = (
        "Prepare and execute project 'UK-5G-Phase2': "
        "3 full sites, 2 upgrades, and 1 partial site starting 2026-12-01."
    )
    final_report = orchestrator_agent.run(kam_prompt)
    print("\n================ EXECUTIVE SUMMARY ================\n")
    print(final_report)