"""
Deterministic smoke test: runs the whole project WITHOUT the LLM agents (RAG triage still uses Qwen).
Use it to check the tools and data are correct before running the agentic version.

    python main.py --reset
"""
import sys
import json
import sqlite3
from database.setup_databases import init_all_databases, ORCHESTRATOR_DB
from tools.orchestrator_tools import plan_multi_site_rollout
from tools.project_tools import (source_and_inbound_project_shortages,
                                 execute_project_dispatch_and_field,
                                 reconcile_and_invoice_project)

PROJECT = "UK-5G-Phase2"


def print_ledger():
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    print(f"\n{'WO':<22}{'SKU':<15}{'Plan':>5}{'Disp':>5}{'Inst':>5}{'Quar':>5}{'Ret':>5}  Status")
    for r in conn.execute("""SELECT work_order_id, sku, qty_planned, qty_dispatched, qty_installed,
                             qty_quarantined, qty_returned, reconciliation_status
                             FROM reconciliation_ledger ORDER BY work_order_id, id"""):
        print(f"{r[0]:<22}{r[1]:<15}{r[2]:>5}{r[3]:>5}{r[4]:>5}{r[5]:>5}{r[6]:>5}  {r[7]}")
    conn.close()


if __name__ == "__main__":
    init_all_databases(force_reset="--reset" in sys.argv)
    mode = "random" if "--random" in sys.argv else "demo"

    plan = plan_multi_site_rollout(PROJECT, full_sites=3, upgrades=2, partial_sites=1, start_date="2026-12-01")
    print("[Plan]", plan["status"], "|", plan["total_sites_scheduled"], "sites |", plan["start_date"], "to", plan["end_date"])
    for wo in plan["work_orders"]:
        print("   ", wo)

    print("[Inventory]", json.dumps(source_and_inbound_project_shortages(PROJECT), indent=2))
    print("[Field]", json.dumps(execute_project_dispatch_and_field(PROJECT, field_mode=mode), indent=2))
    print("[Finance]", json.dumps(reconcile_and_invoice_project(PROJECT), indent=2))
    print_ledger()
