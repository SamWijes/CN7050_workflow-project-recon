"""
Run the LangGraph multi agent system from the terminal.

    python run_agents.py --reset                      # full project, pauses for human sign off
    python run_agents.py --reset --random             # random field faults
    python run_agents.py --reset --auto-approve       # approve everything (for quick tests)
    python run_agents.py ask inventory "How many BBU-01 are in stock?"
"""
import sys
import uuid
import json

from database.setup_databases import init_all_databases
from agents.sim_config import SIM
from agents.graph import build_graph, start_project, resume_project, pending_interrupt, ask_agent

PROJECT = "UK-5G-Phase2"
KAM_REQUEST = ("Prepare and execute project 'UK-5G-Phase2': 3 full sites, 2 upgrades and "
               "1 partial site starting 2026-12-01.")


def ask_human(payload: dict, auto: bool) -> dict:
    """Terminal version of the approvals inbox. The Streamlit UI replaces this in Phase 4."""
    print("\n" + "=" * 70 + "\nHUMAN REVIEW REQUIRED\n" + "=" * 70)
    decisions = {"approver": "KAM", "invoices": {}, "escalations": {}}
    for inv in payload["invoices"]:
        flag = f"  ANOMALIES: {inv['anomalies']}" if inv["status"] == "HELD_ANOMALY" else ""
        print(f"Invoice {inv['invoice_id']}  £{inv['total_gbp']:,.2f}  [{inv['status']}]{flag}")
        answer = "y" if auto else input("  Approve? [y/n]: ").strip().lower()
        decisions["invoices"][inv["invoice_id"]] = "approve" if answer == "y" else "reject"
    for esc in payload["escalations"]:
        print(f"Escalation {esc['work_order_id']}  status {esc['status']}")
        note = "auto noted" if auto else input("  Note for the record: ").strip()
        decisions["escalations"][esc["work_order_id"]] = note or "noted"
    return decisions


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "ask":
        print(ask_agent(sys.argv[2], " ".join(sys.argv[3:])))
        sys.exit(0)

    init_all_databases(force_reset="--reset" in sys.argv)
    SIM["field_mode"] = "random" if "--random" in sys.argv else "demo"
    auto = "--auto-approve" in sys.argv

    graph = build_graph()
    thread_id = f"{PROJECT}-{uuid.uuid4().hex[:6]}"
    print(f"Thread: {thread_id}")

    result = start_project(graph, KAM_REQUEST, PROJECT, thread_id)
    while (payload := pending_interrupt(result)) is not None:
        result = resume_project(graph, thread_id, ask_human(payload, auto))

    print("\n================ EXECUTIVE SUMMARY ================\n")
    print(result.get("summary"))