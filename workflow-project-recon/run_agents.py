"""
Run the LangGraph multi agent system from the terminal.

    python run_agents.py --reset                          # full project, you approve and reply
    python run_agents.py --reset --auto                   # approve everything, simulated replies
    python run_agents.py --reset --auto --scenario-demo   # inject the demo error scenarios
    python run_agents.py --reset --scenarios SITE-002=qty_mismatch,SITE-004=faulty_lost
    python run_agents.py --reset --random                 # random background field faults
    python run_agents.py ask inventory "How many BBU-01 are in stock?"
"""
import sys
import uuid

from database.setup_databases import init_all_databases
from agents.sim_config import SIM
from agents.graph import build_graph, start_project, resume_project, pending_interrupt, ask_agent
from simulation.scenarios import DEMO_ASSIGNMENT, parse_assignment, reset_truth
from simulation.humans import simulate_reply
from tools.mailbox_tools import deliver_reply

PROJECT = "UK-5G-Phase2"
KAM_REQUEST = ("Prepare and execute project 'UK-5G-Phase2': 3 full sites, 2 upgrades and "
               "1 partial site starting 2026-12-01.")


def _arg(name: str):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def handle_pause(payload: dict, auto: bool) -> dict:
    """Terminal version of the approvals inbox and mailboxes. The Streamlit UI replaces this in Phase 4."""
    print("\n" + "=" * 70 + "\nHUMAN INPUT REQUIRED\n" + "=" * 70)

    for msg in payload.get("awaiting_replies", []):
        print(f"\nEmail #{msg['message_id']} to {msg['recipient']}  [{msg['work_order_id']}]")
        print(f"Subject: {msg['subject']}\n{msg['body']}\n")
        choice = "s" if auto else input("  [s]imulate reply, [t]ype a reply, or [l]eave unanswered: ").strip().lower()
        if choice == "t":
            res = deliver_reply(msg["message_id"], input("  Reply text: "))
        elif choice == "s":
            res = simulate_reply(msg)
        else:
            continue
        print(f"  -> reply delivered (#{res.get('reply_id')})")

    decisions = {"approver": "KAM", "invoices": {}, "escalations": {}}
    for inv in payload["invoices"]:
        flag = f"  ANOMALIES: {inv['anomalies']}" if inv["status"] == "HELD_ANOMALY" else ""
        print(f"Invoice {inv['invoice_id']}  £{inv['total_gbp']:,.2f}  [{inv['status']}]{flag}")
        answer = "y" if auto else input("  Approve? [y/n]: ").strip().lower()
        decisions["invoices"][inv["invoice_id"]] = "approve" if answer == "y" else "reject"
    for esc in payload["escalations"]:
        print(f"Escalation {esc['work_order_id']}  status {esc['status']}")
        note = "KAM reviewed: write off and raise with subcontractor" if auto else input("  Decision / note: ").strip()
        decisions["escalations"][esc["work_order_id"]] = note or "noted"
    return decisions


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "ask":
        print(ask_agent(sys.argv[2], " ".join(sys.argv[3:])))
        sys.exit(0)

    if "--reset" in sys.argv:
        init_all_databases(force_reset=True)
        reset_truth()
    else:
        init_all_databases()

    SIM["field_mode"] = "random" if "--random" in sys.argv else "demo"
    SIM["site_scenarios"] = DEMO_ASSIGNMENT if "--scenario-demo" in sys.argv else parse_assignment(_arg("--scenarios") or "")
    auto = "--auto" in sys.argv
    print("Scenarios:", SIM["site_scenarios"] or "none (happy path)")

    graph = build_graph()
    thread_id = f"{PROJECT}-{uuid.uuid4().hex[:6]}"
    print(f"Thread: {thread_id}")

    result = start_project(graph, KAM_REQUEST, PROJECT, thread_id)
    while (payload := pending_interrupt(result)) is not None:
        result = resume_project(graph, thread_id, handle_pause(payload, auto))

    print("\n================ EXECUTIVE SUMMARY ================\n")
    print(result.get("summary"))