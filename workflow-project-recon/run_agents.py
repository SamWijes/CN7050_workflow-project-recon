"""
Run the LangGraph multi agent system from the terminal.

    python run_agents.py --reset                          # you answer every pause yourself
    python run_agents.py --reset --auto                   # automatic answers, approvals and simulated replies
    python run_agents.py --reset --auto --scenario-demo   # inject the demo error scenarios
    python run_agents.py --reset --scenarios SITE-002=qty_mismatch_vague,SITE-003=faulty_lost
    python run_agents.py --reset --kam "2 full sites and 1 upgrade from 2026-12-01, reconcile every 2 sites"
    python run_agents.py --reset --auto --auto-answer "every 2 sites"
    python run_agents.py request-recon UK-5G-Phase2       # ask for a reconciliation while a run is going
    python run_agents.py ask inventory "How many BBU-01 are in stock?"
"""
import sys
import uuid

from database.setup_databases import init_all_databases
from agents.sim_config import SIM
from agents.graph import build_graph, start_project, resume_project, pending_interrupt, ask_agent
from agents import hitl
from simulation.scenarios import DEMO_ASSIGNMENT, parse_assignment, reset_truth
from simulation.humans import simulate_reply
from tools.mailbox_tools import deliver_reply

PROJECT = "UK-5G-Phase2"
KAM_REQUEST = ("Prepare and execute project 'UK-5G-Phase2': 3 full sites, 2 upgrades and "
               "1 partial site starting 2026-12-01.")


def _arg(name: str):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def handle_pause(payload: dict, auto: bool, auto_answer: str = None) -> dict:
    """Terminal version of the approvals inbox, question inbox and mailboxes (the UI replaces this)."""
    print("\n" + "=" * 70 + "\nHUMAN INPUT REQUIRED\n" + "=" * 70)
    decisions = {"approver": "KAM", "batches": {}, "answers": {}, "escalations": {}}

    for q in payload.get("questions", []):
        print(f"\nQuestion #{q['question_id']} from {q['agent']}" + (f" about {q['work_order_id']}" if q['work_order_id'] else ""))
        print(f"  {q['question']}")
        for i, opt in enumerate(q["options"], 1):
            print(f"    {i}. {opt}")
        if auto:
            ans = auto_answer or (q["options"][0] if q["options"] else "Use your best judgement.")
            print(f"  -> auto answer: {ans}")
        else:
            raw = input("  Your answer (option number or free text): ").strip()
            ans = q["options"][int(raw) - 1] if raw.isdigit() and 0 < int(raw) <= len(q["options"]) else raw
        decisions["answers"][q["question_id"]] = ans

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

    for b in payload.get("batches", []):
        flag = "  ANOMALIES FLAGGED" if b["has_anomalies"] else ""
        print(f"\nConsolidated invoice {b['batch_id']}  £{b['total_gbp']:,.2f}  sites: {', '.join(b['work_orders'])}{flag}")
        answer = "y" if auto else input("  Approve this batch? [y/n]: ").strip().lower()
        decisions["batches"][b["batch_id"]] = "approve" if answer == "y" else "reject"

    for esc in payload.get("escalations", []):
        print(f"\nEscalation {esc['work_order_id']}  status {esc['status']}")
        note = "KAM reviewed: write off and raise with subcontractor" if auto else input("  Decision / note: ").strip()
        decisions["escalations"][esc["work_order_id"]] = note or "noted"

    if payload.get("installed_awaiting_reconciliation"):
        sites = payload["installed_awaiting_reconciliation"]
        print(f"\n{len(sites)} installed site(s) waiting for a reconciliation request: {', '.join(sites)}")
        decisions["request_reconciliation"] = auto or input("  Reconcile them now? [y/n]: ").strip().lower() == "y"
    return decisions


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "ask":
        print(ask_agent(sys.argv[2], " ".join(sys.argv[3:])))
        sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "request-recon":
        wos = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        print(hitl.request_reconciliation(sys.argv[2], "KAM (CLI)", wos))
        sys.exit(0)

    if "--reset" in sys.argv:
        init_all_databases(force_reset=True)
        reset_truth()
    else:
        init_all_databases()

    SIM["field_mode"] = "random" if "--random" in sys.argv else "demo"
    SIM["site_scenarios"] = DEMO_ASSIGNMENT if "--scenario-demo" in sys.argv else parse_assignment(_arg("--scenarios") or "")
    auto = "--auto" in sys.argv
    kam = _arg("--kam") or KAM_REQUEST
    print("KAM request:", kam)
    print("Scenarios:", SIM["site_scenarios"] or "none (happy path)")

    graph = build_graph()
    thread_id = f"{PROJECT}-{uuid.uuid4().hex[:6]}"
    print(f"Thread: {thread_id}")

    result = start_project(graph, kam, PROJECT, thread_id)
    while (payload := pending_interrupt(result)) is not None:
        result = resume_project(graph, thread_id, handle_pause(payload, auto, _arg("--auto-answer")))

    print("\n================ EXECUTIVE SUMMARY ================\n")
    print(result.get("summary"))