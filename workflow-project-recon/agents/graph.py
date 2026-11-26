"""
LangGraph orchestration layer.

    START -> planner -> supervisor <-> {inventory, logistics, field, finance, human_review} -> finish -> END

The supervisor (Main Orchestrator) looks at the live state of every site and decides at runtime which
agent works next and on which sites. Sites move independently, so one site can be invoicing while
another is still being sourced. Human sign off uses a LangGraph interrupt: the graph pauses, its state is
checkpointed to SQLite, and it resumes when the KAM or client answers (from the CLI now, the UI later).
"""
import operator
import sqlite3
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from agents.llm import get_llm
from agents.events import log_event
from agents.workers import build_workers, DISPLAY_NAMES
from tools.orchestrator_tools import ORCHESTRATOR_DB, plan_multi_site_rollout, set_work_order_status
from tools.project_tools import get_project_work_orders
from tools.finance_tools import record_invoice_decision
from tools.mailbox_tools import awaiting_replies

CHECKPOINT_DB = "databases/graph_checkpoints.db"
MAX_STEPS = 60          # hard stop for the supervisor loop
MAX_ATTEMPTS = 3        # same site stuck at the same stage this many times -> human

# Which agent can move a site forward from each stage
STAGE_OWNER = {
    "PLANNED": "inventory",
    "STOCK_READY": "logistics",
    "DELIVERED": "field",
    "REPLACEMENT_PENDING": "field",
    "INSTALLED": "finance",
    "RECONCILED": "finance",
    "RECON_DISCREPANCY": "field",      # Field agent investigates and emails whoever holds the items
    "REPLY_RECEIVED": "field",         # Field agent reads the reply and resolves
    "AWAITING_SIGNOFF": "human_review",
    "AWAITING_REPLY": "human_review",  # graph pauses until the person answers
    "NEEDS_ATTENTION": "human_review",
}
TERMINAL = {"CLOSED", "INVOICE_REJECTED", "ESCALATED"}
PIPELINE_ORDER = ["inventory", "logistics", "field", "finance", "human_review"]


# ---------------------------------------------------------------------------
# State and structured outputs
# ---------------------------------------------------------------------------

class ProjectState(TypedDict, total=False):
    kam_request: str
    project_name: str
    plan: dict
    next_agent: str
    task: dict
    steps: int
    attempts: dict
    events: Annotated[list, operator.add]
    summary: str


class PlanRequest(BaseModel):
    full_sites: int = Field(description="Number of full sites requested")
    upgrades: int = Field(description="Number of upgrade sites requested")
    partial_sites: int = Field(description="Number of partial (cabinet only) sites requested")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")


class RouteDecision(BaseModel):
    next_agent: Literal["inventory", "logistics", "field", "finance", "human_review", "FINISH"]
    work_order_ids: list[str] = Field(description="Work orders the chosen agent should handle now")
    instruction: str = Field(description="Short instruction for the chosen agent")
    reason: str = Field(description="Why this agent and these sites are next")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sites(project_name: str) -> list:
    return get_project_work_orders(project_name)


def _site_table(sites: list) -> str:
    rows = [f"{s['work_order_id']} | {s['site_category']} | {s['scheduled_date']} | {s['status']} | "
            f"next owner: {STAGE_OWNER.get(s['status'], 'done' if s['status'] in TERMINAL else '?')}"
            for s in sites]
    return "\n".join(rows)


def _fallback(options: dict) -> RouteDecision:
    """Deterministic safety net when the LLM returns an invalid route."""
    agent = next(a for a in PIPELINE_ORDER if a in options)
    return RouteDecision(next_agent=agent, work_order_ids=options[agent],
                         instruction=f"Progress these sites: {', '.join(options[agent])}",
                         reason="fallback rule: earliest pipeline stage with work waiting")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: ProjectState) -> dict:
    """Main Orchestrator, planning role: turns the KAM's free text request into a scheduled rollout."""
    llm = get_llm().with_structured_output(PlanRequest)
    req = llm.invoke([
        ("system", "Extract the rollout request. Use 0 for any site type not mentioned. Dates as YYYY-MM-DD."),
        ("user", state["kam_request"]),
    ])
    plan = plan_multi_site_rollout(state["project_name"], req.full_sites, req.upgrades, req.partial_sites,
                                   req.start_date)
    log_event("Main Orchestrator", "PROJECT_PLANNED",
              {"request": req.model_dump(), "sites": plan["total_sites_scheduled"],
               "dates": [plan["start_date"], plan["end_date"]]})
    return {"plan": {k: plan[k] for k in ("status", "total_sites_scheduled", "start_date", "end_date", "work_orders")},
            "steps": 0, "attempts": {},
            "events": [{"agent": "Main Orchestrator", "action": "PROJECT_PLANNED",
                        "detail": f"{plan['total_sites_scheduled']} sites, {plan['start_date']} to {plan['end_date']}"}]}


def supervisor_node(state: ProjectState) -> dict:
    """Main Orchestrator, coordination role: decides who works next, on which sites."""
    sites = _sites(state["project_name"])
    steps = state.get("steps", 0) + 1
    attempts = dict(state.get("attempts", {}))

    # Sites stuck at the same stage too many times are handed to a human
    for s in sites:
        if attempts.get(f"{s['work_order_id']}:{s['status']}", 0) >= MAX_ATTEMPTS:
            set_work_order_status(s["work_order_id"], "NEEDS_ATTENTION")
            log_event("Main Orchestrator", "STUCK_ESCALATED", s["status"], s["work_order_id"])
    sites = _sites(state["project_name"])

    options = {}
    for s in sites:
        owner = STAGE_OWNER.get(s["status"])
        if owner:
            options.setdefault(owner, []).append(s["work_order_id"])

    if not options or steps > MAX_STEPS:
        return {"next_agent": "FINISH", "steps": steps}
    
    if len(options) == 1:  # only one agent can act: no decision to make, save an LLM call
        decision = _fallback(options)
        decision.reason = "only one agent has work available"
        for wo in decision.work_order_ids:
            status = next(s["status"] for s in sites if s["work_order_id"] == wo)
            attempts[f"{wo}:{status}"] = attempts.get(f"{wo}:{status}", 0) + 1
        log_event("Main Orchestrator", f"ROUTE -> {decision.next_agent}",
                  {"work_orders": decision.work_order_ids, "reason": decision.reason})
        return {"next_agent": decision.next_agent, "steps": steps, "attempts": attempts,
                "task": {"work_order_ids": decision.work_order_ids, "instruction": decision.instruction},
                "events": [{"agent": "Main Orchestrator", "action": f"ROUTE -> {decision.next_agent}",
                            "detail": f"{decision.work_order_ids}: {decision.reason}"}]}
    
    llm = get_llm().with_structured_output(RouteDecision)
    try:
        decision = llm.invoke([
            ("system",
             "You are the Main Orchestrator of a telecom rollout. Each site moves through stages: "
             "PLANNED -> STOCK_READY -> DELIVERED -> INSTALLED -> RECONCILED -> AWAITING_SIGNOFF -> CLOSED. "
             "Choose ONE agent to act next and the work orders it should handle. Only choose work orders "
             "batch several sites for the same agent when you can. Sites that fail reconciliation go to the "
             "field agent, who investigates and emails the person holding the items. Choose human_review "
             "when invoices wait for sign off, a site needs a human decision, or the only remaining work is "
             "waiting for email replies."),
            ("user", f"Project: {state['project_name']}\nSites:\n{_site_table(sites)}\n\n"
                     f"Agents with work available: {options}"),
        ])
        valid = [w for w in decision.work_order_ids if w in options.get(decision.next_agent, [])]
        if decision.next_agent == "FINISH" or not valid:
            decision = _fallback(options)
        else:
            decision.work_order_ids = valid
    except Exception as e:  # malformed JSON from the model, Ollama hiccup, etc.
        log_event("Main Orchestrator", "ROUTING_ERROR", str(e))
        decision = _fallback(options)

    for wo in decision.work_order_ids:
        status = next(s["status"] for s in sites if s["work_order_id"] == wo)
        attempts[f"{wo}:{status}"] = attempts.get(f"{wo}:{status}", 0) + 1

    log_event("Main Orchestrator", f"ROUTE -> {decision.next_agent}",
              {"work_orders": decision.work_order_ids, "reason": decision.reason})
    return {"next_agent": decision.next_agent, "steps": steps, "attempts": attempts,
            "task": {"work_order_ids": decision.work_order_ids, "instruction": decision.instruction},
            "events": [{"agent": "Main Orchestrator", "action": f"ROUTE -> {decision.next_agent}",
                        "detail": f"{decision.work_order_ids}: {decision.reason}"}]}


# What "done" means for each stage, spelled out for a small model
STAGE_GOALS = {
    "PLANNED": "Make it STOCK_READY: check_site_stock, order_site_shortages if anything is short, then "
               "confirm_site_stock_ready.",
    "STOCK_READY": "Deliver it: call dispatch_site.",
    "DELIVERED": "Install it: call run_site_installation, then request_replacement for each fault that "
                 "needs a replacement.",
    "REPLACEMENT_PENDING": "Call run_site_installation to list outstanding faults, then request_replacement "
                           "for each one.",
    "INSTALLED": "Call reconcile_site, then create_invoice if it reconciled.",
    "RECONCILED": "Call create_invoice.",
    "RECON_DISCREPANCY": "Call investigate_discrepancy, then send_email to the contact it gives you.",
    "REPLY_RECEIVED": "Call read_replies, then record_site_return, book_late_quarantine or escalate_to_kam "
                      "according to exactly what the reply says.",
}
MAX_NUDGES = 2


def _status(work_order_id: str) -> str:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    row = conn.execute("SELECT status FROM work_orders WHERE work_order_id = ?", (work_order_id,)).fetchone()
    conn.close()
    return row[0] if row else "UNKNOWN"


def make_worker_node(name: str, agent):
    """Runs the agent one work order at a time, and nudges it if it stops before the stage changes."""
    def node(state: ProjectState) -> dict:
        task, events = state["task"], []
        for wo in task["work_order_ids"]:
            start = _status(wo)
            # One work order per prompt: never show the model other site IDs it might act on by mistake
            messages = [HumanMessage(
                f"Work order: {wo}\nProject: {state['project_name']}\nCurrent stage: {start}\n"
                f"To finish you MUST: {STAGE_GOALS.get(start, 'move it to its next stage.')}")]
            tools_used, reply = [], ""
            for attempt in range(MAX_NUDGES + 1):
                
                try:
                    result = agent.invoke({"messages": messages}, config={"recursion_limit": 40})
                except Exception as e:  # timeout, malformed tool call, Ollama error: log and move on
                    log_event(DISPLAY_NAMES[name], "AGENT_ERROR", str(e)[:300], wo)
                    break

                messages = result["messages"]
                tools_used = [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
                reply = messages[-1].content
                if _status(wo) != start:
                    break
                if attempt < MAX_NUDGES:
                    log_event(DISPLAY_NAMES[name], "NUDGED", f"still {start} after attempt {attempt + 1}", wo)
                    messages = messages + [HumanMessage(
                        f"{wo} is still {start}. You have not finished. Do not explain, call the next tool now: "
                        f"{STAGE_GOALS.get(start, '')}")]
            log_event(DISPLAY_NAMES[name], "TASK_DONE",
                      {"tools": tools_used, "stage": f"{start} -> {_status(wo)}", "reply": reply[:400]}, wo)
            events.append({"agent": DISPLAY_NAMES[name], "action": "TASK_DONE",
                           "detail": f"{wo}: {start} -> {_status(wo)} | tools={tools_used}"})
        return {"events": events}
    return node


def human_review_node(state: ProjectState) -> dict:
    """Pauses the whole graph until the KAM / client answers. State is checkpointed meanwhile."""
    sites = _sites(state["project_name"])
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    invoices = [dict(zip(["invoice_id", "work_order_id", "total_gbp", "status", "anomalies"], r))
                for r in conn.execute("""SELECT invoice_id, work_order_id, grand_total_gbp, status, anomalies_json
                                         FROM invoices WHERE status IN ('PENDING_APPROVAL', 'HELD_ANOMALY')""")]
    conn.close()
    escalations = [{"work_order_id": s["work_order_id"], "status": s["status"]}
                   for s in sites if s["status"] == "NEEDS_ATTENTION"]
    waiting = awaiting_replies(state["project_name"])

    log_event("Main Orchestrator", "AWAITING_HUMAN",
              {"invoices": [i["invoice_id"] for i in invoices], "escalations": escalations,
               "emails_awaiting_reply": [m["message_id"] for m in waiting]})

    # ---- graph pauses here; resumes with Command(resume=decisions) ----
    # Replies to emails are delivered into the mailbox (simulated or typed in the UI) before resuming.
    decisions = interrupt({"type": "human_review", "invoices": invoices, "escalations": escalations,
                           "awaiting_replies": waiting})
    applied = []
    for inv_id, verdict in decisions.get("invoices", {}).items():
        res = record_invoice_decision(inv_id, decisions.get("approver", "KAM"), verdict == "approve")
        applied.append(res)
        log_event("Human (KAM)", f"INVOICE_{res['status']}", res, inv_id.replace("INV-", ""))
    for wo, note in decisions.get("escalations", {}).items():
        set_work_order_status(wo, "ESCALATED")
        log_event("Human (KAM)", "ESCALATION_NOTED", note, wo)
        applied.append({"work_order_id": wo, "note": note})

    return {"events": [{"agent": "Human (KAM)", "action": "REVIEW_DONE", "detail": str(applied)[:300]}]}


def finish_node(state: ProjectState) -> dict:
    sites = _sites(state["project_name"])
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    approved = conn.execute("SELECT COALESCE(SUM(grand_total_gbp), 0), COUNT(*) FROM invoices WHERE status = 'APPROVED'"
                            ).fetchone()
    conn.close()
    table = _site_table(sites)
    try:
        summary = get_llm().invoke([
            ("system", "Write a 5 line executive summary for the Key Accounts Manager. Use only the facts given."),
            ("user", f"Final site status:\n{table}\nApproved invoices: {approved[1]}, total £{approved[0]:,.2f}"),
        ]).content
    except Exception:
        summary = table
    log_event("Main Orchestrator", "PROJECT_SUMMARY", summary)
    return {"summary": summary}


def route_from_supervisor(state: ProjectState) -> str:
    return "finish" if state["next_agent"] == "FINISH" else state["next_agent"]


# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    workers = build_workers()
    g = StateGraph(ProjectState)
    g.add_node("planner", planner_node)
    g.add_node("supervisor", supervisor_node)
    for name, agent in workers.items():
        g.add_node(name, make_worker_node(name, agent))
        g.add_edge(name, "supervisor")
    g.add_node("human_review", human_review_node)
    g.add_node("finish", finish_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor,
                            ["inventory", "logistics", "field", "finance", "human_review", "finish"])
    g.add_edge("human_review", "supervisor")
    g.add_edge("finish", END)

    if checkpointer is None:
        checkpointer = SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))
    return g.compile(checkpointer=checkpointer)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}


def start_project(graph, kam_request: str, project_name: str, thread_id: str) -> dict:
    """Runs until the first human interrupt (or the end). Returns the graph result."""
    return graph.invoke({"kam_request": kam_request, "project_name": project_name}, _config(thread_id))


def resume_project(graph, thread_id: str, decisions: dict) -> dict:
    """Continues a paused project with the human's decisions."""
    return graph.invoke(Command(resume=decisions), _config(thread_id))


def pending_interrupt(result: dict) -> Optional[dict]:
    items = result.get("__interrupt__") or []
    return items[0].value if items else None


def ask_agent(agent_name: str, question: str) -> str:
    """Ask any specialist agent a question at any time, outside the project flow.
    It answers from the live databases and its knowledge base."""
    agent = build_workers()[agent_name]
    result = agent.invoke({"messages": [HumanMessage(question)]}, config={"recursion_limit": 30})
    log_event(DISPLAY_NAMES[agent_name], "ANSWERED_QUERY", {"question": question})
    return result["messages"][-1].content