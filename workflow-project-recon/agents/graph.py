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
    "AWAITING_SIGNOFF": "human_review",
    "RECON_DISCREPANCY": "human_review",
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

    llm = get_llm().with_structured_output(RouteDecision)
    try:
        decision = llm.invoke([
            ("system",
             "You are the Main Orchestrator of a telecom rollout. Each site moves through stages: "
             "PLANNED -> STOCK_READY -> DELIVERED -> INSTALLED -> RECONCILED -> AWAITING_SIGNOFF -> CLOSED. "
             "Choose ONE agent to act next and the work orders it should handle. Only choose work orders "
             "whose next owner is that agent. Prefer moving sites with the earliest scheduled date, and "
             "batch several sites for the same agent when you can. Choose human_review when invoices are "
             "waiting for sign off or a site needs a human decision."),
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


def make_worker_node(name: str, agent):
    def node(state: ProjectState) -> dict:
        task = state["task"]
        prompt = (f"{task['instruction']}\nWork orders: {', '.join(task['work_order_ids'])}\n"
                  f"Project: {state['project_name']}")
        result = agent.invoke({"messages": [HumanMessage(prompt)]}, config={"recursion_limit": 60})
        tools_used = [tc["name"] for m in result["messages"] if isinstance(m, AIMessage) for tc in m.tool_calls]
        reply = result["messages"][-1].content
        log_event(DISPLAY_NAMES[name], "TASK_DONE", {"tools": tools_used, "reply": reply[:500]})
        return {"events": [{"agent": DISPLAY_NAMES[name], "action": "TASK_DONE",
                            "detail": f"tools={tools_used} | {reply[:300]}"}]}
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
                   for s in sites if s["status"] in ("RECON_DISCREPANCY", "NEEDS_ATTENTION")]

    log_event("Main Orchestrator", "AWAITING_HUMAN",
              {"invoices": [i["invoice_id"] for i in invoices], "escalations": escalations})

    # ---- graph pauses here; resumes with Command(resume=decisions) ----
    decisions = interrupt({"type": "human_review", "invoices": invoices, "escalations": escalations})

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