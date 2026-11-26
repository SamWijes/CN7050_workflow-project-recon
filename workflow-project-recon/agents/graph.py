"""
LangGraph orchestration layer.

    START -> planner -> supervisor <-> {inventory, logistics, field, finance, human_review} -> finish -> END

The supervisor (Main Orchestrator) looks at the live state of every site and decides at runtime which
agent works next. Sites move independently. Procurement runs once for the whole project; reconciliation
runs in batches according to the project's policy (from the KAM request, the contract, or by asking the
KAM) or whenever a human requests it. Any agent can ask a human a question: that site (or task) pauses,
the graph interrupts, and the same agent acts on the answer when the graph resumes.
"""
import operator
import sqlite3
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from agents.llm import get_llm, get_structured_llm
from agents.events import log_event
from agents.workers import build_workers, get_stage_agent, DISPLAY_NAMES
from agents import hitl
from tools.orchestrator_tools import ORCHESTRATOR_DB, plan_multi_site_rollout, set_work_order_status
from tools.project_tools import get_project_work_orders
from tools.mailbox_tools import awaiting_replies

CHECKPOINT_DB = "databases/graph_checkpoints.db"
MAX_STEPS = 60          # hard stop for the supervisor loop
MAX_ATTEMPTS = 3        # same site stuck at the same stage this many times -> human
MAX_NUDGES = 2

POLICY_TASK = "PROJECT:RECON_POLICY"   # pseudo work order id for project level tasks

# Which agent moves a site forward from each stage (INSTALLED / RECONCILED are batched separately)
STAGE_OWNER = {
    "PLANNED": "inventory",
    "STOCK_READY": "logistics",
    "DELIVERED": "field",
    "REPLACEMENT_PENDING": "field",
    "RECON_DISCREPANCY": "field",
    "REPLY_RECEIVED": "field",
    "AWAITING_SIGNOFF": "human_review",
    "AWAITING_REPLY": "human_review",
    "AWAITING_HUMAN_ANSWER": "human_review",
    "NEEDS_ATTENTION": "human_review",
}
TERMINAL = {"CLOSED", "INVOICE_REJECTED", "ESCALATED"}
BEFORE_INSTALL = {"PLANNED", "STOCK_READY", "DISPATCHING", "DELIVERED", "INSTALLING", "REPLACEMENT_PENDING"}
PIPELINE_ORDER = ["inventory", "logistics", "field", "finance", "human_review"]

# What "done" means for each task, spelled out for a small model
STAGE_GOALS = {
    "PROJECT_PROCUREMENT": "Call check_project_stock, then plan_project_procurement once for the whole project. "
                           "If it reports late or failed orders, ask_human what to do.",
    "RECON_POLICY": "Call get_reconciliation_policy. If unspecified, search_knowledge(source='contract') for the "
                    "billing frequency rule. If the contract does not fix it, ask_human with options. If you have "
                    "the KAM's answer, call set_reconciliation_policy with it.",
    "RECON_BATCH": "Call reconcile_batch for these work orders, then create_batch_invoice for the reconciled ones.",
    "PLANNED": "Make it STOCK_READY: check_site_stock, order_site_shortages if anything is short, then "
               "confirm_site_stock_ready.",
    "STOCK_READY": "Deliver it: call dispatch_site.",
    "DELIVERED": "Install it: call run_site_installation, then request_replacement for each fault that "
                 "needs a replacement.",
    "REPLACEMENT_PENDING": "Call run_site_installation to list outstanding faults, then request_replacement "
                           "for each one.",
    "RECON_DISCREPANCY": "Call investigate_discrepancy, then send_email to the contact it gives you.",
    "REPLY_RECEIVED": "Call read_replies, then record_site_return, book_late_quarantine or escalate_to_kam "
                      "according to exactly what the reply says. If the reply is vague, ask_human.",
}


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
    recon_policy: Literal["end_of_project", "every_n_sites", "on_request", "unspecified"] = Field(
        default="unspecified",
        description="Reconciliation / billing frequency ONLY if the request states it, otherwise 'unspecified'")
    recon_every_n: int = Field(default=0, description="N for every_n_sites, else 0")


class RouteDecision(BaseModel):
    next_agent: Literal["inventory", "logistics", "field", "finance", "human_review", "FINISH"]
    work_order_ids: list[str] = Field(description="Work orders (or task ids) the chosen agent should handle now")
    reason: str = Field(default="", description="One short sentence")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sites(project_name: str) -> list:
    return get_project_work_orders(project_name)


def _status(work_order_id: str) -> str:
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    row = conn.execute("SELECT status FROM work_orders WHERE work_order_id = ?", (work_order_id,)).fetchone()
    conn.close()
    return row[0] if row else "UNKNOWN"


def _previously_reconciled(work_order_id: str) -> bool:
    """A site that already went through reconciliation once (e.g. fixed after an investigation)."""
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    n = conn.execute("SELECT COUNT(*) FROM reconciliation_ledger WHERE work_order_id = ? "
                     "AND reconciliation_status != 'PENDING'", (work_order_id,)).fetchone()[0]
    conn.close()
    return n > 0


def _due_for_reconciliation(project_name: str, sites: list) -> tuple:
    """Which installed sites should be reconciled now, per the policy or a human request."""
    policy = hitl.get_policy(project_name)
    waiting = [s["work_order_id"] for s in sites if s["status"] in ("INSTALLED", "RECONCILED")]
    if not waiting:
        return [], [], policy
    rechecks = [w for w in waiting if _previously_reconciled(w)]          # always reconcile again straight away
    fresh = [w for w in waiting if w not in rechecks]
    requests = hitl.open_recon_requests(project_name)
    requested = set()
    for r in requests:
        requested |= set(r["scope"] or fresh)
    still_installing = any(s["status"] in BEFORE_INSTALL for s in sites)

    due = set(rechecks) | (requested & set(fresh))
    if policy["policy"] == "end_of_project" and not still_installing:
        due |= set(fresh)
    elif policy["policy"] == "every_n_sites":
        n = max(1, policy["every_n"])
        if len(fresh) >= n:
            due |= set(sorted(fresh)[:n])            # exactly N sites per batch
        elif not still_installing:
            due |= set(fresh)                        # final partial batch
    return sorted(due), [r["request_id"] for r in requests], policy


def _work_options(project_name: str, sites: list) -> tuple:
    """Who can act right now, and on what."""
    options = {}
    for s in sites:
        owner = STAGE_OWNER.get(s["status"])
        if owner:
            options.setdefault(owner, []).append(s["work_order_id"])

    due, request_ids, policy = _due_for_reconciliation(project_name, sites)
    installed = [s for s in sites if s["status"] in ("INSTALLED", "RECONCILED")]
    if policy["policy"] == "unspecified":
        # Finance settles the policy early, in parallel with procurement, so the KAM is asked once up front
        if not hitl.has_open_project_question(project_name, DISPLAY_NAMES["finance"]):
            options.setdefault("finance", []).append(POLICY_TASK)
    elif due:
        options.setdefault("finance", []).extend(due)

    if hitl.open_questions(project_name) or hitl.pending_batches(project_name):
        options.setdefault("human_review", [])
    # On request policy, nothing else left to do: ask the KAM whether to reconcile now
    if not options and installed and policy["policy"] == "on_request":
        options["human_review"] = []
    return options, request_ids


def _site_table(sites: list) -> str:
    return "\n".join(f"{s['work_order_id']} | {s['site_category']} | {s['scheduled_date']} | {s['status']}"
                     for s in sites)


def _fallback(options: dict) -> RouteDecision:
    """Deterministic safety net when the LLM returns an invalid route."""
    agent = next(a for a in PIPELINE_ORDER if a in options)
    return RouteDecision(next_agent=agent, work_order_ids=options[agent],
                         reason="fallback rule: earliest pipeline stage with work waiting")


def _route(decision: RouteDecision, attempts: dict, sites: list, steps: int) -> dict:
    status_of = {s["work_order_id"]: s["status"] for s in sites}
    for wo in decision.work_order_ids:
        key = f"{wo}:{status_of.get(wo, 'TASK')}"
        attempts[key] = attempts.get(key, 0) + 1
    log_event("Main Orchestrator", f"ROUTE -> {decision.next_agent}",
              {"work_orders": decision.work_order_ids, "reason": decision.reason})
    return {"next_agent": decision.next_agent, "steps": steps, "attempts": attempts,
            "task": {"work_order_ids": decision.work_order_ids},
            "events": [{"agent": "Main Orchestrator", "action": f"ROUTE -> {decision.next_agent}",
                        "detail": f"{decision.work_order_ids}: {decision.reason}"}]}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: ProjectState) -> dict:
    """Main Orchestrator, planning role: turns the KAM's free text request into a scheduled rollout."""
    req = get_structured_llm(PlanRequest).invoke([
        ("system", "Extract the rollout request. Use 0 for any site type not mentioned. Dates as YYYY-MM-DD. "
                   "Only set recon_policy if the request explicitly says when to reconcile or bill."),
        ("user", state["kam_request"]),
    ])
    plan = plan_multi_site_rollout(state["project_name"], req.full_sites, req.upgrades, req.partial_sites,
                                   req.start_date)
    if req.recon_policy != "unspecified":
        hitl.set_policy(state["project_name"], req.recon_policy, req.recon_every_n, "KAM request")
    log_event("Main Orchestrator", "PROJECT_PLANNED",
              {"request": req.model_dump(), "sites": plan["total_sites_scheduled"],
               "dates": [plan["start_date"], plan["end_date"]]})
    return {"plan": {k: plan[k] for k in ("status", "total_sites_scheduled", "start_date", "end_date", "work_orders")},
            "steps": 0, "attempts": {},
            "events": [{"agent": "Main Orchestrator", "action": "PROJECT_PLANNED",
                        "detail": f"{plan['total_sites_scheduled']} sites, recon policy {req.recon_policy}"}]}


def supervisor_node(state: ProjectState) -> dict:
    """Main Orchestrator, coordination role: decides who works next, on which sites."""
    project = state["project_name"]
    steps = state.get("steps", 0) + 1
    attempts = dict(state.get("attempts", {}))

    # Sites stuck at the same stage too many times are handed to a human
    for s in _sites(project):
        if attempts.get(f"{s['work_order_id']}:{s['status']}", 0) >= MAX_ATTEMPTS and s["status"] not in TERMINAL:
            set_work_order_status(s["work_order_id"], "NEEDS_ATTENTION")
            log_event("Main Orchestrator", "STUCK_ESCALATED", s["status"], s["work_order_id"])
    # Safety net: if Finance could not settle a policy after several tries, use the contract default
    if attempts.get(f"{POLICY_TASK}:TASK", 0) >= MAX_ATTEMPTS and hitl.get_policy(project)["policy"] == "unspecified":
        hitl.set_policy(project, "end_of_project", 0, "default after no decision")
        log_event("Main Orchestrator", "RECON_POLICY_DEFAULTED", "end_of_project")
    sites = _sites(project)
    options, request_ids = _work_options(project, sites)

    if not options or steps > MAX_STEPS:
        return {"next_agent": "FINISH", "steps": steps}

    # Requested reconciliations are being scheduled now
    if request_ids and "finance" in options:
        hitl.close_recon_requests(request_ids)

    if len(options) == 1:  # only one agent can act: no decision to make, save an LLM call
        agent = next(iter(options))
        return _route(RouteDecision(next_agent=agent, work_order_ids=options[agent],
                                    reason="only one agent has work available"), attempts, sites, steps)

    try:
        decision = get_structured_llm(RouteDecision).invoke([
            ("system",
             "You are the Main Orchestrator of a telecom rollout. Choose ONE agent to act next and the work "
             "orders (or task ids) it should handle, taken ONLY from that agent's list. Prefer the earliest "
             "scheduled sites and batch work for the same agent. Choose human_review when invoices wait for "
             "sign off, questions wait for the KAM, or the only remaining work is waiting for replies."),
            ("user", f"Project: {project}\nSites:\n{_site_table(sites)}\n\nAgents with work available: {options}"),
        ])
        valid = [w for w in decision.work_order_ids if w in options.get(decision.next_agent, [])]
        if decision.next_agent == "FINISH" or decision.next_agent not in options:
            decision = _fallback(options)
        elif decision.next_agent != "human_review":
            decision.work_order_ids = valid or options[decision.next_agent]
    except Exception as e:  # malformed output, timeout, Ollama hiccup
        log_event("Main Orchestrator", "ROUTING_ERROR", str(e)[:300])
        decision = _fallback(options)
    return _route(decision, attempts, sites, steps)


def _run_task(name: str, task_key: str, subject: str, project: str, done_check, answers: list) -> dict:
    """Run one agent task with stage scoped tools, the 'done' checklist, KAM answers and nudges."""
    goal = STAGE_GOALS.get(task_key, "move it to its next stage.")
    answer_text = "".join(f"\nEarlier you asked the KAM: {a['question']}\nThe KAM answered: {a['answer']}\n"
                          f"Act on this answer now." for a in answers)
    messages = [HumanMessage(f"{subject}\nProject: {project}\nTask: {task_key}{answer_text}\nTo finish you MUST: {goal}")]
    agent = get_stage_agent(name, task_key)
    tools_used, reply = [], ""
    for attempt in range(MAX_NUDGES + 1):
        try:
            result = agent.invoke({"messages": messages}, config={"recursion_limit": 40})
        except Exception as e:  # timeout, malformed tool call, Ollama error: log and move on
            log_event(DISPLAY_NAMES[name], "AGENT_ERROR", str(e)[:300])
            break
        messages = result["messages"]
        tools_used = [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
        reply = messages[-1].content
        if done_check():
            break
        if attempt < MAX_NUDGES:
            log_event(DISPLAY_NAMES[name], "NUDGED", f"{task_key} not finished after attempt {attempt + 1}")
            messages = messages + [HumanMessage(
                f"The task is not finished. Do not explain, call the next tool now: {goal}")]
    return {"tools": tools_used, "reply": reply[:400]}


def make_worker_node(name: str):
    def node(state: ProjectState) -> dict:
        project = state["project_name"]
        wos = state["task"]["work_order_ids"]
        events = []

        def record(task, subject_ids, result, change):
            log_event(DISPLAY_NAMES[name], "TASK_DONE", {**result, "task": task, "stage": change},
                      subject_ids if isinstance(subject_ids, str) else None)
            events.append({"agent": DISPLAY_NAMES[name], "action": "TASK_DONE",
                           "detail": f"{task}: {change} | tools={result['tools']}"})

        # ---- Finance, project level: work out the reconciliation policy ----
        if name == "finance" and POLICY_TASK in wos:
            answers = hitl.take_answers(DISPLAY_NAMES[name], project)
            done = lambda: hitl.get_policy(project)["policy"] != "unspecified" or \
                hitl.has_open_project_question(project, DISPLAY_NAMES[name])
            res = _run_task(name, "RECON_POLICY", "Determine the reconciliation policy for this project.",
                            project, done, answers)
            record("RECON_POLICY", None, res, hitl.get_policy(project)["policy"])
            wos = [w for w in wos if w != POLICY_TASK]
            if not wos:
                return {"events": events}

        # ---- Finance, batch reconciliation + one consolidated invoice ----
        if name == "finance":
            done = lambda: all(_status(w) not in ("INSTALLED", "RECONCILED") for w in wos)
            res = _run_task(name, "RECON_BATCH", f"Reconciliation batch: {wos}", project, done, [])
            record("RECON_BATCH", None, res, {w: _status(w) for w in wos})
            return {"events": events}

        # ---- Inventory, project level procurement when several sites are PLANNED ----
        planned = [w for w in wos if _status(w) == "PLANNED"]
        if name == "inventory" and len(planned) > 1:
            done = lambda: any(_status(w) != "PLANNED" for w in planned) or \
                hitl.has_open_project_question(project, DISPLAY_NAMES[name])
            answers = hitl.take_answers(DISPLAY_NAMES[name], project)
            res = _run_task(name, "PROJECT_PROCUREMENT", f"Planned sites: {planned}", project, done, answers)
            record("PROJECT_PROCUREMENT", None, res, {w: _status(w) for w in planned})
            return {"events": events}   # any site still PLANNED is routed again (then handled per site)

        # ---- Site level tasks, one work order per prompt ----
        for wo in wos:
            start = _status(wo)
            answers = hitl.take_answers(DISPLAY_NAMES[name], project, wo)
            res = _run_task(name, start, f"Work order: {wo}\nCurrent stage: {start}", project,
                            lambda: _status(wo) != start, answers)
            record(start, wo, res, f"{start} -> {_status(wo)}")
        return {"events": events}
    return node


def human_review_node(state: ProjectState) -> dict:
    """Pauses the whole graph until a human answers. State is checkpointed meanwhile."""
    project = state["project_name"]
    sites = _sites(project)
    batches = hitl.pending_batches(project)
    questions = hitl.open_questions(project)
    escalations = [{"work_order_id": s["work_order_id"], "status": s["status"]}
                   for s in sites if s["status"] == "NEEDS_ATTENTION"]
    waiting = awaiting_replies(project)
    policy = hitl.get_policy(project)
    installed = [s["work_order_id"] for s in sites if s["status"] in ("INSTALLED", "RECONCILED")]
    recon_prompt = installed if policy["policy"] == "on_request" else []

    log_event("Main Orchestrator", "AWAITING_HUMAN",
              {"batches": [b["batch_id"] for b in batches], "questions": [q["question_id"] for q in questions],
               "escalations": escalations, "emails_awaiting_reply": [m["message_id"] for m in waiting]})

    # ---- graph pauses here; resumes with Command(resume=decisions) ----
    # Email replies, answers and reconciliation requests can also arrive directly in the database (UI / CLI).
    decisions = interrupt({"type": "human_review", "batches": batches, "questions": questions,
                           "escalations": escalations, "awaiting_replies": waiting,
                           "installed_awaiting_reconciliation": recon_prompt})

    approver = decisions.get("approver", "KAM")
    for batch_id, verdict in decisions.get("batches", {}).items():
        res = hitl.decide_batch(batch_id, approver, verdict == "approve")
        log_event("Human (KAM)", f"BATCH_{res['status']}", res)
    for qid, text in decisions.get("answers", {}).items():
        res = hitl.answer(int(qid), text)
        log_event("Human (KAM)", "ANSWERED", {"question_id": qid, "answer": text}, res.get("work_order_id"))
    for wo, note in decisions.get("escalations", {}).items():
        set_work_order_status(wo, "ESCALATED")
        log_event("Human (KAM)", "ESCALATION_NOTED", note, wo)
    if decisions.get("request_reconciliation"):
        hitl.request_reconciliation(project, approver)
        log_event("Human (KAM)", "RECONCILIATION_REQUESTED", None)
    return {"events": [{"agent": "Human (KAM)", "action": "REVIEW_DONE", "detail": str(decisions)[:300]}]}


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
            ("user", f"Final site status:\n{table}\nApproved site invoices: {approved[1]}, total £{approved[0]:,.2f}"),
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
    g = StateGraph(ProjectState)
    g.add_node("planner", planner_node)
    g.add_node("supervisor", supervisor_node)
    for name in ("inventory", "logistics", "field", "finance"):
        g.add_node(name, make_worker_node(name))
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
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 300}


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
    """Ask any specialist agent a question at any time, outside the project flow."""
    agent = build_workers()[agent_name]
    result = agent.invoke({"messages": [HumanMessage(question)]}, config={"recursion_limit": 30})
    log_event(DISPLAY_NAMES[agent_name], "ANSWERED_QUERY", {"question": question})
    return result["messages"][-1].content