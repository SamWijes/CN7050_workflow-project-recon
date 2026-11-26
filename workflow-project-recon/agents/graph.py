"""
LangGraph orchestration layer.

    START -> planner -> supervisor <-> {inventory, logistics, field, finance, human_review, advance_calendar}
                                     -> finish -> END

The supervisor (Main Orchestrator) looks at the live state of every site and the business date, and decides
at runtime which agent works next. Work is gated by date: inbound batches are received on their ETA, kits are
collected the day before installation, installation happens on the install date. When nobody can act and
nothing waits for a human, the business calendar jumps to the next event.

Any agent can ask a human: that site (or task) pauses, the graph interrupts, and the same agent acts on the
answer when the graph resumes. Questions for humans are always raised before other work.
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
from agents.agent_tools import replacement_in_transit
from agents import hitl
from tools import clock
from tools.orchestrator_tools import ORCHESTRATOR_DB, plan_multi_site_rollout, set_work_order_status
from tools.project_tools import get_project_work_orders
from tools.mailbox_tools import awaiting_replies
from tools.logistics_tools import due_shipments, next_shipment_date

CHECKPOINT_DB = "databases/graph_checkpoints.db"
MAX_STEPS = 150         # hard stop for the supervisor loop (calendar jumps included)
MAX_ATTEMPTS = 3        # same task stuck in the same state this many times -> human
MAX_NUDGES = 2

# Project level tasks (pseudo work order ids)
T_POLICY, T_REJECTIONS = "PROJECT:RECON_POLICY", "PROJECT:REJECTIONS"
T_PROCUREMENT = "PROJECT:PROCUREMENT"
T_INBOUND, T_KITTING, T_COLLECTION = "PROJECT:INBOUND", "PROJECT:KITTING", "PROJECT:COLLECTION"
TASK_KEY = {T_POLICY: "RECON_POLICY", T_REJECTIONS: "REVIEW_REJECTION", T_PROCUREMENT: "PROJECT_PROCUREMENT",
            T_INBOUND: "INBOUND", T_KITTING: "KITTING", T_COLLECTION: "COLLECTION"}

HUMAN_STAGES = {"AWAITING_SIGNOFF", "AWAITING_REPLY", "AWAITING_HUMAN_ANSWER", "NEEDS_ATTENTION"}
TERMINAL = {"CLOSED", "ESCALATED", "INVOICE_CANCELLED"}
BEFORE_INSTALL = {"PLANNED", "ON_ORDER", "STOCK_READY", "KIT_READY", "COLLECTING", "COLLECTED", "INSTALLING",
                  "REPLACEMENT_PENDING"}
PIPELINE_ORDER = ["inventory", "logistics", "field", "finance", "human_review"]

# What "done" means for each task, spelled out for a small model
STAGE_GOALS = {
    "PROJECT_PROCUREMENT": "Call check_project_stock, then plan_project_procurement once. If it reports conflicts "
                           "(lines held because they cannot arrive in time), ask_human with options. If you have the "
                           "KAM's answer, call plan_project_procurement with strategy 'fastest' or 'accept_delay'.",
    "INBOUND": "Call receive_inbound to receive everything due at the warehouse today.",
    "KITTING": "Call prepare_kits to stage a kit for every STOCK_READY site.",
    "COLLECTION": "Call issue_kits to hand today's kits to the subcontractors collecting them.",
    "RECON_POLICY": "Call get_reconciliation_policy. If unspecified, search_knowledge(source='contract') for the "
                    "billing frequency rule. If the contract does not fix it, ask_human with options. If you have "
                    "the KAM's answer, call set_reconciliation_policy with it.",
    "RECON_BATCH": "Call reconcile_batch for these work orders, then create_batch_invoice for the reconciled ones.",
    "REVIEW_REJECTION": "Call get_rejected_batches, then explain_batch. Based on the KAM's reason, resubmit_batch with "
                        "an explanation, cancel_batch, or ask_human if the reason is unclear.",
    "COLLECTED": "Install it: call run_site_installation, then request_replacement for each fault that needs one.",
    "REPLACEMENT_PENDING": "Call run_site_installation to list outstanding faults, then request_replacement for each.",
    "RECON_DISCREPANCY": "Call investigate_discrepancy, then send_email to the contact it gives you.",
    "REPLY_RECEIVED": "Call read_replies, then act on exactly what the reply says, quoting it as evidence_quote: "
                      "record_site_return, report_installed_claim, book_late_quarantine or escalate_to_kam. If the "
                      "reply is vague, ask_human. If the KAM accepted an installed claim, call correct_installed_quantity.",
}


# ---------------------------------------------------------------------------
# State and structured outputs
# ---------------------------------------------------------------------------

class ProjectState(TypedDict, total=False):
    kam_request: str
    project_name: str
    kickoff_days: int
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
    rechecks = [w for w in waiting if _previously_reconciled(w)]      # fixed sites are reconciled again at once
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
            due |= set(sorted(fresh)[:n])
        elif not still_installing:
            due |= set(fresh)
    return sorted(due), [r["request_id"] for r in requests], policy


def _work_options(project_name: str, sites: list) -> tuple:
    """Who can act right now (given the business date), and on what."""
    today = clock.today()
    options = {}

    def add(agent, item):
        if item not in options.setdefault(agent, []):
            options[agent].append(item)

    for s in sites:
        st, wo = s["status"], s["work_order_id"]
        if st == "PLANNED":
            add("inventory", T_PROCUREMENT)
        elif st == "STOCK_READY":
            add("logistics", T_KITTING)
        elif st == "KIT_READY" and clock.collection_date(s["scheduled_date"]) <= today:
            add("logistics", T_COLLECTION)
        elif st == "COLLECTED" and s["scheduled_date"] <= today:
            add("field", wo)
        elif st == "REPLACEMENT_PENDING" and not replacement_in_transit(wo):
            add("field", wo)
        elif st in ("RECON_DISCREPANCY", "REPLY_RECEIVED"):
            add("field", wo)
        elif st in HUMAN_STAGES:
            options.setdefault("human_review", [])
    if due_shipments(today):
        add("logistics", T_INBOUND)

    due, request_ids, policy = _due_for_reconciliation(project_name, sites)
    if policy["policy"] == "unspecified":
        if not hitl.has_open_project_question(project_name, DISPLAY_NAMES["finance"]):
            add("finance", T_POLICY)
    elif due:
        for w in due:
            add("finance", w)
    if hitl.rejected_batches(project_name):
        add("finance", T_REJECTIONS)

    if hitl.open_questions(project_name) or hitl.pending_batches(project_name) or hitl.open_escalations(project_name):
        options.setdefault("human_review", [])
    installed = [s for s in sites if s["status"] in ("INSTALLED", "RECONCILED")]
    if not options and installed and policy["policy"] == "on_request":
        options["human_review"] = []
    return options, request_ids


def _next_event_date(project_name: str, sites: list):
    """The next date on which something happens: an inbound ETA, a kit collection or an installation."""
    today = clock.today()
    dates = [next_shipment_date(today)]
    for s in sites:
        if s["status"] in ("STOCK_READY", "KIT_READY"):
            dates.append(clock.collection_date(s["scheduled_date"]))
        if s["status"] in ("COLLECTED", "STOCK_READY", "KIT_READY"):
            dates.append(s["scheduled_date"])
    future = [d for d in dates if d and d > today]
    return min(future) if future else None


def _site_table(sites: list) -> str:
    return "\n".join(f"{s['work_order_id']} | {s['site_category']} | install {s['scheduled_date']} | {s['status']}"
                     for s in sites)


def _subjects(item: str, sites: list) -> str:
    """What a task is working on right now, so 'stuck' means the same task on the same things."""
    stage = {T_PROCUREMENT: ("PLANNED",), T_KITTING: ("STOCK_READY",), T_COLLECTION: ("KIT_READY",)}.get(item)
    if stage:
        return ",".join(s["work_order_id"] for s in sites if s["status"] in stage)
    if item == T_INBOUND:
        return ",".join(d["shipment_id"] for d in due_shipments(clock.today()))
    status_of = {s["work_order_id"]: s["status"] for s in sites}
    return status_of.get(item, "TASK")


def _fallback(options: dict) -> RouteDecision:
    agent = next(a for a in PIPELINE_ORDER if a in options)
    return RouteDecision(next_agent=agent, work_order_ids=options[agent],
                         reason="fallback rule: earliest pipeline stage with work waiting")


def _route(decision: RouteDecision, attempts: dict, sites: list, steps: int) -> dict:
    for item in decision.work_order_ids:
        key = f"{item}|{_subjects(item, sites)}"
        attempts[key] = attempts.get(key, 0) + 1
    log_event("Main Orchestrator", f"ROUTE -> {decision.next_agent}",
              {"work": decision.work_order_ids, "reason": decision.reason, "business_date": clock.today()})
    return {"next_agent": decision.next_agent, "steps": steps, "attempts": attempts,
            "task": {"work_order_ids": decision.work_order_ids},
            "events": [{"agent": "Main Orchestrator", "action": f"ROUTE -> {decision.next_agent}",
                        "detail": f"{decision.work_order_ids}: {decision.reason}"}]}


def _escalate_stuck(project: str, sites: list, attempts: dict):
    """A task tried MAX_ATTEMPTS times in a row on the same things without progress goes to the KAM."""
    # Only consecutive attempts count: once a site has moved on, its older counts are cleared
    status_of = {s["work_order_id"]: s["status"] for s in sites}
    for key in list(attempts):
        item, subj = key.split("|", 1)
        if item in status_of and status_of[item] != subj:
            del attempts[key]
    for key, n in list(attempts.items()):
        if n < MAX_ATTEMPTS:
            continue
        item, subj = key.split("|", 1)
        attempts[key] = -99   # do not escalate twice
        if item == T_POLICY and hitl.get_policy(project)["policy"] == "unspecified":
            hitl.set_policy(project, "end_of_project", 0, "default after no decision")
            log_event("Main Orchestrator", "RECON_POLICY_DEFAULTED", "end_of_project")
            continue
        targets = subj.split(",") if item in (T_PROCUREMENT, T_KITTING, T_COLLECTION) else [item]
        for wo in targets:
            if wo.startswith("WO-") and _status(wo) not in TERMINAL | HUMAN_STAGES:
                hitl.raise_escalation(project, wo, "Main Orchestrator",
                                      f"No progress after {MAX_ATTEMPTS} attempts at stage {_status(wo)}")
                log_event("Main Orchestrator", "STUCK_ESCALATED", item, wo)


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
    clock.set_today(clock.shift(req.start_date, -state.get("kickoff_days", 7)))   # project kick off date
    plan = plan_multi_site_rollout(state["project_name"], req.full_sites, req.upgrades, req.partial_sites,
                                   req.start_date)
    if req.recon_policy != "unspecified":
        hitl.set_policy(state["project_name"], req.recon_policy, req.recon_every_n, "KAM request")
    log_event("Main Orchestrator", "PROJECT_PLANNED",
              {"request": req.model_dump(), "sites": plan["total_sites_scheduled"],
               "dates": [plan["start_date"], plan["end_date"]], "business_date": clock.today()})
    return {"plan": {k: plan[k] for k in ("status", "total_sites_scheduled", "start_date", "end_date", "work_orders")},
            "steps": 0, "attempts": {},
            "events": [{"agent": "Main Orchestrator", "action": "PROJECT_PLANNED",
                        "detail": f"{plan['total_sites_scheduled']} sites, recon policy {req.recon_policy}"}]}


def supervisor_node(state: ProjectState) -> dict:
    """Main Orchestrator, coordination role: decides who works next, on what."""
    project = state["project_name"]
    steps = state.get("steps", 0) + 1
    attempts = dict(state.get("attempts", {}))
    _escalate_stuck(project, _sites(project), attempts)
    sites = _sites(project)
    options, request_ids = _work_options(project, sites)

    if steps > MAX_STEPS:
        return {"next_agent": "FINISH", "steps": steps, "attempts": attempts}
    if not options:
        nxt = _next_event_date(project, sites)
        return {"next_agent": "advance_calendar" if nxt else "FINISH", "steps": steps, "attempts": attempts}

    if request_ids and "finance" in options:
        hitl.close_recon_requests(request_ids)

    # 1) Anything only a human can answer is raised first: people are the slowest part of the system
    if hitl.open_questions(project):
        return _route(RouteDecision(next_agent="human_review", work_order_ids=[],
                                    reason="questions are waiting for the KAM"), attempts, sites, steps)
    if T_POLICY in options.get("finance", []):
        return _route(RouteDecision(next_agent="finance", work_order_ids=[T_POLICY],
                                    reason="settle the reconciliation policy first"), attempts, sites, steps)

    # 2) Only one agent can act: no decision to make, save an LLM call
    if len(options) == 1:
        agent = next(iter(options))
        return _route(RouteDecision(next_agent=agent, work_order_ids=options[agent],
                                    reason="only one agent has work available"), attempts, sites, steps)

    # 3) Real choice: the LLM decides
    try:
        decision = get_structured_llm(RouteDecision).invoke([
            ("system",
             "You are the Main Orchestrator of a telecom rollout. Choose ONE agent to act next and the work "
             "orders (or task ids) it should handle, taken ONLY from that agent's list. Prefer work for the earliest "
             "install dates, and batch work for the same agent. Choose human_review when invoices wait for sign "
             "off, escalations wait for the KAM, or the only remaining work is waiting for replies."),
            ("user", f"Project: {project}\nBusiness date: {clock.today()}\nSites:\n{_site_table(sites)}\n\n"
                     f"Agents with work available: {options}"),
        ])
        valid = [w for w in decision.work_order_ids if w in options.get(decision.next_agent, [])]
        if decision.next_agent == "FINISH" or decision.next_agent not in options:
            decision = _fallback(options)
        elif decision.next_agent != "human_review":
            decision.work_order_ids = valid or options[decision.next_agent]
    except Exception as e:
        log_event("Main Orchestrator", "ROUTING_ERROR", str(e)[:300])
        decision = _fallback(options)
    return _route(decision, attempts, sites, steps)


def advance_calendar_node(state: ProjectState) -> dict:
    """Nobody can act and nobody is waiting for a human: move the business date to the next event."""
    nxt = _next_event_date(state["project_name"], _sites(state["project_name"]))
    old = clock.today()
    clock.set_today(nxt)
    log_event("Business Calendar", f"DATE {old} -> {nxt}", None)
    return {"events": [{"agent": "Business Calendar", "action": "ADVANCE", "detail": f"{old} -> {nxt}"}]}


def _run_task(name: str, task_key: str, subject: str, project: str, done_check, answers: list) -> dict:
    """Run one agent task with stage scoped tools, the 'done' checklist, KAM answers and nudges."""
    goal = STAGE_GOALS.get(task_key, "move it to its next stage.")
    answer_text = "".join(f"\nEarlier you asked the KAM: {a['question']}\nThe KAM answered: {a['answer']}\n"
                          f"Act on this answer now." for a in answers)
    messages = [HumanMessage(f"{subject}\nProject: {project}\nBusiness date: {clock.today()}\nTask: {task_key}"
                             f"{answer_text}\nTo finish you MUST: {goal}")]
    agent = get_stage_agent(name, task_key)
    tools_used, reply = [], ""
    for attempt in range(MAX_NUDGES + 1):
        try:
            result = agent.invoke({"messages": messages}, config={"recursion_limit": 40})
        except Exception as e:
            log_event(DISPLAY_NAMES[name], "AGENT_ERROR", str(e)[:300])
            break
        messages = result["messages"]
        tools_used = [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
        reply = messages[-1].content
        if done_check():
            break
        if attempt < MAX_NUDGES:
            log_event(DISPLAY_NAMES[name], "NUDGED", f"{task_key} not finished after attempt {attempt + 1}")
            messages = messages + [HumanMessage(f"The task is not finished. Do not explain, call the next tool now: {goal}")]
    return {"tools": tools_used, "reply": reply[:400]}


def make_worker_node(name: str):
    def node(state: ProjectState) -> dict:
        project = state["project_name"]
        items = state["task"]["work_order_ids"]
        events = []
        me = DISPLAY_NAMES[name]

        def record(task, wo, result, change):
            log_event(me, "TASK_DONE", {**result, "task": task, "result": change}, wo)
            events.append({"agent": me, "action": "TASK_DONE", "detail": f"{task}: {change} | tools={result['tools']}"})

        def stage_count(statuses, dated=False):
            return [s for s in _sites(project) if s["status"] in statuses and
                    (not dated or clock.collection_date(s["scheduled_date"]) <= clock.today())]

        project_checks = {
            T_POLICY: lambda: hitl.get_policy(project)["policy"] != "unspecified" or hitl.has_open_project_question(project, me),
            T_REJECTIONS: lambda: not hitl.rejected_batches(project) or hitl.has_open_project_question(project, me),
            T_PROCUREMENT: lambda: not stage_count(("PLANNED",)) or hitl.has_open_project_question(project, me),
            T_INBOUND: lambda: not due_shipments(clock.today()),
            T_KITTING: lambda: not stage_count(("STOCK_READY",)),
            T_COLLECTION: lambda: not stage_count(("KIT_READY",), dated=True),
        }
        # ---- project level tasks ----
        for item in [i for i in items if i in TASK_KEY]:
            answers = hitl.take_answers(me, project)
            res = _run_task(name, TASK_KEY[item], f"Project task: {TASK_KEY[item]}", project, project_checks[item], answers)
            record(TASK_KEY[item], None, res, "done" if project_checks[item]() else "not finished")
        sites_in_task = [i for i in items if i not in TASK_KEY]

        # ---- Finance: batch reconciliation and one consolidated invoice ----
        if name == "finance" and sites_in_task:
            done = lambda: all(_status(w) not in ("INSTALLED", "RECONCILED") for w in sites_in_task)
            res = _run_task(name, "RECON_BATCH", f"Reconciliation batch: {sites_in_task}", project, done, [])
            record("RECON_BATCH", None, res, {w: _status(w) for w in sites_in_task})
            return {"events": events}

        # ---- Site level tasks, one work order per prompt ----
        for wo in sites_in_task:
            start = _status(wo)
            answers = hitl.take_answers(me, project, wo)
            res = _run_task(name, start, f"Work order: {wo}\nCurrent stage: {start}", project,
                            lambda: _status(wo) != start, answers)
            record(start, wo, res, f"{start} -> {_status(wo)}")
        return {"events": events}
    return node


def human_review_node(state: ProjectState) -> dict:
    """Pauses the whole graph until a human answers. State is checkpointed meanwhile."""
    project = state["project_name"]
    sites = _sites(project)
    policy = hitl.get_policy(project)
    installed = [s["work_order_id"] for s in sites if s["status"] in ("INSTALLED", "RECONCILED")]
    payload = {"type": "human_review", "business_date": clock.today(),
               "questions": hitl.open_questions(project), "batches": hitl.pending_batches(project),
               "escalations": hitl.open_escalations(project), "awaiting_replies": awaiting_replies(project),
               "installed_awaiting_reconciliation": installed if policy["policy"] == "on_request" else []}
    log_event("Main Orchestrator", "AWAITING_HUMAN",
              {k: (len(v) if isinstance(v, list) else v) for k, v in payload.items()})

    # ---- graph pauses here; resumes with Command(resume=decisions) ----
    decisions = interrupt(payload)

    approver = decisions.get("approver", "KAM")
    for qid, text in decisions.get("answers", {}).items():
        res = hitl.answer(int(qid), text)
        log_event("Human (KAM)", "ANSWERED", {"question_id": qid, "answer": text}, res.get("work_order_id"))
    for batch_id, verdict in decisions.get("batches", {}).items():
        v = verdict if isinstance(verdict, dict) else {"verdict": verdict}
        res = hitl.decide_batch(batch_id, approver, v["verdict"] == "approve", v.get("reason", ""))
        log_event("Human (KAM)", f"BATCH_{res['status']}", {**res, "reason": v.get("reason", "")})
    for wo, decision in decisions.get("escalations", {}).items():
        d = decision.lower()
        if "write" in d:
            done = hitl.write_off(wo)
            log_event("Human (KAM)", "WRITE_OFF_APPROVED", done, wo)
        elif "chase" in d or "investigate" in d:
            set_work_order_status(wo, "RECON_DISCREPANCY")
            hitl.note_for_agent(project, DISPLAY_NAMES["field"], wo, decision)
            log_event("Human (KAM)", "CHASE_AGAIN", decision, wo)
        else:
            set_work_order_status(wo, "ESCALATED")
            log_event("Human (KAM)", "SITE_CLOSED_UNBILLED", decision, wo)
        hitl.resolve_escalation(wo, decision)
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
            ("user", f"Business date: {clock.today()}\nFinal site status:\n{table}\n"
                     f"Approved site invoices: {approved[1]}, total £{approved[0]:,.2f}"),
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
    g.add_node("advance_calendar", advance_calendar_node)
    g.add_node("finish", finish_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor,
                            ["inventory", "logistics", "field", "finance", "human_review", "advance_calendar", "finish"])
    g.add_edge("human_review", "supervisor")
    g.add_edge("advance_calendar", "supervisor")
    g.add_edge("finish", END)

    if checkpointer is None:
        checkpointer = SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))
    return g.compile(checkpointer=checkpointer)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 600}


def start_project(graph, kam_request: str, project_name: str, thread_id: str, kickoff_days: int = 7) -> dict:
    """Runs until the first human interrupt (or the end). Returns the graph result."""
    return graph.invoke({"kam_request": kam_request, "project_name": project_name, "kickoff_days": kickoff_days},
                        _config(thread_id))


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