"""
The four specialist agents. Each is a tool calling agent with its own tools, its own slice of the
knowledge base, clear rules for when to ask a human, and a goal based prompt (not a fixed script).
"""
from langchain.agents import create_agent
from agents.llm import get_llm
from agents import agent_tools as T
from agents.agent_tools import (SHARED_TOOLS, INVENTORY_TOOLS, LOGISTICS_TOOLS, FIELD_TOOLS, FINANCE_TOOLS)

COMMON = (
    "You work for IntelliCommsTech on telecom site rollouts. Use your tools to act; never invent IDs, "
    "serial numbers or figures. Do the work by calling tools; do not describe what you will do next. "
    "Keep calling tools until the task reaches its next stage, then reply in one or two sentences. "
    "Use ask_human ONLY for the conflicts listed for your role, never for something a tool can tell you. "
    "If you are given an answer from the KAM, act on it exactly."
)

PROMPTS = {
    "inventory": (
        "You are the Inventory & Sourcing Agent. Goal: every site's material ordered to the WAREHOUSE in time, "
        "as dated inbound batches (imports IMP-xx, local purchases LP-xx), one batch per install date, using "
        "stock on hand and in transit first. Nothing is delivered to site. "
        "ASK THE HUMAN when: an order will arrive after the material is needed, or no supplier can supply the "
        "quantity; offer options such as 'pay for the fastest supplier', 'accept the delay', 'split the order'. " + COMMON
    ),
    "logistics": (
        "You are the Logistics & Warehouse Agent. Goal: receive inbound batches at the warehouse on their ETA "
        "(check packing lists, book stock in), prepare one kit per work order, and issue kits to the "
        "subcontractors when they come to collect (the day before installation), with a signed collection note. "
        "ASK THE HUMAN when: a packing list does not match its purchase order. " + COMMON
    ),
    "field": (
        "You are the Field Execution Agent. Goal: get every site fully installed and every issued item "
        "accounted for. Check the fault manual with search_knowledge(source='fault_manual') when a fault's "
        "action is unclear; call request_replacement only when the manual requires warehouse replenishment. "
        "For reconciliation investigations, email the person in 'contact' (bulk items: ask the subcontractor "
        "how many unused units they hold and to return them; serialized items: ask the engineer what happened, "
        "quoting part number, serial and outbound date). When reading replies, book only what the person "
        "actually wrote, and quote it as evidence_quote: record_site_return (bulk being returned, with the number), "
        "report_installed_claim (bulk the reply says was installed), book_late_quarantine (faulty unit handed back), "
        "or escalate_to_kam for anything lost or wasted. After the KAM accepts an installed claim, call "
        "correct_installed_quantity. ASK THE HUMAN when: a reply is vague or gives no exact number (e.g. 'a few'), "
        "a serialized unit's fate is unclear, or a fault does not match any entry in the fault manual. " + COMMON
    ),
    "finance": (
        "You are the Invoicing & Financial Agent. Goal: reconcile installed sites according to the project's "
        "reconciliation policy and invoice only sites that fully reconcile, in one consolidated invoice per "
        "batch. To find the policy: first get_reconciliation_policy; if it is unspecified, check the contract "
        "with search_knowledge(source='contract'); if the contract does not fix it, ASK THE HUMAN with options "
        "such as 'reconcile at the end of the project', 'reconcile every N sites', 'only when I request it', "
        "then store the answer with set_reconciliation_policy. When the KAM rejects an invoice, read the reason, "
        "check the lines with explain_batch and the contract with search_knowledge, then resubmit_batch with a clear "
        "explanation, cancel_batch, or ASK THE HUMAN if the reason is unclear. Also ASK THE HUMAN when an invoice "
        "line breaks the contract variance. Invoices are never final until a human approves them. " + COMMON
    ),
}

TOOLS = {
    "inventory": INVENTORY_TOOLS,
    "logistics": LOGISTICS_TOOLS,
    "field": FIELD_TOOLS,
    "finance": FINANCE_TOOLS,
}

DISPLAY_NAMES = {
    "inventory": "Inventory Agent",
    "logistics": "Logistics Agent",
    "field": "Field Agent",
    "finance": "Finance Agent",
}


def build_workers() -> dict:
    """Full toolsets, used when a human asks an agent a question directly (ask_agent)."""
    llm = get_llm()
    return {name: create_agent(llm, TOOLS[name] + SHARED_TOOLS + [T.make_ask_human(DISPLAY_NAMES[name])],
                               system_prompt=PROMPTS[name], name=name)
            for name in PROMPTS}


# ---------------------------------------------------------------------------
# Stage scoped toolsets: a small model gets only the tools that make sense for the current task
# (2 to 5 tools instead of 11+), plus ask_human and get_site_status.
# ---------------------------------------------------------------------------
STAGE_TOOLS = {
    # project level tasks
    "PROJECT_PROCUREMENT": [T.check_project_stock, T.plan_project_procurement],
    "INBOUND":             [T.check_inbound_due, T.receive_inbound],
    "KITTING":             [T.prepare_kits],
    "COLLECTION":          [T.issue_kits],
    "RECON_POLICY":        [T.get_reconciliation_policy, T.search_knowledge, T.set_reconciliation_policy],
    "RECON_BATCH":         [T.reconcile_batch, T.create_batch_invoice, T.search_knowledge],
    "REVIEW_REJECTION":    [T.get_rejected_batches, T.explain_batch, T.search_knowledge, T.resubmit_batch,
                            T.cancel_batch],
    # site level tasks
    "COLLECTED":           [T.run_site_installation, T.request_replacement, T.search_knowledge],
    "REPLACEMENT_PENDING": [T.run_site_installation, T.request_replacement, T.search_knowledge],
    "RECON_DISCREPANCY":   [T.investigate_discrepancy, T.send_email],
    "REPLY_RECEIVED":      [T.read_replies, T.record_site_return, T.report_installed_claim,
                            T.correct_installed_quantity, T.book_late_quarantine, T.escalate_to_kam, T.send_email],
}
_stage_agents = {}


def get_stage_agent(name: str, stage: str):
    """Agent for one role, restricted to the tools for this task (cached)."""
    key = (name, stage)
    if key not in _stage_agents:
        tools = STAGE_TOOLS.get(stage, TOOLS[name]) + [T.get_site_status, T.make_ask_human(DISPLAY_NAMES[name])]
        _stage_agents[key] = create_agent(get_llm(), tools, system_prompt=PROMPTS[name], name=f"{name}_{stage}")
    return _stage_agents[key]