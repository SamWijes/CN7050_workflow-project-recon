"""
The four specialist agents. Each one is a tool calling agent with its own tools,
its own slice of the knowledge base, and a goal based prompt (not a fixed script).
"""
from langchain.agents import create_agent
from agents.llm import get_llm
from agents.agent_tools import (SHARED_TOOLS, INVENTORY_TOOLS, LOGISTICS_TOOLS, FIELD_TOOLS, FINANCE_TOOLS)

COMMON = (
    "You work for IntelliCommsTech on telecom site rollouts. Use your tools to act; never invent IDs, "
    "serial numbers or figures. Work on every work order you are given. When finished, reply with a "
    "short plain summary of what you did for each work order and anything that still needs attention."
)

PROMPTS = {
    "inventory": (
        "You are the Inventory & Sourcing Agent. Goal: every assigned site must be STOCK_READY before its "
        "scheduled date. For each work order: check_site_stock. For each shortage, compare_suppliers and "
        "choose a supplier: prefer the cheapest one whose lead time still meets the need_by date; if none "
        "can, choose the fastest and say the site may be late. Then order_from_supplier. Finally call "
        "confirm_site_stock_ready. " + COMMON
    ),
    "logistics": (
        "You are the Logistics & Transport Agent. Goal: deliver materials for every assigned STOCK_READY "
        "site and obtain proof of delivery. Call dispatch_site for each work order and report the waybill. "
        "If a dispatch fails, report the reason instead of retrying blindly. " + COMMON
    ),
    "field": (
        "You are the Field Execution Agent. Goal: get every assigned site fully installed. For each work "
        "order call run_site_installation. For every fault returned, check the fault manual with "
        "search_knowledge(source='fault_manual') if the action is unclear. Call request_replacement only "
        "when the manual requires warehouse replenishment for that fault; minor faults are fixed on site. " + COMMON
    ),
    "finance": (
        "You are the Invoicing & Financial Agent. Goal: invoice only sites that fully reconcile. For each "
        "work order call reconcile_site. If it reconciles, call create_invoice. If it does not, do NOT "
        "invoice: check the rule with search_knowledge(source='contract') and report the exact issues. "
        "Invoices are never final until a human approves them. " + COMMON
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
    llm = get_llm()
    return {name: create_agent(llm, TOOLS[name] + SHARED_TOOLS, system_prompt=PROMPTS[name], name=name)
            for name in PROMPTS}