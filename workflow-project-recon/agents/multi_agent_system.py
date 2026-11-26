from database.setup_databases import init_all_databases
from agents.base_agent import BaseAgent
from tools.orchestrator_tools import plan_multi_site_rollout
from tools.project_tools import (
    source_and_inbound_project_shortages,
    execute_project_dispatch_and_field,
    reconcile_and_invoice_project,
)


# ---------------------------------------------------------------------------
# Instantiate Specialist Agents
# ---------------------------------------------------------------------------

inventory_logistics_agent = BaseAgent(
    name="Inventory & Logistics Agent",
    system_prompt="You manage warehouse stock, IntelliCommsTech/3rd-party sourcing, customs clearance, and inbound verification.",
    tools=[{
        "type": "function",
        "function": {
            "name": "source_and_inbound_project_shortages",
            "description": "Check aggregated project BOM, order shortages, clear customs, and inbound into WMS.",
            "parameters": {
                "type": "object",
                "properties": {"project_name": {"type": "string"}},
                "required": ["project_name"],
            },
        },
    }],
    tool_map={"source_and_inbound_project_shortages": source_and_inbound_project_shortages},
)

field_agent = BaseAgent(
    name="Field Execution Agent",
    system_prompt="You coordinate site dispatch, engineer Field App confirmations, RAG fault triage, and faulty unit replacements.",
    tools=[{
        "type": "function",
        "function": {
            "name": "execute_project_dispatch_and_field",
            "description": "Dispatch materials, run Field App simulation, RAG fault triage, and replacements for a project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string"},
                    "field_mode": {"type": "string", "enum": ["demo", "random"]},
                },
                "required": ["project_name"],
            },
        },
    }],
    tool_map={"execute_project_dispatch_and_field": execute_project_dispatch_and_field},
)

finance_agent = BaseAgent(
    name="Finance & Reconciliation Agent",
    system_prompt="You perform 3-way reconciliation (Dispatched = Installed + Quarantined) and issue client invoices.",
    tools=[{
        "type": "function",
        "function": {
            "name": "reconcile_and_invoice_project",
            "description": "Reconcile and invoice all work orders for a project.",
            "parameters": {
                "type": "object",
                "properties": {"project_name": {"type": "string"}},
                "required": ["project_name"],
            },
        },
    }],
    tool_map={"reconcile_and_invoice_project": reconcile_and_invoice_project},
)


# ---------------------------------------------------------------------------
# Main Orchestrator (Supervisor) Agent
# ---------------------------------------------------------------------------

orchestrator_agent = BaseAgent(
    name="Main Orchestrator Agent",
    system_prompt=(
        "You are the Main Orchestrator for IntelliCommsTech. "
        "Execute these 4 tools ONCE each, in order, using the exact project_name from the user: "
        "1) plan_multi_site_rollout "
        "2) delegate_to_inventory_logistics "
        "3) delegate_to_field_execution "
        "4) delegate_to_finance "
        "After all 4 succeed, present a concise executive summary with the actual numbers returned."
    ),
    tools=[
        {
            "type": "function",
            "function": {
                "name": "plan_multi_site_rollout",
                "description": "Schedule a multi-site rollout across available Subcons and Engineers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string"},
                        "full_sites": {"type": "integer"},
                        "upgrades": {"type": "integer"},
                        "partial_sites": {"type": "integer"},
                        "start_date": {"type": "string"},
                    },
                    "required": ["project_name", "full_sites", "upgrades", "partial_sites"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delegate_to_inventory_logistics",
                "description": "Delegate stock checking, sourcing, and inbound clearance for a project.",
                "parameters": {
                    "type": "object",
                    "properties": {"project_name": {"type": "string"}},
                    "required": ["project_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delegate_to_field_execution",
                "description": "Delegate site dispatch, field app installs, and RAG fault triage for a project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string"},
                        "field_mode": {"type": "string", "enum": ["demo", "random"]},
                    },
                    "required": ["project_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delegate_to_finance",
                "description": "Delegate 3-way reconciliation and invoicing for a project.",
                "parameters": {
                    "type": "object",
                    "properties": {"project_name": {"type": "string"}},
                    "required": ["project_name"],
                },
            },
        },
    ],
    tool_map={
        "plan_multi_site_rollout": plan_multi_site_rollout,
        "delegate_to_inventory_logistics": lambda project_name: {
            "agent_reply": inventory_logistics_agent.run(
                f"Call source_and_inbound_project_shortages for project_name='{project_name}'."
            )
        },
        "delegate_to_field_execution": lambda project_name, field_mode="demo": {
            "agent_reply": field_agent.run(
                f"Call execute_project_dispatch_and_field for project_name='{project_name}' with field_mode='{field_mode}'."
            )
        },
        "delegate_to_finance": lambda project_name: {
            "agent_reply": finance_agent.run(
                f"Call reconcile_and_invoice_project for project_name='{project_name}'."
            )
        },
    },
    max_steps=6,
)


if __name__ == "__main__":
    init_all_databases(force_reset=True)
    # Test with 6 sites first (2 days across 5 teams)
    kam_prompt = (
        "Prepare and execute project 'UK-5G-Phase2': "
        "3 full sites, 2 upgrades, and 1 partial site starting 2026-12-01."
    )
    final_report = orchestrator_agent.run(kam_prompt)
    print("\n================ EXECUTIVE SUMMARY ================\n")
    print(final_report)