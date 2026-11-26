"""
Simulated engineers and subcontractors. They answer agent emails in free text, using the hidden
scenario truth. The agent has to read the text and work out what to do.
"""
import re
import sqlite3
from agents.llm import get_llm
from tools.mailbox_tools import deliver_reply, ORCHESTRATOR_DB
from tools.inventory_tools import get_sku_description
from simulation.scenarios import get_truth


def _facts_for(message: dict) -> tuple:
    """Returns (instructions for the persona, plain fallback reply) from the hidden scenario truth."""
    truths = get_truth(message["work_order_id"])
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    who = conn.execute("SELECT resource_id FROM resources WHERE email = ?", (message["recipient"],)).fetchone()
    conn.close()
    me = who[0] if who else message["recipient"]
    for t in truths:
        if t["holder"] != me:
            continue
        d = f"{get_sku_description(t['sku'])} ({t['sku']})"

        if t["reply"] == "vague_then_exact":
            conn = sqlite3.connect(ORCHESTRATOR_DB)
            earlier = conn.execute("SELECT COUNT(*) FROM mailbox WHERE work_order_id = ? AND direction = 'IN'",
                                   (message["work_order_id"],)).fetchone()[0]
            conn.close()
            if earlier == 0:   # first reply: no number at all
                return ("You have a few of the conduits left over and will drop them back at the warehouse. "
                        "Do NOT give a number.",
                        "Hi, yeah we've got a few of those left over, I'll drop them back at the warehouse.")
            t = dict(t, reply="returns_balance")
        if t["reply"] == "returns_balance":
            n = t["missing"]
            return (f"You still have {n} unused {d} from the site and will return all {n} to the warehouse today.",
                    f"Hi, I still have {n} unused {d} from the site. I will return all {n} to the warehouse today.")
        if t["reply"] == "wastage":
            back, waste = t["missing"] // 2, t["missing"] - t["missing"] // 2
            return (f"You only have {back} unused {d} to return; the other {waste} were offcuts wasted during "
                    f"installation. You are returning {back} today.",
                    f"Hi, I only have {back} unused {d} to return. The other {waste} were offcuts wasted during "
                    f"installation. I am returning {back} today.")
        if t["reply"] == "returned_late":
            return (f"The faulty {d} serial {t['serial']} was left in your van; you dropped it at the warehouse "
                    f"quarantine bay today.",
                    f"Hi, sorry, the faulty {d} serial {t['serial']} was left in my van. I dropped it at the "
                    f"warehouse quarantine bay today.")
        if t["reply"] == "lost":
            return (f"You cannot find the faulty {d} serial {t['serial']}; you searched the van and the site.",
                    f"Hi, I have searched the van and the site but the faulty {d} serial {t['serial']} "
                    f"cannot be located.")
    return ("As far as you know everything issued to you was installed as reported.",
            "Hi, as far as I know everything issued to me was installed as reported.")


def simulate_reply(message: dict, use_llm: bool = True) -> dict:
    facts, fallback = _facts_for(message)
    body = None
    if use_llm:
        try:
            body = get_llm(temperature=0.7).invoke([
                ("system", "You are writing an email reply as a busy telecom field worker. 2 to 4 sentences, "
                           "informal but polite. Include every number and serial number in the facts exactly. "
                           "Do not add a subject line."),
                ("user", f"Email you received:\n{message['body']}\n\nFacts for your reply: {facts}"),
            ]).content.strip()
        except Exception:
            body = None
    # Guard: if the model dropped the numbers, use the plain version so the scenario stays correct
        needed = re.findall(r"\d+|SN-[A-Z-]+-[0-9A-F]+", facts)
    vague = "Do NOT give a number" in facts
    leaked = "Do NOT" in body if body else False          # persona instruction copied into the email
    if not body or leaked or not all(tok in body for tok in needed) or (vague and re.search(r"\d", body)):
        body = fallback
    return deliver_reply(message["message_id"], body)