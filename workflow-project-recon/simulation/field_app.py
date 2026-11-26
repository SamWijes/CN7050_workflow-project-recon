"""Simulated Field App: the normal report from tools.field_tools, plus any scenario injection."""
from tools.field_tools import simulate_field_app_report, FAULT_NOTES
from simulation.scenarios import SCENARIOS, record_truth


def simulate_site_report(wo: dict, dispatched_items: list, field_mode: str, scenario: str) -> dict:
    report = simulate_field_app_report(wo["work_order_id"], wo["site_id"], wo["engineer_id"],
                                       dispatched_items, mode=field_mode)
    cfg = SCENARIOS.get(scenario, {})

    if scenario.startswith("qty_mismatch"):
        for item in report["installed_items"]:
            if item["sku"] == cfg["bulk_sku"] and item["qty"] > 1:
                missing = max(1, round(item["qty"] * cfg["missing_pct"]))
                item["qty"] -= missing                      # engineer reports fewer installed
                record_truth(wo["work_order_id"], scenario,
                             {"sku": item["sku"], "missing": missing, "holder": wo["subcon_id"], "reply": cfg["reply"]})
                break

    if scenario.startswith("faulty_"):
        for i, item in enumerate(report["installed_items"]):
            if item["sku"] == cfg["fault_sku"] and item.get("serial_number"):
                sn = item["serial_number"]
                report["installed_items"].pop(i)
                report["faulty_items"].append({
                    "sku": item["sku"], "serial_number": sn, "qty": 1,
                    "fault_log": f"{wo['site_id']}: " + FAULT_NOTES[item["sku"]].format(sn=sn),
                    "unit_returned": False,                 # engineer keeps the faulty unit
                })
                record_truth(wo["work_order_id"], scenario,
                             {"sku": item["sku"], "serial": sn, "holder": wo["engineer_id"], "reply": cfg["reply"]})
                break
    return report