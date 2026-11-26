import json
import sqlite3
import uuid
from tools.inventory_tools import find_available_bay, get_sku_description

SUPPLIERS_DB = "databases/suppliers.db"
LOGISTICS_DB = "databases/logistics_transport.db"


def schedule_inbound_shipment(po_id: str, pallets_required: int, inbound_date: str, serialized: bool = True) -> dict:
    """Clears customs, reserves a WH bay, generates the supplier packing list and logs the shipment."""
    sup_conn = sqlite3.connect(SUPPLIERS_DB)
    po = sup_conn.execute("SELECT supplier_name, sku, qty_ordered FROM purchase_orders WHERE po_id = ?",
                          (po_id,)).fetchone()
    sup_conn.close()
    if not po:
        return {"status": "FAILED", "reason": f"PO {po_id} not found."}
    supplier_name, sku, qty_ordered = po

    zone = "Active Hardware Zone" if serialized else "Bulk & Civil Zone"
    bay = find_available_bay(pallets_required, zone=zone)
    if not bay["available"]:
        return {"status": "DELAYED", "reason": bay["reason"]}

    item_desc = get_sku_description(sku)
    prefix = sku.rsplit("-", 1)[0]
    if serialized:
        packing_list = [{"sku": sku, "item_description": item_desc,
                         "serial_number": f"SN-{prefix}-{uuid.uuid4().hex[:6].upper()}", "qty": 1}
                        for _ in range(qty_ordered)]
    else:
        packing_list = [{"sku": sku, "item_description": item_desc, "serial_number": None, "qty": qty_ordered}]

    shipment_id = f"SHP-{uuid.uuid4().hex[:6].upper()}"
    conn = sqlite3.connect(LOGISTICS_DB)
    conn.execute("""
        INSERT INTO inbound_shipments
        (shipment_id, po_id, supplier_name, clearance_status, inbound_date, assigned_bay,
         pallets_required, packing_list_json, status)
        VALUES (?, ?, ?, 'CLEARED', ?, ?, ?, ?, 'IN_TRANSIT')
    """, (shipment_id, po_id, supplier_name, inbound_date, bay["bay_id"], pallets_required, json.dumps(packing_list)))
    conn.commit()
    conn.close()
    return {"status": "CLEARED_AND_SCHEDULED", "shipment_id": shipment_id, "po_id": po_id,
            "supplier_name": supplier_name, "inbound_date": inbound_date,
            "assigned_bay": bay["bay_id"], "packing_list": packing_list}


def create_outbound_waybill(work_order_id: str, site_id: str, dispatched_items: list, scheduled_date: str,
                            carrier: str = "IntelliFleet Logistics") -> dict:
    waybill_id = f"WB-{site_id}-{uuid.uuid4().hex[:4].upper()}"
    conn = sqlite3.connect(LOGISTICS_DB)
    conn.execute("""
        INSERT INTO outbound_deliveries
        (waybill_id, work_order_id, site_id, carrier_name, scheduled_delivery_date, delivery_status, dispatched_items_json)
        VALUES (?, ?, ?, ?, ?, 'IN_TRANSIT', ?)
    """, (waybill_id, work_order_id, site_id, carrier, scheduled_date, json.dumps(dispatched_items)))
    conn.commit()
    conn.close()
    return {"status": "WAYBILL_CREATED", "waybill_id": waybill_id, "site_id": site_id,
            "carrier": carrier, "scheduled_delivery_date": scheduled_date}


def confirm_site_delivery(waybill_id: str, received_by: str, delivered_at: str) -> dict:
    """Proof of delivery: engineer signs for the consignment on site."""
    conn = sqlite3.connect(LOGISTICS_DB)
    cur = conn.execute("""
        UPDATE outbound_deliveries SET delivery_status = 'DELIVERED', received_by_engineer = ?, delivered_at = ?
        WHERE waybill_id = ?
    """, (received_by, delivered_at, waybill_id))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        return {"status": "FAILED", "reason": f"Waybill {waybill_id} not found."}
    return {"status": "DELIVERED", "waybill_id": waybill_id, "received_by": received_by, "delivered_at": delivered_at}


def get_delivered_quantities(work_order_id: str) -> dict:
    """Sum of delivered quantities per SKU from signed waybills (the carrier leg of 3 way reconciliation)."""
    conn = sqlite3.connect(LOGISTICS_DB)
    rows = conn.execute("""
        SELECT dispatched_items_json FROM outbound_deliveries
        WHERE work_order_id = ? AND delivery_status = 'DELIVERED'
    """, (work_order_id,)).fetchall()
    conn.close()
    totals = {}
    for (items_json,) in rows:
        for item in json.loads(items_json):
            totals[item["sku"]] = totals.get(item["sku"], 0) + item["qty"]
    return totals
