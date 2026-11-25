import json
import sqlite3
import uuid
from datetime import datetime
from tools.inventory_tools import find_available_bay, check_bom_stock, query_suppliers, create_purchase_order, verify_and_inbound_packing_list

SUPPLIERS_DB = "databases/suppliers.db"
LOGISTICS_DB = "databases/logistics_transport.db"
WMS_DB = "databases/wms_inventory.db"


def schedule_inbound_shipment(po_id: str, pallets_required: int, inbound_date: str, serialized: bool = True) -> dict:
    """
    Coordinates customs clearance, reserves a WH bay, generates the supplier packing list
    (with serial codes for electronic items), and logs the shipment in logistics_transport.db.
    """
    sup_conn = sqlite3.connect(SUPPLIERS_DB)
    po = sup_conn.execute(
        "SELECT supplier_name, sku, qty_ordered FROM purchase_orders WHERE po_id = ?", (po_id,)
    ).fetchone()
    sup_conn.close()

    if not po:
        return {"status": "FAILED", "reason": f"PO {po_id} not found."}

    supplier_name, sku, qty_ordered = po

    # Coordinate with Inventory for WH bay spacing
    zone = "Active Hardware Zone" if serialized else "Bulk & Civil Zone"
    bay = find_available_bay(pallets_required, zone=zone)
    if not bay["available"]:
        return {"status": "DELAYED", "reason": bay["reason"]}

    # Look up item description
    wms_conn = sqlite3.connect(WMS_DB)
    desc_row = wms_conn.execute(
        "SELECT item_description FROM wms_ledger WHERE sku = ? LIMIT 1", (sku,)
    ).fetchone()
    wms_conn.close()
    item_desc = desc_row[0] if desc_row else sku

    # Generate packing list (1 row per serial for electronics, 1 bulk row otherwise)
    packing_list = []
    if serialized:
        for i in range(qty_ordered):
            sn = f"SN-{sku.split('-')[1]}-{uuid.uuid4().hex[:4].upper()}"
            packing_list.append({
                "sku": sku,
                "item_description": item_desc,
                "serial_number": sn,
                "qty": 1,
            })
    else:
        packing_list.append({
            "sku": sku,
            "item_description": item_desc,
            "serial_number": None,
            "qty": qty_ordered,
        })

    shipment_id = f"SHP-{uuid.uuid4().hex[:6].upper()}"
    conn = sqlite3.connect(LOGISTICS_DB)
    conn.execute("""
        INSERT INTO inbound_shipments
        (shipment_id, po_id, supplier_name, clearance_status, inbound_date, assigned_bay, pallets_required, packing_list_json, status)
        VALUES (?, ?, ?, 'CLEARED', ?, ?, ?, ?, 'IN_TRANSIT')
    """, (shipment_id, po_id, supplier_name, inbound_date, bay["bay_id"], pallets_required, json.dumps(packing_list)))
    conn.commit()
    conn.close()

    return {
        "status": "CLEARED_AND_SCHEDULED",
        "shipment_id": shipment_id,
        "po_id": po_id,
        "supplier_name": supplier_name,
        "inbound_date": inbound_date,
        "assigned_bay": bay["bay_id"],
        "packing_list": packing_list,
    }


def create_outbound_waybill(work_order_id: str, site_id: str, dispatched_items: list, scheduled_date: str, carrier: str = "IntelliFleet Logistics") -> dict:
    """Creates an outbound delivery waybill for site dispatch."""
    waybill_id = f"WB-{site_id}-{uuid.uuid4().hex[:4].upper()}"
    conn = sqlite3.connect(LOGISTICS_DB)
    conn.execute("""
        INSERT INTO outbound_deliveries
        (waybill_id, work_order_id, site_id, carrier_name, scheduled_delivery_date, delivery_status, dispatched_items_json)
        VALUES (?, ?, ?, ?, ?, 'IN_TRANSIT', ?)
    """, (waybill_id, work_order_id, site_id, carrier, scheduled_date, json.dumps(dispatched_items)))
    conn.commit()
    conn.close()

    return {
        "status": "WAYBILL_CREATED",
        "waybill_id": waybill_id,
        "site_id": site_id,
        "carrier": carrier,
        "scheduled_delivery_date": scheduled_date,
    }


if __name__ == "__main__":
    # End-to-end test: Full Site BOM shortage -> PO -> Logistics Clearance -> Packing List Inbound
    with open("mock_apis/bom_templates.json") as f:
        full_site_bom = json.load(f)["full_site"]

    print("1. Checking WMS stock for full_site...")
    stock_res = check_bom_stock(full_site_bom)
    print(json.dumps(stock_res["shortages"], indent=2))

    for shortage in stock_res["shortages"]:
        print(f"\n2. Querying suppliers for {shortage['sku']} (short {shortage['shortage_qty']})...")
        suppliers = query_suppliers(shortage["sku"], shortage["shortage_qty"])
        chosen = suppliers[0]  # Fastest lead time
        print(f"   Selected: {chosen['supplier_name']} ({chosen['supplier_type']}) - £{chosen['unit_price_gbp']}")

        po = create_purchase_order(chosen["supplier_id"], shortage["sku"], shortage["shortage_qty"])
        print("3. Purchase Order Created:", po["po_id"])

        shipment = schedule_inbound_shipment(po["po_id"], pallets_required=1, inbound_date="2026-11-30", serialized=shortage["serialized"])
        print("4. Logistics Cleared & Scheduled:", shipment["shipment_id"], "-> Bay:", shipment["assigned_bay"])

        inbound = verify_and_inbound_packing_list(shipment["shipment_id"], shipment["packing_list"])
        print("5. Inventory Cross-Check & Inbound:", json.dumps(inbound, indent=2))