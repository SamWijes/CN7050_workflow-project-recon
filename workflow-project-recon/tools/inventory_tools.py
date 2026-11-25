import json
import sqlite3
import uuid
from datetime import datetime

WMS_DB = "databases/wms_inventory.db"
SUPPLIERS_DB = "databases/suppliers.db"
LOGISTICS_DB = "databases/logistics_transport.db"


def check_bom_stock(bom: list) -> dict:
    """Checks current warehouse stock against a BOM and returns shortages."""
    conn = sqlite3.connect(WMS_DB)
    cur = conn.cursor()

    stock_map = {
        row[0]: row[1]
        for row in cur.execute("SELECT sku, available_qty FROM current_stock")
    }
    conn.close()

    shortages = []
    available_summary = []

    for item in bom:
        sku = item["sku"]
        needed = item["qty"]
        on_hand = stock_map.get(sku, 0)
        available_summary.append({"sku": sku, "needed": needed, "on_hand": on_hand})
        if on_hand < needed:
            shortages.append({
                "sku": sku,
                "needed": needed,
                "on_hand": on_hand,
                "shortage_qty": needed - on_hand,
                "serialized": item.get("serialized", False),
            })

    return {
        "all_in_stock": len(shortages) == 0,
        "stock_check": available_summary,
        "shortages": shortages,
    }


def query_suppliers(sku: str, qty_needed: int) -> list:
    """Queries IntelliCommmsTech (INTERNAL) and 3RD_PARTY suppliers with enough stock."""
    conn = sqlite3.connect(SUPPLIERS_DB)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT supplier_id, supplier_name, supplier_type, stock_available, unit_price_gbp, lead_time_days
        FROM supplier_catalog
        WHERE sku = ? AND stock_available >= ?
        ORDER BY lead_time_days ASC, unit_price_gbp ASC
    """, (sku, qty_needed)).fetchall()
    conn.close()

    return [
        {
            "supplier_id": r[0],
            "supplier_name": r[1],
            "supplier_type": r[2],
            "stock_available": r[3],
            "unit_price_gbp": r[4],
            "lead_time_days": r[5],
        }
        for r in rows
    ]


def create_purchase_order(supplier_id: str, sku: str, qty: int) -> dict:
    """Creates a PO in suppliers.db and deducts ordered qty from supplier stock."""
    conn = sqlite3.connect(SUPPLIERS_DB)
    cur = conn.cursor()

    row = cur.execute("""
        SELECT supplier_name, supplier_type, stock_available, unit_price_gbp
        FROM supplier_catalog
        WHERE supplier_id = ? AND sku = ?
    """, (supplier_id, sku)).fetchone()

    if not row or row[2] < qty:
        conn.close()
        return {"status": "FAILED", "reason": f"Insufficient stock at {supplier_id} for {sku}."}

    supplier_name, supplier_type, _, unit_price = row
    prefix = "PO-INT" if supplier_type == "INTERNAL" else "PO-EXT"
    po_id = f"{prefix}-{uuid.uuid4().hex[:6].upper()}"
    total_cost = round(unit_price * qty, 2)
    today = datetime.now().strftime("%Y-%m-%d")

    cur.execute("""
        INSERT INTO purchase_orders
        (po_id, supplier_id, supplier_name, supplier_type, sku, qty_ordered, total_cost_gbp, status, order_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'ORDERED', ?)
    """, (po_id, supplier_id, supplier_name, supplier_type, sku, qty, total_cost, today))

    cur.execute("""
        UPDATE supplier_catalog
        SET stock_available = stock_available - ?
        WHERE supplier_id = ? AND sku = ?
    """, (qty, supplier_id, sku))

    conn.commit()
    conn.close()

    return {
        "status": "ORDERED",
        "po_id": po_id,
        "supplier_id": supplier_id,
        "supplier_name": supplier_name,
        "supplier_type": supplier_type,
        "sku": sku,
        "qty_ordered": qty,
        "total_cost_gbp": total_cost,
    }


def find_available_bay(pallets_needed: int, zone: str = "Active Hardware Zone") -> dict:
    """Finds a warehouse bay with enough free pallet capacity."""
    conn = sqlite3.connect(WMS_DB)
    cur = conn.cursor()
    row = cur.execute("""
        SELECT bay_id, (max_pallets - used_pallets) AS free_pallets
        FROM wh_bays
        WHERE zone = ? AND (max_pallets - used_pallets) >= ?
        ORDER BY free_pallets DESC LIMIT 1
    """, (zone, pallets_needed)).fetchone()
    conn.close()

    if not row:
        return {"available": False, "reason": f"No bay in '{zone}' can fit {pallets_needed} pallets."}
    return {"available": True, "bay_id": row[0], "free_pallets": row[1]}


def verify_and_inbound_packing_list(shipment_id: str, packing_list: list) -> dict:
    """
    Cross-checks Logistics packing list against the original PO.
    On match: writes +1 (serialized) or +qty (bulk) rows to wms_ledger and marks RECEIVED.
    On mismatch: flags DISCREPANCY without altering stock.
    """
    log_conn = sqlite3.connect(LOGISTICS_DB)
    log_cur = log_conn.cursor()
    ship = log_cur.execute("""
        SELECT po_id, supplier_name, assigned_bay, pallets_required, clearance_status
        FROM inbound_shipments WHERE shipment_id = ?
    """, (shipment_id,)).fetchone()

    if not ship:
        log_conn.close()
        return {"status": "FAILED", "reason": f"Shipment {shipment_id} not found."}

    po_id, supplier_name, assigned_bay, pallets_required, clearance_status = ship
    if clearance_status != "CLEARED":
        log_conn.close()
        return {"status": "BLOCKED", "reason": f"Shipment {shipment_id} is not cleared ({clearance_status})."}

    sup_conn = sqlite3.connect(SUPPLIERS_DB)
    po = sup_conn.execute(
        "SELECT sku, qty_ordered FROM purchase_orders WHERE po_id = ?", (po_id,)
    ).fetchone()

    if not po:
        log_conn.close()
        sup_conn.close()
        return {"status": "FAILED", "reason": f"PO {po_id} not found."}

    expected_sku, expected_qty = po
    actual_qty = sum(item["qty"] for item in packing_list if item["sku"] == expected_sku)

    # Cross-check quantities
    if actual_qty != expected_qty:
        log_cur.execute(
            "UPDATE inbound_shipments SET status = 'DISCREPANCY' WHERE shipment_id = ?",
            (shipment_id,),
        )
        log_conn.commit()
        log_conn.close()
        sup_conn.close()
        return {
            "status": "DISCREPANCY",
            "shipment_id": shipment_id,
            "po_id": po_id,
            "expected_qty": expected_qty,
            "actual_qty": actual_qty,
            "message": "Packing list quantity does not match PO. Supplier dispute raised.",
        }

    # Match confirmed -> Inbound into wms_ledger
    today = datetime.now().strftime("%Y-%m-%d")
    wms_conn = sqlite3.connect(WMS_DB)
    wms_cur = wms_conn.cursor()

    for item in packing_list:
        wms_cur.execute("""
            INSERT INTO wms_ledger
            (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
            VALUES (?, ?, ?, ?, ?, ?, 'INBOUND', ?)
        """, (
            item["sku"],
            item["item_description"],
            item.get("serial_number"),
            item["qty"],
            today,
            f"WH Inbound ({supplier_name})",
            po_id,
        ))

    wms_cur.execute(
        "UPDATE wh_bays SET used_pallets = used_pallets + ? WHERE bay_id = ?",
        (pallets_required, assigned_bay),
    )
    wms_conn.commit()
    wms_conn.close()

    # Update shipment & PO statuses
    log_cur.execute("UPDATE inbound_shipments SET status = 'RECEIVED' WHERE shipment_id = ?", (shipment_id,))
    log_conn.commit()
    log_conn.close()

    sup_conn.execute("UPDATE purchase_orders SET status = 'RECEIVED' WHERE po_id = ?", (po_id,))
    sup_conn.commit()
    sup_conn.close()

    return {
        "status": "RECEIVED_CONFIRMED",
        "shipment_id": shipment_id,
        "po_id": po_id,
        "bay_assigned": assigned_bay,
        "items_inbounded": len(packing_list),
        "total_qty": actual_qty,
    }


def dispatch_bom_from_wms(work_order_id: str, site_id: str, bom: list, collector: str) -> dict:
    """Logs -1 per serialized unit and -qty for bulk items in wms_ledger."""
    check = check_bom_stock(bom)
    if not check["all_in_stock"]:
        return {"status": "FAILED", "reason": "Insufficient stock.", "shortages": check["shortages"]}

    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(WMS_DB)
    cur = conn.cursor()
    dispatched = []

    for item in bom:
        sku = item["sku"]
        qty = item["qty"]
        desc = cur.execute(
            "SELECT item_description FROM wms_ledger WHERE sku = ? LIMIT 1", (sku,)
        ).fetchone()[0]

        if item.get("serialized", False):
            # Pick available serial numbers
            serials = [
                r[0] for r in cur.execute("""
                    SELECT serial_number FROM wms_ledger
                    WHERE sku = ? AND serial_number IS NOT NULL
                    GROUP BY serial_number
                    HAVING SUM(qty) > 0
                    LIMIT ?
                """, (sku, qty)).fetchall()
            ]
            for sn in serials:
                cur.execute("""
                    INSERT INTO wms_ledger
                    (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
                    VALUES (?, ?, ?, -1, ?, ?, 'OUTBOUND', ?)
                """, (sku, desc, sn, today, collector, site_id))
                dispatched.append({"sku": sku, "serial_number": sn, "qty": 1})
        else:
            cur.execute("""
                INSERT INTO wms_ledger
                (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
                VALUES (?, ?, NULL, ?, ?, ?, 'OUTBOUND', ?)
            """, (sku, desc, -qty, today, collector, site_id))
            dispatched.append({"sku": sku, "serial_number": None, "qty": qty})

    conn.commit()
    conn.close()
    return {"status": "DISPATCHED", "work_order_id": work_order_id, "site_id": site_id, "items": dispatched}