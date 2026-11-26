import sqlite3
import uuid
from datetime import datetime

WMS_DB = "databases/wms_inventory.db"
SUPPLIERS_DB = "databases/suppliers.db"
LOGISTICS_DB = "databases/logistics_transport.db"


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_sku_description(sku: str) -> str:
    conn = sqlite3.connect(WMS_DB)
    row = conn.execute("SELECT item_description FROM sku_master WHERE sku = ?", (sku,)).fetchone()
    conn.close()
    return row[0] if row else sku


def check_bom_stock(bom: list) -> dict:
    """Checks warehouse stock against a BOM. Serialized items count only good, unquarantined serials."""
    conn = sqlite3.connect(WMS_DB)
    bulk = {sku: qty for sku, qty in conn.execute("SELECT sku, available_qty FROM current_stock")}
    serial_counts = {sku: n for sku, n in conn.execute(
        "SELECT sku, COUNT(*) FROM available_serials GROUP BY sku")}
    conn.close()

    shortages, summary = [], []
    for item in bom:
        sku, needed = item["sku"], item["qty"]
        on_hand = serial_counts.get(sku, 0) if item.get("serialized") else bulk.get(sku, 0)
        summary.append({"sku": sku, "needed": needed, "on_hand": on_hand})
        if on_hand < needed:
            shortages.append({"sku": sku, "needed": needed, "on_hand": on_hand,
                              "shortage_qty": needed - on_hand,
                              "serialized": item.get("serialized", False)})
    return {"all_in_stock": not shortages, "stock_check": summary, "shortages": shortages}


def query_suppliers(sku: str, qty_needed: int) -> list:
    """Returns suppliers that can fill the qty, fastest first, then cheapest."""
    conn = sqlite3.connect(SUPPLIERS_DB)
    rows = conn.execute("""
        SELECT supplier_id, supplier_name, supplier_type, stock_available, unit_price_gbp, lead_time_days
        FROM supplier_catalog
        WHERE sku = ? AND stock_available >= ?
        ORDER BY lead_time_days ASC, unit_price_gbp ASC
    """, (sku, qty_needed)).fetchall()
    conn.close()
    keys = ["supplier_id", "supplier_name", "supplier_type", "stock_available", "unit_price_gbp", "lead_time_days"]
    return [dict(zip(keys, r)) for r in rows]


def create_purchase_order(supplier_id: str, sku: str, qty: int) -> dict:
    """Creates a PO and deducts the ordered qty from supplier stock."""
    conn = sqlite3.connect(SUPPLIERS_DB)
    cur = conn.cursor()
    row = cur.execute("""
        SELECT supplier_name, supplier_type, stock_available, unit_price_gbp
        FROM supplier_catalog WHERE supplier_id = ? AND sku = ?
    """, (supplier_id, sku)).fetchone()
    if not row or row[2] < qty:
        conn.close()
        return {"status": "FAILED", "reason": f"Insufficient stock at {supplier_id} for {sku}."}

    supplier_name, supplier_type, _, unit_price = row
    po_id = f"{'PO-INT' if supplier_type == 'INTERNAL' else 'PO-EXT'}-{uuid.uuid4().hex[:6].upper()}"
    total_cost = round(unit_price * qty, 2)
    cur.execute("""
        INSERT INTO purchase_orders
        (po_id, supplier_id, supplier_name, supplier_type, sku, qty_ordered, total_cost_gbp, status, order_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'ORDERED', ?)
    """, (po_id, supplier_id, supplier_name, supplier_type, sku, qty, total_cost, _today()))
    cur.execute("UPDATE supplier_catalog SET stock_available = stock_available - ? WHERE supplier_id = ? AND sku = ?",
                (qty, supplier_id, sku))
    conn.commit()
    conn.close()
    return {"status": "ORDERED", "po_id": po_id, "supplier_id": supplier_id, "supplier_name": supplier_name,
            "supplier_type": supplier_type, "sku": sku, "qty_ordered": qty, "total_cost_gbp": total_cost}


def find_available_bay(pallets_needed: int, zone: str = "Active Hardware Zone") -> dict:
    conn = sqlite3.connect(WMS_DB)
    row = conn.execute("""
        SELECT bay_id, (max_pallets - used_pallets) AS free_pallets FROM wh_bays
        WHERE zone = ? AND (max_pallets - used_pallets) >= ?
        ORDER BY free_pallets DESC LIMIT 1
    """, (zone, pallets_needed)).fetchone()
    conn.close()
    if not row:
        return {"available": False, "reason": f"No bay in '{zone}' can fit {pallets_needed} pallets."}
    return {"available": True, "bay_id": row[0], "free_pallets": row[1]}


def verify_and_inbound_packing_list(shipment_id: str, packing_list: list, txn_date: str = None) -> dict:
    """Cross checks the packing list against the PO. Match: books stock in. Mismatch: DISCREPANCY."""
    txn_date = txn_date or _today()
    log_conn = sqlite3.connect(LOGISTICS_DB)
    ship = log_conn.execute("""
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
    po = sup_conn.execute("SELECT sku, qty_ordered FROM purchase_orders WHERE po_id = ?", (po_id,)).fetchone()
    if not po:
        log_conn.close()
        sup_conn.close()
        return {"status": "FAILED", "reason": f"PO {po_id} not found."}

    expected_sku, expected_qty = po
    actual_qty = sum(i["qty"] for i in packing_list if i["sku"] == expected_sku)
    if actual_qty != expected_qty:
        log_conn.execute("UPDATE inbound_shipments SET status = 'DISCREPANCY' WHERE shipment_id = ?", (shipment_id,))
        log_conn.commit()
        log_conn.close()
        sup_conn.close()
        return {"status": "DISCREPANCY", "shipment_id": shipment_id, "po_id": po_id,
                "expected_qty": expected_qty, "actual_qty": actual_qty,
                "message": "Packing list quantity does not match PO. Supplier dispute raised."}

    wms_conn = sqlite3.connect(WMS_DB)
    for item in packing_list:
        wms_conn.execute("""
            INSERT INTO wms_ledger
            (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
            VALUES (?, ?, ?, ?, ?, ?, 'INBOUND', ?)
        """, (item["sku"], item["item_description"], item.get("serial_number"), item["qty"],
              txn_date, f"WH Inbound ({supplier_name})", po_id))
    wms_conn.execute("UPDATE wh_bays SET used_pallets = used_pallets + ? WHERE bay_id = ?",
                     (pallets_required, assigned_bay))
    wms_conn.commit()
    wms_conn.close()

    log_conn.execute("UPDATE inbound_shipments SET status = 'RECEIVED' WHERE shipment_id = ?", (shipment_id,))
    log_conn.commit()
    log_conn.close()
    sup_conn.execute("UPDATE purchase_orders SET status = 'RECEIVED' WHERE po_id = ?", (po_id,))
    sup_conn.commit()
    sup_conn.close()

    return {"status": "RECEIVED_CONFIRMED", "shipment_id": shipment_id, "po_id": po_id,
            "bay_assigned": assigned_bay, "items_inbounded": len(packing_list), "total_qty": actual_qty}


def dispatch_bom_from_wms(work_order_id: str, site_id: str, bom: list, collector: str, txn_date: str = None) -> dict:
    """Books stock out: one row per serial for serialized items, one row per bulk line."""
    check = check_bom_stock(bom)
    if not check["all_in_stock"]:
        return {"status": "FAILED", "reason": "Insufficient stock.", "shortages": check["shortages"]}

    txn_date = txn_date or _today()
    conn = sqlite3.connect(WMS_DB)
    cur = conn.cursor()
    dispatched = []

    for item in bom:
        sku, qty = item["sku"], item["qty"]
        desc = get_sku_description(sku)
        if item.get("serialized"):
            # Only good serials: never pick a unit that has ever been quarantined
            serials = [r[0] for r in cur.execute(
                "SELECT serial_number FROM available_serials WHERE sku = ? ORDER BY serial_number LIMIT ?",
                (sku, qty))]
            if len(serials) < qty:
                conn.rollback()
                conn.close()
                return {"status": "FAILED", "reason": f"Only {len(serials)} good serials for {sku}, need {qty}."}
            for sn in serials:
                cur.execute("""
                    INSERT INTO wms_ledger
                    (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
                    VALUES (?, ?, ?, -1, ?, ?, 'OUTBOUND', ?)
                """, (sku, desc, sn, txn_date, collector, work_order_id))
                dispatched.append({"sku": sku, "serial_number": sn, "qty": 1})
        else:
            cur.execute("""
                INSERT INTO wms_ledger
                (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
                VALUES (?, ?, NULL, ?, ?, ?, 'OUTBOUND', ?)
            """, (sku, desc, -qty, txn_date, collector, work_order_id))
            dispatched.append({"sku": sku, "serial_number": None, "qty": qty})

    conn.commit()
    conn.close()
    return {"status": "DISPATCHED", "work_order_id": work_order_id, "site_id": site_id,
            "dispatch_date": txn_date, "items": dispatched}


def record_return(work_order_id: str, sku: str, qty: int, returned_by: str,
                  serial_number: str = None, txn_date: str = None) -> dict:
    """Books unused material returned from site back into stock (RETURN) and updates the ledger."""
    from tools.orchestrator_tools import record_ledger_quantity
    txn_date = txn_date or _today()
    conn = sqlite3.connect(WMS_DB)
    conn.execute("""
        INSERT INTO wms_ledger
        (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
        VALUES (?, ?, ?, ?, ?, ?, 'RETURN', ?)
    """, (sku, get_sku_description(sku), serial_number, qty, txn_date, returned_by, work_order_id))
    conn.commit()
    conn.close()
    record_ledger_quantity(work_order_id, [{"sku": sku, "qty": qty}], "qty_returned")
    return {"status": "RETURNED", "work_order_id": work_order_id, "sku": sku, "qty": qty}


def get_outbound_history(work_order_id: str, sku: str = None) -> list:
    """Who collected what and when, for a work order. Used when chasing unaccounted items."""
    conn = sqlite3.connect(WMS_DB)
    q = ("SELECT sku, item_description, serial_number, -qty, transaction_date, handled_by FROM wms_ledger "
         "WHERE transaction_type = 'OUTBOUND' AND reference = ?")
    args = [work_order_id]
    if sku:
        q += " AND sku = ?"
        args.append(sku)
    rows = conn.execute(q, args).fetchall()
    conn.close()
    keys = ["sku", "description", "serial_number", "qty", "outbound_date", "collected_by"]
    return [dict(zip(keys, r)) for r in rows]
