import json
import sqlite3
import uuid
from tools.inventory_tools import find_available_bay, get_sku_description

SUPPLIERS_DB = "databases/suppliers.db"
LOGISTICS_DB = "databases/logistics_transport.db"


def schedule_inbound_shipment(po_id: str, pallets_required: float, inbound_date: str, serialized: bool = True) -> dict:
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


# ---------------------------------------------------------------------------
# Warehouse based model: inbound batches arrive at the WH on their ETA date, subcontractors
# (main kits) and engineers (replacements) COLLECT from the WH and sign a collection note.
# ---------------------------------------------------------------------------

def _lconn():
    conn = sqlite3.connect(LOGISTICS_DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS shipment_batches (
            shipment_id TEXT PRIMARY KEY,
            batch_ref TEXT NOT NULL,          -- e.g. IMP-01 (import) or LP-01 (local purchase)
            project_name TEXT
        );
        CREATE TABLE IF NOT EXISTS collection_notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_order_id TEXT NOT NULL,
            sku TEXT NOT NULL,
            serial_number TEXT,
            qty INTEGER NOT NULL,
            collected_by TEXT NOT NULL,
            collected_at TEXT NOT NULL
        );
    """)
    return conn


def tag_shipment(shipment_id: str, batch_ref: str, project_name: str = None) -> None:
    conn = _lconn()
    conn.execute("INSERT OR REPLACE INTO shipment_batches VALUES (?, ?, ?)", (shipment_id, batch_ref, project_name))
    conn.commit()
    conn.close()


def in_transit_quantities() -> dict:
    """Stock ordered and on its way to the warehouse (not yet received)."""
    conn = _lconn()
    rows = conn.execute("SELECT packing_list_json FROM inbound_shipments WHERE status = 'IN_TRANSIT'").fetchall()
    conn.close()
    totals = {}
    for (pl,) in rows:
        for item in json.loads(pl):
            totals[item["sku"]] = totals.get(item["sku"], 0) + item["qty"]
    return totals


def due_shipments(as_of: str) -> list:
    """Inbound shipments whose ETA is on or before as_of and that are not received yet."""
    conn = _lconn()
    rows = conn.execute("""SELECT s.shipment_id, s.po_id, s.supplier_name, s.inbound_date, s.packing_list_json,
                                  COALESCE(b.batch_ref, '')
                           FROM inbound_shipments s LEFT JOIN shipment_batches b ON b.shipment_id = s.shipment_id
                           WHERE s.status = 'IN_TRANSIT' AND s.inbound_date <= ? ORDER BY s.inbound_date""",
                        (as_of,)).fetchall()
    conn.close()
    return [{"shipment_id": r[0], "po_id": r[1], "supplier": r[2], "eta": r[3],
             "packing_list": json.loads(r[4]), "batch_ref": r[5]} for r in rows]


def next_shipment_date(after: str):
    conn = _lconn()
    row = conn.execute("SELECT MIN(inbound_date) FROM inbound_shipments WHERE status = 'IN_TRANSIT' AND inbound_date > ?",
                       (after,)).fetchone()
    conn.close()
    return row[0] if row else None


def record_collection(work_order_id: str, items: list, collected_by: str, collected_at: str) -> int:
    """Collection note signed by the subcontractor or engineer at the warehouse counter."""
    conn = _lconn()
    conn.executemany("""INSERT INTO collection_notes (work_order_id, sku, serial_number, qty, collected_by, collected_at)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                     [(work_order_id, i["sku"], i.get("serial_number"), i["qty"], collected_by, collected_at) for i in items])
    conn.commit()
    conn.close()
    return len(items)


def get_collected_quantities(work_order_id: str) -> dict:
    """Quantities signed for on collection notes (the 'handover' leg of 3 way reconciliation)."""
    conn = _lconn()
    rows = conn.execute("SELECT sku, SUM(qty) FROM collection_notes WHERE work_order_id = ? GROUP BY sku",
                        (work_order_id,)).fetchall()
    conn.close()
    return {sku: q for sku, q in rows}


# Reconciliation reads the handover quantities through this name
get_delivered_quantities = get_collected_quantities