import os
import sys
import json
import sqlite3
from datetime import datetime

os.makedirs("databases", exist_ok=True)
os.makedirs("mock_apis", exist_ok=True)
os.makedirs("outbox", exist_ok=True)

ORCHESTRATOR_DB = "databases/orchestrator_ledger.db"
WMS_DB = "databases/wms_inventory.db"
SUPPLIERS_DB = "databases/suppliers.db"
LOGISTICS_DB = "databases/logistics_transport.db"
BOM_FILE = "mock_apis/bom_templates.json"


def setup_orchestrator_db(force_reset: bool = False):
    if os.path.exists(ORCHESTRATOR_DB) and not force_reset:
        print(f"Using existing {ORCHESTRATOR_DB}")
        return

    if os.path.exists(ORCHESTRATOR_DB):
        os.remove(ORCHESTRATOR_DB)

    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()

    # 5 Subcontractors (Subcon-A to E) & 5 Internal Engineers (Engineer-A to E)
    cur.execute("""
    CREATE TABLE resources (
        resource_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('SUBCONTRACTOR', 'ENGINEER')),
        email TEXT NOT NULL
    )
    """)

    # Enforces: every site has 1 Subcon + 1 Engineer, and neither can be double-booked on the same date
    cur.execute("""
    CREATE TABLE work_orders (
        work_order_id TEXT PRIMARY KEY,
        project_name TEXT NOT NULL,
        site_id TEXT NOT NULL,
        site_category TEXT NOT NULL CHECK(site_category IN ('full_site', 'partial_site', 'upgrade')),
        scheduled_date TEXT NOT NULL,
        subcon_id TEXT NOT NULL,
        engineer_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'PLANNED',
        FOREIGN KEY (subcon_id) REFERENCES resources(resource_id),
        FOREIGN KEY (engineer_id) REFERENCES resources(resource_id),
        UNIQUE(subcon_id, scheduled_date),
        UNIQUE(engineer_id, scheduled_date)
    )
    """)

    # 3-way reconciliation ledger: Dispatched = Installed + Quarantined
    cur.execute("""
    CREATE TABLE reconciliation_ledger (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        work_order_id TEXT NOT NULL,
        site_id TEXT NOT NULL,
        sku TEXT NOT NULL,
        qty_planned INTEGER NOT NULL,
        qty_dispatched INTEGER DEFAULT 0,
        qty_installed INTEGER DEFAULT 0,
        qty_quarantined INTEGER DEFAULT 0,
        reconciliation_status TEXT DEFAULT 'PENDING',
        FOREIGN KEY (work_order_id) REFERENCES work_orders(work_order_id)
    )
    """)

    # Simulated email audit log (Subcons, Engineers, WH)
    cur.execute("""
    CREATE TABLE email_outbox (
        email_id INTEGER PRIMARY KEY AUTOINCREMENT,
        work_order_id TEXT,
        recipient_email TEXT NOT NULL,
        recipient_role TEXT NOT NULL,
        subject TEXT NOT NULL,
        body TEXT NOT NULL,
        sent_at TEXT NOT NULL
    )
    """)

    resources = [
        ("Subcon-A", "Subcontractor Team A", "SUBCONTRACTOR", "subcon.a@contractors.local"),
        ("Subcon-B", "Subcontractor Team B", "SUBCONTRACTOR", "subcon.b@contractors.local"),
        ("Subcon-C", "Subcontractor Team C", "SUBCONTRACTOR", "subcon.c@contractors.local"),
        ("Subcon-D", "Subcontractor Team D", "SUBCONTRACTOR", "subcon.d@contractors.local"),
        ("Subcon-E", "Subcontractor Team E", "SUBCONTRACTOR", "subcon.e@contractors.local"),
        ("Engineer-A", "Internal Engineer A", "ENGINEER", "engineer.a@intellicommstech.local"),
        ("Engineer-B", "Internal Engineer B", "ENGINEER", "engineer.b@intellicommstech.local"),
        ("Engineer-C", "Internal Engineer C", "ENGINEER", "engineer.c@intellicommstech.local"),
        ("Engineer-D", "Internal Engineer D", "ENGINEER", "engineer.d@intellicommstech.local"),
        ("Engineer-E", "Internal Engineer E", "ENGINEER", "engineer.e@intellicommstech.local"),
    ]

    cur.executemany(
        "INSERT INTO resources (resource_id, name, role, email) VALUES (?, ?, ?, ?)",
        resources,
    )

    conn.commit()
    conn.close()
    print(f"Created and seeded {ORCHESTRATOR_DB}")


def setup_wms_db(force_reset: bool = False):
    if os.path.exists(WMS_DB) and not force_reset:
        print(f"Using existing {WMS_DB}")
        return

    if os.path.exists(WMS_DB):
        os.remove(WMS_DB)

    conn = sqlite3.connect(WMS_DB)
    cur = conn.cursor()

    # Ledger: +1/-1 per serialized unit; +qty/-qty for bulk local purchases
    cur.execute("""
    CREATE TABLE wms_ledger (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku TEXT NOT NULL,
        item_description TEXT NOT NULL,
        serial_number TEXT,
        qty INTEGER NOT NULL,
        transaction_date TEXT NOT NULL,
        handled_by TEXT NOT NULL,
        transaction_type TEXT NOT NULL CHECK(transaction_type IN ('INBOUND', 'OUTBOUND', 'QUARANTINE')),
        reference TEXT
    )
    """)

    # Live stock view (excludes quarantined units)
    cur.execute("""
    CREATE VIEW current_stock AS
    SELECT
        sku,
        item_description,
        SUM(qty) AS available_qty
    FROM wms_ledger
    WHERE transaction_type != 'QUARANTINE'
    GROUP BY sku, item_description
    """)

    # Warehouse bays for Inventory <-> Logistics inbound spacing coordination
    cur.execute("""
    CREATE TABLE wh_bays (
        bay_id TEXT PRIMARY KEY,
        zone TEXT NOT NULL,
        max_pallets INTEGER NOT NULL,
        used_pallets INTEGER NOT NULL
    )
    """)

    today = datetime.now().strftime("%Y-%m-%d")

    initial_ledger = [
        # Serialized Telecom Hardware (+1 per row)
        # Note: Only 2 SKU-RU-5G available (1 already dispatched) so a full_site (needs 3) triggers sourcing
        ("SKU-RU-5G",  "5G Radio Unit (Standard)", "SN-RU-1001", 1, today, "John Doe (WH Inbound)", "INBOUND", "PO-INIT-01"),
        ("SKU-RU-5G",  "5G Radio Unit (Standard)", "SN-RU-1002", 1, today, "John Doe (WH Inbound)", "INBOUND", "PO-INIT-01"),
        ("SKU-RU-5G",  "5G Radio Unit (Standard)", "SN-RU-1003", 1, today, "John Doe (WH Inbound)", "INBOUND", "PO-INIT-01"),
        ("SKU-RU-5G",  "5G Radio Unit (Standard)", "SN-RU-1003", -1, today, "Subcon-A", "OUTBOUND", "SITE-099"),

        ("SKU-BBU-01", "Baseband Unit", "SN-BBU-2001", 1, today, "John Doe (WH Inbound)", "INBOUND", "PO-INIT-01"),
        ("SKU-BBU-01", "Baseband Unit", "SN-BBU-2002", 1, today, "John Doe (WH Inbound)", "INBOUND", "PO-INIT-01"),

        ("SKU-ANT-04", "Sector Antenna (Tri-band)", "SN-ANT-3001", 1, today, "John Doe (WH Inbound)", "INBOUND", "PO-INIT-02"),
        ("SKU-ANT-04", "Sector Antenna (Tri-band)", "SN-ANT-3002", 1, today, "John Doe (WH Inbound)", "INBOUND", "PO-INIT-02"),
        ("SKU-ANT-04", "Sector Antenna (Tri-band)", "SN-ANT-3003", 1, today, "John Doe (WH Inbound)", "INBOUND", "PO-INIT-02"),
        ("SKU-ANT-04", "Sector Antenna (Tri-band)", "SN-ANT-3004", 1, today, "John Doe (WH Inbound)", "INBOUND", "PO-INIT-02"),

        # Bulk / Local Purchases (No serial number)
        ("SKU-UPVC-20", "20mm uPVC Conduit Pipe (Local)", None, 1000, today, "Sarah Lee (Local Procurement)", "INBOUND", "LP-2026-01"),
        ("SKU-FBR-10",  "Fiber Patch Cord 10m (Bulk)",    None, 150,  today, "Sarah Lee (Local Procurement)", "INBOUND", "LP-2026-02"),
        ("SKU-CBL-TIE", "UV Cable Ties Pack (Bulk)",      None, 500,  today, "Sarah Lee (Local Procurement)", "INBOUND", "LP-2026-03"),
    ]

    cur.executemany("""
    INSERT INTO wms_ledger
    (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, initial_ledger)

    bays = [
        ("BAY-A1", "Active Hardware Zone", 20, 12),
        ("BAY-A2", "Active Hardware Zone", 20, 4),
        ("BAY-B1", "Bulk & Civil Zone",    40, 15),
        ("BAY-Q1", "Faulty Quarantine",    10, 0),
    ]

    cur.executemany(
        "INSERT INTO wh_bays (bay_id, zone, max_pallets, used_pallets) VALUES (?, ?, ?, ?)",
        bays,
    )

    conn.commit()
    conn.close()
    print(f"Created and seeded {WMS_DB}")


def setup_suppliers_db(force_reset: bool = False):
    if os.path.exists(SUPPLIERS_DB) and not force_reset:
        print(f"Using existing {SUPPLIERS_DB}")
        return

    if os.path.exists(SUPPLIERS_DB):
        os.remove(SUPPLIERS_DB)

    conn = sqlite3.connect(SUPPLIERS_DB)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE supplier_catalog (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_id TEXT NOT NULL,
        supplier_name TEXT NOT NULL,
        supplier_type TEXT NOT NULL CHECK(supplier_type IN ('INTERNAL', '3RD_PARTY')),
        sku TEXT NOT NULL,
        item_description TEXT NOT NULL,
        stock_available INTEGER NOT NULL,
        unit_price_gbp REAL NOT NULL,
        lead_time_days INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE purchase_orders (
        po_id TEXT PRIMARY KEY,
        supplier_id TEXT NOT NULL,
        supplier_name TEXT NOT NULL,
        supplier_type TEXT NOT NULL,
        sku TEXT NOT NULL,
        qty_ordered INTEGER NOT NULL,
        total_cost_gbp REAL NOT NULL,
        status TEXT NOT NULL,
        order_date TEXT NOT NULL
    )
    """)

    catalog = [
        # Internal Company Imports (IntelliCommmsTech HQ)
        ("SUP-INT", "IntelliCommmsTech", "INTERNAL", "SKU-RU-5G",  "5G Radio Unit (Standard)",  25, 2100.00, 5),
        ("SUP-INT", "IntelliCommmsTech", "INTERNAL", "SKU-BBU-01", "Baseband Unit",             15, 2800.00, 5),
        ("SUP-INT", "IntelliCommmsTech", "INTERNAL", "SKU-ANT-04", "Sector Antenna (Tri-band)", 30, 750.00,  4),

        # 3rd-Party External Hardware Suppliers (faster lead times, higher unit cost)
        ("SUP-A", "Apex Telecom Hardware", "3RD_PARTY", "SKU-RU-5G",  "5G Radio Unit (Standard)",  5, 2400.00, 3),
        ("SUP-B", "RapidTel Express UK",   "3RD_PARTY", "SKU-RU-5G",  "5G Radio Unit (Standard)",  2, 2450.00, 1),
        ("SUP-A", "Apex Telecom Hardware", "3RD_PARTY", "SKU-BBU-01", "Baseband Unit",             4, 3050.00, 3),
        ("SUP-B", "RapidTel Express UK",   "3RD_PARTY", "SKU-BBU-01", "Baseband Unit",             1, 3100.00, 1),
        ("SUP-A", "Apex Telecom Hardware", "3RD_PARTY", "SKU-ANT-04", "Sector Antenna (Tri-band)", 8, 840.00,  2),

        # Local 3rd-Party Bulk Supplies
        ("SUP-C", "Essex Civil & Bulk Supplies", "3RD_PARTY", "SKU-UPVC-20", "20mm uPVC Conduit Pipe (Local)", 5000, 1.50, 1),
        ("SUP-C", "Essex Civil & Bulk Supplies", "3RD_PARTY", "SKU-FBR-10",  "Fiber Patch Cord 10m (Bulk)",    500,  22.00, 1),
        ("SUP-C", "Essex Civil & Bulk Supplies", "3RD_PARTY", "SKU-CBL-TIE", "UV Cable Ties Pack (Bulk)",      2000, 4.00,  1),
    ]

    cur.executemany("""
    INSERT INTO supplier_catalog
    (supplier_id, supplier_name, supplier_type, sku, item_description, stock_available, unit_price_gbp, lead_time_days)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, catalog)

    conn.commit()
    conn.close()
    print(f"Created and seeded {SUPPLIERS_DB}")


def setup_logistics_db(force_reset: bool = False):
    if os.path.exists(LOGISTICS_DB) and not force_reset:
        print(f"Using existing {LOGISTICS_DB}")
        return

    if os.path.exists(LOGISTICS_DB):
        os.remove(LOGISTICS_DB)

    conn = sqlite3.connect(LOGISTICS_DB)
    cur = conn.cursor()

    # Inbound imports & supplier deliveries (clearance, WH bay, packing list handoff)
    cur.execute("""
    CREATE TABLE inbound_shipments (
        shipment_id TEXT PRIMARY KEY,
        po_id TEXT NOT NULL,
        supplier_name TEXT NOT NULL,
        clearance_status TEXT NOT NULL CHECK(clearance_status IN ('PENDING', 'CLEARED', 'HELD')),
        inbound_date TEXT NOT NULL,
        assigned_bay TEXT NOT NULL,
        pallets_required INTEGER NOT NULL,
        packing_list_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'IN_TRANSIT'
    )
    """)

    # Outbound dispatches from WH to telecom sites
    cur.execute("""
    CREATE TABLE outbound_deliveries (
        waybill_id TEXT PRIMARY KEY,
        work_order_id TEXT NOT NULL,
        site_id TEXT NOT NULL,
        carrier_name TEXT NOT NULL,
        scheduled_delivery_date TEXT NOT NULL,
        delivery_status TEXT NOT NULL DEFAULT 'SCHEDULED',
        received_by_engineer TEXT,
        dispatched_items_json TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()
    print(f"Created {LOGISTICS_DB}")


def setup_bom_templates(force_reset: bool = False):
    if os.path.exists(BOM_FILE) and not force_reset:
        print(f"Using existing {BOM_FILE}")
        return

    bom_templates = {
        "full_site": [
            {"sku": "SKU-RU-5G",   "qty": 3,  "serialized": True},
            {"sku": "SKU-BBU-01",  "qty": 1,  "serialized": True},
            {"sku": "SKU-ANT-04",  "qty": 3,  "serialized": True},
            {"sku": "SKU-FBR-10",  "qty": 10, "serialized": False},
            {"sku": "SKU-UPVC-20", "qty": 50, "serialized": False},
            {"sku": "SKU-CBL-TIE", "qty": 10, "serialized": False},
        ],
        "partial_site": [
            {"sku": "SKU-RU-5G",   "qty": 1,  "serialized": True},
            {"sku": "SKU-ANT-04",  "qty": 1,  "serialized": True},
            {"sku": "SKU-FBR-10",  "qty": 4,  "serialized": False},
            {"sku": "SKU-UPVC-20", "qty": 20, "serialized": False},
        ],
        "upgrade": [
            {"sku": "SKU-RU-5G",   "qty": 2,  "serialized": True},
            {"sku": "SKU-FBR-10",  "qty": 6,  "serialized": False},
            {"sku": "SKU-CBL-TIE", "qty": 4,  "serialized": False},
        ],
    }

    with open(BOM_FILE, "w", encoding="utf-8") as f:
        json.dump(bom_templates, f, indent=2)
    print(f"Created {BOM_FILE}")


def init_all_databases(force_reset: bool = False):
    setup_orchestrator_db(force_reset=force_reset)
    setup_wms_db(force_reset=force_reset)
    setup_suppliers_db(force_reset=force_reset)
    setup_logistics_db(force_reset=force_reset)
    setup_bom_templates(force_reset=force_reset)


def verify_setup():
    conn = sqlite3.connect(WMS_DB)
    print("\n--- Current Warehouse Stock ---")
    for row in conn.execute("SELECT sku, item_description, available_qty FROM current_stock"):
        print(f"  {row[0]:<12} | {row[1]:<32} | Qty: {row[2]}")

    print("\n--- Available Serialized Units ---")
    for row in conn.execute("""
        SELECT sku, serial_number
        FROM wms_ledger
        WHERE serial_number IS NOT NULL
        GROUP BY sku, serial_number
        HAVING SUM(qty) > 0
    """):
        print(f"  {row[0]:<12} | {row[1]}")

    print("\n--- Warehouse Bays ---")
    for row in conn.execute("SELECT bay_id, zone, used_pallets, max_pallets FROM wh_bays"):
        print(f"  {row[0]:<8} | {row[1]:<22} | {row[2]}/{row[3]} pallets used")
    conn.close()

    conn = sqlite3.connect(ORCHESTRATOR_DB)
    print("\n--- Registered Teams ---")
    for row in conn.execute("SELECT resource_id, role, email FROM resources"):
        print(f"  {row[0]:<12} | {row[1]:<14} | {row[2]}")
    conn.close()


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    init_all_databases(force_reset=reset)
    verify_setup()