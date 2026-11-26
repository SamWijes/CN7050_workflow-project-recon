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

# ---------------------------------------------------------------------------
# Single source of truth for every SKU used in the system
# (sku, description, serialized, unit, contract_rate_gbp, allowed_variance_pct, billable)
# ---------------------------------------------------------------------------
SKU_MASTER = [
    # Cabinet kit (all serialized)
    ("CAB-OUT-01",    "Outdoor Telecom Cabinet",              1, "pcs", 4200.00, 0.0, 1),
    ("BAT-48V-01",    "48V 100Ah Lithium Battery",            1, "pcs",  380.00, 0.0, 1),
    ("PSU-01",        "Power Supply Unit",                    1, "pcs",  650.00, 0.0, 1),
    ("BBU-01",        "Baseband Unit",                        1, "pcs", 3100.00, 0.0, 1),
    ("CARD-CC-01",    "Control & Clock (CC) Card",            1, "pcs", 1250.00, 0.0, 1),
    ("CARD-BBN-01",   "Baseband Processing (BBN) Card",       1, "pcs",  980.00, 0.0, 1),
    ("RECT-01",       "Rectifier Shelf",                      1, "pcs",  720.00, 0.0, 1),
    ("SMR-01",        "Switch Mode Rectifier (SMR) Module",   1, "pcs",  310.00, 0.0, 1),
    # Site passive materials (bulk, not serialized)
    ("FBR-CBL-01",    "Fiber Optic Cable",                    0, "pcs",   45.00, 2.0, 1),
    ("MNT-BRK-01",    "Mounting Bracket",                     0, "pcs",  120.00, 2.0, 1),
    ("BASE-PLT-01",   "Cabinet Base Plate",                   0, "pcs",  260.00, 2.0, 1),
    ("UPVC-20",       "20mm uPVC Conduit",                    0, "pcs",    1.50, 5.0, 1),
    ("PWR-CBL-2C-20", "20mm 2 Core Power Cable",              0, "m",      3.20, 5.0, 1),
    ("TIE-150",       "150mm UV Cable Tie",                   0, "pcs",    0.05, 0.0, 0),  # consumable
]

LABOUR_RATES = [
    ("LAB-FULL",    "full_site",    "Full Site Installation Labour",    1800.00),
    ("LAB-PARTIAL", "partial_site", "Cabinet Installation Labour",      1100.00),
    ("LAB-UPGRADE", "upgrade",      "BBU Upgrade Labour",                600.00),
]

BOM_TEMPLATES = {
    "kits": {
        "cabinet": [
            {"sku": "CAB-OUT-01",  "qty": 1},
            {"sku": "BAT-48V-01",  "qty": 8},
            {"sku": "PSU-01",      "qty": 1},
            {"sku": "BBU-01",      "qty": 1},
            {"sku": "CARD-CC-01",  "qty": 1},
            {"sku": "CARD-BBN-01", "qty": 1},
            {"sku": "RECT-01",     "qty": 1},
            {"sku": "SMR-01",      "qty": 3},
        ],
        "bbu_swap": [
            {"sku": "PSU-01",      "qty": 1},
            {"sku": "BBU-01",      "qty": 1},
            {"sku": "CARD-CC-01",  "qty": 1},
            {"sku": "CARD-BBN-01", "qty": 1},
        ],
    },
    "full_site": {
        "kits": ["cabinet"],
        "fixed": [
            {"sku": "FBR-CBL-01",  "qty": 3},
            {"sku": "MNT-BRK-01",  "qty": 3},
            {"sku": "BASE-PLT-01", "qty": 1},
        ],
        "ranged": [
            {"sku": "TIE-150", "min": 100, "max": 200},
        ],
        # One random "cable run" factor per site keeps conduit and cable proportional
        "linked": {
            "factor": "cable_run",
            "items": [
                {"sku": "UPVC-20",       "min": 20, "max": 90},
                {"sku": "PWR-CBL-2C-20", "min": 20, "max": 180},
            ],
        },
    },
    "partial_site": {"kits": ["cabinet"]},
    "upgrade": {"kits": ["bbu_swap"]},
}


def _fresh(path, force_reset):
    if os.path.exists(path) and not force_reset:
        print(f"Using existing {path}")
        return False
    if os.path.exists(path):
        os.remove(path)
    return True


def _create_sku_master(cur):
    cur.execute("""
    CREATE TABLE sku_master (
        sku TEXT PRIMARY KEY,
        item_description TEXT NOT NULL,
        serialized INTEGER NOT NULL,
        unit TEXT NOT NULL
    )
    """)
    cur.executemany(
        "INSERT INTO sku_master VALUES (?, ?, ?, ?)",
        [(s, d, ser, u) for s, d, ser, u, *_ in SKU_MASTER],
    )


def setup_orchestrator_db(force_reset: bool = False):
    if not _fresh(ORCHESTRATOR_DB, force_reset):
        return
    conn = sqlite3.connect(ORCHESTRATOR_DB)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE resources (
        resource_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('SUBCONTRACTOR', 'ENGINEER')),
        email TEXT NOT NULL
    )
    """)

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

    # Reconciliation: Dispatched = Installed + Quarantined + Returned
    cur.execute("""
    CREATE TABLE reconciliation_ledger (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        work_order_id TEXT NOT NULL,
        site_id TEXT NOT NULL,
        sku TEXT NOT NULL,
        serialized INTEGER NOT NULL,
        qty_planned INTEGER NOT NULL,
        qty_dispatched INTEGER DEFAULT 0,
        qty_installed INTEGER DEFAULT 0,
        qty_quarantined INTEGER DEFAULT 0,
        qty_returned INTEGER DEFAULT 0,
        reconciliation_status TEXT DEFAULT 'PENDING',
        UNIQUE(work_order_id, sku),
        FOREIGN KEY (work_order_id) REFERENCES work_orders(work_order_id)
    )
    """)

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

    cur.execute("""
    CREATE TABLE contract_rates (
        code TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        rate_type TEXT NOT NULL CHECK(rate_type IN ('MATERIAL', 'LABOUR')),
        site_category TEXT,
        unit_rate_gbp REAL NOT NULL,
        allowed_variance_pct REAL NOT NULL,
        billable INTEGER NOT NULL
    )
    """)
    cur.executemany(
        "INSERT INTO contract_rates VALUES (?, ?, 'MATERIAL', NULL, ?, ?, ?)",
        [(s, d, rate, var, bill) for s, d, _, _, rate, var, bill in SKU_MASTER],
    )
    cur.executemany(
        "INSERT INTO contract_rates VALUES (?, ?, 'LABOUR', ?, ?, 0.0, 1)",
        [(code, desc, cat, rate) for code, cat, desc, rate in LABOUR_RATES],
    )

    cur.execute("""
    CREATE TABLE invoices (
        invoice_id TEXT PRIMARY KEY,
        work_order_id TEXT NOT NULL,
        grand_total_gbp REAL NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('PENDING_APPROVAL', 'APPROVED', 'REJECTED', 'HELD_ANOMALY')),
        anomalies_json TEXT,
        created_date TEXT NOT NULL,
        approved_by TEXT,
        approved_at TEXT
    )
    """)

    resources = []
    for letter in "ABCDE":
        resources.append((f"Subcon-{letter}", f"Subcontractor Team {letter}", "SUBCONTRACTOR",
                          f"subcon.{letter.lower()}@contractors.local"))
    for letter in "ABCDE":
        resources.append((f"Engineer-{letter}", f"Internal Engineer {letter}", "ENGINEER",
                          f"engineer.{letter.lower()}@intellicommstech.local"))
    cur.executemany("INSERT INTO resources VALUES (?, ?, ?, ?)", resources)

    conn.commit()
    conn.close()
    print(f"Created and seeded {ORCHESTRATOR_DB}")


def setup_wms_db(force_reset: bool = False):
    if not _fresh(WMS_DB, force_reset):
        return
    conn = sqlite3.connect(WMS_DB)
    cur = conn.cursor()
    _create_sku_master(cur)

    cur.execute("""
    CREATE TABLE wms_ledger (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku TEXT NOT NULL,
        item_description TEXT NOT NULL,
        serial_number TEXT,
        qty INTEGER NOT NULL,
        transaction_date TEXT NOT NULL,
        handled_by TEXT NOT NULL,
        transaction_type TEXT NOT NULL CHECK(transaction_type IN ('INBOUND', 'OUTBOUND', 'QUARANTINE', 'RETURN')),
        reference TEXT
    )
    """)

    # Available stock: quarantined units never count as available
    cur.execute("""
    CREATE VIEW current_stock AS
    SELECT sku, item_description, SUM(qty) AS available_qty
    FROM wms_ledger
    WHERE transaction_type != 'QUARANTINE'
    GROUP BY sku, item_description
    """)

    # Serials that are physically on the shelf and not faulty
    cur.execute("""
    CREATE VIEW available_serials AS
    SELECT sku, serial_number
    FROM wms_ledger
    WHERE serial_number IS NOT NULL
    GROUP BY sku, serial_number
    HAVING SUM(CASE WHEN transaction_type != 'QUARANTINE' THEN qty ELSE 0 END) > 0
       AND SUM(CASE WHEN transaction_type = 'QUARANTINE' THEN 1 ELSE 0 END) = 0
    """)

    cur.execute("""
    CREATE TABLE wh_bays (
        bay_id TEXT PRIMARY KEY,
        zone TEXT NOT NULL,
        max_pallets INTEGER NOT NULL,
        used_pallets INTEGER NOT NULL
    )
    """)

    seed_date = "2026-11-20"
    desc = {s: d for s, d, *_ in SKU_MASTER}

    # Opening stock: deliberately short on some cabinet items so sourcing triggers
    serial_stock = {
        "CAB-OUT-01": 2, "BAT-48V-01": 16, "PSU-01": 3, "BBU-01": 3,
        "CARD-CC-01": 3, "CARD-BBN-01": 3, "RECT-01": 2, "SMR-01": 6,
    }
    bulk_stock = {
        "FBR-CBL-01": 10, "MNT-BRK-01": 6, "BASE-PLT-01": 2,
        "UPVC-20": 300, "PWR-CBL-2C-20": 600, "TIE-150": 1000,
    }

    rows = []
    for sku, count in serial_stock.items():
        prefix = sku.rsplit("-", 1)[0]  # CARD-CC-01 -> CARD-CC, keeps serials unique per part
        for i in range(1, count + 1):
            rows.append((sku, desc[sku], f"SN-{prefix}-{1000 + i}", 1, seed_date,
                         "John Doe (WH Inbound)", "INBOUND", "PO-OPENING"))
    for sku, qty in bulk_stock.items():
        rows.append((sku, desc[sku], None, qty, seed_date,
                     "Sarah Lee (Local Procurement)", "INBOUND", "LP-OPENING"))

    cur.executemany("""
    INSERT INTO wms_ledger
    (sku, item_description, serial_number, qty, transaction_date, handled_by, transaction_type, reference)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    cur.executemany("INSERT INTO wh_bays VALUES (?, ?, ?, ?)", [
        ("BAY-A1", "Active Hardware Zone", 20, 12),
        ("BAY-A2", "Active Hardware Zone", 20, 4),
        ("BAY-B1", "Bulk & Civil Zone",    40, 15),
        ("BAY-Q1", "Faulty Quarantine",    10, 0),
    ])

    conn.commit()
    conn.close()
    print(f"Created and seeded {WMS_DB}")


def setup_suppliers_db(force_reset: bool = False):
    if not _fresh(SUPPLIERS_DB, force_reset):
        return
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

    desc = {s: d for s, d, *_ in SKU_MASTER}
    rate = {s: r for s, _, _, _, r, _, _ in SKU_MASTER}
    catalog = []
    for sku, d, serialized, *_ in SKU_MASTER:
        contract = rate[sku]
        if serialized:
            # Internal import: cheapest, slowest. 3rd party: dearer, faster, smaller stock.
            catalog.append(("SUP-INT", "IntelliCommsTech", "INTERNAL", sku, d, 60, round(contract * 0.80, 2), 5))
            catalog.append(("SUP-A", "Apex Telecom Hardware", "3RD_PARTY", sku, d, 10, round(contract * 0.92, 2), 2))
            catalog.append(("SUP-B", "RapidTel Express UK", "3RD_PARTY", sku, d, 4, round(contract * 0.98, 2), 1))
        else:
            catalog.append(("SUP-C", "Essex Civil & Bulk Supplies", "3RD_PARTY", sku, d, 10000, round(contract * 0.75, 2), 1))

    cur.executemany("""
    INSERT INTO supplier_catalog
    (supplier_id, supplier_name, supplier_type, sku, item_description, stock_available, unit_price_gbp, lead_time_days)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, catalog)

    conn.commit()
    conn.close()
    print(f"Created and seeded {SUPPLIERS_DB}")


def setup_logistics_db(force_reset: bool = False):
    if not _fresh(LOGISTICS_DB, force_reset):
        return
    conn = sqlite3.connect(LOGISTICS_DB)
    cur = conn.cursor()

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

    cur.execute("""
    CREATE TABLE outbound_deliveries (
        waybill_id TEXT PRIMARY KEY,
        work_order_id TEXT NOT NULL,
        site_id TEXT NOT NULL,
        carrier_name TEXT NOT NULL,
        scheduled_delivery_date TEXT NOT NULL,
        delivery_status TEXT NOT NULL DEFAULT 'SCHEDULED',
        received_by_engineer TEXT,
        delivered_at TEXT,
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
    with open(BOM_FILE, "w", encoding="utf-8") as f:
        json.dump(BOM_TEMPLATES, f, indent=2)
    print(f"Created {BOM_FILE}")


def init_all_databases(force_reset: bool = False):
    setup_orchestrator_db(force_reset)
    setup_wms_db(force_reset)
    setup_suppliers_db(force_reset)
    setup_logistics_db(force_reset)
    setup_bom_templates(force_reset)


def verify_setup():
    conn = sqlite3.connect(WMS_DB)
    print("\n--- Current Warehouse Stock ---")
    for sku, d, qty in conn.execute("SELECT sku, item_description, available_qty FROM current_stock ORDER BY sku"):
        print(f"  {sku:<14} | {d:<36} | Qty: {qty}")
    conn.close()


if __name__ == "__main__":
    init_all_databases(force_reset="--reset" in sys.argv)
    verify_setup()
