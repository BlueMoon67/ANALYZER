"""
pos_correlate.py — POS transaction ↔ visitor session correlation.

Reads pos_transactions.csv and marks visitor sessions as converted when a
visitor was in the BILLING zone in the 5-minute window before the transaction.

Usage:
    python pos_correlate.py --pos pos_transactions.csv --api http://localhost:8000

This runs after clip processing so that events are already ingested.
Correlation is purely time-window based (no customer_id in POS data).
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone, timedelta

import urllib.request
import urllib.error
import sys as _sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so `app` package is importable
_PROJECT_ROOT = str(_Path(__file__).parent.parent)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("pos_correlate")

BILLING_WINDOW_MINUTES = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pos",  required=True, help="Path to pos_transactions.csv")
    p.add_argument("--api",  default="http://localhost:8000", help="API base URL")
    return p.parse_args()


def api_get(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def load_transactions(csv_path: str) -> list[dict]:
    txns = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txns.append({
                "store_id":     row["store_id"].strip(),
                "transaction_id": row["transaction_id"].strip(),
                "timestamp":    row["timestamp"].strip(),
                "basket_value": float(row["basket_value_inr"]),
            })
    return txns


def get_billing_visitors_in_window(
    api_base: str, store_id: str, txn_ts: str
) -> list[str]:
    """
    Query the events store for visitors in BILLING zone in the 5-min window
    before the transaction. This hits the raw events table via a future
    /stores/{id}/billing-visitors endpoint — for now we call /metrics and
    cross-reference with locally cached events.

    Since this is a batch script (not live), it directly queries SQLite via
    the same SQLAlchemy engine used by the API.
    """
    # Import here so this script can be run standalone
    from app.database import get_conn, events_table
    from sqlalchemy import select

    txn_dt = _parse_ts(txn_ts)
    window_start = (txn_dt - timedelta(minutes=BILLING_WINDOW_MINUTES)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    window_end = txn_ts

    with get_conn() as conn:
        rows = conn.execute(
            select(events_table.c.visitor_id)
            .where(
                events_table.c.store_id   == store_id,
                events_table.c.is_staff   == False,
                events_table.c.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
                events_table.c.zone_id    == "BILLING",
                events_table.c.timestamp  >= window_start,
                events_table.c.timestamp  <= window_end,
            )
            .distinct()
        ).fetchall()

    return [r[0] for r in rows]


def mark_converted(store_id: str, visitor_id: str):
    from app.ingestion import mark_visitor_converted
    mark_visitor_converted(store_id, visitor_id)
    logger.debug("Marked converted: store=%s visitor=%s", store_id, visitor_id)


def main():
    args = parse_args()
    txns = load_transactions(args.pos)
    logger.info("Loaded %d POS transactions", len(txns))

    converted_pairs: set[tuple[str, str]] = set()

    for txn in txns:
        store_id = txn["store_id"]
        txn_ts   = txn["timestamp"]

        visitors = get_billing_visitors_in_window(args.api, store_id, txn_ts)
        for vis_id in visitors:
            key = (store_id, vis_id)
            if key not in converted_pairs:
                mark_converted(store_id, vis_id)
                converted_pairs.add(key)
                logger.info(
                    "Converted: store=%s visitor=%s txn=%s value=%.2f",
                    store_id, vis_id, txn["transaction_id"], txn["basket_value"],
                )

    logger.info(
        "POS correlation complete. %d unique visitor-sessions marked converted.",
        len(converted_pairs),
    )


def _parse_ts(ts: str) -> datetime:
    if ts.endswith("Z"):
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


if __name__ == "__main__":
    main()
