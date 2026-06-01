"""
database.py — SQLAlchemy Core database layer for Store Intelligence API.

Uses SQLite by default (swappable to PostgreSQL via DATABASE_URL env var).
All schema is created on startup via create_all().

Design choice: SQLAlchemy Core (not ORM) for explicit SQL control and easier
async migration path if needed. Connection pooling via StaticPool for SQLite.
"""

import os
import logging
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, text,
    Table, Column, MetaData,
    String, Integer, Float, Boolean,
    Index, UniqueConstraint,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

logger = logging.getLogger("database")

# ─────────────────────────────────────────────────────────────────────────────
# Engine — lazy init so DATABASE_URL can be overridden before import (e.g. tests)
# ─────────────────────────────────────────────────────────────────────────────

def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
    return create_engine(url, pool_pre_ping=True, echo=False)


def _get_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./store_intelligence.db")


# Module-level engine — created once, but DATABASE_URL is read at first use
# via a lazy proxy so tests can set os.environ before importing this module.
_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = _get_database_url()
        _engine = _make_engine(url)
        logger.info("Database engine created (url=%s)", url.split("@")[-1])
    return _engine


metadata = MetaData()


def get_engine() -> Engine:
    """Return the (lazily-created) engine. Call this instead of using the module-level engine."""
    return _get_engine()

# ─────────────────────────────────────────────────────────────────────────────
# Table definitions
# ─────────────────────────────────────────────────────────────────────────────

events_table = Table(
    "events",
    metadata,
    Column("event_id",   String(36), primary_key=True),
    Column("store_id",   String(64), nullable=False, index=True),
    Column("camera_id",  String(64), nullable=False),
    Column("visitor_id", String(32), nullable=False, index=True),
    Column("event_type", String(32), nullable=False, index=True),
    # Store as ISO string for portability; index for range queries
    Column("timestamp",  String(32), nullable=False, index=True),
    Column("zone_id",    String(64), nullable=True),
    Column("dwell_ms",   Integer,    nullable=False, default=0),
    Column("is_staff",   Boolean,    nullable=False, default=False),
    Column("confidence", Float,      nullable=False, default=1.0),
    # Metadata stored as JSON string
    Column("queue_depth",  Integer, nullable=True),
    Column("sku_zone",     String(64), nullable=True),
    Column("session_seq",  Integer, nullable=False, default=0),
    # Ingestion bookkeeping
    Column("ingested_at", String(32), nullable=False),

    UniqueConstraint("event_id", name="uq_event_id"),
    Index("ix_store_ts", "store_id", "timestamp"),
    Index("ix_store_type", "store_id", "event_type"),
    Index("ix_visitor_store", "visitor_id", "store_id"),
)

# Hourly visitor session summary (materialised for fast metrics)
sessions_table = Table(
    "visitor_sessions",
    metadata,
    Column("id",              Integer,    primary_key=True, autoincrement=True),
    Column("store_id",        String(64), nullable=False, index=True),
    Column("visitor_id",      String(32), nullable=False),
    Column("entry_ts",        String(32), nullable=True),
    Column("exit_ts",         String(32), nullable=True),
    Column("converted",       Boolean,    nullable=False, default=False),
    Column("was_in_billing",  Boolean,    nullable=False, default=False),
    Column("abandoned_queue", Boolean,    nullable=False, default=False),
    Column("is_reentry",      Boolean,    nullable=False, default=False),

    Index("ix_sess_store_visitor", "store_id", "visitor_id"),
)

# Daily conversion snapshots for anomaly baseline
daily_stats_table = Table(
    "daily_stats",
    metadata,
    Column("id",               Integer,    primary_key=True, autoincrement=True),
    Column("store_id",         String(64), nullable=False),
    Column("date",             String(10), nullable=False),   # YYYY-MM-DD
    Column("unique_visitors",  Integer,    nullable=False, default=0),
    Column("conversions",      Integer,    nullable=False, default=0),
    Column("conversion_rate",  Float,      nullable=False, default=0.0),

    UniqueConstraint("store_id", "date", name="uq_store_date"),
    Index("ix_daily_store_date", "store_id", "date"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def create_tables():
    """Create all tables if they don't exist."""
    metadata.create_all(get_engine())
    logger.info("Database tables ready (url=%s)", _get_database_url().split("@")[-1])


def check_db() -> bool:
    """Return True if the database connection is healthy."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError as exc:
        logger.error("DB health check failed: %s", exc)
        return False


def reset_engine():
    """Reset the cached engine — call this in tests when DATABASE_URL changes."""
    global _engine
    _engine = None


def get_conn():
    """Context manager yielding a live connection. Use with `with get_conn() as conn:`."""
    return get_engine().connect()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def now_utc_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(ts_str: str) -> datetime:
    """Parse ISO-8601 UTC string to datetime (always UTC-aware)."""
    if ts_str.endswith("Z"):
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    return datetime.fromisoformat(ts_str).astimezone(timezone.utc)
