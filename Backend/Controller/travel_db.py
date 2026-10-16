"""
travel_db.py
SQLAlchemy models and helpers for the Travel Planner feature.
Plans are identified by a short human-friendly ID (e.g. "ABC123").
Anyone with the ID can view and edit — designed for small-group collaboration.
"""

import uuid
import random
from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

DB_PATH = '_data/market.db'   # reuse the shared SQLite file

engine = create_engine(
    f'sqlite:///{DB_PATH}',
    echo=False,
    connect_args={'check_same_thread': False},
)
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Unambiguous uppercase alphanumeric chars (no 0/O, 1/I/L)
_ID_CHARS = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'


def _gen_plan_id():
    return ''.join(random.choices(_ID_CHARS, k=6))


def new_plan_id(session):
    """A free plan id. Exposed so the same id can name the plan's bill."""
    for _ in range(20):
        pid = _gen_plan_id()
        if not session.get(TravelPlan, pid):
            return pid
    return _gen_plan_id()


# ── Models ────────────────────────────────────────────────────────────────────

class TravelPlan(Base):
    __tablename__ = 'travel_plans'
    id         = Column(String(10),  primary_key=True)
    name       = Column(Text,        nullable=False)
    created_by = Column(String(64),  nullable=False)
    num_days   = Column(Integer,     nullable=False, default=1)
    created_at = Column(DateTime,    default=datetime.utcnow)
    # A plan carries its own bill (see travel_bill.py). Stored here rather than
    # looked up by convention so an old plan without one reads as None, not as
    # a missing bill to chase.
    bill_id    = Column(String(10), nullable=True)


class TravelEntry(Base):
    __tablename__ = 'travel_entries'
    id            = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    plan_id       = Column(String(10), nullable=False, index=True)
    day_number    = Column(Integer,    nullable=False, default=1)
    type          = Column(String(32), nullable=False, default='other')
    time_start    = Column(String(8))    # "09:00"
    time_end      = Column(String(8))    # "11:30"
    name          = Column(Text,        nullable=False)
    address       = Column(Text)
    notes         = Column(Text)
    display_order = Column(Integer,    default=0)
    created_at    = Column(DateTime,   default=datetime.utcnow)


def init_travel_db():
    """Create tables if they don't exist. Called once on app startup."""
    Base.metadata.create_all(engine)
    _migrate_columns()


def _migrate_columns():
    """Additive, idempotent column migrations — the pattern market_db uses.

    `create_all` only creates whole tables, so a column added after a table
    already exists has to be ALTERed in here. Safe to run on every boot.
    """
    from sqlalchemy import text as _text
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(_text('PRAGMA table_info(travel_plans)'))}
        if 'bill_id' not in cols:
            conn.execute(_text('ALTER TABLE travel_plans ADD COLUMN bill_id VARCHAR(10)'))


# ── Serialisers ───────────────────────────────────────────────────────────────

def _entry_dict(e):
    return {
        'id':            e.id,
        'plan_id':       e.plan_id,
        'day_number':    e.day_number,
        'type':          e.type,
        'time_start':    e.time_start,
        'time_end':      e.time_end,
        'name':          e.name,
        'address':       e.address,
        'notes':         e.notes,
        'display_order': e.display_order,
        'created_at':    e.created_at.isoformat() if e.created_at else None,
    }


def _plan_dict(p, entries=None, bill_id=None):
    return {
        'id':         p.id,
        'name':       p.name,
        'created_by': p.created_by,
        'num_days':   p.num_days,
        'created_at': p.created_at.isoformat() if p.created_at else None,
        'bill_id':    bill_id if bill_id is not None else getattr(p, 'bill_id', None),
        'entries':    [_entry_dict(e) for e in (entries or [])],
    }


# ── Plan helpers ──────────────────────────────────────────────────────────────

def create_plan(name, username):
    """Create a plan and the bill that will hold its shared expenses.

    Every plan gets a bill up front (id equal to the plan id). Giving them the
    same code means one string opens both the itinerary and the pot, and the
    Expenses tab is never "not set up yet".
    """
    from Backend.Controller import bill_db
    with Session() as s:
        pid = new_plan_id(s)
        plan = TravelPlan(id=pid, name=name, created_by=username, num_days=1, bill_id=pid)
        s.add(plan)
        s.commit()
        s.refresh(plan)
    # The bill lives in bill_db's table but on the same SQLite file.
    bill_db.create_bill_for_plan(pid, name, username)
    return get_plan(pid)


def set_plan_bill_id(plan_id, bill_id):
    with Session() as s:
        plan = s.get(TravelPlan, plan_id.upper())
        if not plan:
            return False
        plan.bill_id = bill_id
        s.commit()
        return True


def bill_id_for(plan_id):
    with Session() as s:
        plan = s.get(TravelPlan, plan_id.upper())
        return plan.bill_id if plan else None


def get_plan(plan_id):
    with Session() as s:
        plan = s.get(TravelPlan, plan_id.upper())
        if not plan:
            return None
        entries = (
            s.query(TravelEntry)
             .filter_by(plan_id=plan.id)
             .order_by(TravelEntry.day_number, TravelEntry.display_order, TravelEntry.time_start)
             .all()
        )
        return _plan_dict(plan, entries, bill_id=plan.bill_id)


def get_my_plans(username):
    with Session() as s:
        plans = (
            s.query(TravelPlan)
             .filter_by(created_by=username)
             .order_by(TravelPlan.created_at.desc())
             .all()
        )
        return [_plan_dict(p) for p in plans]


def update_plan(plan_id, name=None, num_days=None):
    with Session() as s:
        plan = s.get(TravelPlan, plan_id.upper())
        if not plan:
            return False
        if name is not None:
            plan.name = name
        if num_days is not None:
            plan.num_days = max(1, int(num_days))
        s.commit()
        return True


def delete_plan(plan_id, username):
    """Remove the itinerary and the bill that went with it.

    Horizon's call (2026-10-16): the two are one thing, so deleting the plan
    deletes the expenses too. The plan owner's word is final here — a
    collaborator editing entries still cannot delete the whole trip.
    """
    from Backend.Controller import bill_db
    with Session() as s:
        plan = s.get(TravelPlan, plan_id.upper())
        if not plan or plan.created_by != username:
            return False
        bill_id = plan.bill_id
        s.query(TravelEntry).filter_by(plan_id=plan_id.upper()).delete()
        s.delete(plan)
        s.commit()
    if bill_id:
        # delete_bill is owner-scoped; the plan creator made the bill, so the
        # same username passes. Force through anyway — the plan is gone and an
        # orphan bill would be unreachable.
        bill_db.delete_bill_force(bill_id)
    return True


# ── Entry helpers ─────────────────────────────────────────────────────────────

def add_entry(plan_id, day_number, type_, time_start, time_end, name, address, notes, display_order=None):
    with Session() as s:
        plan = s.get(TravelPlan, plan_id.upper())
        if not plan:
            return None
        # Auto-append to end of the day's list when no order is specified
        if display_order is None:
            from sqlalchemy import func as _func
            max_order = s.query(_func.max(TravelEntry.display_order)).filter_by(
                plan_id=plan_id.upper(), day_number=int(day_number)
            ).scalar()
            display_order = (max_order + 1) if max_order is not None else 0
        entry = TravelEntry(
            id=str(uuid.uuid4()),
            plan_id=plan_id.upper(),
            day_number=int(day_number),
            type=type_ or 'other',
            time_start=time_start or None,
            time_end=time_end or None,
            name=name,
            address=address or None,
            notes=notes or None,
            display_order=int(display_order),
        )
        s.add(entry)
        s.commit()
        s.refresh(entry)
        return _entry_dict(entry)


def update_entry(entry_id, plan_id, **kwargs):
    with Session() as s:
        entry = s.query(TravelEntry).filter_by(id=entry_id, plan_id=plan_id.upper()).first()
        if not entry:
            return None
        allowed = {'day_number', 'type', 'time_start', 'time_end', 'name', 'address', 'notes', 'display_order'}
        for k, v in kwargs.items():
            if k in allowed:
                # Store empty strings as None for optional fields
                if k in ('time_start', 'time_end', 'address', 'notes') and v == '':
                    setattr(entry, k, None)
                else:
                    setattr(entry, k, v)
        s.commit()
        s.refresh(entry)
        return _entry_dict(entry)


def reorder_entries(plan_id, orders):
    """Batch-update display_order for entries in a plan.
    orders: [{'id': '<uuid>', 'display_order': 0}, ...]
    """
    with Session() as s:
        plan = s.get(TravelPlan, plan_id.upper())
        if not plan:
            return False
        for item in orders:
            entry = s.query(TravelEntry).filter_by(
                id=item['id'], plan_id=plan_id.upper()
            ).first()
            if entry:
                entry.display_order = int(item['display_order'])
        s.commit()
        return True


def delete_entry(entry_id, plan_id):
    with Session() as s:
        entry = s.query(TravelEntry).filter_by(id=entry_id, plan_id=plan_id.upper()).first()
        if not entry:
            return False
        s.delete(entry)
        s.commit()
        return True
