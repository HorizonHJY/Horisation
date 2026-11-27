"""
bill_db.py
SQLAlchemy models and helpers for the Bill Split feature.
Bills are identified by a short human-friendly ID (e.g. "ABC123").
Anyone with the ID can view and edit - designed for small-group collaboration.
"""

import uuid
import json
import random
from datetime import datetime

from sqlalchemy import Column, Float, Integer, String, Text, DateTime, UniqueConstraint, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine

DB_PATH = '_data/market.db'

engine = create_engine(
    f'sqlite:///{DB_PATH}',
    echo=False,
    connect_args={'check_same_thread': False},
)
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()

_ID_CHARS = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'


def _gen_bill_id():
    return ''.join(random.choices(_ID_CHARS, k=6))


# ── Models ────────────────────────────────────────────────────────────────────

class Bill(Base):
    __tablename__ = 'bill_splits'
    id         = Column(String(10), primary_key=True)
    name       = Column(Text,       nullable=False)
    created_by = Column(String(64), nullable=False)
    created_at = Column(DateTime,   default=datetime.utcnow)


class BillMember(Base):
    """A bill someone joined from a share link. The creator lives on
    bills.created_by; this table only holds the people who joined."""
    __tablename__ = 'bill_members'
    id       = Column(Integer, primary_key=True, autoincrement=True)
    bill_id  = Column(String(10), nullable=False, index=True)
    username = Column(String(64), nullable=False, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('bill_id', 'username', name='uq_bill_member'),
    )


class BillParticipant(Base):
    __tablename__ = 'bill_participants'
    id      = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    bill_id = Column(String(10), nullable=False, index=True)
    name    = Column(String(64), nullable=False)


class BillExpense(Base):
    __tablename__ = 'bill_expenses'
    id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    bill_id      = Column(String(10), nullable=False, index=True)
    desc         = Column(Text,       nullable=False)
    amount       = Column(Float,      nullable=False)
    paid_by      = Column(String(64), nullable=False)
    split_among  = Column(Text,       nullable=False)  # JSON array
    created_at   = Column(DateTime,   default=datetime.utcnow)


def init_bill_db():
    Base.metadata.create_all(engine)


# ── Serialisers ───────────────────────────────────────────────────────────────

def _expense_dict(e):
    return {
        'id':          e.id,
        'bill_id':     e.bill_id,
        'desc':        e.desc,
        'amount':      e.amount,
        'paidBy':      e.paid_by,
        'splitAmong':  json.loads(e.split_among),
        'created_at':  e.created_at.isoformat() if e.created_at else None,
    }


def _bill_dict(b, participants=None, expenses=None):
    return {
        'id':           b.id,
        'name':         b.name,
        'created_by':   b.created_by,
        'created_at':   b.created_at.isoformat() if b.created_at else None,
        'participants': [p.name for p in (participants or [])],
        'expenses':     [_expense_dict(e) for e in (expenses or [])],
    }


# ── Bill helpers ──────────────────────────────────────────────────────────────

def create_bill(name, username):
    with Session() as s:
        for _ in range(10):
            bid = _gen_bill_id()
            if not s.get(Bill, bid):
                break
        bill = Bill(id=bid, name=name, created_by=username)
        s.add(bill)
        s.commit()
        s.refresh(bill)
        return _bill_dict(bill)


def create_bill_for_plan(plan_id, name, username):
    """The bill behind a travel plan's Expenses tab.

    Same id as the plan, so the id already in the page's URL opens both. The
    creator is a participant from the start (Horizon's call) — they are on the
    trip by definition, and an empty participants list would block the first
    expense. Idempotent: if the bill is already there, only the founder is
    ensured as a participant.
    """
    with Session() as s:
        bill = s.get(Bill, plan_id.upper())
        if not bill:
            bill = Bill(id=plan_id.upper(), name=f'{name} · 费用', created_by=username)
            s.add(bill)
            s.commit()
    add_participant(plan_id, username)
    return get_bill(plan_id)


def bill_members(bill_id):
    """Accounts that can open this bill: the creator plus everyone who joined
    from a share link. Display names come from the user table."""
    from sqlalchemy import text
    with Session() as s:
        bill = s.get(Bill, bill_id.upper())
        if not bill:
            return []
        owner = s.execute(
            text('select coalesce(display_name, username) from user where username = :u'),
            {'u': bill.created_by}).fetchone()
        out = [{'username': bill.created_by,
                'display_name': (owner[0] if owner else bill.created_by),
                'is_owner': True}]
        rows = s.execute(
            text('select bm.username, coalesce(u.display_name, bm.username) '
                 'from bill_members bm left join user u on u.username = bm.username '
                 'where bm.bill_id = :b order by bm.added_at'),
            {'b': bill.id}).fetchall()
        for username, display in rows:
            out.append({'username': username, 'display_name': display or username, 'is_owner': False})
        return out


def get_bill(bill_id):
    with Session() as s:
        bill = s.get(Bill, bill_id.upper())
        if not bill:
            return None
        participants = s.query(BillParticipant).filter_by(bill_id=bill.id).all()
        expenses     = (
            s.query(BillExpense)
             .filter_by(bill_id=bill.id)
             .order_by(BillExpense.created_at)
             .all()
        )
        out = _bill_dict(bill, participants, expenses)
        out['members'] = bill_members(bill.id)
        return out


def get_my_bills(username):
    """Owned bills plus the ones joined from a share link."""
    with Session() as s:
        bills = s.query(Bill).filter_by(created_by=username).all()
        joined = {bid for (bid,) in
                  s.query(BillMember.bill_id).filter_by(username=username).all()}
        if joined:
            bills += s.query(Bill).filter(Bill.id.in_(joined)).all()
        seen, result = set(), []
        for b in sorted(bills, key=lambda x: x.created_at or datetime.min, reverse=True):
            if b.id in seen:
                continue
            seen.add(b.id)
            participants = s.query(BillParticipant).filter_by(bill_id=b.id).all()
            expenses     = s.query(BillExpense).filter_by(bill_id=b.id).all()
            out = _bill_dict(b, participants, expenses)
            out['members'] = bill_members(b.id)
            result.append(out)
        return result


def add_bill_member(bill_id, username):
    """Join a shared bill. Idempotent; the creator is not written here."""
    with Session() as s:
        bill = s.get(Bill, bill_id.upper())
        if not bill:
            return None
        if bill.created_by != username:
            exists = (s.query(BillMember)
                      .filter_by(bill_id=bill.id, username=username).first())
            if not exists:
                s.add(BillMember(bill_id=bill.id, username=username))
                s.commit()
        return get_bill(bill.id)


def update_bill_name(bill_id, name):
    with Session() as s:
        bill = s.get(Bill, bill_id.upper())
        if not bill:
            return False
        bill.name = name
        s.commit()
        return True


def delete_bill(bill_id, username):
    with Session() as s:
        bill = s.get(Bill, bill_id.upper())
        if not bill or bill.created_by != username:
            return False
        s.query(BillParticipant).filter_by(bill_id=bill_id.upper()).delete()
        s.query(BillExpense).filter_by(bill_id=bill_id.upper()).delete()
        s.delete(bill)
        s.commit()
        return True


def delete_bill_force(bill_id):
    """Delete a bill regardless of who owns it.

    Only for the travel-plan path, where the plan's own delete has already run
    and an orphaned bill would be unreachable. Never call this from a route
    that takes a bare bill id from the client.
    """
    with Session() as s:
        bill = s.get(Bill, bill_id.upper())
        if not bill:
            return False
        s.query(BillParticipant).filter_by(bill_id=bill_id.upper()).delete()
        s.query(BillExpense).filter_by(bill_id=bill_id.upper()).delete()
        s.delete(bill)
        s.commit()
        return True


# ── Participant helpers ───────────────────────────────────────────────────────

def add_participant(bill_id, name):
    with Session() as s:
        bill = s.get(Bill, bill_id.upper())
        if not bill:
            return False
        existing = s.query(BillParticipant).filter_by(bill_id=bill_id.upper(), name=name).first()
        if existing:
            return True  # already there
        p = BillParticipant(bill_id=bill_id.upper(), name=name)
        s.add(p)
        s.commit()
        return True


def remove_participant(bill_id, name):
    with Session() as s:
        s.query(BillParticipant).filter_by(bill_id=bill_id.upper(), name=name).delete()
        s.commit()
        return True


# ── Expense helpers ───────────────────────────────────────────────────────────

def add_expense(bill_id, desc, amount, paid_by, split_among):
    with Session() as s:
        bill = s.get(Bill, bill_id.upper())
        if not bill:
            return None
        e = BillExpense(
            bill_id=bill_id.upper(),
            desc=desc,
            amount=float(amount),
            paid_by=paid_by,
            split_among=json.dumps(split_among),
        )
        s.add(e)
        s.commit()
        s.refresh(e)
        return _expense_dict(e)


def update_expense(bill_id, expense_id, desc, amount, paid_by, split_among):
    with Session() as s:
        e = s.query(BillExpense).filter_by(id=expense_id, bill_id=bill_id.upper()).first()
        if not e:
            return None
        e.desc = desc
        e.amount = float(amount)
        e.paid_by = paid_by
        e.split_among = json.dumps(split_among)
        s.commit()
        s.refresh(e)
        return _expense_dict(e)


def delete_expense(bill_id, expense_id):
    with Session() as s:
        e = s.query(BillExpense).filter_by(id=expense_id, bill_id=bill_id.upper()).first()
        if not e:
            return False
        s.delete(e)
        s.commit()
        return True
