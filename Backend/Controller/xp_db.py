"""
xp_db.py
经验值 + 等级. Two tables:

  user_xp    — one row per person, the running total. The level is never
               stored: it is computed from `xp` against LEVELS, so a change
               to the table below re-levels everyone on the next request.
  xp_events  — one row per grant. UNIQUE(username, kind, ref) is the whole
               idempotency story: the same action (same ref) can hit award()
               as many times as it likes and only the first insert counts.

What earns what (daily routine caps at 41; trades and reviews are on top):

  daily_sign   +5   ref = YYYY-MM-DD           once a day   (今日一签)
  listing      +10  ref = listing id           3 a day
  deal         +30  ref = listing id           once per item sold, both parties
  review       +10  ref = deal id              once per deal, reviewer only
  wall         +2   ref = wall post id         3 a day
  streak7      +20  ref = YYYY-MM-DD           when the draw streak hits 7, 14, 21 …

The streak itself is read off `tarot_daily` (one row per person per day they
drew) — nothing is duplicated here, so a day that was drawn before this
module existed still counts.

Days are America/Chicago business days, the same clock as the daily card.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Integer, String, UniqueConstraint, create_engine, event,
    func, text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

from Backend.Service.ai.service import BUSINESS_TZ, today_key


# ── Level table (定稿) ───────────────────────────────────────────────────────
# (level, name, xp needed to reach it). Ascending; the first row is 0.
LEVELS = [
    (1, '新村民',   0),
    (2, '老村民',   100),
    (3, '村民代表', 450),
    (4, '村委委员', 1500),
    (5, '村支书',   4000),
    (6, '村长',     30000),
]

# The hidden tag: an account that has been around a full year.
NATIVE_TAG   = '土著'
NATIVE_DAYS  = 365

# Rules: kind -> (delta, per-day cap or None)
RULES = {
    'daily_sign': (5,  1),
    'listing':    (10, 3),
    'deal':       (30, None),
    'review':     (10, None),
    'wall':       (2,  3),
    'streak7':    (20, None),
}
STREAK_BONUS_EVERY = 7


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = '_data/market.db'   # the shared SQLite file

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()


class UserXP(Base):
    __tablename__ = 'user_xp'
    username   = Column(String(100), primary_key=True)
    xp         = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class XPEvent(Base):
    __tablename__ = 'xp_events'
    id         = Column(Integer, primary_key=True, autoincrement=True)
    username   = Column(String(100), nullable=False, index=True)
    kind       = Column(String(32),  nullable=False)
    ref        = Column(String(64),  nullable=False)
    delta      = Column(Integer,     nullable=False)
    created_at = Column(DateTime,    nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint('username', 'kind', 'ref', name='uq_xp_event'),
    )


def init_xp_db():
    """Create the two tables if missing. Additive only; nothing else is touched."""
    Base.metadata.create_all(engine)


# ── Level math ───────────────────────────────────────────────────────────────

def level_for(xp: int) -> dict:
    """Everything the UI needs about a total: level, name, next step, progress."""
    xp = max(0, int(xp or 0))
    cur = LEVELS[0]
    nxt = None
    for i, row in enumerate(LEVELS):
        if xp >= row[2]:
            cur = row
            nxt = LEVELS[i + 1] if i + 1 < len(LEVELS) else None
        else:
            break
    if nxt is None:
        progress = 1.0
    else:
        span = nxt[2] - cur[2]
        progress = (xp - cur[2]) / span if span > 0 else 1.0
        progress = max(0.0, min(1.0, progress))
    return {
        'xp':              xp,
        'level':           cur[0],
        'level_name':      cur[1],
        'level_xp':        cur[2],
        'next_level_name': nxt[1] if nxt else None,
        'next_xp':         nxt[2] if nxt else None,
        'progress':        round(progress, 4),
    }


def hidden_tag_for(created_at) -> Optional[str]:
    """'土著' for an account older than a year; None otherwise.
    `created_at` is the user's created_at as stored (naive UTC) or its ISO string."""
    if not created_at:
        return None
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            return None
    if created_at.tzinfo is not None:
        created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
    if _utcnow() - created_at >= timedelta(days=NATIVE_DAYS):
        return NATIVE_TAG
    return None


# ── Reads ────────────────────────────────────────────────────────────────────

def get_xp(username: str) -> int:
    with Session() as s:
        row = s.get(UserXP, username)
        return int(row.xp) if row else 0


def _day_bounds_utc(day: str) -> tuple[datetime, datetime]:
    """Naive-UTC [start, end) of one Chicago business day."""
    d = datetime.fromisoformat(day).replace(tzinfo=BUSINESS_TZ)
    start = d.astimezone(timezone.utc).replace(tzinfo=None)
    return start, start + timedelta(days=1)


def count_today(username: str, kind: str, day: Optional[str] = None) -> int:
    day = day or today_key()
    start, end = _day_bounds_utc(day)
    with Session() as s:
        return s.query(func.count(XPEvent.id)).filter(
            XPEvent.username == username,
            XPEvent.kind == kind,
            XPEvent.created_at >= start,
            XPEvent.created_at < end,
        ).scalar() or 0


def draw_days(username: str, since_day: str) -> set[str]:
    """Days (YYYY-MM-DD) on or after `since_day` the person drew a daily card."""
    with Session() as s:
        rows = s.execute(
            text('SELECT day FROM tarot_daily WHERE username = :u AND day >= :d'),
            {'u': username, 'd': since_day},
        ).fetchall()
        return {r[0] for r in rows}


def streak_info(username: str, day: Optional[str] = None) -> dict:
    """Consecutive draw days ending today (or yesterday, if today is not drawn
    yet — the streak is alive until midnight). Also the last-7-days dots."""
    day = day or today_key()
    today = datetime.fromisoformat(day).date()
    # 400 days is plenty; a streak longer than that would already be unusual.
    since = (today - timedelta(days=400)).isoformat()
    days = draw_days(username, since)

    cursor = today
    if today.isoformat() not in days:
        cursor = today - timedelta(days=1)
    streak = 0
    while cursor.isoformat() in days:
        streak += 1
        cursor -= timedelta(days=1)

    week = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        week.append({'day': d.isoformat(), 'drawn': d.isoformat() in days})
    return {'streak_days': streak, 'week': week, 'today_drawn': today.isoformat() in days}


def summary(username: str, created_at=None) -> dict:
    """The /api/auth/xp payload."""
    out = level_for(get_xp(username))
    out.update(streak_info(username))
    out['hidden_tag'] = hidden_tag_for(created_at)
    return out


def public_summary(username: str, created_at=None) -> dict:
    """What a public profile gets: level, name, hidden tag. No xp number."""
    lv = level_for(get_xp(username))
    return {
        'level':      lv['level'],
        'level_name': lv['level_name'],
        'hidden_tag': hidden_tag_for(created_at),
    }


# ── Writes ───────────────────────────────────────────────────────────────────

def award(username: str, kind: str, ref: str, *, day: Optional[str] = None) -> bool:
    """Grant the points for one action. Returns True if points were added,
    False if this (username, kind, ref) was already paid or the day cap is hit.

    Idempotent by construction: the unique index refuses a second insert for
    the same ref, so a retried request, a double click or a replayed job can
    never pay twice. The cap check is a plain count; the one race it leaves
    (two different refs landing in the same instant at the cap) is worth at
    most one extra grant and is not worth a table lock.
    """
    if kind not in RULES or not username or not ref:
        return False
    delta, cap = RULES[kind]
    ref = str(ref)[:64]

    if cap is not None and count_today(username, kind, day) >= cap:
        return False

    with Session() as s:
        s.add(XPEvent(username=username, kind=kind, ref=ref, delta=delta))
        try:
            s.flush()
        except IntegrityError:
            s.rollback()
            return False
        row = s.get(UserXP, username)
        if row is None:
            s.add(UserXP(username=username, xp=delta))
        else:
            row.xp = int(row.xp or 0) + delta
        s.commit()
        return True


def award_daily_sign(username: str, day: Optional[str] = None) -> dict:
    """Called when the daily card is shown. Pays +5 for the day (once) and,
    when the streak lands on a multiple of 7, the +20 bonus (once per day)."""
    day = day or today_key()
    signed = award(username, 'daily_sign', day, day=day)
    bonus = False
    info = streak_info(username, day)
    if info['today_drawn'] and info['streak_days'] > 0 \
            and info['streak_days'] % STREAK_BONUS_EVERY == 0:
        bonus = award(username, 'streak7', day, day=day)
    return {'signed': signed, 'bonus': bonus, 'streak_days': info['streak_days']}


def recompute(username: str) -> int:
    """Rebuild the total from the event log. Not used on the request path;
    here so the stored number is never the only copy of the truth."""
    with Session() as s:
        total = s.query(func.coalesce(func.sum(XPEvent.delta), 0)).filter(
            XPEvent.username == username).scalar() or 0
        row = s.get(UserXP, username)
        if row is None:
            s.add(UserXP(username=username, xp=int(total)))
        else:
            row.xp = int(total)
        s.commit()
        return int(total)
