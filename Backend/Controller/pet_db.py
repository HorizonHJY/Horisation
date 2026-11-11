"""
pet_db.py
电子猫 (the virtual cat). One cat per person, ten breeds, three meters, three
chores a day, and a visitor's pat. Two tables:

  pets              — one row per owner. breed is fixed at adoption; name can
                      change. hunger / mood / bond are 0–100 and are stored as
                      of `last_decay_day`: every read first rolls the meters
                      forward by the days elapsed (DECAY per day each), so
                      there is no scheduler and a cat that nobody opened for a
                      month costs nothing until someone looks.
  pet_interactions  — one row per (owner, actor, kind, day). UNIQUE on all
                      four is the whole once-a-day story: feed / play / brush
                      by the owner, `touch` by a visitor. The second insert
                      hits the constraint and nothing is paid.

Rules the code keeps, by decision (PRODUCT.md 2026-11-11):
  - the cat never dies, never loses a stage, never resets. Neglect only
    lowers the meters; care brings them back.
  - stage is read off bond alone: 幼崽 < 40, 成年 40–79, 老友 ≥ 80. Because
    bond only decays to 0 and never below, and the stage is recomputed,
    a cat can in principle step back from 老友 to 成年 after a long absence;
    the floor rule below stops that: the highest stage reached is remembered
    and shown — the meter may drop, the title does not.
  - XP: each chore pays +2 through xp_db (`pet_care`, 3/day), a visitor's
    touch pays the visitor +1 (`pet_touch`, 3/day). Both ride the existing
    (username, kind, ref) idempotency.

Days are America/Chicago business days, the same clock as the daily card.
"""

from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Integer, String, UniqueConstraint, create_engine, event,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

from Backend.Service.ai.service import today_key


# ── The fixed tables ─────────────────────────────────────────────────────────
# slug -> (中文, English). The slug is what the row stores and the SVG keys on.
BREEDS = {
    'blue':    ('蓝猫',   'British Blue'),
    'calico':  ('三花',   'Calico'),
    'golden':  ('金渐层', 'Golden Shaded'),
    'silver':  ('银渐层', 'Silver Shaded'),
    'orange':  ('橘猫',   'Orange Tabby'),
    'tuxedo':  ('奶牛猫', 'Tuxedo'),
    'black':   ('黑猫',   'Black Cat'),
    'tabby':   ('狸花',   'Chinese Li Hua'),
    'ragdoll': ('布偶',   'Ragdoll'),
    'siamese': ('暹罗',   'Siamese'),
}
BREED_ORDER = list(BREEDS)

MAX_NAME_CHARS = 12

# action -> (meter, gain). One of each per day.
CARE = {
    'feed':  ('hunger', 30),
    'play':  ('mood',   25),
    'brush': ('bond',   12),
}
TOUCH_BOND = 1            # a visitor's pat, to the cat
# Per calendar day of absence. Hunger goes first, the bond last.
DECAY = {'hunger': 12, 'mood': 10, 'bond': 8}
START = {'hunger': 70, 'mood': 70, 'bond': 20}

# (stage slug, 中文, English, min bond)
STAGES = [
    ('kitten', '幼崽', 'Kitten', 0),
    ('adult',  '成年', 'Grown',  40),
    ('elder',  '老友', 'Old friend', 80),
]
STAGE_INDEX = {s[0]: i for i, s in enumerate(STAGES)}

# XP rule names, see xp_db.RULES
XP_CARE  = 'pet_care'
XP_TOUCH = 'pet_touch'


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = '_data/market.db'   # the shared SQLite file

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Pet(Base):
    __tablename__ = 'pets'
    username       = Column(String(100), primary_key=True)
    breed          = Column(String(16),  nullable=False)
    name           = Column(String(40),  nullable=False)
    hunger         = Column(Integer,     nullable=False, default=START['hunger'])
    mood           = Column(Integer,     nullable=False, default=START['mood'])
    bond           = Column(Integer,     nullable=False, default=START['bond'])
    # Highest stage index ever reached; the title never goes back down.
    stage_max      = Column(Integer,     nullable=False, default=0)
    last_decay_day = Column(String(10),  nullable=False)   # YYYY-MM-DD, Chicago
    created_at     = Column(DateTime,    nullable=False, default=_utcnow)
    updated_at     = Column(DateTime,    nullable=False, default=_utcnow, onupdate=_utcnow)


class PetInteraction(Base):
    __tablename__ = 'pet_interactions'
    id         = Column(Integer, primary_key=True, autoincrement=True)
    owner      = Column(String(100), nullable=False, index=True)
    actor      = Column(String(100), nullable=False, index=True)
    kind       = Column(String(16),  nullable=False)   # feed | play | brush | touch
    day        = Column(String(10),  nullable=False)
    created_at = Column(DateTime,    nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint('owner', 'actor', 'kind', 'day', name='uq_pet_interaction'),
    )


def init_pet_db():
    """Create the two tables if missing. Additive only; nothing else is touched."""
    Base.metadata.create_all(engine)


# ── Pure helpers ─────────────────────────────────────────────────────────────

def _clamp(v) -> int:
    return max(0, min(100, int(v)))


def stage_for(bond: int) -> str:
    cur = STAGES[0][0]
    for slug, _zh, _en, floor in STAGES:
        if bond >= floor:
            cur = slug
    return cur


def _stage_dict(idx: int) -> dict:
    slug, zh, en, floor = STAGES[idx]
    return {'stage': slug, 'stage_zh': zh, 'stage_en': en}


def _days_between(a: str, b: str) -> int:
    try:
        return max(0, (date.fromisoformat(b) - date.fromisoformat(a)).days)
    except ValueError:
        return 0


def apply_decay(meters: dict, days: int) -> dict:
    """Pure: roll three meters forward by `days` of absence."""
    if days <= 0:
        return dict(meters)
    return {k: _clamp(meters[k] - DECAY[k] * days) for k in DECAY}


def clean_name(name) -> Optional[str]:
    name = ' '.join(str(name or '').split())
    if not name or len(name) > MAX_NAME_CHARS:
        return None
    return name


def mood_line(p: dict) -> str:
    """One sentence for the profile card, from the lowest meter."""
    h, m, b = p['hunger'], p['mood'], p['bond']
    if h < 30:
        return '肚子饿了，想吃饭'
    if m < 30:
        return '有点蔫，想玩一会儿'
    if b < 30:
        return '还在认生，多梳梳毛'
    if min(h, m, b) >= 80:
        return '吃饱了，正在打呼噜'
    return '今天状态不错'


# ── Reads ────────────────────────────────────────────────────────────────────

def _settle(s, row: Pet, day: str) -> None:
    """Roll the row forward to `day`. Writes only when a day has passed."""
    days = _days_between(row.last_decay_day, day)
    if days <= 0:
        return
    m = apply_decay({'hunger': row.hunger, 'mood': row.mood, 'bond': row.bond}, days)
    row.hunger, row.mood, row.bond = m['hunger'], m['mood'], m['bond']
    row.last_decay_day = day
    s.commit()


def _row_dict(row: Pet) -> dict:
    stage_idx = max(int(row.stage_max or 0), STAGE_INDEX[stage_for(row.bond)])
    zh, en = BREEDS.get(row.breed, (row.breed, row.breed))
    out = {
        'breed': row.breed, 'breed_zh': zh, 'breed_en': en,
        'name': row.name,
        'hunger': int(row.hunger), 'mood': int(row.mood), 'bond': int(row.bond),
        'created_at': row.created_at.isoformat() if row.created_at else None,
    }
    out.update(_stage_dict(stage_idx))
    out['line'] = mood_line(out)
    return out


def _done_today(s, owner: str, actor: str, day: str) -> set[str]:
    rows = s.query(PetInteraction.kind).filter(
        PetInteraction.owner == owner, PetInteraction.actor == actor,
        PetInteraction.day == day).all()
    return {r[0] for r in rows}


def get_pet(username: str, day: Optional[str] = None) -> Optional[dict]:
    """The owner's view: the cat plus what has been done today. None if no cat."""
    day = day or today_key()
    with Session() as s:
        row = s.get(Pet, username)
        if row is None:
            return None
        _settle(s, row, day)
        out = _row_dict(row)
        done = _done_today(s, username, username, day)
        out['today'] = {k: (k in done) for k in CARE}
        out['day'] = day
        return out


def public_card(owner: str, viewer: Optional[str] = None, day: Optional[str] = None) -> Optional[dict]:
    """What a public profile shows: breed, name, stage, bond — and whether this
    viewer has already touched it today. None if the person has no cat."""
    day = day or today_key()
    with Session() as s:
        row = s.get(Pet, owner)
        if row is None:
            return None
        _settle(s, row, day)
        d = _row_dict(row)
        out = {k: d[k] for k in ('breed', 'breed_zh', 'breed_en', 'name', 'stage', 'stage_zh', 'stage_en', 'bond')}
        out['touched_today'] = bool(viewer) and 'touch' in _done_today(s, owner, viewer, day)
        return out


# ── Writes ───────────────────────────────────────────────────────────────────

def adopt(username: str, breed: str, name: str, day: Optional[str] = None) -> tuple[Optional[dict], Optional[str]]:
    """(pet, None) on success; (None, error) otherwise. One cat per person, ever."""
    day = day or today_key()
    if breed not in BREEDS:
        return None, 'Pick one of the ten breeds.'
    name = clean_name(name)
    if not name:
        return None, f'Name is 1 to {MAX_NAME_CHARS} characters.'
    with Session() as s:
        if s.get(Pet, username) is not None:
            return None, 'You already have a cat.'
        s.add(Pet(username=username, breed=breed, name=name, last_decay_day=day, **START))
        try:
            s.commit()
        except IntegrityError:
            s.rollback()
            return None, 'You already have a cat.'
    return get_pet(username, day), None


def rename(username: str, name: str) -> tuple[Optional[dict], Optional[str]]:
    name = clean_name(name)
    if not name:
        return None, f'Name is 1 to {MAX_NAME_CHARS} characters.'
    with Session() as s:
        row = s.get(Pet, username)
        if row is None:
            return None, 'No cat yet.'
        row.name = name
        s.commit()
    return get_pet(username), None


def _record(s, owner: str, actor: str, kind: str, day: str) -> bool:
    s.add(PetInteraction(owner=owner, actor=actor, kind=kind, day=day))
    try:
        s.flush()
        return True
    except IntegrityError:
        s.rollback()
        return False


def care(username: str, action: str, day: Optional[str] = None) -> tuple[Optional[dict], Optional[str], bool]:
    """One chore. Returns (pet, error, applied). `applied` is False when the
    chore was already done today — the meters do not move again, and the
    caller must not pay XP for it."""
    day = day or today_key()
    if action not in CARE:
        return None, 'Unknown action.', False
    meter, gain = CARE[action]
    with Session() as s:
        row = s.get(Pet, username)
        if row is None:
            return None, 'No cat yet.', False
        _settle(s, row, day)
        if not _record(s, username, username, action, day):
            return get_pet(username, day), None, False
        setattr(row, meter, _clamp(getattr(row, meter) + gain))
        row.stage_max = max(int(row.stage_max or 0), STAGE_INDEX[stage_for(row.bond)])
        s.commit()
    return get_pet(username, day), None, True


def touch(owner: str, actor: str, day: Optional[str] = None) -> tuple[Optional[dict], Optional[str], bool]:
    """A visitor's pat: +1 bond to the cat, once per visitor per day.
    Returns (public card, error, applied)."""
    day = day or today_key()
    if owner == actor:
        return None, 'You cannot pet your own cat here — go to /pet.', False
    with Session() as s:
        row = s.get(Pet, owner)
        if row is None:
            return None, 'This person has no cat.', False
        _settle(s, row, day)
        if not _record(s, owner, actor, 'touch', day):
            return public_card(owner, actor, day), None, False
        row.bond = _clamp(row.bond + TOUCH_BOND)
        row.stage_max = max(int(row.stage_max or 0), STAGE_INDEX[stage_for(row.bond)])
        s.commit()
    return public_card(owner, actor, day), None, True
