"""
tod_db.py
Truth or Dare: the question bank, who drew what, and the 1–5 star feedback.

  tod_draws    — one row per (username, prompt, day). Proves a prompt was
                 actually dealt before it can be rated.
  tod_ratings  — one row per (username, prompt). The first rating counts;
                 repeats update the score but never pay XP twice.

The bank itself is a JSON file next to the code (Backend/data/tod_prompts.json)
so the content is reviewable in a diff. Ratings live in the DB and are
aggregated on read; a prompt that averages below HIDE_AVG with at least
HIDE_MIN ratings drops out of the deal pool.
"""

import json
import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Integer, String, UniqueConstraint, create_engine,
    event, func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH  = os.path.join(BASE_DIR, '_data', 'market.db')
BANK_PATH = os.path.join(BASE_DIR, 'Backend', 'data', 'tod_prompts.json')

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()

LEVELS = ('ice', 'normal', 'adult')
KINDS  = ('truth', 'dare')

# A prompt this weak, with enough votes, stops being dealt.
HIDE_AVG = 2.5
HIDE_MIN = 5


class TodDraw(Base):
    __tablename__ = 'tod_draws'
    id         = Column(Integer, primary_key=True, autoincrement=True)
    username   = Column(String(100), nullable=False, index=True)
    prompt_id  = Column(String(40),  nullable=False, index=True)
    day        = Column(String(10),  nullable=False)
    created_at = Column(DateTime,    nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint('username', 'prompt_id', 'day', name='uq_tod_draw'),
    )


class TodRating(Base):
    __tablename__ = 'tod_ratings'
    id         = Column(Integer, primary_key=True, autoincrement=True)
    username   = Column(String(100), nullable=False, index=True)
    prompt_id  = Column(String(40),  nullable=False, index=True)
    rating     = Column(Integer,     nullable=False)
    tags       = Column(String(200), nullable=True)
    created_at = Column(DateTime,    nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint('username', 'prompt_id', name='uq_tod_rating'),
    )


def init_tod_db():
    Base.metadata.create_all(engine)


# ── Bank ─────────────────────────────────────────────────────────────────────

_bank_cache = None


def load_bank() -> list[dict]:
    """Read the JSON once per process. Missing or broken file means no prompts,
    never a crash on the game page."""
    global _bank_cache
    if _bank_cache is not None:
        return _bank_cache
    try:
        with open(BANK_PATH, encoding='utf-8') as f:
            data = json.load(f)
        rows = data.get('prompts') or []
    except (OSError, ValueError):
        rows = []
    _bank_cache = [r for r in rows if r.get('id') and r.get('kind') in KINDS
                   and r.get('level') in LEVELS and r.get('text')]
    return _bank_cache


def reload_bank():
    global _bank_cache
    _bank_cache = None
    return load_bank()


def _stats_map() -> dict[str, dict]:
    with Session() as s:
        rows = (s.query(TodRating.prompt_id, func.avg(TodRating.rating), func.count(TodRating.id))
                .group_by(TodRating.prompt_id).all())
        return {pid: {'avg': round(float(a or 0), 2), 'count': int(c or 0)}
                for pid, a, c in rows}


def hidden_ids() -> set[str]:
    return {pid for pid, st in _stats_map().items()
            if st['count'] >= HIDE_MIN and st['avg'] < HIDE_AVG}


def prompts_for(level: str | None) -> list[dict]:
    """Prompts the deal pool may use. `level` None means every level."""
    hidden = hidden_ids()
    return [p for p in load_bank()
            if p['id'] not in hidden and (level is None or p['level'] == level)]


def get_prompt(prompt_id: str) -> dict | None:
    for p in load_bank():
        if p['id'] == prompt_id:
            return p
    return None


# ── Draws ────────────────────────────────────────────────────────────────────

def record_draw(username: str, prompt_id: str, day: str) -> bool:
    """Remember that this prompt reached this person today. Idempotent."""
    with Session() as s:
        s.add(TodDraw(username=username, prompt_id=prompt_id, day=day))
        try:
            s.commit()
            return True
        except Exception:
            s.rollback()
            return False


def has_drawn(username: str, prompt_id: str) -> bool:
    with Session() as s:
        return s.query(TodDraw.id).filter(
            TodDraw.username == username, TodDraw.prompt_id == prompt_id).first() is not None


# ── Ratings ──────────────────────────────────────────────────────────────────

def my_ratings(username: str) -> dict[str, int]:
    with Session() as s:
        return {pid: int(r) for pid, r in
                s.query(TodRating.prompt_id, TodRating.rating)
                .filter(TodRating.username == username).all()}


def rate(username: str, prompt_id: str, rating: int, tags: list[str] | None = None) -> bool:
    """Store or update one rating. Returns True only when this is a new rating,
    which is what the caller uses to decide whether XP is due."""
    tag_text = ','.join(tags or [])[:200] or None
    with Session() as s:
        row = s.query(TodRating).filter_by(username=username, prompt_id=prompt_id).first()
        if row:
            row.rating = rating
            row.tags = tag_text
            s.commit()
            return False
        s.add(TodRating(username=username, prompt_id=prompt_id, rating=rating, tags=tag_text))
        s.commit()
        return True


def stats() -> dict:
    """Admin view: every prompt with its average, plus the ones already hidden."""
    sm = _stats_map()
    out = []
    for p in load_bank():
        st = sm.get(p['id'], {'avg': 0, 'count': 0})
        out.append({**p, **st, 'hidden': st['count'] >= HIDE_MIN and st['avg'] < HIDE_AVG})
    out.sort(key=lambda r: (r['hidden'] is False, r['avg']))
    return {'prompts': out, 'hide_avg': HIDE_AVG, 'hide_min': HIDE_MIN}
