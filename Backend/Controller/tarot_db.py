"""
tarot_db.py
The `tarot_readings` table: every spread that was drawn, and — if the reader
asked — the question, the model's answer, and how well they said it fit.

This is business data and stays with the app; `ai_usage` (the AI layer's
accounting) is a different table with a different owner. Kept apart on
purpose: this one grows into history, sharing, and one day a training set.

One row per /draw, whether or not a reading is ever requested. The rows that
never get one are data too — how many people drew and did not ask.
"""

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, event, Index
from sqlalchemy.orm import sessionmaker, declarative_base


# Naive UTC, like every other DateTime column in market.db — but without the
# deprecated utcnow().
def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = '_data/market.db'   # reuse the shared SQLite file

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()


class TarotReading(Base):
    __tablename__ = 'tarot_readings'
    id             = Column(String(36), primary_key=True)
    username       = Column(String(64), nullable=False)
    spread_json    = Column(Text, nullable=False)        # [{position, card}] × 3, as drawn
    question       = Column(Text, nullable=True)
    reading_raw    = Column(Text, nullable=True)         # the model's reply, untouched
    reading_json   = Column(Text, nullable=True)         # parsed; NULL if it broke the schema
    model          = Column(String(80), nullable=True)
    prompt_version = Column(String(40), nullable=True)
    rating         = Column(Integer, nullable=True)      # 1–5
    rating_note    = Column(Text, nullable=True)
    rated_at       = Column(DateTime, nullable=True)
    created_at     = Column(DateTime, default=_utcnow)   # drawn
    read_at        = Column(DateTime, nullable=True)             # interpreted

    __table_args__ = (Index('idx_tarot_user_day', 'username', 'created_at'),)

    def to_dict(self, *, include_spread=True) -> dict:
        d = {
            'id': self.id,
            'question': self.question,
            'reading': json.loads(self.reading_json) if self.reading_json else None,
            'reading_text': self.reading_raw if self.reading_raw and not self.reading_json else None,
            'model': self.model,
            'prompt_version': self.prompt_version,
            'rating': self.rating,
            'rating_note': self.rating_note,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'read_at': self.read_at.isoformat() if self.read_at else None,
        }
        if include_spread:
            d['spread'] = json.loads(self.spread_json)
        return d


def init_tarot_db() -> None:
    Base.metadata.create_all(engine)


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_reading(username: str, spread: list) -> str:
    rid = uuid.uuid4().hex
    with Session() as s:
        s.add(TarotReading(id=rid, username=username, spread_json=json.dumps(spread, ensure_ascii=False)))
        s.commit()
    return rid


def get_reading(rid: str, username: str) -> TarotReading | None:
    """Only the owner's. A wrong username is the same as a wrong id."""
    with Session() as s:
        r = s.get(TarotReading, rid)
        if r is None or r.username != username:
            return None
        s.expunge(r)
        return r


def save_interpretation(rid: str, *, question: str | None, raw: str, data: dict | None,
                        model: str, prompt_version: str) -> None:
    with Session() as s:
        r = s.get(TarotReading, rid)
        if r is None:
            return
        r.question = question
        r.reading_raw = raw
        r.reading_json = json.dumps(data, ensure_ascii=False) if data is not None else None
        r.model = model
        r.prompt_version = prompt_version
        r.read_at = _utcnow()
        s.commit()


def save_rating(rid: str, username: str, rating: int, note: str | None) -> bool:
    with Session() as s:
        r = s.get(TarotReading, rid)
        if r is None or r.username != username or r.read_at is None:
            return False
        r.rating = rating
        r.rating_note = note
        r.rated_at = _utcnow()
        s.commit()
        return True


def list_readings(username: str, *, limit: int = 20, before: str | None = None) -> list[dict]:
    """The owner's interpreted readings, newest first. Draws that were never
    read are skipped — there is nothing to look back at."""
    with Session() as s:
        q = (s.query(TarotReading)
              .filter(TarotReading.username == username, TarotReading.read_at.isnot(None)))
        if before:
            q = q.filter(TarotReading.read_at < datetime.fromisoformat(before))
        rows = q.order_by(TarotReading.read_at.desc()).limit(limit).all()
        return [r.to_dict() for r in rows]
