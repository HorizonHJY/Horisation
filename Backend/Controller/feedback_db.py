"""Where the reader says whether a reading landed.

One row per (feature, reading, person): a thumbs verdict plus optional reason
chips. Not the same thing as tarot's 1-5 stars — those score the reading, this
answers "was it right". The aggregated tallies are fed back into the prompt as
a short correction block (Backend/Service/ai/feedback.py), so a recurring
complaint changes the next reading rather than sitting in a table.
"""
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, String, Text, UniqueConstraint,
                        create_engine, event)
from sqlalchemy.orm import declarative_base, sessionmaker


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = '_data/market.db'   # same shared SQLite file as the other tables
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()

FEATURES = ('tarot', 'bazi')
VERDICTS = ('up', 'down')
MAX_REASONS = 6
MAX_REASON_CHARS = 40
MAX_NOTE_CHARS = 200


class ReadingFeedback(Base):
    __tablename__ = 'ai_reading_feedback'
    __table_args__ = (UniqueConstraint('feature', 'reading_id', 'username', name='uq_feedback_once'),)

    id             = Column(String(36), primary_key=True)
    feature        = Column(String(16), nullable=False)
    reading_id     = Column(String(36), nullable=False)
    username       = Column(String(64), nullable=False)
    verdict        = Column(String(8), nullable=False)    # 'up' | 'down'
    reasons_json   = Column(Text, nullable=True)          # ["太笼统", ...]
    note           = Column(Text, nullable=True)          # one line, optional
    prompt_version = Column(String(40), nullable=True)    # snapshot of the reading
    model          = Column(String(80), nullable=True)
    created_at     = Column(DateTime, default=_utcnow)
    updated_at     = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'feature': self.feature,
            'reading_id': self.reading_id,
            'verdict': self.verdict,
            'reasons': json.loads(self.reasons_json) if self.reasons_json else [],
            'note': self.note,
            'prompt_version': self.prompt_version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


def init_feedback_db() -> None:
    Base.metadata.create_all(engine)


def upsert(*, feature: str, reading_id: str, username: str, verdict: str,
           reasons: list[str] | None, note: str | None,
           prompt_version: str | None = None, model: str | None = None) -> dict:
    """One vote per reading, changed rather than duplicated."""
    reasons = [r[:MAX_REASON_CHARS] for r in (reasons or []) if str(r).strip()][:MAX_REASONS]
    note = (note or '').strip()[:MAX_NOTE_CHARS] or None
    with Session() as s:
        row = (s.query(ReadingFeedback)
                .filter(ReadingFeedback.feature == feature,
                        ReadingFeedback.reading_id == reading_id,
                        ReadingFeedback.username == username)
                .one_or_none())
        if row is None:
            row = ReadingFeedback(id=uuid.uuid4().hex, feature=feature,
                                  reading_id=reading_id, username=username)
            s.add(row)
        row.verdict = verdict
        row.reasons_json = json.dumps(reasons, ensure_ascii=False) if reasons else None
        row.note = note
        if prompt_version:
            row.prompt_version = prompt_version
        if model:
            row.model = model
        s.commit()
        s.refresh(row)
        return row.to_dict()


def get_mine(feature: str, reading_id: str, username: str) -> dict | None:
    with Session() as s:
        row = (s.query(ReadingFeedback)
                .filter(ReadingFeedback.feature == feature,
                        ReadingFeedback.reading_id == reading_id,
                        ReadingFeedback.username == username)
                .one_or_none())
        return row.to_dict() if row else None


def recent_negative(feature: str, limit: int = 200) -> list[dict]:
    """Down votes, newest first. Only the chips — never the free-text note.

    The note can name someone's health, job or family, and the aggregated
    lessons are shown to the model while it writes for a different person.
    Reason chips are ours and carry no personal detail.
    """
    with Session() as s:
        rows = (s.query(ReadingFeedback)
                .filter(ReadingFeedback.feature == feature,
                        ReadingFeedback.verdict == 'down')
                .order_by(ReadingFeedback.updated_at.desc())
                .limit(limit)
                .all())
        return [{'reasons': json.loads(r.reasons_json) if r.reasons_json else [],
                 'prompt_version': r.prompt_version} for r in rows]


def stats(feature: str) -> dict:
    with Session() as s:
        rows = (s.query(ReadingFeedback)
                .filter(ReadingFeedback.feature == feature).all())
        up = sum(1 for r in rows if r.verdict == 'up')
        down = sum(1 for r in rows if r.verdict == 'down')
        tally: dict[str, int] = {}
        for r in rows:
            if r.verdict != 'down':
                continue
            for reason in (json.loads(r.reasons_json) if r.reasons_json else []):
                tally[reason] = tally.get(reason, 0) + 1
        return {'up': up, 'down': down, 'total': up + down,
                'top_reasons': sorted(tally.items(), key=lambda kv: -kv[1])[:8]}

