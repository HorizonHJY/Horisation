"""mbti_db.py — MBTI test results.

One row per completed test. A user may re-take; the latest row (by taken_at)
is their "current" result. Earlier rows stay for the history view.

Visibility is per-row because someone may re-take under a new mood and want
only the latest to be public, with history private.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime, String, Text, Integer, Index, create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = '_data/market.db'

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False,
                       connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()


class MBTIResult(Base):
    __tablename__ = 'mbti_results'

    id          = Column(String(36),  primary_key=True)
    username    = Column(String(100), nullable=False, index=True)
    type_       = Column('type', String(4), nullable=False)
    counts_json = Column(Text, nullable=False)   # {"E":12,"I":8,...}
    mode        = Column(String(10), nullable=False)   # 'short' | 'full'
    visibility  = Column(String(10), nullable=False, default='friends')  # private | friends | public
    taken_at    = Column(DateTime, nullable=False, default=_utcnow, index=True)

    __table_args__ = (Index('idx_mbti_user_taken', 'username', 'taken_at'),)

    def to_dict(self) -> dict:
        return {
            'id':         self.id,
            'username':   self.username,
            'type':       self.type_,
            'counts':     json.loads(self.counts_json),
            'mode':       self.mode,
            'visibility': self.visibility,
            'taken_at':   self.taken_at.isoformat() if self.taken_at else None,
        }


def init_mbti_db() -> None:
    Base.metadata.create_all(engine)


def record(username: str, type_: str, counts: dict, mode: str,
           visibility: str = 'friends') -> dict:
    row = MBTIResult(
        id=uuid.uuid4().hex, username=username, type_=type_,
        counts_json=json.dumps(counts), mode=mode, visibility=visibility,
    )
    with Session() as s:
        s.add(row); s.commit(); s.refresh(row)
        return row.to_dict()


def latest(username: str) -> Optional[dict]:
    """Caller's most recent result (private included — this is their own)."""
    with Session() as s:
        row = (s.query(MBTIResult)
                 .filter_by(username=username)
                 .order_by(MBTIResult.taken_at.desc())
                 .first())
        return row.to_dict() if row else None


def latest_visible(username: str, viewer: Optional[str], is_friend: bool) -> Optional[dict]:
    """Someone else looking at `username`'s result — honour visibility.
    `viewer` may be None (guest). public → everyone; friends → viewer must be
    a friend or the owner; private → owner only."""
    with Session() as s:
        row = (s.query(MBTIResult)
                 .filter_by(username=username)
                 .order_by(MBTIResult.taken_at.desc())
                 .first())
        if row is None:
            return None
        v = row.visibility
        if v == 'public':                           return row.to_dict()
        if v == 'friends' and (viewer == username or is_friend): return row.to_dict()
        if v == 'private' and viewer == username:   return row.to_dict()
        return None


def history(username: str, limit: int = 20) -> list[dict]:
    with Session() as s:
        rows = (s.query(MBTIResult)
                 .filter_by(username=username)
                 .order_by(MBTIResult.taken_at.desc())
                 .limit(limit).all())
        return [r.to_dict() for r in rows]


def set_latest_visibility(username: str, visibility: str) -> bool:
    """Change visibility on the user's latest row."""
    if visibility not in ('private', 'friends', 'public'):
        return False
    with Session() as s:
        row = (s.query(MBTIResult)
                 .filter_by(username=username)
                 .order_by(MBTIResult.taken_at.desc())
                 .first())
        if row is None:
            return False
        row.visibility = visibility
        s.commit(); return True


def wipe(username: str) -> int:
    """Delete every result for the caller — resets as if never tested."""
    with Session() as s:
        n = s.query(MBTIResult).filter_by(username=username).delete()
        s.commit(); return n
