"""
bazi_db.py
The `bazi_readings` table: one row per chart someone cast, and — if they asked —
the model's reading of it.

Birth date, time and place are personal data. They are kept only so the page
can show your chart again and the reading can be asked for later; only the
owner can read a row, and DELETE /api/bazi/mine removes every row of yours.

A reading takes up to two minutes, longer than nginx will hold a request open,
so it runs in the background: `status` goes NULL → 'pending' → NULL (done,
`read_at` set) or 'error' (`error_json` says why, and whether a retry is free).
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, String, Text, create_engine, event, Index
from sqlalchemy.orm import sessionmaker, declarative_base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = '_data/market.db'   # reuse the shared SQLite file

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()

# A pending reading older than this is treated as lost (the process restarted
# mid-call) and may be asked for again.
PENDING_TTL = timedelta(minutes=4)


class BaziReading(Base):
    __tablename__ = 'bazi_readings'
    id             = Column(String(36), primary_key=True)
    username       = Column(String(64), nullable=False)
    chart_json     = Column(Text, nullable=False)        # chart.build_chart output, input included
    status         = Column(String(16), nullable=True)   # NULL | 'pending' | 'error'
    started_at     = Column(DateTime, nullable=True)
    error_json     = Column(Text, nullable=True)
    reading_raw    = Column(Text, nullable=True)
    reading_json   = Column(Text, nullable=True)         # parsed; NULL if it broke the schema
    model          = Column(String(80), nullable=True)
    prompt_version = Column(String(40), nullable=True)
    created_at     = Column(DateTime, default=_utcnow)
    read_at        = Column(DateTime, nullable=True)

    __table_args__ = (Index('idx_bazi_user_created', 'username', 'created_at'),)

    def pending(self) -> bool:
        return self.status == 'pending' and self.started_at is not None \
            and _utcnow() - self.started_at < PENDING_TTL

    def to_dict(self) -> dict:
        d = {
            'id': self.id,
            'chart': json.loads(self.chart_json),
            'state': 'done' if self.read_at else 'pending' if self.pending()
                     else 'error' if self.status == 'error' else 'none',
            'reading': json.loads(self.reading_json) if self.reading_json else None,
            'reading_text': self.reading_raw if self.reading_raw and not self.reading_json else None,
            'error': json.loads(self.error_json) if self.status == 'error' and self.error_json else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'read_at': self.read_at.isoformat() if self.read_at else None,
        }
        return d


def init_bazi_db() -> None:
    Base.metadata.create_all(engine)


def create(username: str, chart: dict) -> str:
    rid = uuid.uuid4().hex
    with Session() as s:
        s.add(BaziReading(id=rid, username=username, chart_json=json.dumps(chart, ensure_ascii=False)))
        s.commit()
    return rid


def get(rid: str, username: str) -> BaziReading | None:
    """Only the owner's. A wrong username is the same as a wrong id."""
    with Session() as s:
        r = s.get(BaziReading, rid)
        if r is None or r.username != username:
            return None
        s.expunge(r)
        return r


def latest(username: str) -> BaziReading | None:
    with Session() as s:
        r = (s.query(BaziReading).filter(BaziReading.username == username)
              .order_by(BaziReading.created_at.desc()).first())
        if r is not None:
            s.expunge(r)
        return r


def claim(rid: str) -> bool:
    """Mark a reading pending, unless it is already done or already running.
    The check and the write are one UPDATE, so two clicks start one call."""
    now = _utcnow()
    with Session() as s:
        n = (s.query(BaziReading)
              .filter(BaziReading.id == rid, BaziReading.read_at.is_(None))
              .filter((BaziReading.status.is_(None)) | (BaziReading.status == 'error')
                      | (BaziReading.started_at < now - PENDING_TTL))
              .update({'status': 'pending', 'started_at': now, 'error_json': None},
                      synchronize_session=False))
        s.commit()
        return n == 1


def finish(rid: str, *, raw: str, data: dict | None, model: str, prompt_version: str) -> None:
    with Session() as s:
        r = s.get(BaziReading, rid)
        if r is None:
            return
        r.status = None
        r.reading_raw = raw
        r.reading_json = json.dumps(data, ensure_ascii=False) if data is not None else None
        r.model = model
        r.prompt_version = prompt_version
        r.read_at = _utcnow()
        s.commit()


def fail(rid: str, error: dict) -> None:
    with Session() as s:
        r = s.get(BaziReading, rid)
        if r is None:
            return
        r.status = 'error'
        r.error_json = json.dumps(error, ensure_ascii=False)
        s.commit()


def delete_all(username: str) -> int:
    with Session() as s:
        n = s.query(BaziReading).filter(BaziReading.username == username).delete()
        s.commit()
        return n
