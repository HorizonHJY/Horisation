"""
bazi_db.py
`bazi_readings`: one row per chart someone cast, its reading, and how the
reader marked the reading's checks. `bazi_followups`: questions asked about a
chart afterwards, and their answers.

Deleting is a hide, not a delete (Horizon, 2026-10-12): the reader's "delete"
sets `deleted_at`, the row drops out of everything they can see, and the data
stays for analysing and improving the readings. The page says so.

A reading or an answer can take longer than nginx holds a request, so both
run in the background: `status` goes NULL → 'pending' → NULL (done) or
'error' (`error_json` says why, and whether a retry is free).
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, String, Text, create_engine, event, Index, text
from sqlalchemy.orm import sessionmaker, declarative_base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = '_data/market.db'   # reuse the shared SQLite file

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()

# A pending job older than this is treated as lost (the process restarted
# mid-call) and may be asked for again.
PENDING_TTL = timedelta(minutes=4)


def _state(row) -> str:
    if row.status == 'pending' and row.started_at and _utcnow() - row.started_at < PENDING_TTL:
        return 'pending'
    if row.status == 'error':
        return 'error'
    return 'done' if row.done_at else 'none'


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
    verify_json    = Column(Text, nullable=True)         # {"0": "yes", "2": "no"} — the reader's marks
    created_at     = Column(DateTime, default=_utcnow)
    read_at        = Column(DateTime, nullable=True)
    deleted_at     = Column(DateTime, nullable=True)     # hidden from the reader; kept for analysis

    __table_args__ = (Index('idx_bazi_user_created', 'username', 'created_at'),)

    @property
    def done_at(self):
        return self.read_at

    def verify(self) -> dict:
        return json.loads(self.verify_json) if self.verify_json else {}

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'chart': json.loads(self.chart_json),
            'state': _state(self),
            'reading': json.loads(self.reading_json) if self.reading_json else None,
            'reading_text': self.reading_raw if self.reading_raw and not self.reading_json else None,
            'error': json.loads(self.error_json) if self.status == 'error' and self.error_json else None,
            'verify': self.verify(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'read_at': self.read_at.isoformat() if self.read_at else None,
        }

    def summary(self) -> dict:
        """One line of the history list."""
        c = json.loads(self.chart_json)
        r = json.loads(self.reading_json) if self.reading_json else None
        return {
            'id': self.id,
            'bazi': ' '.join(p['gan'] + p['zhi'] for p in c['pillars']),
            'gender_label': c['gender_label'],
            'birth': c['solar_time']['clock'],
            'place': c['input']['place'],
            'verdict': r['verdict'] if r else None,
            'state': _state(self),
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class BaziFollowup(Base):
    __tablename__ = 'bazi_followups'
    id             = Column(String(36), primary_key=True)
    reading_id     = Column(String(36), nullable=False)
    username       = Column(String(64), nullable=False)
    question       = Column(Text, nullable=False)
    status         = Column(String(16), nullable=True)
    started_at     = Column(DateTime, nullable=True)
    error_json     = Column(Text, nullable=True)
    answer         = Column(Text, nullable=True)
    model          = Column(String(80), nullable=True)
    prompt_version = Column(String(40), nullable=True)
    created_at     = Column(DateTime, default=_utcnow)
    answered_at    = Column(DateTime, nullable=True)

    __table_args__ = (Index('idx_bazi_fu_reading', 'reading_id', 'created_at'),)

    @property
    def done_at(self):
        return self.answered_at

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'state': _state(self),
            'error': json.loads(self.error_json) if self.status == 'error' and self.error_json else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


def init_bazi_db() -> None:
    Base.metadata.create_all(engine)
    # create_all does not add columns to a table that already exists.
    with engine.begin() as c:
        cols = {row[1] for row in c.execute(text('PRAGMA table_info(bazi_readings)'))}
        for col in ('verify_json TEXT', 'deleted_at DATETIME'):
            if col.split()[0] not in cols:
                c.execute(text(f'ALTER TABLE bazi_readings ADD COLUMN {col}'))


# ── Readings ──────────────────────────────────────────────────────────

def create(username: str, chart: dict) -> str:
    rid = uuid.uuid4().hex
    with Session() as s:
        s.add(BaziReading(id=rid, username=username, chart_json=json.dumps(chart, ensure_ascii=False)))
        s.commit()
    return rid


def get(rid: str, username: str) -> BaziReading | None:
    """Only the owner's, and only if they have not deleted it. A wrong
    username is the same as a wrong id."""
    with Session() as s:
        r = s.get(BaziReading, rid)
        if r is None or r.username != username or r.deleted_at is not None:
            return None
        s.expunge(r)
        return r


def _mine(s, username: str):
    return (s.query(BaziReading)
             .filter(BaziReading.username == username, BaziReading.deleted_at.is_(None))
             .order_by(BaziReading.created_at.desc()))


def latest(username: str) -> BaziReading | None:
    with Session() as s:
        r = _mine(s, username).first()
        if r is not None:
            s.expunge(r)
        return r


def list_mine(username: str, limit: int = 50) -> list[dict]:
    with Session() as s:
        return [r.summary() for r in _mine(s, username).limit(limit).all()]


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


def set_verify(rid: str, username: str, marks: dict) -> bool:
    with Session() as s:
        r = s.get(BaziReading, rid)
        if r is None or r.username != username or r.deleted_at is not None:
            return False
        r.verify_json = json.dumps(marks)
        s.commit()
        return True


def hide(rid: str, username: str) -> bool:
    with Session() as s:
        n = (s.query(BaziReading)
              .filter(BaziReading.id == rid, BaziReading.username == username, BaziReading.deleted_at.is_(None))
              .update({'deleted_at': _utcnow()}, synchronize_session=False))
        s.commit()
        return n == 1


def hide_all(username: str) -> int:
    with Session() as s:
        n = (s.query(BaziReading)
              .filter(BaziReading.username == username, BaziReading.deleted_at.is_(None))
              .update({'deleted_at': _utcnow()}, synchronize_session=False))
        s.commit()
        return n


# ── Follow-up questions ──────────────────────────────────────────────

def followups(rid: str) -> list[BaziFollowup]:
    with Session() as s:
        rows = (s.query(BaziFollowup).filter(BaziFollowup.reading_id == rid)
                 .order_by(BaziFollowup.created_at).all())
        for r in rows:
            s.expunge(r)
        return rows


def ask(rid: str, username: str, question: str) -> BaziFollowup:
    """A new question, already marked pending — the caller starts the job."""
    fid = uuid.uuid4().hex
    with Session() as s:
        f = BaziFollowup(id=fid, reading_id=rid, username=username, question=question,
                         status='pending', started_at=_utcnow())
        s.add(f)
        s.commit()
        s.refresh(f)
        s.expunge(f)
        return f


def ask_screened(rid: str, username: str, question: str, reply: str, *, kind: str) -> BaziFollowup:
    """A question answered by the scope screen, not the model — recorded all
    the same (model='screen:<kind>'), so the data shows what people ask."""
    now = _utcnow()
    with Session() as s:
        f = BaziFollowup(id=uuid.uuid4().hex, reading_id=rid, username=username, question=question,
                         answer=reply, model=f'screen:{kind}', started_at=now, answered_at=now)
        s.add(f)
        s.commit()
        s.refresh(f)
        s.expunge(f)
        return f


def get_followup(fid: str, username: str) -> BaziFollowup | None:
    with Session() as s:
        f = s.get(BaziFollowup, fid)
        if f is None or f.username != username:
            return None
        r = s.get(BaziReading, f.reading_id)
        if r is None or r.deleted_at is not None:
            return None
        s.expunge(f)
        return f


def answer(fid: str, *, text_: str, model: str, prompt_version: str) -> None:
    with Session() as s:
        f = s.get(BaziFollowup, fid)
        if f is None:
            return
        f.status = None
        f.answer = text_
        f.model = model
        f.prompt_version = prompt_version
        f.answered_at = _utcnow()
        s.commit()


def fail_followup(fid: str, error: dict) -> None:
    with Session() as s:
        f = s.get(BaziFollowup, fid)
        if f is None:
            return
        f.status = 'error'
        f.error_json = json.dumps(error, ensure_ascii=False)
        s.commit()
