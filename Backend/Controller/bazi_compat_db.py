"""bazi_compat.py / bazi_compat_db.py

`bazi_compat_readings`: 一行一次合盘（两张命盘的 id + 合盘算术 + AI 解读 +
reader 的验证）。`bazi_compat_followups`: 合盘后的追问。

同 bazi_db：delete 是 hide（Horizon 2026-10-12），数据保留分析用；pending
超过 PENDING_TTL 视为丢失，可重跑。
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, String, Text, create_engine, event, Index, text
from sqlalchemy.orm import sessionmaker, declarative_base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = '_data/market.db'

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()

PENDING_TTL = timedelta(minutes=4)


def _state(row) -> str:
    if row.status == 'pending' and row.started_at and _utcnow() - row.started_at < PENDING_TTL:
        return 'pending'
    if row.status == 'error':
        return 'error'
    return 'done' if row.done_at else 'none'


class BaziCompatReading(Base):
    __tablename__ = 'bazi_compat_readings'
    id             = Column(String(36), primary_key=True)
    username       = Column(String(64), nullable=False)
    chart_a_id     = Column(String(36), nullable=False)   # 本命 reading id
    chart_b_id     = Column(String(36), nullable=False)   # 对象 reading id
    chart_a_json   = Column(Text, nullable=False)         # 快照：当时的 chart
    chart_b_json   = Column(Text, nullable=False)
    compat_json    = Column(Text, nullable=False)         # compat.build_compat 输出
    status         = Column(String(16), nullable=True)
    started_at     = Column(DateTime, nullable=True)
    error_json     = Column(Text, nullable=True)
    reading_raw    = Column(Text, nullable=True)
    reading_json   = Column(Text, nullable=True)
    model          = Column(String(80), nullable=True)
    prompt_version = Column(String(40), nullable=True)
    verify_json    = Column(Text, nullable=True)
    created_at     = Column(DateTime, default=_utcnow)
    read_at        = Column(DateTime, nullable=True)
    deleted_at     = Column(DateTime, nullable=True)

    __table_args__ = (Index('idx_bazi_compat_user_created', 'username', 'created_at'),)

    @property
    def done_at(self):
        return self.read_at

    def verify(self) -> dict:
        return json.loads(self.verify_json) if self.verify_json else {}

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'chart_a': json.loads(self.chart_a_json),
            'chart_b': json.loads(self.chart_b_json),
            'chart_a_id': self.chart_a_id,
            'chart_b_id': self.chart_b_id,
            'compat': json.loads(self.compat_json),
            'state': _state(self),
            'reading': json.loads(self.reading_json) if self.reading_json else None,
            'reading_text': self.reading_raw if self.reading_raw and not self.reading_json else None,
            'error': json.loads(self.error_json) if self.status == 'error' and self.error_json else None,
            'verify': self.verify(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'read_at': self.read_at.isoformat() if self.read_at else None,
        }

    def summary(self) -> dict:
        a = json.loads(self.chart_a_json)
        b = json.loads(self.chart_b_json)
        c = json.loads(self.compat_json)
        r = json.loads(self.reading_json) if self.reading_json else None
        return {
            'id': self.id,
            'self_bazi':  ' '.join(p['gan'] + p['zhi'] for p in a['pillars']),
            'other_bazi': ' '.join(p['gan'] + p['zhi'] for p in b['pillars']),
            'self_birth':  a['solar_time']['clock'],
            'other_birth': b['solar_time']['clock'],
            'score': c.get('score'),
            'verdict': r['verdict'] if r else None,
            'state': _state(self),
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class BaziCompatFollowup(Base):
    __tablename__ = 'bazi_compat_followups'
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

    __table_args__ = (Index('idx_bazi_compat_fu_reading', 'reading_id', 'created_at'),)

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


def init_bazi_compat_db() -> None:
    Base.metadata.create_all(engine)


# ── Readings ──────────────────────────────────────────────────────────

def create(username: str, chart_a_id: str, chart_a: dict, chart_b_id: str, chart_b: dict, compat: dict) -> str:
    rid = uuid.uuid4().hex
    with Session() as s:
        s.add(BaziCompatReading(
            id=rid, username=username,
            chart_a_id=chart_a_id, chart_b_id=chart_b_id,
            chart_a_json=json.dumps(chart_a, ensure_ascii=False),
            chart_b_json=json.dumps(chart_b, ensure_ascii=False),
            compat_json=json.dumps(compat, ensure_ascii=False),
        ))
        s.commit()
    return rid


def get(rid: str, username: str) -> BaziCompatReading | None:
    with Session() as s:
        r = s.get(BaziCompatReading, rid)
        if r is None or r.username != username or r.deleted_at is not None:
            return None
        s.expunge(r)
        return r


def _mine(s, username: str):
    return (s.query(BaziCompatReading)
             .filter(BaziCompatReading.username == username, BaziCompatReading.deleted_at.is_(None))
             .order_by(BaziCompatReading.created_at.desc()))


def latest(username: str) -> BaziCompatReading | None:
    with Session() as s:
        r = _mine(s, username).first()
        if r is not None:
            s.expunge(r)
        return r


def list_mine(username: str, limit: int = 50) -> list[dict]:
    with Session() as s:
        return [r.summary() for r in _mine(s, username).limit(limit).all()]


def claim(rid: str) -> bool:
    """同 bazi_db.claim：原子锁，防止双击重复跑 AI。"""
    now = _utcnow()
    with Session() as s:
        n = (s.query(BaziCompatReading)
              .filter(BaziCompatReading.id == rid, BaziCompatReading.read_at.is_(None))
              .filter((BaziCompatReading.status.is_(None)) | (BaziCompatReading.status == 'error')
                      | (BaziCompatReading.started_at < now - PENDING_TTL))
              .update({'status': 'pending', 'started_at': now, 'error_json': None},
                      synchronize_session=False))
        s.commit()
        return n == 1


def finish(rid: str, *, raw: str, data: dict | None, model: str, prompt_version: str) -> None:
    with Session() as s:
        r = s.get(BaziCompatReading, rid)
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
        r = s.get(BaziCompatReading, rid)
        if r is None:
            return
        r.status = 'error'
        r.error_json = json.dumps(error, ensure_ascii=False)
        s.commit()


def set_verify(rid: str, username: str, marks: dict) -> bool:
    with Session() as s:
        r = s.get(BaziCompatReading, rid)
        if r is None or r.username != username or r.deleted_at is not None:
            return False
        r.verify_json = json.dumps(marks)
        s.commit()
        return True


def hide(rid: str, username: str) -> bool:
    with Session() as s:
        n = (s.query(BaziCompatReading)
              .filter(BaziCompatReading.id == rid, BaziCompatReading.username == username,
                      BaziCompatReading.deleted_at.is_(None))
              .update({'deleted_at': _utcnow()}, synchronize_session=False))
        s.commit()
        return n == 1


def hide_all(username: str) -> int:
    with Session() as s:
        n = (s.query(BaziCompatReading)
              .filter(BaziCompatReading.username == username, BaziCompatReading.deleted_at.is_(None))
              .update({'deleted_at': _utcnow()}, synchronize_session=False))
        s.commit()
        return n


# ── Follow-up ──────────────────────────────────────────────

def followups(rid: str) -> list[BaziCompatFollowup]:
    with Session() as s:
        rows = (s.query(BaziCompatFollowup).filter(BaziCompatFollowup.reading_id == rid)
                 .order_by(BaziCompatFollowup.created_at).all())
        for r in rows:
            s.expunge(r)
        return rows


def ask(rid: str, username: str, question: str) -> BaziCompatFollowup:
    fid = uuid.uuid4().hex
    with Session() as s:
        f = BaziCompatFollowup(id=fid, reading_id=rid, username=username, question=question,
                               status='pending', started_at=_utcnow())
        s.add(f); s.commit(); s.refresh(f); s.expunge(f)
        return f


def ask_screened(rid: str, username: str, question: str, reply: str, *, kind: str) -> BaziCompatFollowup:
    now = _utcnow()
    with Session() as s:
        f = BaziCompatFollowup(id=uuid.uuid4().hex, reading_id=rid, username=username, question=question,
                               answer=reply, model=f'screen:{kind}', started_at=now, answered_at=now)
        s.add(f); s.commit(); s.refresh(f); s.expunge(f)
        return f


def get_followup(fid: str, username: str) -> BaziCompatFollowup | None:
    with Session() as s:
        f = s.get(BaziCompatFollowup, fid)
        if f is None or f.username != username:
            return None
        r = s.get(BaziCompatReading, f.reading_id)
        if r is None or r.deleted_at is not None:
            return None
        s.expunge(f)
        return f


def answer(fid: str, *, text_: str, model: str, prompt_version: str) -> None:
    with Session() as s:
        f = s.get(BaziCompatFollowup, fid)
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
        f = s.get(BaziCompatFollowup, fid)
        if f is None:
            return
        f.status = 'error'
        f.error_json = json.dumps(error, ensure_ascii=False)
        s.commit()
