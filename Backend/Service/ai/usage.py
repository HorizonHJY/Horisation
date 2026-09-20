"""The `ai_usage` table: every call, succeeded or not, and what it cost.

This table belongs to the AI layer, not to the app — if the layer is ever
split into its own service the table goes with it. That is why it has its
own engine here rather than importing market_db's, and why the engine can be
swapped for tests.

Success and failure are both rows. Quota counts only ok=1 (see quota.py), so
a timeout never costs a user their turn — but it is still on record, or you
would never know the failure rate or what was burned for nothing.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, create_engine, event, func
from sqlalchemy.orm import declarative_base, sessionmaker


# Naive UTC, like every other DateTime column in market.db — but without the
# deprecated utcnow().
def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DB_PATH = os.path.join(_BASE_DIR, '_data', 'market.db')

Base = declarative_base()


class AIUsageRow(Base):
    __tablename__ = 'ai_usage'
    id                = Column(String(36), primary_key=True)
    feature           = Column(String(40), nullable=False)
    username          = Column(String(64), nullable=False)
    quota_key         = Column(String(120), nullable=True)
    model             = Column(String(80), nullable=False)
    prompt_version    = Column(String(40), nullable=True)
    prompt_tokens     = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    cost_usd          = Column(Float, default=0.0)
    latency_ms        = Column(Integer, nullable=True)
    attempts          = Column(Integer, default=1)
    ok                = Column(Integer, nullable=False)
    error_kind        = Column(String(20), nullable=True)
    created_at        = Column(DateTime, default=_utcnow)

    __table_args__ = (
        Index('idx_ai_usage_quota', 'quota_key', 'ok'),
        Index('idx_ai_usage_feature_day', 'feature', 'created_at'),
    )


# USD per million tokens. Checked 2026-09-20 against the vendors' pricing
# pages; when this drifts, cost_usd on old rows stays as it was computed —
# it is a record of what we believed at the time, not a live figure.
PRICES_PER_M = {
    'deepseek-chat':               (0.27, 1.10),
    'deepseek-reasoner':           (0.55, 2.19),
    'claude-haiku-4-5-20251001':   (1.00, 5.00),
    'claude-sonnet-5':             (3.00, 15.00),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICES_PER_M.get(model, (0.0, 0.0))
    return round(prompt_tokens / 1e6 * inp + completion_tokens / 1e6 * out, 6)


@dataclass(frozen=True)
class AIUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int | None
    attempts: int

    def as_dict(self) -> dict:
        return {
            'model': self.model, 'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens, 'cost_usd': self.cost_usd,
            'latency_ms': self.latency_ms, 'attempts': self.attempts,
        }


class UsageStore:
    """Writes and counts. Constructed once by AIService; tests hand it an
    in-memory engine."""

    def __init__(self, engine=None):
        if engine is None:
            engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
            event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
        self.engine = engine
        self.Session = sessionmaker(bind=engine)

    def init(self) -> None:
        Base.metadata.create_all(self.engine)

    def record(self, *, feature, username, quota_key, model, prompt_version,
               prompt_tokens=0, completion_tokens=0, latency_ms=None, attempts=1,
               ok: bool, error_kind: str | None = None) -> AIUsage:
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        with self.Session() as s:
            s.add(AIUsageRow(
                id=str(uuid.uuid4()), feature=feature, username=username, quota_key=quota_key,
                model=model, prompt_version=prompt_version,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                cost_usd=cost, latency_ms=latency_ms, attempts=attempts,
                ok=1 if ok else 0, error_kind=error_kind,
            ))
            s.commit()
        return AIUsage(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                       cost_usd=cost, latency_ms=latency_ms, attempts=attempts)

    def count_ok(self, quota_key: str) -> int:
        with self.Session() as s:
            return s.query(func.count(AIUsageRow.id)).filter(
                AIUsageRow.quota_key == quota_key, AIUsageRow.ok == 1).scalar() or 0

    def count_ok_since(self, feature: str, since: datetime) -> int:
        with self.Session() as s:
            return s.query(func.count(AIUsageRow.id)).filter(
                AIUsageRow.feature == feature, AIUsageRow.ok == 1,
                AIUsageRow.created_at >= since).scalar() or 0
