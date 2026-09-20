"""Who may call how often.

The AI layer counts; the caller decides what a count is scoped to by the
quota_key it passes (`user:zwy:tarot:2026-09-20` → one bucket per person per
feature per business day). The limits live in one dict here, per feature,
per role, with a default — and a per-feature daily cap for the whole site,
which is the thing that protects the bill when someone finds a loop.

`None` as a limit means unlimited. Unlimited is still logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


# Naive UTC, like every other DateTime column in market.db — but without the
# deprecated utcnow().
def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# feature → role → calls per quota_key. '*' is the default.
LIMITS: dict[str, dict[str, int | None]] = {
    'tarot': {
        '*':       1,
        'vip':     3,
        'svip':    3,
        'admin':   None,
        'horizon': None,
    },
}

# feature → ok calls per site per rolling 24h. The safety net, not a product
# rule: 11 people cannot legitimately get near it.
GLOBAL_DAILY_CAP: dict[str, int] = {
    'tarot': 100,
}


@dataclass(frozen=True)
class QuotaState:
    used: int
    limit: int | None
    remaining: int | None            # None when unlimited
    reset_at: str | None             # ISO date the bucket rolls over, if the key carries one

    def as_dict(self) -> dict:
        return {'used': self.used, 'limit': self.limit, 'remaining': self.remaining, 'reset_at': self.reset_at}


class QuotaExceeded(Exception):
    def __init__(self, state: QuotaState):
        super().__init__('quota exceeded')
        self.state = state


class GlobalCapReached(Exception):
    pass


def limit_for(feature: str, role: str | None) -> int | None:
    table = LIMITS.get(feature, {'*': None})
    return table.get(role or '*', table.get('*'))


def _reset_from_key(quota_key: str | None) -> str | None:
    # Keys end in the business date when they are per-day; that date's end is
    # the reset. Anything else we do not pretend to know.
    if not quota_key:
        return None
    tail = quota_key.rsplit(':', 1)[-1]
    try:
        datetime.strptime(tail, '%Y-%m-%d')
    except ValueError:
        return None
    return tail


def check(store, *, feature: str, role: str | None, quota_key: str | None) -> QuotaState:
    """Raise if this call may not proceed. Returns the state as it stands
    BEFORE the call, so a caller can show 'x of y left' honestly."""
    cap = GLOBAL_DAILY_CAP.get(feature)
    if cap is not None:
        since = _utcnow() - timedelta(hours=24)
        if store.count_ok_since(feature, since) >= cap:
            raise GlobalCapReached()

    limit = limit_for(feature, role)
    used = store.count_ok(quota_key) if quota_key else 0
    state = QuotaState(
        used=used, limit=limit,
        remaining=None if limit is None else max(0, limit - used),
        reset_at=_reset_from_key(quota_key),
    )
    if quota_key and limit is not None and used >= limit:
        raise QuotaExceeded(state)
    return state


def after(state: QuotaState) -> QuotaState:
    """The state once one more call has succeeded."""
    return QuotaState(
        used=state.used + 1, limit=state.limit,
        remaining=None if state.limit is None else max(0, state.limit - state.used - 1),
        reset_at=state.reset_at,
    )
