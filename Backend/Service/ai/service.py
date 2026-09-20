"""AIService: the single door.

    result = ai.run(feature='tarot', user='zwy', role='vip',
                    quota_key=f'user:zwy:tarot:{ai.today_key()}',
                    payload={'spread': ..., 'question': ...})

Behind it, in order: the kill switch, the site-wide cap, the caller's quota,
the vendor call (retried once on timeout or 5xx), the parse, the usage row.
Nothing above this file knows which vendor answered or what it cost; nothing
below it knows who asked or why.

Business day is America/Chicago, decided here and only here. Every quota key
that carries a date gets it from today_key().
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from . import client as default_client
from . import features, quota
from .usage import AIUsage, UsageStore

log = logging.getLogger('ai')

BUSINESS_TZ = ZoneInfo('America/Chicago')

# Retried once: the vendor was slow or broken. Not retried: it told us no
# (rate limit), or we asked wrong (config). Both still get a usage row.
RETRY_KINDS = frozenset({'timeout', 'provider'})

# What the user sees. Never the vendor's words, never a stack trace.
USER_MESSAGES = {
    'disabled':   'AI 功能暂时关闭。',
    'quota':      '今天的解读次数用完了，明天再来。',
    'global_cap': '今日解读已约满，明天再来。',
    'timeout':    '解读超时了，请再试一次——这次不算次数。',
    'provider':   '解读服务暂时没有回应，请稍后再试——这次不算次数。',
    'rate_limit': '解读服务正忙，请稍后再试——这次不算次数。',
    'config':     'AI 功能还没配置好。',
}


@dataclass(frozen=True)
class AIResult:
    ok: bool
    text: str | None
    data: dict | None
    error: str | None
    error_kind: str | None
    usage: AIUsage | None
    quota: quota.QuotaState | None
    prompt_version: str | None

    def as_dict(self) -> dict:
        return {
            'ok': self.ok, 'text': self.text, 'data': self.data,
            'error': self.error, 'error_kind': self.error_kind,
            'usage': self.usage.as_dict() if self.usage else None,
            'quota': self.quota.as_dict() if self.quota else None,
            'prompt_version': self.prompt_version,
        }


def today_key() -> str:
    """The business date, for quota keys. The only place the timezone lives."""
    return datetime.now(BUSINESS_TZ).date().isoformat()


def enabled() -> bool:
    return os.environ.get('AI_ENABLED', '1').strip().lower() not in ('0', 'false', 'no', 'off')


class AIService:
    def __init__(self, *, chat=None, store: UsageStore | None = None, config=None):
        # Injectable so tests never touch the network or the real database.
        self._chat = chat or default_client.chat
        self._store = store or UsageStore()
        self._config = config

    def init(self) -> None:
        self._store.init()

    @property
    def store(self) -> UsageStore:
        return self._store

    def run(self, *, feature: str, user: str, role: str | None, quota_key: str | None, payload: dict) -> AIResult:
        if not feature:
            raise ValueError('ai.run() needs a feature name — it is how usage and cost are attributed')
        feat = features.get(feature)
        pv = feat.prompt_version

        if not enabled():
            return self._fail('disabled', feat, quota_state=None)

        try:
            state = quota.check(self._store, feature=feature, role=role, quota_key=quota_key)
        except quota.GlobalCapReached:
            return self._fail('global_cap', feat, quota_state=None)
        except quota.QuotaExceeded as e:
            return self._fail('quota', feat, quota_state=e.state)

        try:
            cfg = self._config or default_client.load_config()
        except default_client.ProviderError as e:
            log.error('ai config: %s', e)
            return self._fail('config', feat, quota_state=state)

        messages = feat.build(payload)
        attempts = 0
        last_err: default_client.ProviderError | None = None
        resp = None
        while attempts < 2 and resp is None:
            attempts += 1
            try:
                resp = self._chat(messages, model=cfg.model, max_tokens=feat.max_tokens,
                                  temperature=feat.temperature, json_mode=feat.json_mode, config=cfg)
            except default_client.ProviderError as e:
                last_err = e
                if e.kind not in RETRY_KINDS:
                    break

        if resp is None:
            kind = last_err.kind if last_err else 'provider'
            log.warning('ai %s failed (%s) after %d attempt(s): %s', feature, kind, attempts, last_err)
            usage = self._store.record(feature=feature, username=user, quota_key=quota_key, model=cfg.model,
                                       prompt_version=pv, attempts=attempts, ok=False, error_kind=kind)
            return AIResult(ok=False, text=None, data=None, error=USER_MESSAGES.get(kind, USER_MESSAGES['provider']),
                            error_kind=kind, usage=usage, quota=state, prompt_version=pv)

        data = feat.parse(resp.text)
        # The model answered and was paid for: that is a success and it counts,
        # even if it wandered off the schema. The caller still gets the text.
        usage = self._store.record(feature=feature, username=user, quota_key=quota_key, model=resp.model,
                                   prompt_version=pv, prompt_tokens=resp.prompt_tokens,
                                   completion_tokens=resp.completion_tokens, latency_ms=resp.latency_ms,
                                   attempts=attempts, ok=True, error_kind=None if data is not None else 'parse')
        if data is None:
            log.warning('ai %s: reply did not match schema (kept as text)', feature)
        return AIResult(ok=True, text=resp.text, data=data, error=None,
                        error_kind=None if data is not None else 'parse',
                        usage=usage, quota=quota.after(state), prompt_version=pv)

    @staticmethod
    def _fail(kind: str, feat: features.Feature, *, quota_state) -> AIResult:
        return AIResult(ok=False, text=None, data=None, error=USER_MESSAGES[kind], error_kind=kind,
                        usage=None, quota=quota_state, prompt_version=feat.prompt_version)
