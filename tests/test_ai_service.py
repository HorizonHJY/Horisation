"""The AI layer, with the vendor faked and the database in memory.

Run:  python -m pytest tests/test_ai_service.py -q
Needs only sqlalchemy + requests + pytest; never imports app.py.
"""

import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.Service.ai import service as svc               # noqa: E402
from Backend.Service.ai.client import ChatResponse, ProviderConfig, ProviderError   # noqa: E402
from Backend.Service.ai.prompts import tarot as tp          # noqa: E402
from Backend.Service.ai.usage import UsageStore              # noqa: E402

CFG = ProviderConfig(provider='deepseek', api_key='test', model='deepseek-chat')

SPREAD = [
    {'position': {'key': 'past', 'label': 'Past', 'label_zh': '过去'},
     'card': {'id': 'm08', 'name': 'Strength', 'name_zh': '力量', 'keywords_zh': '柔中带刚', 'keywords': 'Power', 'description': 'A woman closes the jaws of a lion. The chain of flowers. More text.'}},
    {'position': {'key': 'present', 'label': 'Present', 'label_zh': '现在'},
     'card': {'id': 'c13', 'name': 'Queen of Cups', 'name_zh': '圣杯王后', 'keywords_zh': '同理心', 'keywords': 'Good, fair woman', 'description': ''}},
    {'position': {'key': 'future', 'label': 'Future', 'label_zh': '未来'},
     'card': {'id': 'm21', 'name': 'The World', 'name_zh': '世界', 'keywords_zh': '圆满', 'keywords': 'Assured success', 'description': 'Dance.'}},
]

GOOD_JSON = json.dumps({
    'past': '过去那张…', 'present': '现在…', 'future': '未来…', 'summary': '整体…', 'next_step': '明天给她发条消息。',
}, ensure_ascii=False)


def reply(text, *, p=300, c=200, model='deepseek-chat'):
    return ChatResponse(text=text, prompt_tokens=p, completion_tokens=c, model=model, latency_ms=1234)


class FakeChat:
    """Scripted vendor: each call pops the next item — a ChatResponse to return
    or a ProviderError to raise. Records what it was asked."""
    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def __call__(self, messages, **kw):
        self.calls.append({'messages': messages, **kw})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def store():
    s = UsageStore(engine=create_engine('sqlite:///:memory:'))
    s.init()
    return s


def make(store, *script):
    fake = FakeChat(*script)
    return svc.AIService(chat=fake, store=store, config=CFG), fake


def run(ai, **over):
    args = dict(feature='tarot', user='zwy', role='user', quota_key='user:zwy:tarot:2026-09-20',
                payload={'spread': SPREAD, 'question': '我该换工作吗'})
    args.update(over)
    return ai.run(**args)


# ── the door ──────────────────────────────────────────────────────────────

def test_feature_name_is_mandatory(store):
    ai, _ = make(store, reply(GOOD_JSON))
    with pytest.raises(ValueError):
        ai.run(feature='', user='u', role='user', quota_key=None, payload={})
    with pytest.raises(KeyError):
        ai.run(feature='nope', user='u', role='user', quota_key=None, payload={})


def test_happy_path_parses_and_records(store):
    ai, fake = make(store, reply(GOOD_JSON))
    r = run(ai)
    assert r.ok and r.data['next_step'] == '明天给她发条消息。' and r.error_kind is None
    assert r.usage.attempts == 1 and r.usage.cost_usd > 0
    assert r.quota.used == 1 and r.quota.remaining == 0 and r.quota.reset_at == '2026-09-20'
    assert r.prompt_version == tp.PROMPT_VERSION
    assert store.count_ok('user:zwy:tarot:2026-09-20') == 1
    # what the vendor was asked: json mode, the spread, the question, not the whole deck
    call = fake.calls[0]
    assert call['json_mode'] is True
    user_msg = call['messages'][-1]['content']
    assert '我该换工作吗' in user_msg and 'Strength' in user_msg and '力量' in user_msg
    assert 'The Magician' not in user_msg


# ── quota ─────────────────────────────────────────────────────────────────

def test_quota_blocks_second_call_for_plain_user(store):
    ai, fake = make(store, reply(GOOD_JSON), reply(GOOD_JSON))
    assert run(ai).ok
    r = run(ai)
    assert not r.ok and r.error_kind == 'quota' and r.quota.remaining == 0
    assert len(fake.calls) == 1, 'the vendor must not be called when quota is exhausted'


def test_vip_gets_three_and_horizon_unlimited(store):
    ai, _ = make(store, *[reply(GOOD_JSON)] * 10)
    for i in range(3):
        assert run(ai, user='v', role='vip', quota_key=f'user:v:tarot:2026-09-20').ok
    assert run(ai, user='v', role='vip', quota_key='user:v:tarot:2026-09-20').error_kind == 'quota'
    for i in range(5):
        r = run(ai, user='h', role='horizon', quota_key=f'user:h:tarot:2026-09-20')
        assert r.ok and r.quota.limit is None and r.quota.remaining is None
    assert store.count_ok('user:h:tarot:2026-09-20') == 5, 'unlimited is still logged'


def test_new_day_new_bucket(store):
    ai, _ = make(store, reply(GOOD_JSON), reply(GOOD_JSON))
    assert run(ai, quota_key='user:zwy:tarot:2026-09-20').ok
    assert run(ai, quota_key='user:zwy:tarot:2026-09-21').ok


def test_failure_is_logged_but_not_charged(store):
    ai, fake = make(store, ProviderError('timeout', 't'), ProviderError('timeout', 't'), reply(GOOD_JSON))
    r = run(ai)
    assert not r.ok and r.error_kind == 'timeout' and r.usage.attempts == 2
    assert '不算次数' in r.error
    assert store.count_ok('user:zwy:tarot:2026-09-20') == 0
    # the failed attempt left a row, so the failure rate is knowable
    with store.Session() as s:
        from Backend.Service.ai.usage import AIUsageRow
        rows = s.query(AIUsageRow).all()
        assert len(rows) == 1 and rows[0].ok == 0 and rows[0].error_kind == 'timeout' and rows[0].attempts == 2
    # and the user can try again
    assert run(ai).ok


def test_retry_once_on_5xx_then_succeed(store):
    ai, fake = make(store, ProviderError('provider', '502'), reply(GOOD_JSON))
    r = run(ai)
    assert r.ok and r.usage.attempts == 2 and len(fake.calls) == 2


def test_no_retry_on_rate_limit_or_config(store):
    ai, fake = make(store, ProviderError('rate_limit', 'x'))
    r = run(ai)
    assert not r.ok and r.error_kind == 'rate_limit' and len(fake.calls) == 1
    ai2, fake2 = make(store, ProviderError('config', 'bad key'))
    r2 = run(ai2, user='b', quota_key='user:b:tarot:2026-09-20')
    assert not r2.ok and r2.error_kind == 'config' and len(fake2.calls) == 1


def test_global_cap(store, monkeypatch):
    from Backend.Service.ai import quota
    monkeypatch.setitem(quota.GLOBAL_DAILY_CAP, 'tarot', 2)
    ai, fake = make(store, *[reply(GOOD_JSON)] * 5)
    assert run(ai, user='a', role='horizon', quota_key='user:a:tarot:2026-09-20').ok
    assert run(ai, user='b', role='horizon', quota_key='user:b:tarot:2026-09-20').ok
    r = run(ai, user='c', role='horizon', quota_key='user:c:tarot:2026-09-20')
    assert not r.ok and r.error_kind == 'global_cap' and len(fake.calls) == 2


def test_kill_switch(store, monkeypatch):
    monkeypatch.setenv('AI_ENABLED', '0')
    ai, fake = make(store, reply(GOOD_JSON))
    r = run(ai)
    assert not r.ok and r.error_kind == 'disabled' and len(fake.calls) == 0


# ── parse ─────────────────────────────────────────────────────────────────

def test_off_schema_reply_is_kept_as_text_and_still_charged(store):
    ai, _ = make(store, reply('The Strength card suggests… (no json here)'))
    r = run(ai)
    assert r.ok and r.data is None and r.text.startswith('The Strength') and r.error_kind == 'parse'
    assert store.count_ok('user:zwy:tarot:2026-09-20') == 1, 'paid for → counts'


def test_parse_tolerates_fences_and_prose():
    fenced = "Here you go:\n```json\n" + GOOD_JSON + "\n```\nHope it helps."
    assert tp.parse(fenced)['summary'] == '整体…'
    assert tp.parse('{"past": "x"}') is None                 # missing keys
    assert tp.parse('{"past":"","present":"a","future":"b","summary":"c","next_step":"d"}') is None
    assert tp.parse('not json') is None


def test_prompt_uses_two_sentences_of_waite_and_optional_question():
    msgs = tp.build(SPREAD, None)
    u = msgs[-1]['content']
    assert '没有写下问题' in u and 'More text.' not in u and 'A woman closes the jaws' in u
    assert tp.build(SPREAD, '要不要搬家')[-1]['content'].count('要不要搬家') == 1


# ── the business day ──────────────────────────────────────────────────────

def test_today_key_is_chicago_not_utc(monkeypatch):
    # 2026-09-21 02:30 UTC is 2026-09-20 21:30 in Chicago (CDT). A UTC date
    # would roll the quota over at 7pm local; the key must say the 20th.
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 9, 21, 2, 30, tzinfo=ZoneInfo('UTC'))
            return base.astimezone(tz) if tz else base.replace(tzinfo=None)
    monkeypatch.setattr(svc, 'datetime', FakeDT)
    assert svc.today_key() == '2026-09-20'
