"""八字: the chart arithmetic, the reply parser, and the cast → read flow.

Run:  python -m pytest tests/test_bazi.py -q
The model is faked and the background task runs inline; the database is a
temp SQLite file.
"""

import json
import os
import sys
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.Controller import auth_controller, bazi_controller, bazi_db   # noqa: E402
from Backend.Service.ai.prompts import bazi as bp                          # noqa: E402
from Backend.Service.bazi.chart import build_chart, true_solar_time, zi_zuo, year_ganzhi   # noqa: E402

TODAY = date(2026, 10, 12)


def gz(chart):
    return [p['gan'] + p['zhi'] for p in chart['pillars']]


# ── Chart ────────────────────────────────────────────────────────────

def test_matches_wenzhen_on_a_known_chart():
    """问真八字, 1998-03-09 21:34 乾造 — every row of its table."""
    c = build_chart(birth=datetime(1998, 3, 9, 21, 34), longitude=120, utc_offset=8, male=True,
                    place='x', today=TODAY)
    assert gz(c) == ['戊寅', '乙卯', '乙卯', '丁亥']
    p = c['pillars']
    assert [x['main_star'] for x in p] == ['正财', '比肩', '元男', '食神']
    assert [x['di_shi'] for x in p] == ['帝旺', '临官', '临官', '死']
    assert [x['zi_zuo'] for x in p] == ['长生', '临官', '临官', '胎']
    assert [x['xun_kong'] for x in p] == ['申酉', '子丑', '子丑', '午未']
    assert [x['na_yin'] for x in p] == ['城头土', '大溪水', '大溪水', '屋上土']
    assert [h['star'] for h in p[0]['hidden']] == ['劫财', '伤官', '正财']


def test_true_solar_time_moves_a_xinjiang_birth_into_the_earlier_hour():
    """石河子, 21:30 Beijing time: clock says 亥, the sun says 戌."""
    c = build_chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, utc_offset=8, male=True,
                    place='石河子', today=TODAY)
    assert c['solar_time']['true'].endswith('19:02')
    assert gz(c) == ['戊寅', '乙卯', '丙辰', '戊戌']
    assert [d['ganzhi'] for d in c['yun']['da_yun'][:3]] == ['丙辰', '丁巳', '戊午']
    assert c['yun']['start']['date'] == '2006-09-10'
    assert [d['ganzhi'] for d in c['yun']['da_yun'] if d['current']] == ['戊午']
    assert c['pillars'][1]['shen_sha'] == ['桃花'] and c['pillars'][3]['shen_sha'] == ['华盖']


def test_a_female_chart_runs_luck_backward():
    c = build_chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, utc_offset=8, male=False,
                    place='石河子', today=TODAY)
    assert c['gender_label'] == '坤造'
    assert [d['ganzhi'] for d in c['yun']['da_yun'][:3]] == ['甲寅', '癸丑', '壬子']


def test_china_summer_time_1986_to_1991_is_taken_off():
    st = true_solar_time(datetime(1988, 7, 1, 12, 0), 120, 8)
    assert st['dst_applied'] and st['true'].startswith('1988-07-01 10:5')
    assert not true_solar_time(datetime(1992, 7, 1, 12, 0), 120, 8)['dst_applied']


def test_small_tables():
    assert zi_zuo('丙', '辰') == '冠带' and zi_zuo('乙', '卯') == '临官' and zi_zuo('丁', '亥') == '胎'
    assert year_ganzhi(1998) == '戊寅' and year_ganzhi(2026) == '丙午'


# ── Prompt ───────────────────────────────────────────────────────────

GOOD = {
    'verdict': '印旺身强，食神吐秀',
    'strength': {'level': '身强', 'reasons': ['丙生卯月得令', '寅为长生']},
    'pattern': {'name': '正印格', 'explain': '卯月乙透。'},
    'gods': {'use': {'element': '金', 'why': '食神生财'}, 'like': {'element': '土', 'why': '泄秀'},
             'avoid': {'element': '木、火', 'why': '印比过旺'}, 'idle': {'element': '水', 'why': ''}},
    'da_yun': [{'ganzhi': '丙辰', 'theme': '比肩争强', 'reading': '…'}],
    'liu_nian': [{'year': 2018, 'ganzhi': '戊戌', 'tag': '情绪', 'event': '…'},
                 {'year': 2016, 'ganzhi': '丙申', 'tag': '迁移', 'event': '…'}],
    'verify': ['2016 年是否离家远行？'],
    'summary': '一生主线在食神生财。',
}


def test_the_model_is_handed_the_finished_chart():
    c = build_chart(birth=datetime(1998, 3, 10, 21, 30), longitude=86.0, utc_offset=8, male=True,
                    place='石河子', today=TODAY)
    user = bp.build(c)[1]['content']
    assert '乾造，八字：戊寅 乙卯 丙辰 戊戌' in user
    assert '2006年9岁起运后第一个大运是丙辰' in user and '当前大运：戊午' in user


def test_parse_keeps_the_shape_and_orders_the_years():
    out = bp.parse('```json\n' + json.dumps(GOOD, ensure_ascii=False) + '\n```')
    assert out['strength']['level'] == '身强'
    assert [x['year'] for x in out['liu_nian']] == [2016, 2018]


def test_parse_rejects_a_reply_missing_the_core():
    bad = dict(GOOD, verdict='')
    assert bp.parse(json.dumps(bad, ensure_ascii=False)) is None
    assert bp.parse('不是 JSON') is None


# ── Flow ─────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path}/b.db')
    monkeypatch.setattr(bazi_db, 'engine', engine)
    monkeypatch.setattr(bazi_db, 'Session', sessionmaker(bind=engine))
    bazi_db.init_bazi_db()
    users = {'tok-a': {'username': 'a', 'role': 'user'}, 'tok-b': {'username': 'b', 'role': 'user'}}
    monkeypatch.setattr(auth_controller.user_manager, 'validate_session', users.get)
    # Background work runs inline, so a poll right after sees the result.
    monkeypatch.setattr(bazi_controller.socketio, 'start_background_task', lambda f, *a: f(*a))
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        return SimpleNamespace(ok=True, text=json.dumps(GOOD, ensure_ascii=False), data=bp.parse(json.dumps(GOOD)),
                               usage=SimpleNamespace(model='fake'), prompt_version=bp.PROMPT_VERSION,
                               error=None, error_kind=None, quota=None)
    monkeypatch.setattr(bazi_controller.ai, 'run', fake_run)

    app = Flask(__name__)
    app.secret_key = 'test'
    app.register_blueprint(bazi_controller.bazi_bp)
    c = app.test_client()
    with c.session_transaction() as s:
        s['session_token'] = 'tok-a'
    c.calls = calls
    return c


BODY = {'date': '1998-03-10', 'time': '21:30', 'place': '石河子', 'gender': 'male'}


def test_cast_then_read_then_read_again_costs_one_call(client):
    d = client.post('/api/bazi/chart', json=BODY).get_json()
    assert d['ok'] and gz(d['chart']) == ['戊寅', '乙卯', '丙辰', '戊戌']
    rid = d['reading_id']

    r = client.post('/api/bazi/reading', json={'reading_id': rid}).get_json()
    assert r['item']['state'] == 'done' and r['item']['reading']['verdict'] == GOOD['verdict']
    client.post('/api/bazi/reading', json={'reading_id': rid})
    assert len(client.calls) == 1, 'a finished reading is not asked for twice'
    assert client.calls[0]['payload']['chart']['pillars'][3]['gan'] == '戊', 'the stored chart, not the client'

    assert client.get('/api/bazi/latest').get_json()['item']['id'] == rid


def test_input_is_checked(client):
    assert client.post('/api/bazi/chart', json=dict(BODY, gender='x')).status_code == 400
    assert client.post('/api/bazi/chart', json=dict(BODY, date='2999-01-01')).status_code == 400
    assert client.post('/api/bazi/chart', json=dict(BODY, place='火星')).status_code == 400
    assert client.post('/api/bazi/chart', json=dict(BODY, place='火星', longitude=200)).status_code == 400
    assert client.post('/api/bazi/chart', json=dict(BODY, place='Boston', longitude=-71.1, utc_offset=-5)).status_code == 200


def test_charts_are_private_and_can_be_deleted(client):
    rid = client.post('/api/bazi/chart', json=BODY).get_json()['reading_id']
    with client.session_transaction() as s:
        s['session_token'] = 'tok-b'
    assert client.get(f'/api/bazi/reading/{rid}').status_code == 404
    assert client.post('/api/bazi/reading', json={'reading_id': rid}).status_code == 404
    with client.session_transaction() as s:
        s['session_token'] = 'tok-a'
    assert client.delete('/api/bazi/mine').get_json()['deleted'] == 1
    assert client.get('/api/bazi/latest').get_json()['item'] is None
