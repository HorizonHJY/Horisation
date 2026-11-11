"""电子猫: one cat each, chores once a day, decay on read, nothing ever dies.

Run:  python -m pytest tests/test_pet.py -q
The database is a temp SQLite file; the session check is faked.
"""

import os
import sys

import pytest
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.Controller import auth_controller, pet_controller, pet_db, xp_db   # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path}/t.db')
    for mod in (pet_db, xp_db):
        monkeypatch.setattr(mod, 'engine', engine)
        monkeypatch.setattr(mod, 'Session', sessionmaker(bind=engine))
    pet_db.init_pet_db()
    xp_db.init_xp_db()
    return engine


@pytest.fixture
def client(db, monkeypatch):
    users = {'tok-a': {'username': 'a', 'role': 'user'}, 'tok-b': {'username': 'b', 'role': 'user'}}
    monkeypatch.setattr(auth_controller.user_manager, 'validate_session', users.get)
    monkeypatch.setattr(pet_db, 'today_key', lambda: '2026-11-11')
    app = Flask(__name__)
    app.secret_key = 'test'
    app.register_blueprint(pet_controller.pet_bp)
    c = app.test_client()

    def as_user(tok):
        with c.session_transaction() as s:
            s['session_token'] = tok
    c.as_user = as_user
    c.as_user('tok-a')
    return c


# ── pet_db ───────────────────────────────────────────────────────────────────

def test_adopt_once_and_breed_is_fixed(db):
    pet, err = pet_db.adopt('a', 'calico', '  花  花 ', day='2026-11-01')
    assert err is None and pet['name'] == '花 花' and pet['breed'] == 'calico'
    assert pet['stage'] == 'kitten' and pet['bond'] == 20
    again, err = pet_db.adopt('a', 'black', 'x', day='2026-11-01')
    assert again is None and 'already' in err
    assert pet_db.get_pet('a', '2026-11-01')['breed'] == 'calico'
    # rename keeps the breed
    renamed, err = pet_db.rename('a', '小黑')
    assert err is None and renamed['name'] == '小黑' and renamed['breed'] == 'calico'


def test_name_limits(db):
    assert pet_db.adopt('a', 'blue', '', day='2026-11-01')[1]
    assert pet_db.adopt('a', 'blue', 'x' * 13, day='2026-11-01')[1]
    assert pet_db.adopt('a', 'blue', 'x' * 12, day='2026-11-01')[1] is None
    assert pet_db.adopt('b', 'unicorn', 'x', day='2026-11-01')[1]


def test_care_once_a_day_and_meters_move(db):
    pet_db.adopt('a', 'orange', '橘子', day='2026-11-01')
    pet, err, applied = pet_db.care('a', 'feed', day='2026-11-01')
    assert applied and pet['hunger'] == 100 and pet['today']['feed'] is True
    pet, err, applied = pet_db.care('a', 'feed', day='2026-11-01')
    assert applied is False and pet['hunger'] == 100
    pet, _, applied = pet_db.care('a', 'brush', day='2026-11-01')
    assert applied and pet['bond'] == 32
    # a new day, a new chance
    pet, _, applied = pet_db.care('a', 'feed', day='2026-11-02')
    assert applied


def test_decay_is_computed_on_read_and_never_below_zero(db):
    pet_db.adopt('a', 'blue', '灰灰', day='2026-11-01')
    pet = pet_db.get_pet('a', '2026-11-04')          # 3 days later
    assert (pet['hunger'], pet['mood'], pet['bond']) == (70 - 36, 70 - 30, 0)
    # read again on the same day: no double decay
    assert pet_db.get_pet('a', '2026-11-04')['hunger'] == 34
    far = pet_db.get_pet('a', '2027-11-04')          # a year away
    assert far['hunger'] == 0 and far['mood'] == 0 and far['bond'] == 0
    assert far['stage'] == 'kitten'                  # alive, just a bit sad


def test_stage_only_goes_up(db):
    pet_db.adopt('a', 'siamese', '阿暹', day='2026-11-01')
    # brush daily: +12 - 8 per day net +4 → bond climbs
    for i in range(1, 25):
        pet, _, _ = pet_db.care('a', 'brush', day=f'2026-11-{i:02d}')
    assert pet['bond'] >= 80 and pet['stage'] == 'elder'
    # vanish for two months: meters fall, title stays
    later = pet_db.get_pet('a', '2027-02-01')
    assert later['bond'] == 0 and later['stage'] == 'elder'


def test_touch_once_per_visitor_per_day_not_self(db):
    pet_db.adopt('a', 'ragdoll', '布布', day='2026-11-01')
    card, err, applied = pet_db.touch('a', 'b', day='2026-11-01')
    assert applied and card['bond'] == 21 and card['touched_today'] is True
    card, err, applied = pet_db.touch('a', 'b', day='2026-11-01')
    assert applied is False and card['bond'] == 21
    card, err, applied = pet_db.touch('a', 'a', day='2026-11-01')
    assert card is None and err
    assert pet_db.touch('nobody', 'b', day='2026-11-01')[1]
    assert pet_db.public_card('a', 'c', '2026-11-01')['touched_today'] is False
    assert pet_db.public_card('zzz') is None


# ── routes + XP ──────────────────────────────────────────────────────────────

def test_routes_pay_xp_once(client):
    r = client.get('/api/pet/me')
    assert r.status_code == 200 and r.get_json()['pet'] is None

    r = client.post('/api/pet/adopt', json={'breed': 'tuxedo', 'name': '奶牛'})
    assert r.status_code == 201
    r = client.post('/api/pet/adopt', json={'breed': 'black', 'name': 'x'})
    assert r.status_code == 409

    for action in ('feed', 'play', 'brush'):
        r = client.post('/api/pet/care', json={'action': action}).get_json()
        assert r['applied'] is True and r['xp_added'] is True and r['xp_delta'] == 2
    r = client.post('/api/pet/care', json={'action': 'feed'}).get_json()
    assert r['applied'] is False and r['xp_added'] is False
    assert xp_db.get_xp('a') == 6
    assert client.post('/api/pet/care', json={'action': 'hug'}).status_code == 400

    me = client.get('/api/pet/me').get_json()['pet']
    assert me['today'] == {'feed': True, 'play': True, 'brush': True}

    # self-pat is refused before anything is touched
    assert client.post('/api/pet/pet', json={'username': 'a'}).status_code == 403

    client.as_user('tok-b')
    r = client.post('/api/pet/pet', json={'username': 'a'}).get_json()
    assert r['applied'] and r['xp_added'] and r['pet']['bond'] == 33
    r = client.post('/api/pet/pet', json={'username': 'a'}).get_json()
    assert r['applied'] is False and r['xp_added'] is False
    assert xp_db.get_xp('b') == 1
    assert client.post('/api/pet/pet', json={'username': 'ghost'}).status_code == 404
    # b has no cat: care says so
    assert client.post('/api/pet/care', json={'action': 'feed'}).status_code == 404
