"""The tarot draw: the server shuffles, the reader's seat decides the card.

Run:  python -m pytest tests/test_tarot_pick.py -q
The database is a temp SQLite file; the session check is faked.
"""

import json
import os
import sys

import pytest
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.Controller import auth_controller, tarot_controller, tarot_db   # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path}/t.db')
    monkeypatch.setattr(tarot_db, 'engine', engine)
    monkeypatch.setattr(tarot_db, 'Session', sessionmaker(bind=engine))
    tarot_db.init_tarot_db()

    users = {'tok-a': {'username': 'a', 'role': 'user'}, 'tok-b': {'username': 'b', 'role': 'user'}}
    monkeypatch.setattr(auth_controller.user_manager, 'validate_session', users.get)

    app = Flask(__name__)
    app.secret_key = 'test'
    app.register_blueprint(tarot_controller.tarot_bp)
    c = app.test_client()
    with c.session_transaction() as s:
        s['session_token'] = 'tok-a'
    return c


def draw(c):
    d = c.post('/api/tarot/draw').get_json()
    assert d['ok'] and 'spread' not in d, 'the draw must not reveal any card'
    return d['reading_id']


def pick(c, rid, seat):
    return c.post('/api/tarot/pick', json={'reading_id': rid, 'seat': seat})


def order_of(rid):
    with tarot_db.Session() as s:
        return json.loads(s.get(tarot_db.TarotReading, rid).deck_order_json)


def test_the_seat_you_touch_is_the_card_you_get(client):
    rid = draw(client)
    order = order_of(rid)
    assert sorted(order) == sorted(c['id'] for c in tarot_controller._deck())

    got = [pick(client, rid, seat).get_json() for seat in (5, 40, 77)]
    assert [g['pick']['card']['id'] for g in got] == [order[5], order[40], order[77]]
    assert [g['pick']['position']['key'] for g in got] == ['past', 'present', 'future']
    assert [g['done'] for g in got] == [False, False, True]


def test_a_seat_cannot_be_taken_twice_or_a_fourth_card_taken(client):
    rid = draw(client)
    assert pick(client, rid, 3).status_code == 200
    assert pick(client, rid, 3).status_code == 409
    assert pick(client, rid, 4).status_code == 200
    assert pick(client, rid, 5).status_code == 200
    assert pick(client, rid, 6).status_code == 409


def test_bad_seats_and_other_peoples_spreads_are_refused(client):
    rid = draw(client)
    for seat in (-1, 78, '3', True, None):
        assert pick(client, rid, seat).status_code == 400
    with client.session_transaction() as s:
        s['session_token'] = 'tok-b'
    assert pick(client, rid, 0).status_code == 404


def test_reading_waits_for_all_three_cards(client):
    rid = draw(client)
    pick(client, rid, 0)
    r = client.post('/api/tarot/reading', json={'reading_id': rid})
    assert r.status_code == 409


def test_two_shuffles_are_not_the_same_order(client):
    assert order_of(draw(client)) != order_of(draw(client))
