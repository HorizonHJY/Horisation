"""Travel plan expenses: a plan carries a bill, and the two die together.

Run:  python -m pytest tests/test_travel_bill.py -q
The database is a temp SQLite file; the session check is faked.
"""

import os
import sys

import pytest
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.Controller import auth_controller, bill_db, travel_db, travel_controller   # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path}/t.db')

    # Two modules share one SQLite file in production (each opens its own
    # engine on _data/market.db). Point both at the same temp file here.
    for mod in (travel_db, bill_db):
        monkeypatch.setattr(mod, 'engine', engine)
        monkeypatch.setattr(mod, 'Session', sessionmaker(bind=engine))

    travel_db.init_travel_db()
    bill_db.init_bill_db()

    users = {'tok-a': {'username': 'alice', 'role': 'user'},
             'tok-b': {'username': 'bob', 'role': 'user'}}
    monkeypatch.setattr(auth_controller.user_manager, 'validate_session', users.get)
    monkeypatch.setattr('Backend.Controller.market_db.get_friends', lambda me: [])

    app = Flask(__name__)
    app.secret_key = 'test'
    app.register_blueprint(travel_controller.travel_bp)
    c = app.test_client()
    with c.session_transaction() as s:
        s['session_token'] = 'tok-a'
    return c


def make_plan(c, name='Kyoto'):
    d = c.post('/api/travel/plans', json={'name': name}).get_json()
    assert d['ok'], d
    return d['plan']


def test_a_new_plan_comes_with_a_bill_that_opens_on_the_same_id(client):
    plan = make_plan(client)
    assert plan['bill_id'] == plan['id'], 'one code should open both'

    d = client.get(f"/api/travel/plans/{plan['id']}/bill").get_json()
    assert d['ok'], d
    # The creator was put on the trip by default.
    assert d['bill']['participants'] == ['alice']
    assert d['bill']['expenses'] == []


def test_expenses_and_participants_are_reachable_through_the_plan(client):
    plan = make_plan(client)
    pid = plan['id']

    d = client.post(f'/api/travel/plans/{pid}/bill/participants', json={'name': 'bob'}).get_json()
    assert d['ok'] and 'bob' in d['bill']['participants']

    d = client.post(f'/api/travel/plans/{pid}/bill/expenses', json={
        'desc': '拉面', 'amount': 30, 'paidBy': 'alice', 'splitAmong': ['alice', 'bob'],
    }).get_json()
    assert d['ok'], d
    assert d['expense']['amount'] == 30
    assert len(d['bill']['expenses']) == 1

    # The same bill is visible through the bill API too — one row, two doors.
    d2 = client.get(f'/api/travel/plans/{pid}/bill').get_json()
    assert len(d2['bill']['expenses']) == 1


def test_deleting_the_plan_takes_the_bill_with_it(client):
    plan = make_plan(client)
    pid = plan['id']
    client.post(f'/api/travel/plans/{pid}/bill/expenses', json={
        'desc': 'taxi', 'amount': 12, 'paidBy': 'alice', 'splitAmong': ['alice'],
    })

    d = client.delete(f'/api/travel/plans/{pid}').get_json()
    assert d['ok'], d

    assert travel_db.get_plan(pid) is None
    assert bill_db.get_bill(pid) is None, 'no orphan bill should survive the plan'


def test_a_plan_from_before_the_feature_gets_its_bill_on_first_visit(client):
    """An old plan has bill_id = NULL; opening Expenses mints one and records it."""
    with travel_db.Session() as s:
        s.add(travel_db.TravelPlan(id='OLD123', name='Old Trip', created_by='alice', num_days=2))
        s.commit()

    d = client.get('/api/travel/plans/OLD123/bill').get_json()
    assert d['ok'], d
    assert d['bill']['id'] == 'OLD123'
    assert d['bill']['participants'] == ['alice']

    # The backfill is written down, so the next read is a plain lookup.
    assert travel_db.bill_id_for('OLD123') == 'OLD123'


def test_the_creator_default_is_not_duplicated(client):
    plan = make_plan(client)
    pid = plan['id']
    client.post(f'/api/travel/plans/{pid}/bill/participants', json={'name': 'alice'})
    d = client.get(f'/api/travel/plans/{pid}/bill').get_json()
    assert d['bill']['participants'].count('alice') == 1


def test_a_bad_expense_is_rejected(client):
    pid = make_plan(client)['id']
    d = client.post(f'/api/travel/plans/{pid}/bill/expenses', json={
        'desc': '', 'amount': 5, 'paidBy': 'alice', 'splitAmong': ['alice'],
    })
    assert d.status_code == 400
    d = client.post(f'/api/travel/plans/{pid}/bill/expenses', json={
        'desc': 'x', 'amount': 0, 'paidBy': 'alice', 'splitAmong': ['alice'],
    })
    assert d.status_code == 400
