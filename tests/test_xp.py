"""经验值 + 等级: grants are idempotent, caps hold, levels come from the table.

Run:  python -m pytest tests/test_xp.py -q
The database is a temp SQLite file; never imports app.py.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.Controller import tarot_db, xp_db   # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path}/t.db')
    for mod in (tarot_db, xp_db):
        monkeypatch.setattr(mod, 'engine', engine)
        monkeypatch.setattr(mod, 'Session', sessionmaker(bind=engine))
    tarot_db.init_tarot_db()
    xp_db.init_xp_db()
    return engine


def test_level_table_is_exact():
    assert xp_db.level_for(0)['level_name'] == '新村民'
    assert xp_db.level_for(99)['level'] == 1
    assert xp_db.level_for(100)['level_name'] == '老村民'
    assert xp_db.level_for(450)['level_name'] == '村民代表'
    assert xp_db.level_for(1500)['level_name'] == '村委委员'
    assert xp_db.level_for(4000)['level_name'] == '村支书'
    top = xp_db.level_for(30000)
    assert top['level_name'] == '村长' and top['next_xp'] is None and top['progress'] == 1.0
    mid = xp_db.level_for(275)                       # halfway 100 → 450
    assert mid['next_level_name'] == '村民代表' and mid['next_xp'] == 450
    assert mid['progress'] == 0.5


def test_same_ref_pays_once(db):
    assert xp_db.award('ann', 'listing', 'L1') is True
    assert xp_db.award('ann', 'listing', 'L1') is False
    assert xp_db.get_xp('ann') == 10


def test_daily_caps(db):
    for i in range(3):
        assert xp_db.award('ann', 'listing', f'L{i}') is True
    assert xp_db.award('ann', 'listing', 'L-fourth') is False
    for i in range(3):
        assert xp_db.award('ann', 'wall', f'W{i}') is True
    assert xp_db.award('ann', 'wall', 'W-fourth') is False
    assert xp_db.get_xp('ann') == 30 + 6


def test_deal_pays_both_review_pays_reviewer(db):
    # The trade grant is keyed on the listing (see market_controller.mark_sold).
    assert xp_db.award('ann', 'deal', 'L1') and xp_db.award('ben', 'deal', 'L1')
    assert xp_db.award('ann', 'deal', 'L1') is False
    assert xp_db.award('ann', 'review', 'D1') is True
    assert xp_db.award('ann', 'review', 'D1') is False
    assert xp_db.get_xp('ann') == 40 and xp_db.get_xp('ben') == 30


def _draw(username, day):
    tarot_db.set_daily(username, day, card_id='c1', card_name='x',
                       upright=True, text_='t', source='fallback')


def test_streak_and_seven_day_bonus(db):
    today = xp_db.today_key()
    d = datetime.fromisoformat(today).date()
    for i in range(6, 0, -1):
        _draw('ann', (d - timedelta(days=i)).isoformat())
    # Six days behind us, today not yet drawn: streak alive at 6.
    info = xp_db.streak_info('ann', today)
    assert info['streak_days'] == 6 and info['today_drawn'] is False
    assert [w['drawn'] for w in info['week']] == [True] * 6 + [False]

    _draw('ann', today)
    r = xp_db.award_daily_sign('ann', today)
    assert r == {'signed': True, 'bonus': True, 'streak_days': 7}
    assert xp_db.get_xp('ann') == 25
    # Reload: nothing is paid twice.
    r2 = xp_db.award_daily_sign('ann', today)
    assert r2['signed'] is False and r2['bonus'] is False
    assert xp_db.get_xp('ann') == 25


def test_broken_streak_is_zero(db):
    today = datetime.fromisoformat(xp_db.today_key()).date()
    _draw('ben', (today - timedelta(days=3)).isoformat())
    assert xp_db.streak_info('ben')['streak_days'] == 0


def test_native_tag_after_a_year():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert xp_db.hidden_tag_for((now - timedelta(days=366)).isoformat()) == '土著'
    assert xp_db.hidden_tag_for((now - timedelta(days=30)).isoformat()) is None
    assert xp_db.hidden_tag_for('') is None


def test_recompute_matches_log(db):
    xp_db.award('ann', 'deal', 'D9')
    xp_db.award('ann', 'wall', 'W9')
    assert xp_db.recompute('ann') == 32 == xp_db.get_xp('ann')
