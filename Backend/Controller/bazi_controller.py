"""
bazi_controller.py
四柱八字 — cast a chart, then (optionally) have it read.

The chart is arithmetic and happens here, in Backend/Service/bazi/chart.py.
The reading is the AI layer's (feature 'bazi', same quota/usage/kill switch as
tarot) and is asked for by reading_id only: the chart the model reads is the
one the server computed and stored, never one the client sent.

A reading takes up to two minutes — longer than nginx holds a request — so
POST /reading starts it in the background and returns at once; the page polls
GET /reading/<id>.
"""

import logging
import re
from datetime import date, datetime

from flask import Blueprint, jsonify, request

from Backend.Controller.auth_controller import login_required
from Backend.Controller import bazi_db
from Backend.Controller.socketio_instance import socketio
from Backend.Service.ai import ai, today_key
from Backend.Service.bazi.chart import build_chart
from Backend.Service.bazi.cities import CITIES

log = logging.getLogger(__name__)

bazi_bp = Blueprint('bazi', __name__, url_prefix='/api/bazi')

_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_TIME = re.compile(r'^\d{2}:\d{2}$')
_ERROR_STATUS = {'quota': 429, 'global_cap': 503, 'disabled': 503, 'config': 503}


def _me() -> tuple[str, str | None]:
    u = request.current_user
    return u['username'], u.get('role')


def _bad(msg: str, status: int = 400):
    return jsonify({'ok': False, 'error': msg}), status


@bazi_bp.route('/cities', methods=['GET'])
@login_required
def cities():
    return jsonify({'ok': True, 'cities': [{'name': k, 'longitude': v} for k, v in CITIES.items()]})


@bazi_bp.route('/chart', methods=['POST'])
@login_required
def cast():
    """Body: { date: YYYY-MM-DD, time: HH:MM, place, longitude?, utc_offset?, gender: male|female }.
    `time` is clock time at birth in `utc_offset` (default +8, Beijing time).
    A known `place` supplies the longitude; otherwise `longitude` is required."""
    b = request.get_json(silent=True) or {}
    d, t = str(b.get('date') or ''), str(b.get('time') or '')
    if not _DATE.match(d) or not _TIME.match(t):
        return _bad('请填写出生日期和时间。')
    try:
        birth = datetime.strptime(f'{d} {t}', '%Y-%m-%d %H:%M')
    except ValueError:
        return _bad('出生日期或时间不对。')
    if not (1900 <= birth.year and birth.date() <= date.today()):
        return _bad('出生日期需在 1900 年到今天之间。')

    gender = b.get('gender')
    if gender not in ('male', 'female'):
        return _bad('请选择性别——大运的顺逆要看它。')

    place = str(b.get('place') or '').strip()[:40]
    lon = CITIES.get(place)
    if lon is None:
        try:
            lon = float(b.get('longitude'))
        except (TypeError, ValueError):
            return _bad('列表里没有这个城市，请填出生地经度。')
        if not -180 <= lon <= 180:
            return _bad('经度应在 -180 到 180 之间。')
    try:
        offset = float(b.get('utc_offset', 8))
    except (TypeError, ValueError):
        return _bad('时区不对。')
    if not -12 <= offset <= 14:
        return _bad('时区应在 UTC-12 到 UTC+14 之间。')

    chart = build_chart(birth=birth, longitude=lon, utc_offset=offset, male=gender == 'male',
                        place=place or f'{lon}°')
    username, _ = _me()
    rid = bazi_db.create(username, chart)
    return jsonify({'ok': True, 'reading_id': rid, 'chart': chart})


@bazi_bp.route('/latest', methods=['GET'])
@login_required
def latest():
    """Your most recent chart and its reading state, so the page can come back to it."""
    username, _ = _me()
    row = bazi_db.latest(username)
    return jsonify({'ok': True, 'item': row.to_dict() if row else None})


def _run_reading(rid: str, username: str, role: str | None, chart: dict) -> None:
    try:
        result = ai.run(
            feature='bazi', user=username, role=role,
            quota_key=f'user:{username}:bazi:{today_key()}',
            payload={'chart': chart},
        )
    except Exception:
        log.exception('bazi reading %s crashed', rid)
        bazi_db.fail(rid, {'error': '解读出错了，请再试一次。', 'error_kind': 'provider', 'retryable': True})
        return
    if result.ok:
        bazi_db.finish(rid, raw=result.text, data=result.data, model=result.usage.model,
                       prompt_version=result.prompt_version)
    else:
        bazi_db.fail(rid, {
            'error': result.error, 'error_kind': result.error_kind,
            'retryable': result.error_kind in ('timeout', 'provider', 'rate_limit'),
            'quota': result.quota.as_dict() if result.quota else None,
        })


@bazi_bp.route('/reading', methods=['POST'])
@login_required
def start_reading():
    """Body: { reading_id }. Starts the reading and returns at once; poll
    GET /reading/<id>. A finished reading is returned as it is, free."""
    rid = str((request.get_json(silent=True) or {}).get('reading_id') or '').strip()
    if not rid:
        return _bad('reading_id is required')
    username, role = _me()
    row = bazi_db.get(rid, username)
    if row is None:
        return _bad('没有这张命盘。', 404)
    if row.read_at is None and bazi_db.claim(rid):
        socketio.start_background_task(_run_reading, rid, username, role, row.to_dict()['chart'])
    return jsonify({'ok': True, 'item': bazi_db.get(rid, username).to_dict()})


@bazi_bp.route('/reading/<rid>', methods=['GET'])
@login_required
def poll_reading(rid):
    username, _ = _me()
    row = bazi_db.get(rid, username)
    if row is None:
        return _bad('没有这张命盘。', 404)
    item = row.to_dict()
    if item['state'] == 'error':
        err = item['error'] or {}
        return jsonify({'ok': False, 'item': item, **err}), _ERROR_STATUS.get(err.get('error_kind'), 502)
    return jsonify({'ok': True, 'item': item})


@bazi_bp.route('/mine', methods=['DELETE'])
@login_required
def delete_mine():
    """Forget every chart of yours — birth data, readings, all of it."""
    username, _ = _me()
    return jsonify({'ok': True, 'deleted': bazi_db.delete_all(username)})
