"""
fx_controller.py
Flask Blueprints for FX (换汇):
  - `fx_rate_bp`  → /api/fx/rate, /api/fx/history   (live USD→CNY data)
  - `fx_posts_bp` → /api/market/fx/...              (info-board posts)

Rate source: Frankfurter (api.frankfurter.app), no key required. Cached
30 min in-process; on fetch failure the previous cached value is served
with `stale: true`. If a fetch has never succeeded, `{ok: false}` is
returned so the UI shows a placeholder rather than a blank card.

Posts are an info board only — the platform does NOT broker trades, as
the UI's disclaimer makes explicit.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from datetime import datetime, timezone, date, timedelta
from typing import Optional

from flask import Blueprint, request, jsonify

from Backend.Controller.auth_controller import login_required, login_optional
from Backend.Controller import market_db


fx_rate_bp  = Blueprint('fx_rate',  __name__, url_prefix='/api/fx')
fx_posts_bp = Blueprint('fx_posts', __name__, url_prefix='/api/market/fx')


_RATE_CACHE_TTL   = 30 * 60   # 30 minutes
_HISTORY_DAYS     = 7
_HTTP_TIMEOUT     = 5.0
_FRANKFURTER_BASE = 'https://api.frankfurter.app'

_cache_lock = threading.Lock()
_rate_cache: dict = {'fetched_at': 0.0, 'payload': None}
_hist_cache: dict = {'fetched_at': 0.0, 'payload': None}


def _http_get_json(url: str) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Horisation/FX'})
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode('utf-8'))
    except Exception:
        return None


def _fetch_latest_rate() -> Optional[dict]:
    data = _http_get_json(f'{_FRANKFURTER_BASE}/latest?from=USD&to=CNY')
    if not data or 'rates' not in data or 'CNY' not in data['rates']:
        return None
    try:
        return {'rate': float(data['rates']['CNY']), 'date': str(data.get('date') or '')}
    except (TypeError, ValueError):
        return None


def _fetch_history(days: int = _HISTORY_DAYS) -> Optional[list]:
    today = date.today()
    start = today - timedelta(days=max(days, 2) + 5)
    url = f'{_FRANKFURTER_BASE}/{start.isoformat()}..{today.isoformat()}?from=USD&to=CNY'
    data = _http_get_json(url)
    if not data or not isinstance(data.get('rates'), dict):
        return None
    points: list[dict] = []
    for d_str in sorted(data['rates'].keys()):
        try:
            rate = float(data['rates'][d_str]['CNY'])
        except (TypeError, ValueError, KeyError):
            continue
        points.append({'date': d_str, 'rate': rate})
    if not points:
        return None
    return points[-days:]


@fx_rate_bp.route('/rate', methods=['GET'])
@login_optional
def get_rate():
    now = time.time()
    with _cache_lock:
        cached = _rate_cache['payload']
        fresh  = (now - _rate_cache['fetched_at']) < _RATE_CACHE_TTL and cached is not None
    if fresh:
        return jsonify({**cached, 'stale': False})

    fetched = _fetch_latest_rate()
    if fetched:
        payload = {
            'ok':         True,
            'base':       'USD',
            'quote':      'CNY',
            'rate':       fetched['rate'],
            'date':       fetched['date'],
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'source':     'Frankfurter / ECB',
        }
        with _cache_lock:
            _rate_cache['payload']    = payload
            _rate_cache['fetched_at'] = now
        return jsonify({**payload, 'stale': False})

    with _cache_lock:
        cached = _rate_cache['payload']
    if cached:
        return jsonify({**cached, 'stale': True})
    return jsonify({'ok': False, 'error': 'Rate unavailable right now.'})


@fx_rate_bp.route('/history', methods=['GET'])
@login_optional
def get_history():
    now = time.time()
    with _cache_lock:
        cached = _hist_cache['payload']
        fresh  = (now - _hist_cache['fetched_at']) < _RATE_CACHE_TTL and cached is not None
    if fresh:
        return jsonify({**cached, 'stale': False})

    points = _fetch_history(_HISTORY_DAYS)
    if points:
        payload = {
            'ok':         True,
            'base':       'USD',
            'quote':      'CNY',
            'points':     points,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        with _cache_lock:
            _hist_cache['payload']    = payload
            _hist_cache['fetched_at'] = now
        return jsonify({**payload, 'stale': False})

    with _cache_lock:
        cached = _hist_cache['payload']
    if cached:
        return jsonify({**cached, 'stale': True})
    return jsonify({'ok': False, 'error': 'History unavailable right now.'})


# ── Posts ───────────────────────────────────────────────────────────────────

_DIRECTIONS = {'usd_to_cny', 'cny_to_usd'}
_METHODS    = {'cash', 'transfer', 'either'}


def _parse_post(body: dict) -> tuple[dict, str | None]:
    direction = (body.get('direction') or '').strip()
    method    = (body.get('method') or '').strip()
    area      = (body.get('area') or '').strip()
    note      = (body.get('note') or '').strip()

    if direction not in _DIRECTIONS:
        return {}, '方向只能是 usd_to_cny 或 cny_to_usd。'
    if method not in _METHODS:
        return {}, '方式只能是 cash / transfer / either。'

    try:
        amount = float(body.get('amount'))
    except (TypeError, ValueError):
        return {}, '金额必须是数字。'
    if not (amount > 0):
        return {}, '金额必须大于 0。'

    rate_raw = body.get('rate')
    if rate_raw in (None, '', 'null'):
        rate = None
    else:
        try:
            rate = float(rate_raw)
        except (TypeError, ValueError):
            return {}, '汇率必须是数字,留空表示按实时汇率。'
        if not (rate > 0):
            return {}, '汇率必须大于 0,留空表示按实时汇率。'

    if len(note) > 200:
        return {}, '备注不能超过 200 字。'
    if len(area) > 100:
        return {}, '区域不能超过 100 字。'

    return {
        'direction': direction, 'amount': amount, 'rate': rate,
        'method': method, 'area': area[:100], 'note': note[:200],
    }, None


@fx_posts_bp.route('', methods=['GET'], strict_slashes=False)
@login_optional
def list_fx_posts():
    """Public feed: open posts, newest first; `done`/`removed` hidden."""
    return jsonify({'ok': True, 'items': market_db.list_fx_posts()})


@fx_posts_bp.route('', methods=['POST'], strict_slashes=False)
@login_required
def create_fx_post():
    body = request.get_json(silent=True) or {}
    fields, err = _parse_post(body)
    if err:
        return jsonify({'ok': False, 'error': err}), 400
    me = request.current_user['username']
    pid = market_db.create_fx_post(owner=me, **fields)
    return jsonify({'ok': True, 'item': market_db.get_fx_post(pid)}), 201


@fx_posts_bp.route('/mine', methods=['GET'])
@login_required
def my_fx_posts():
    me = request.current_user['username']
    return jsonify({'ok': True, 'items': market_db.list_my_fx_posts(me)})


@fx_posts_bp.route('/<post_id>', methods=['GET'])
@login_optional
def get_fx_post(post_id):
    item = market_db.get_fx_post(post_id)
    if not item:
        return jsonify({'ok': False, 'error': '帖子不存在。'}), 404
    return jsonify({'ok': True, 'item': item})


@fx_posts_bp.route('/<post_id>', methods=['PUT'])
@login_required
def update_fx_post(post_id):
    body = request.get_json(silent=True) or {}
    fields, err = _parse_post(body)
    if err:
        return jsonify({'ok': False, 'error': err}), 400
    me = request.current_user['username']
    if not market_db.update_fx_post(post_id, me, **fields):
        return jsonify({'ok': False, 'error': '帖子不存在或没有权限。'}), 404
    return jsonify({'ok': True, 'item': market_db.get_fx_post(post_id)})


@fx_posts_bp.route('/<post_id>', methods=['DELETE'])
@login_required
def delete_fx_post(post_id):
    me = request.current_user['username']
    if not market_db.delete_fx_post(post_id, me):
        return jsonify({'ok': False, 'error': '帖子不存在或没有权限。'}), 404
    return jsonify({'ok': True})


@fx_posts_bp.route('/<post_id>/done', methods=['POST'])
@login_required
def mark_fx_done(post_id):
    me = request.current_user['username']
    if not market_db.mark_fx_done(post_id, me):
        return jsonify({'ok': False, 'error': '帖子不存在、已完成或没有权限。'}), 400
    return jsonify({'ok': True, 'item': market_db.get_fx_post(post_id)})


@fx_posts_bp.route('/<post_id>/reopen', methods=['POST'])
@login_required
def reopen_fx(post_id):
    me = request.current_user['username']
    if not market_db.reopen_fx_post(post_id, me):
        return jsonify({'ok': False, 'error': '帖子不存在或当前未完成。'}), 400
    return jsonify({'ok': True, 'item': market_db.get_fx_post(post_id)})
