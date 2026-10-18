"""
bazi_compat_controller.py
八字合盘 — 用自己两张已排好的命盘做合盘。

左=本命（self），右=对象（other）。合盘算术在 Backend/Service/bazi/compat.py；
AI 解读在 bazi_compat / bazi_compat_ask feature（跟 bazi 一样的 claim/poll
后台任务结构）。

路由：
  POST   /api/bazi/compat           {chart_a_id, chart_b_id} → 建合盘并返回
  POST   /api/bazi/compat/reading   {reading_id}             → 启动 AI 解读
  GET    /api/bazi/compat/reading/<rid>                      → 轮询状态
  GET    /api/bazi/compat/readings                           → 历史合盘列表
  GET    /api/bazi/compat/readings/<rid>                     → 读一条
  PUT    /api/bazi/compat/readings/<rid>/verify              → 验证打勾
  POST   /api/bazi/compat/readings/<rid>/ask                 → 追问
  GET    /api/bazi/compat/followups/<fid>                    → 追问轮询
  DELETE /api/bazi/compat/readings/<rid>                     → 隐藏
  DELETE /api/bazi/compat/mine                               → 隐藏全部
"""

import logging
import re
from flask import Blueprint, jsonify, request

from Backend.Controller.auth_controller import login_required
from Backend.Controller import bazi_db, bazi_compat_db
from Backend.Controller.socketio_instance import socketio
from Backend.Service.ai import ai, today_key
from Backend.Service.ai import scope
from Backend.Service.ai.prompts.bazi_compat import MAX_QUESTION_CHARS
from Backend.Service.bazi.compat import build_compat

log = logging.getLogger(__name__)

bazi_compat_bp = Blueprint('bazi_compat', __name__, url_prefix='/api/bazi/compat')

_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_ERROR_STATUS = {'quota': 429, 'global_cap': 503, 'disabled': 503, 'config': 503}

_CRASH = {'error': '出错了，请再试一次。', 'error_kind': 'provider', 'retryable': True}


def _me() -> tuple[str, str | None]:
    u = request.current_user
    return u['username'], u.get('role')


def _bad(msg: str, status: int = 400):
    return jsonify({'ok': False, 'error': msg}), status


def _item(row) -> dict:
    d = row.to_dict()
    d['followups'] = [f.to_dict() for f in bazi_compat_db.followups(row.id)]
    return d


def _error_kind(result) -> dict:
    return {
        'error': result.error, 'error_kind': result.error_kind,
        'retryable': result.error_kind in ('timeout', 'provider', 'rate_limit'),
        'quota': result.quota.as_dict() if result.quota else None,
    }


@bazi_compat_bp.route('', methods=['POST'])
@login_required
def create_compat():
    """Body: { chart_a_id, chart_b_id }. 从自己已排好的两张盘建合盘。"""
    b = request.get_json(silent=True) or {}
    aid = str(b.get('chart_a_id') or '').strip()
    bid = str(b.get('chart_b_id') or '').strip()
    if not aid or not bid:
        return _bad('请选择两张命盘。')
    if aid == bid:
        return _bad('两张命盘不能是同一张。')
    username, _ = _me()
    a = bazi_db.get(aid, username)
    b_row = bazi_db.get(bid, username)
    if a is None or b_row is None:
        return _bad('找不到命盘。', 404)
    chart_a = a.to_dict()['chart']
    chart_b = b_row.to_dict()['chart']
    compat = build_compat(chart_a, chart_b)
    rid = bazi_compat_db.create(username, aid, chart_a, bid, chart_b, compat)
    row = bazi_compat_db.get(rid, username)
    return jsonify({'ok': True, 'reading_id': rid, 'item': _item(row)})


def _run_reading(rid: str, username: str, role: str | None, chart_a: dict, chart_b: dict, compat: dict) -> None:
    try:
        result = ai.run(
            feature='bazi_compat', user=username, role=role,
            quota_key=f'user:{username}:bazi_compat:{today_key()}',
            payload={'chart_a': chart_a, 'chart_b': chart_b, 'compat': compat},
        )
    except Exception:
        log.exception('bazi_compat reading %s crashed', rid)
        bazi_compat_db.fail(rid, _CRASH)
        return
    if result.ok:
        bazi_compat_db.finish(rid, raw=result.text, data=result.data, model=result.usage.model,
                              prompt_version=result.prompt_version)
    else:
        bazi_compat_db.fail(rid, _error_kind(result))


@bazi_compat_bp.route('/reading', methods=['POST'])
@login_required
def start_reading():
    rid = str((request.get_json(silent=True) or {}).get('reading_id') or '').strip()
    if not rid:
        return _bad('reading_id is required')
    username, role = _me()
    row = bazi_compat_db.get(rid, username)
    if row is None:
        return _bad('没有这张合盘。', 404)
    if row.read_at is None and bazi_compat_db.claim(rid):
        d = row.to_dict()
        socketio.start_background_task(_run_reading, rid, username, role,
                                       d['chart_a'], d['chart_b'], d['compat'])
    return jsonify({'ok': True, 'item': _item(bazi_compat_db.get(rid, username))})


@bazi_compat_bp.route('/reading/<rid>', methods=['GET'])
@bazi_compat_bp.route('/readings/<rid>', methods=['GET'])
@login_required
def poll_reading(rid):
    username, _ = _me()
    row = bazi_compat_db.get(rid, username)
    if row is None:
        return _bad('没有这张合盘。', 404)
    item = _item(row)
    if item['state'] == 'error':
        err = item['error'] or {}
        return jsonify({'ok': False, 'item': item, **err}), _ERROR_STATUS.get(err.get('error_kind'), 502)
    return jsonify({'ok': True, 'item': item})


@bazi_compat_bp.route('/readings', methods=['GET'])
@login_required
def history():
    username, _ = _me()
    return jsonify({'ok': True, 'items': bazi_compat_db.list_mine(username)})


@bazi_compat_bp.route('/readings/<rid>/verify', methods=['PUT'])
@login_required
def verify(rid):
    marks = (request.get_json(silent=True) or {}).get('marks')
    if not isinstance(marks, dict):
        return _bad('marks is required')
    clean = {str(k): v for k, v in marks.items() if str(k).isdigit() and int(k) < 20 and v in ('yes', 'no')}
    username, _ = _me()
    if not bazi_compat_db.set_verify(rid, username, clean):
        return _bad('没有这张合盘。', 404)
    return jsonify({'ok': True, 'verify': clean})


def _run_followup(fid: str, username: str, role: str | None, payload: dict) -> None:
    try:
        result = ai.run(
            feature='bazi_compat_ask', user=username, role=role,
            quota_key=f'user:{username}:bazi_compat_ask:{today_key()}',
            payload=payload,
        )
    except Exception:
        log.exception('bazi_compat follow-up %s crashed', fid)
        bazi_compat_db.fail_followup(fid, _CRASH)
        return
    if result.ok:
        bazi_compat_db.answer(fid, text_=result.data['answer'], model=result.usage.model,
                              prompt_version=result.prompt_version)
    else:
        bazi_compat_db.fail_followup(fid, _error_kind(result))


@bazi_compat_bp.route('/readings/<rid>/ask', methods=['POST'])
@login_required
def ask(rid):
    q = _CONTROL.sub('', str((request.get_json(silent=True) or {}).get('question') or '')).strip()
    if not q:
        return _bad('请写下你的问题。')
    if len(q) > MAX_QUESTION_CHARS:
        return _bad(f'问题太长了（最多 {MAX_QUESTION_CHARS} 字）。')
    username, role = _me()
    row = bazi_compat_db.get(rid, username)
    if row is None:
        return _bad('没有这张合盘。', 404)
    if row.read_at is None:
        return _bad('先完成整体合盘解读，再追问。', 409)
    screened = scope.screen(q)
    if screened:
        f = bazi_compat_db.ask_screened(rid, username, q, screened[1], kind=screened[0])
        return jsonify({'ok': True, 'followup': f.to_dict()})
    history_ = [(f.question, f.answer) for f in bazi_compat_db.followups(rid) if f.answer]
    d = row.to_dict()
    payload = {
        'chart_a': d['chart_a'], 'chart_b': d['chart_b'], 'compat': d['compat'],
        'reading': d['reading'], 'reading_text': d['reading_text'],
        'verify': d['verify'], 'history': history_, 'question': q,
    }
    f = bazi_compat_db.ask(rid, username, q)
    socketio.start_background_task(_run_followup, f.id, username, role, payload)
    return jsonify({'ok': True, 'followup': f.to_dict()})


@bazi_compat_bp.route('/followups/<fid>', methods=['GET'])
@login_required
def poll_followup(fid):
    username, _ = _me()
    f = bazi_compat_db.get_followup(fid, username)
    if f is None:
        return _bad('没有这条提问。', 404)
    return jsonify({'ok': True, 'followup': f.to_dict()})


@bazi_compat_bp.route('/readings/<rid>', methods=['DELETE'])
@login_required
def delete_one(rid):
    username, _ = _me()
    if not bazi_compat_db.hide(rid, username):
        return _bad('没有这张合盘。', 404)
    return jsonify({'ok': True})


@bazi_compat_bp.route('/mine', methods=['DELETE'])
@login_required
def delete_mine():
    username, _ = _me()
    return jsonify({'ok': True, 'deleted': bazi_compat_db.hide_all(username)})
