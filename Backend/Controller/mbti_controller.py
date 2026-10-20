"""
mbti_controller.py — the MBTI test.

Two formats (short / full), per-row visibility, re-takes keep history.
The scoring is deterministic and tiny; no AI call, no quota.

Routes
  GET  /api/mbti/questions?mode=short|full       public — the question bank
  GET  /api/mbti/types                           public — the 16 type cards
  POST /api/mbti/submit                          login — {mode, answers:[0|1,...]}
  GET  /api/mbti/me                              login — my latest + my history
  GET  /api/mbti/user/<username>                 optional — latest, visibility-filtered
  PATCH /api/mbti/me/visibility                  login — {visibility: private|friends|public}
  DELETE /api/mbti/me                            login — wipe all my results
"""

from flask import Blueprint, jsonify, request

from Backend.Controller.auth_controller import login_required, login_optional
from Backend.Controller import market_db, mbti_db
from Backend.Service.mbti.questions import get_bank, score
from Backend.Service.mbti.types import TYPES

mbti_bp = Blueprint('mbti', __name__, url_prefix='/api/mbti')

VALID_MODES = {'short', 'full'}
VALID_VIS   = {'private', 'friends', 'public'}


def _bad(msg: str, status: int = 400):
    return jsonify({'ok': False, 'error': msg}), status


@mbti_bp.route('/questions', methods=['GET'])
@login_optional
def questions():
    """Public — anyone can see the questions, even without logging in."""
    mode = (request.args.get('mode') or 'short').strip()
    if mode not in VALID_MODES:
        return _bad('mode must be short or full')
    return jsonify({'ok': True, 'mode': mode, 'questions': get_bank(mode)})


@mbti_bp.route('/types', methods=['GET'])
@login_optional
def types_card():
    """Public — the 16-type reference card. Used by the result page and by
    anyone who wants to see what MBTI the site thinks each letter means."""
    return jsonify({'ok': True, 'types': TYPES})


@mbti_bp.route('/submit', methods=['POST'])
@login_required
def submit():
    body    = request.get_json(silent=True) or {}
    mode    = (body.get('mode') or '').strip()
    answers = body.get('answers')
    if mode not in VALID_MODES:
        return _bad('mode must be short or full')
    if not isinstance(answers, list):
        return _bad('answers must be a list')
    try:
        answers = [int(a) for a in answers]
    except (TypeError, ValueError):
        return _bad('answers must be integers (0 or 1)')
    try:
        result = score(mode, answers)
    except ValueError as e:
        return _bad(str(e))

    me = request.current_user['username']
    row = mbti_db.record(me, result['type'], result['counts'], mode)
    return jsonify({
        'ok': True,
        'result': row,
        'type_card': TYPES.get(row['type']),
    })


@mbti_bp.route('/me', methods=['GET'])
@login_required
def me():
    """Latest + full history. Visibility is honoured on others, not here —
    the caller is always allowed to see their own."""
    me = request.current_user['username']
    latest = mbti_db.latest(me)
    return jsonify({
        'ok': True,
        'result':  latest,
        'history': mbti_db.history(me),
        'type_card': TYPES.get(latest['type']) if latest else None,
    })


@mbti_bp.route('/user/<username>', methods=['GET'])
@login_optional
def by_user(username):
    """Someone else's latest, honouring visibility."""
    viewer = request.current_user['username'] if request.current_user else None
    friend = bool(viewer and viewer != username and market_db.are_friends(viewer, username))
    row = mbti_db.latest_visible(username, viewer, friend)
    if row is None:
        return jsonify({'ok': True, 'result': None, 'type_card': None})
    return jsonify({
        'ok': True,
        'result': row,
        'type_card': TYPES.get(row['type']),
    })


@mbti_bp.route('/me/visibility', methods=['PATCH'])
@login_required
def visibility():
    body = request.get_json(silent=True) or {}
    v    = (body.get('visibility') or '').strip()
    if v not in VALID_VIS:
        return _bad('visibility must be private, friends, or public')
    me = request.current_user['username']
    if not mbti_db.set_latest_visibility(me, v):
        return _bad('no result to update — take the test first', 404)
    return jsonify({'ok': True, 'result': mbti_db.latest(me)})


@mbti_bp.route('/me', methods=['DELETE'])
@login_required
def wipe():
    me = request.current_user['username']
    n  = mbti_db.wipe(me)
    return jsonify({'ok': True, 'deleted': n})
