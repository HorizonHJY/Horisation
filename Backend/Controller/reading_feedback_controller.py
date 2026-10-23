"""Was it right? — one verdict per reading, for tarot and bazi alike.

Separate from /api/feedback, which is feedback about the *site*. This one is
about a specific reading, and it feeds the prompt hook in
Backend/Service/ai/feedback.py.
"""
from __future__ import annotations

import re

from flask import Blueprint, jsonify, request

from Backend.Controller import feedback_db, tarot_db, bazi_db
from Backend.Controller.auth_controller import login_required

reading_feedback_bp = Blueprint('reading_feedback', __name__, url_prefix='/api/reading-feedback')

_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _own_reading(feature: str, reading_id: str, username: str):
    if feature == 'tarot':
        return tarot_db.get_reading(reading_id, username)
    return bazi_db.get(reading_id, username)


@reading_feedback_bp.route('', methods=['POST'])
@login_required
def submit():
    """Body: { feature, reading_id, verdict: 'up'|'down', reasons?: [str], note?: str }

    One row per reading; a second call changes the first. Only the verdict and
    the chips steer future prompts — the note stays in the table.
    """
    body = request.get_json(silent=True) or {}
    feature = str(body.get('feature') or '').strip()
    reading_id = str(body.get('reading_id') or '').strip()
    verdict = str(body.get('verdict') or '').strip()

    if feature not in feedback_db.FEATURES:
        return jsonify({'ok': False, 'error': f"feature must be one of {list(feedback_db.FEATURES)}"}), 400
    if verdict not in feedback_db.VERDICTS:
        return jsonify({'ok': False, 'error': "verdict must be 'up' or 'down'"}), 400
    if not reading_id:
        return jsonify({'ok': False, 'error': 'reading_id is required'}), 400

    reasons = body.get('reasons')
    if reasons is not None and not isinstance(reasons, list):
        return jsonify({'ok': False, 'error': 'reasons must be a list'}), 400
    reasons = [_CONTROL.sub('', str(r)).strip()[:feedback_db.MAX_REASON_CHARS]
               for r in (reasons or []) if str(r).strip()][:feedback_db.MAX_REASONS]

    note = body.get('note')
    if note is not None:
        note = _CONTROL.sub('', str(note)).strip()[:feedback_db.MAX_NOTE_CHARS] or None

    username, _ = request.current_user['username'], None
    row = _own_reading(feature, reading_id, username)
    if row is None:
        return jsonify({'ok': False, 'error': 'No such reading.'}), 404
    if getattr(row, 'read_at', None) is None:
        return jsonify({'ok': False, 'error': 'This reading has not been written yet.'}), 409

    saved = feedback_db.upsert(
        feature=feature, reading_id=reading_id, username=username,
        verdict=verdict, reasons=reasons, note=note,
        prompt_version=getattr(row, 'prompt_version', None),
        model=getattr(row, 'model', None),
    )
    return jsonify({'ok': True, 'feedback': saved})


@reading_feedback_bp.route('', methods=['GET'])
@login_required
def mine():
    """?feature=tarot&reading_id=... — the caller's own verdict, if any."""
    feature = request.args.get('feature', '')
    reading_id = request.args.get('reading_id', '')
    if feature not in feedback_db.FEATURES or not reading_id:
        return jsonify({'ok': False, 'error': 'feature and reading_id are required'}), 400
    row = feedback_db.get_mine(feature, reading_id, request.current_user['username'])
    return jsonify({'ok': True, 'feedback': row})

