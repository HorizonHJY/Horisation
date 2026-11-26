"""
tod_controller.py
Truth or Dare — deal a prompt, record that it happened, take a 1–5 rating.

  GET  /api/tod/prompts?level=ice|normal|adult|random
  POST /api/tod/draw   {prompt_id}          proves the prompt reached the user
  POST /api/tod/rate   {prompt_id, rating, tags[]}
  GET  /api/tod/stats                       admin: averages and hidden list
"""

from flask import Blueprint, jsonify, request

from Backend.Controller.auth_controller import login_required
from Backend.Controller import tod_db, xp_db
from Backend.Service.ai.service import today_key

tod_bp = Blueprint('tod', __name__, url_prefix='/api/tod')


def _me() -> str:
    return request.current_user['username']


@tod_bp.route('/prompts', methods=['GET'])
@login_required
def list_prompts():
    level = (request.args.get('level') or '').strip().lower()
    if level in ('', 'random'):
        level = None
    elif level not in tod_db.LEVELS:
        return jsonify({'ok': False, 'error': 'Unknown level.'}), 400

    return jsonify({
        'ok': True,
        'levels': list(tod_db.LEVELS),
        'prompts': tod_db.prompts_for(level),
        'my_ratings': tod_db.my_ratings(_me()),
    })


@tod_bp.route('/draw', methods=['POST'])
@login_required
def draw():
    data = request.get_json(silent=True) or {}
    prompt = tod_db.get_prompt((data.get('prompt_id') or '').strip())
    if not prompt:
        return jsonify({'ok': False, 'error': 'Unknown prompt.'}), 404

    # The card was actually dealt; the rating endpoint checks this row.
    tod_db.record_draw(_me(), prompt['id'], today_key())
    return jsonify({'ok': True, 'prompt': prompt})


@tod_bp.route('/rate', methods=['POST'])
@login_required
def rate():
    data = request.get_json(silent=True) or {}
    prompt_id = (data.get('prompt_id') or '').strip()
    prompt = tod_db.get_prompt(prompt_id)
    if not prompt:
        return jsonify({'ok': False, 'error': 'Unknown prompt.'}), 404

    try:
        rating = int(data.get('rating'))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'Rating must be 1–5.'}), 400
    if rating < 1 or rating > 5:
        return jsonify({'ok': False, 'error': 'Rating must be 1–5.'}), 400

    tags = data.get('tags')
    if tags is not None and not isinstance(tags, list):
        return jsonify({'ok': False, 'error': 'tags must be a list.'}), 400
    tags = [str(t)[:24] for t in (tags or [])][:3]

    # You can only rate what you were dealt. This is the anti-farm gate.
    if not tod_db.has_drawn(_me(), prompt_id):
        return jsonify({'ok': False, 'error': 'Draw the card before rating it.'}), 403

    first_time = tod_db.rate(_me(), prompt_id, rating, tags)
    rewarded = xp_db.award(_me(), 'tod_rate', prompt_id) if first_time else False
    return jsonify({'ok': True, 'first_time': first_time, 'xp_awarded': rewarded})


@tod_bp.route('/stats', methods=['GET'])
@login_required
def stats():
    role = request.current_user.get('role')
    if role not in ('horizon', 'admin'):
        return jsonify({'ok': False, 'error': 'Admin only.'}), 403
    return jsonify({'ok': True, **tod_db.stats()})
