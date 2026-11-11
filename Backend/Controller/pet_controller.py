"""
pet_controller.py
/api/pet/* — the virtual cat. Thin: validation and the XP hand-off live here,
the meters, the once-a-day constraint and the decay live in pet_db.

  GET   /api/pet/me          my cat + meters + stage + what is done today
  POST  /api/pet/adopt       {breed, name}
  PATCH /api/pet/me          {name}             rename; the breed is fixed
  POST  /api/pet/care        {action: feed|play|brush}
  POST  /api/pet/pet         {username}         a visitor's pat, once a day
  GET   /api/pet/breeds      the ten breeds, for the adoption picker
"""

from flask import Blueprint, jsonify, request

from Backend.Controller.auth_controller import login_required
from Backend.Controller import pet_db, xp_db

pet_bp = Blueprint('pet', __name__, url_prefix='/api/pet')


def _me() -> str:
    return request.current_user['username']


@pet_bp.route('/breeds', methods=['GET'])
@login_required
def breeds():
    return jsonify({'ok': True, 'breeds': [
        {'slug': slug, 'zh': zh, 'en': en} for slug, (zh, en) in pet_db.BREEDS.items()
    ], 'max_name': pet_db.MAX_NAME_CHARS})


@pet_bp.route('/me', methods=['GET'])
@login_required
def me():
    pet = pet_db.get_pet(_me())
    return jsonify({'ok': True, 'pet': pet})


@pet_bp.route('/adopt', methods=['POST'])
@login_required
def adopt():
    data = request.get_json(silent=True) or {}
    pet, err = pet_db.adopt(_me(), str(data.get('breed') or ''), data.get('name'))
    if err:
        code = 409 if 'already' in err else 400
        return jsonify({'ok': False, 'error': err}), code
    return jsonify({'ok': True, 'pet': pet}), 201


@pet_bp.route('/me', methods=['PATCH'])
@login_required
def rename():
    data = request.get_json(silent=True) or {}
    pet, err = pet_db.rename(_me(), data.get('name'))
    if err:
        return jsonify({'ok': False, 'error': err}), 404 if err == 'No cat yet.' else 400
    return jsonify({'ok': True, 'pet': pet})


@pet_bp.route('/care', methods=['POST'])
@login_required
def care():
    data = request.get_json(silent=True) or {}
    action = str(data.get('action') or '')
    username = _me()
    pet, err, applied = pet_db.care(username, action)
    if err:
        return jsonify({'ok': False, 'error': err}), 404 if err == 'No cat yet.' else 400
    xp_added = False
    if applied:
        # ref = day:action, so the same chore on the same day can never pay twice
        # even if the interaction row and the xp row ever disagree.
        xp_added = xp_db.award(username, pet_db.XP_CARE, f"{pet['day']}:{action}", day=pet['day'])
    return jsonify({'ok': True, 'pet': pet, 'applied': applied, 'xp_added': xp_added,
                    'xp_delta': xp_db.RULES[pet_db.XP_CARE][0] if xp_added else 0})


@pet_bp.route('/pet', methods=['POST'])
@login_required
def touch():
    data = request.get_json(silent=True) or {}
    owner = str(data.get('username') or '').strip()
    actor = _me()
    if not owner:
        return jsonify({'ok': False, 'error': 'Who?'}), 400
    if owner == actor:
        return jsonify({'ok': False, 'error': 'That is your own cat.'}), 403
    card, err, applied = pet_db.touch(owner, actor)
    if err:
        return jsonify({'ok': False, 'error': err}), 404
    xp_added = False
    if applied:
        day = pet_db.today_key()
        xp_added = xp_db.award(actor, pet_db.XP_TOUCH, f'{day}:{owner}', day=day)
    return jsonify({'ok': True, 'pet': card, 'applied': applied, 'xp_added': xp_added,
                    'xp_delta': xp_db.RULES[pet_db.XP_TOUCH][0] if xp_added else 0})
