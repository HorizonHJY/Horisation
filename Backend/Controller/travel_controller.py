"""
travel_controller.py
Flask Blueprint for the Travel Planner feature.
All routes require login. Plans are shared by plan ID — anyone with the
ID can view and edit (lightweight collaboration model).
"""

from flask import Blueprint, request, jsonify
from Backend.Controller.auth_controller import login_required
from Backend.Controller import travel_db

travel_bp = Blueprint('travel', __name__, url_prefix='/api/travel')


# ── Plans ─────────────────────────────────────────────────────────────────────

@travel_bp.route('/plans', methods=['POST'])
@login_required
def create_plan():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'Plan name is required.'}), 400
    plan = travel_db.create_plan(name, request.current_user['username'])
    return jsonify({'ok': True, 'plan': plan}), 201


@travel_bp.route('/my', methods=['GET'])
@login_required
def my_plans():
    plans = travel_db.get_my_plans(request.current_user['username'])
    return jsonify({'ok': True, 'plans': plans})


@travel_bp.route('/plans/<plan_id>', methods=['GET'])
@login_required
def get_plan(plan_id):
    plan = travel_db.get_plan(plan_id)
    if not plan:
        return jsonify({'ok': False, 'error': 'Plan not found.'}), 404
    return jsonify({'ok': True, 'plan': plan})


@travel_bp.route('/plans/<plan_id>', methods=['PUT'])
@login_required
def update_plan(plan_id):
    data     = request.get_json() or {}
    name     = data.get('name', '').strip() or None
    num_days = data.get('num_days')
    if name == '':
        return jsonify({'ok': False, 'error': 'Plan name cannot be empty.'}), 400
    ok = travel_db.update_plan(plan_id, name=name, num_days=num_days)
    if not ok:
        return jsonify({'ok': False, 'error': 'Plan not found.'}), 404
    plan = travel_db.get_plan(plan_id)
    return jsonify({'ok': True, 'plan': plan})


@travel_bp.route('/plans/<plan_id>', methods=['DELETE'])
@login_required
def delete_plan(plan_id):
    ok = travel_db.delete_plan(plan_id, request.current_user['username'])
    if not ok:
        return jsonify({'ok': False, 'error': 'Plan not found or permission denied.'}), 404
    return jsonify({'ok': True})


# ── Entries ───────────────────────────────────────────────────────────────────

@travel_bp.route('/plans/<plan_id>/entries/reorder', methods=['PUT'])
@login_required
def reorder_entries(plan_id):
    data   = request.get_json() or {}
    orders = data.get('orders', [])
    if not isinstance(orders, list):
        return jsonify({'ok': False, 'error': 'orders must be a list.'}), 400
    ok = travel_db.reorder_entries(plan_id, orders)
    if not ok:
        return jsonify({'ok': False, 'error': 'Plan not found.'}), 404
    return jsonify({'ok': True})


@travel_bp.route('/plans/<plan_id>/entries', methods=['POST'])
@login_required
def add_entry(plan_id):
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'Name is required.'}), 400
    entry = travel_db.add_entry(
        plan_id=plan_id,
        day_number=int(data.get('day_number', 1)),
        type_=data.get('type', 'other'),
        time_start=data.get('time_start', ''),
        time_end=data.get('time_end', ''),
        name=name,
        address=data.get('address', ''),
        notes=data.get('notes', ''),
        display_order=int(data.get('display_order', 0)),
    )
    if entry is None:
        return jsonify({'ok': False, 'error': 'Plan not found.'}), 404
    return jsonify({'ok': True, 'entry': entry}), 201


@travel_bp.route('/plans/<plan_id>/entries/<entry_id>', methods=['PUT'])
@login_required
def update_entry(plan_id, entry_id):
    data    = request.get_json() or {}
    allowed = {'day_number', 'type', 'time_start', 'time_end', 'name', 'address', 'notes', 'display_order'}
    kwargs  = {k: v for k, v in data.items() if k in allowed}
    if 'day_number'    in kwargs: kwargs['day_number']    = int(kwargs['day_number'])
    if 'display_order' in kwargs: kwargs['display_order'] = int(kwargs['display_order'])
    entry = travel_db.update_en