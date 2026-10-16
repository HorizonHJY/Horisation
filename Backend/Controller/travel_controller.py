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
    entry = travel_db.update_entry(entry_id, plan_id, **kwargs)
    if entry is None:
        return jsonify({'ok': False, 'error': 'Entry not found.'}), 404
    return jsonify({'ok': True, 'entry': entry})


@travel_bp.route('/plans/<plan_id>/entries/<entry_id>', methods=['DELETE'])
@login_required
def delete_entry(plan_id, entry_id):
    ok = travel_db.delete_entry(entry_id, plan_id)
    if not ok:
        return jsonify({'ok': False, 'error': 'Entry not found.'}), 404
    return jsonify({'ok': True})


# ── Expenses (the plan's bill) ─────────────────────────────────────────────
# The Expenses tab speaks the same shapes as /api/bill/*, but scoped to the
# plan id so the page never has to know the bill exists. One code opens both.

def _plan_bill(plan_id):
    """The plan's bill, created on demand for plans made before this feature.

    A plan from before 2026-10-16 has bill_id = NULL. Rather than force a
    migration, the first visit to a plan's Expenses tab mints the bill — and
    records it on the plan so the next visit is a plain lookup.
    """
    from Backend.Controller import bill_db
    plan = travel_db.get_plan(plan_id)
    if not plan:
        return None
    bill_id = plan.get('bill_id')
    if bill_id and bill_db.get_bill(bill_id):
        return bill_db.get_bill(bill_id)
    # Backfill: use the plan id, same as create_plan does.
    bill = bill_db.create_bill_for_plan(plan['id'], plan['name'], plan['created_by'])
    travel_db.set_plan_bill_id(plan['id'], plan['id'])
    return bill


@travel_bp.route('/plans/<plan_id>/bill', methods=['GET'])
@login_required
def get_plan_bill(plan_id):
    bill = _plan_bill(plan_id)
    if not bill:
        return jsonify({'ok': False, 'error': 'Plan not found.'}), 404
    return jsonify({'ok': True, 'bill': bill})


@travel_bp.route('/plans/<plan_id>/bill/participants', methods=['POST'])
@login_required
def add_plan_participant(plan_id):
    from Backend.Controller import bill_db
    bill = _plan_bill(plan_id)
    if not bill:
        return jsonify({'ok': False, 'error': 'Plan not found.'}), 404
    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'Name is required.'}), 400
    bill_db.add_participant(bill['id'], name)
    return jsonify({'ok': True, 'bill': bill_db.get_bill(bill['id'])})


@travel_bp.route('/plans/<plan_id>/bill/participants/<name>', methods=['DELETE'])
@login_required
def remove_plan_participant(plan_id, name):
    from Backend.Controller import bill_db
    bill = _plan_bill(plan_id)
    if not bill:
        return jsonify({'ok': False, 'error': 'Plan not found.'}), 404
    bill_db.remove_participant(bill['id'], name)
    return jsonify({'ok': True, 'bill': bill_db.get_bill(bill['id'])})


@travel_bp.route('/plans/<plan_id>/bill/expenses', methods=['POST'])
@login_required
def add_plan_expense(plan_id):
    from Backend.Controller import bill_db
    bill = _plan_bill(plan_id)
    if not bill:
        return jsonify({'ok': False, 'error': 'Plan not found.'}), 404
    data        = request.get_json() or {}
    desc        = (data.get('desc') or '').strip()
    paid_by     = (data.get('paidBy') or '').strip()
    split_among = data.get('splitAmong', [])
    if not desc:
        return jsonify({'ok': False, 'error': 'Description is required.'}), 400
    try:
        amount = float(data.get('amount'))
        assert amount > 0
    except Exception:
        return jsonify({'ok': False, 'error': 'Invalid amount.'}), 400
    if not paid_by or not split_among:
        return jsonify({'ok': False, 'error': 'paidBy and splitAmong are required.'}), 400
    expense = bill_db.add_expense(bill['id'], desc, amount, paid_by, split_among)
    return jsonify({'ok': True, 'expense': expense, 'bill': bill_db.get_bill(bill['id'])}), 201


@travel_bp.route('/plans/<plan_id>/bill/expenses/<expense_id>', methods=['DELETE'])
@login_required
def delete_plan_expense(plan_id, expense_id):
    from Backend.Controller import bill_db
    bill = _plan_bill(plan_id)
    if not bill:
        return jsonify({'ok': False, 'error': 'Plan not found.'}), 404
    ok = bill_db.delete_expense(bill['id'], expense_id)
    if not ok:
        return jsonify({'ok': False, 'error': 'Expense not found.'}), 404
    return jsonify({'ok': True, 'bill': bill_db.get_bill(bill['id'])})


@travel_bp.route('/friends', methods=['GET'])
@login_required
def plan_friend_options():
    """Friends for the participant picker — name to offer as a datalist.

    The picker still takes free text (Horizon's call): the group is not only
    friends, and typing "妈妈" must work. This is a convenience list, not a
    constraint.
    """
    from Backend.Controller import market_db
    from Backend.Controller.user_manager import user_manager
    me = request.current_user['username']
    users = user_manager._load_users()
    out = []
    for username in market_db.get_friends(me):
        _, u = user_manager._find_user(users, username)
        if u:
            out.append({
                'username':     u.get('username'),
                'display_name': u.get('display_name', u.get('username')),
            })
    return jsonify({'ok': True, 'friends': out})
