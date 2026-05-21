"""
bill_controller.py
Flask Blueprint for the Bill Split feature.
All routes require login. Bills are shared by bill ID (e.g. "ABC123") -
anyone with the ID can view and edit.
"""

from flask import Blueprint, request, jsonify
from Backend.Controller.auth_controller import login_required
from Backend.Controller import bill_db

bill_bp = Blueprint('bill', __name__, url_prefix='/api/bill')


# ── Bills ─────────────────────────────────────────────────────────────────────

@bill_bp.route('/bills', methods=['POST'])
@login_required
def create_bill():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'Bill name is required.'}), 400
    bill = bill_db.create_bill(name, request.current_user['username'])
    return jsonify({'ok': True, 'bill': bill}), 201


@bill_bp.route('/my', methods=['GET'])
@login_required
def my_bills():
    bills = bill_db.get_my_bills(request.current_user['username'])
    return jsonify({'ok': True, 'bills': bills})


@bill_bp.route('/bills/<bill_id>', methods=['GET'])
@login_required
def get_bill(bill_id):
    bill = bill_db.get_bill(bill_id)
    if not bill:
        return jsonify({'ok': False, 'error': 'Bill not found.'}), 404
    return jsonify({'ok': True, 'bill': bill})


@bill_bp.route('/bills/<bill_id>', methods=['PUT'])
@login_required
def update_bill(bill_id):
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'Name cannot be empty.'}), 400
    ok = bill_db.update_bill_name(bill_id, name)
    if not ok:
        return jsonify({'ok': False, 'error': 'Bill not found.'}), 404
    return jsonify({'ok': True})


@bill_bp.route('/bills/<bill_id>', methods=['DELETE'])
@login_required
def delete_bill(bill_id):
    ok = bill_db.delete_bill(bill_id, request.current_user['username'])
    if not ok:
        return jsonify({'ok': False, 'error': 'Bill not found or permission denied.'}), 404
    return jsonify({'ok': True})


# ── Participants ──────────────────────────────────────────────────────────────

@bill_bp.route('/bills/<bill_id>/participants', methods=['POST'])
@login_required
def add_participant(bill_id):
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'Name is required.'}), 400
    ok = bill_db.add_participant(bill_id, name)
    if not ok:
        return jsonify({'ok': False, 'error': 'Bill not found.'}), 404
    return jsonify({'ok': True})


@bill_bp.route('/bills/<bill_id>/participants/<name>', methods=['DELETE'])
@login_required
def remove_participant(bill_id, name):
    bill_db.remove_participant(bill_id, name)
    return jsonify({'ok': True})


# ── Expenses ──────────────────────────────────────────────────────────────────

@bill_bp.route('/bills/<bill_id>/expenses', methods=['POST'])
@login_required
def add_expense(bill_id):
    data       = request.get_json() or {}
    desc       = data.get('desc', '').strip()
    amount     = data.get('amount')
    paid_by    = data.get('paidBy', '').strip()
    split_among = data.get('splitAmong', [])
    if not desc:
        return jsonify({'ok': False, 'error': 'Description is required.'}), 400
    try:
        amount = float(amount)
        assert amount > 0
    except Exception:
        return jsonify({'ok': False, 'error': 'Invalid amount.'}), 400
    if not paid_by or not split_among:
        return jsonify({'ok': False, 'error': 'paidBy and splitAmong are required.'}), 400
    expense = bill_db.add_expense(bill_id, desc, amount, paid_by, split_among)
    if expense is None:
        return jsonify({'ok': False, 'error': 'Bill not found.'}), 404
    return jsonify({'ok': True, 'expense': expense}), 201


@bill_bp.route('/bills/<bill_id>/expenses/<expense_id>', methods=['DELETE'])
@login_required
def delete_expense(bill_id, expense_id):
    ok = bill_db.delete_expense(bill_id, expense_id)
    if not ok:
        return jsonify({'ok': False, 'error': 'Expense not found.'}), 404
    return jsonify({'ok': True})
