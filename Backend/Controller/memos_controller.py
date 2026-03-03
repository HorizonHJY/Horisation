"""
memos_controller.py
Hormemo — per-user memos stored in SQLite (market.db via market_db.py).
"""

from flask import Blueprint, request, jsonify
from .auth_controller import login_required
from .market_db import (
    get_memos, create_memo, get_memo_by_id,
    update_memo, complete_memo, delete_memo, get_memo_statistics,
)

memos_bp = Blueprint('memos', __name__, url_prefix='/api/memos')


@memos_bp.route('/', methods=['GET'])
@login_required
def list_memos():
    username   = request.current_user['username']
    status     = request.args.get('status')
    memo_type  = request.args.get('type')
    priority   = request.args.get('priority')
    limit      = int(request.args.get('limit', 50))
    offset     = int(request.args.get('offset', 0))

    memos, total = get_memos(username, status=status, memo_type=memo_type,
                             priority=priority, limit=limit, offset=offset)
    return jsonify({'ok': True, 'memos': memos, 'total_count': total})


@memos_bp.route('/', methods=['POST'])
@login_required
def create_memo_route():
    username = request.current_user['username']
    data     = request.get_json() or {}

    content = data.get('content', '').strip()
    if not content:
        return jsonify({'ok': False, 'error': 'Content is required'}), 400

    memo_id = create_memo(
        username=username,
        content=content,
        memo_type=data.get('type', 'general'),
        priority=data.get('priority', 'normal'),
        tags=data.get('tags', []),
        due_date=data.get('due_date'),
    )
    return jsonify({'ok': True, 'message': 'Memo created successfully', 'memo_id': memo_id})


@memos_bp.route('/statistics', methods=['GET'])
@login_required
def memo_statistics():
    username = request.current_user['username']
    stats    = get_memo_statistics(username)
    return jsonify({'ok': True, 'statistics': stats})


@memos_bp.route('/<memo_id>', methods=['GET'])
@login_required
def get_memo_route(memo_id):
    username = request.current_user['username']
    memo     = get_memo_by_id(memo_id, username)
    if not memo:
        return jsonify({'ok': False, 'error': 'Memo not found'}), 404
    return jsonify({'ok': True, 'memo': memo})


@memos_bp.route('/<memo_id>', methods=['PUT'])
@login_required
def update_memo_route(memo_id):
    username = request.current_user['username']
    data     = request.get_json() or {}
    fields   = {k: data[k] for k in ('content', 'priority', 'tags', 'due_date', 'status') if k in data}
    if not update_memo(memo_id, username, **fields):
        return jsonify({'ok': False, 'error': 'Memo not found'}), 404
    return jsonify({'ok': True, 'message': 'Memo updated successfully'})


@memos_bp.route('/<memo_id>', methods=['DELETE'])
@login_required
def delete_memo_route(memo_id):
    username = request.current_user['username']
    if not delete_memo(memo_id, username):
        return jsonify({'ok': False, 'error': 'Memo not found'}), 404
    return jsonify({'ok': True, 'message': 'Memo deleted successfully'})


@memos_bp.route('/<memo_id>/complete', methods=['POST'])
@login_required
def complete_memo_route(memo_id):
    username = request.current_user['username']
    if not complete_memo(memo_id, username):
        return jsonify({'ok': False, 'error': 'Memo not found'}), 404
    return jsonify({'ok': True, 'message': 'Memo marked as completed'})
