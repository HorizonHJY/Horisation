"""
feedback_controller.py
Message board — all logged-in users can post and read messages.
"""

from flask import Blueprint, request, jsonify
from Backend.Controller.auth_controller import login_required
from Backend.Controller import market_db
from Backend.Controller.user_manager import user_manager

feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')


@feedback_bp.route('/messages', methods=['GET'])
@login_required
def get_messages():
    messages = market_db.get_messages(limit=200)

    # Attach current avatar_url for each user
    users = user_manager._load_users()
    avatar_map = {
        u.get('username'): u.get('avatar_url')
        for u in users.values()
    }
    for m in messages:
        m['avatar_url'] = avatar_map.get(m['username'])

    return jsonify({'ok': True, 'messages': messages})


@feedback_bp.route('/messages', methods=['POST'])
@login_required
def post_message():
    data    = request.get_json() or {}
    content = data.get('content', '').strip()
    if not content:
        return jsonify({'ok': False, 'error': 'Message cannot be empty.'}), 400
    if len(content) > 500:
        return jsonify({'ok': False, 'error': 'Message too long (max 500 characters).'}), 400

    user    = request.current_user
    message = market_db.post_message(
        username=user['username'],
        display_name=user['display_name'],
        content=content,
    )
    message['avatar_url'] = user.get('avatar_url')
    return jsonify({'ok': True, 'message': message}), 201


@feedback_bp.route('/messages/<message_id>', methods=['DELETE'])
@login_required
def delete_message(message_id):
    user     = request.current_user
    is_admin = 'admin' in user.get('role_info', {}).get('permissions', [])
    ok       = market_db.delete_message(message_id, user['username'], is_admin)
    if not ok:
        return jsonify({'ok': False, 'error': 'Message not found or permission denied.'}), 404
    return jsonify({'ok': True})
