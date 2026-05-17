"""
feedback_controller.py
Message board — all logged-in users can post, reply, and like messages.
"""

from flask import Blueprint, request, jsonify
from Backend.Controller.auth_controller import login_required
from Backend.Controller import market_db
from Backend.Controller.user_manager import user_manager

feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')


def _build_avatar_map():
    users = user_manager._load_users()
    return {u.get('username'): u.get('avatar_url') for u in users.values()}


def _enrich(messages, liked_set, avatar_map):
    """Attach avatar_url and liked_by_me to a list of message dicts (in-place)."""
    for m in messages:
        m['avatar_url'] = avatar_map.get(m['username'])
        m['liked_by_me'] = m['id'] in liked_set
        if m.get('reply_to'):
            m['reply_to']['avatar_url'] = avatar_map.get(m['reply_to']['username'])


@feedback_bp.route('/messages', methods=['GET'])
@login_required
def get_messages():
    page   = max(1, int(request.args.get('page', 1)))
    result = market_db.get_messages(page=page, per_page=5)
    me         = request.current_user['username']
    avatar_map = _build_avatar_map()

    # Collect all ids to batch-check likes
    all_ids = []
    for m in result['messages']:
        all_ids.append(m['id'])
        all_ids.extend(r['id'] for r in m.get('top_replies', []))
    liked_set = market_db.get_liked_ids(me, all_ids)

    for m in result['messages']:
        m['avatar_url']   = avatar_map.get(m['username'])
        m['liked_by_me']  = m['id'] in liked_set
        _enrich(m.get('top_replies', []), liked_set, avatar_map)

    return jsonify({'ok': True, **result})


@feedback_bp.route('/messages', methods=['POST'])
@login_required
def post_message():
    data    = request.get_json() or {}
    content = data.get('content', '').strip()
    if not content:
        return jsonify({'ok': False, 'error': 'Message cannot be empty.'}), 400
    if len(content) > 500:
        return jsonify({'ok': False, 'error': 'Message too long (max 500 characters).'}), 400

    reply_to_id = data.get('reply_to_id') or None
    user        = request.current_user
    message     = market_db.post_message(
        username=user['username'],
        display_name=user['display_name'],
        content=content,
        reply_to_id=reply_to_id,
    )
    message['avatar_url']  = user.get('avatar_url')
    message['liked_by_me'] = False
    return jsonify({'ok': True, 'message': message}), 201


@feedback_bp.route('/messages/<message_id>/replies', methods=['GET'])
@login_required
def get_replies(message_id):
    replies    = market_db.get_replies(message_id)
    me         = request.current_user['username']
    avatar_map = _build_avatar_map()
    liked_set  = market_db.get_liked_ids(me, [r['id'] for r in replies])
    _enrich(replies, liked_set, avatar_map)
    return jsonify({'ok': True, 'replies': replies})


@feedback_bp.route('/messages/<message_id>/like', methods=['POST'])
@login_required
def like_message(message_id):
    me     = request.current_user['username']
    result = market_db.toggle_like(message_id, me)
    if result is None:
        return jsonify({'ok': False, 'error': 'Message not found.'}), 404
    return jsonify({'ok': True, **result})


@feedback_bp.route('/messages/<message_id>', methods=['DELETE'])
@login_required
def delete_message(message_id):
    user     = request.current_user
    is_admin = 'admin' in user.get('role_info', {}).get('permissions', [])
    ok       = market_db.delete_message(message_id, user['username'], is_admin)
    if not ok:
        return jsonify({'ok': False, 'error': 'Message not found or permission denied.'}), 404
    return jsonify({'ok': True})
