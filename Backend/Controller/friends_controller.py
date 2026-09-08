"""
friends_controller.py
Friend system REST API: requests, friendship, chat history, contact info.
"""

from flask import Blueprint, request, jsonify
from .auth_controller import login_required
from .user_manager import user_manager
from . import market_db

friends_bp = Blueprint('friends', __name__, url_prefix='/api/friends')


def _enrich_user(u: dict) -> dict:
    return {
        'username':     u.get('username'),
        'display_name': u.get('display_name', u.get('username')),
        'avatar_url':   u.get('avatar_url'),
    }

def _with_sender_identity(reqs: list) -> list:
    """Attach display name and avatar for each request's sender.

    Applied to pushed requests as well as fetched ones. Without it a request
    that arrives over the socket renders as a bare username with no avatar
    until something re-fetches it, which looks like the notification "did not
    work" even though it did.
    """
    users = user_manager._load_users()
    for r in reqs:
        _, u = user_manager._find_user(users, r['from_user'])
        r['from_display'] = u.get('display_name', r['from_user']) if u else r['from_user']
        r['from_avatar']  = u.get('avatar_url') if u else None
    return reqs


@friends_bp.route('/users', methods=['GET'])
@login_required
def search_users():
    """Search users by username or display name (requires ?q= with 2+ chars)."""
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify({'ok': True, 'users': []})
    me      = request.current_user['username']
    results = user_manager.search_users(q)
    filtered = [u for u in results if u['username'] != me and u.get('is_active', True)]
    return jsonify({'ok': True, 'users': [_enrich_user(u) for u in filtered]})


@friends_bp.route('/requests', methods=['POST'])
@login_required
def send_request():
    body    = request.get_json() or {}
    me      = request.current_user['username']
    to_user = body.get('to_user', '').strip()
    message = body.get('message', '').strip() or None

    if not to_user:
        return jsonify({'ok': False, 'error': 'to_user required'}), 400
    if to_user == me:
        return jsonify({'ok': False, 'error': 'Cannot add yourself'}), 400

    users = user_manager._load_users()
    key, _ = user_manager._find_user(users, to_user)
    if key is None:
        return jsonify({'ok': False, 'error': 'User not found'}), 404

    if market_db.are_friends(me, to_user):
        return jsonify({'ok': False, 'error': 'Already friends'}), 400

    row = market_db.send_friend_request(me, to_user, message)
    if row is None:
        return jsonify({'ok': False, 'error': 'Request already pending'}), 400

    from .friends_socket import notify_friend_request
    # Push the same shape the snapshot endpoint returns, so a pushed request
    # renders identically to a fetched one.
    notify_friend_request(to_user, _with_sender_identity([dict(row)])[0])
    return jsonify({'ok': True, 'request': row}), 201


@friends_bp.route('/requests/pending', methods=['GET'])
@login_required
def get_pending():
    me   = request.current_user['username']
    reqs = _with_sender_identity(market_db.get_pending_requests(me))
    return jsonify({'ok': True, 'requests': reqs})


@friends_bp.route('/requests/sent', methods=['GET'])
@login_required
def get_sent():
    me = request.current_user['username']
    return jsonify({'ok': True, 'requests': market_db.get_sent_requests(me)})


@friends_bp.route('/requests/<req_id>', methods=['PUT'])
@login_required
def respond_request(req_id):
    me     = request.current_user['username']
    action = (request.get_json() or {}).get('action')
    if action not in ('accept', 'reject'):
        return jsonify({'ok': False, 'error': 'action must be accept or reject'}), 400

    ok = market_db.respond_friend_request(req_id, me, accept=(action == 'accept'))
    if not ok:
        return jsonify({'ok': False, 'error': 'Request not found or not yours'}), 404

    if action == 'accept':
        from .friends_socket import notify_friend_accepted
        notify_friend_accepted(me, req_id)
    return jsonify({'ok': True})


@friends_bp.route('/list', methods=['GET'])
@login_required
def get_friends():
    me      = request.current_user['username']
    friends = market_db.get_friends(me)
    users   = user_manager._load_users()
    result  = []
    for username in friends:
        _, u = user_manager._find_user(users, username)
        if u:
            result.append(_enrich_user(u))
    return jsonify({'ok': True, 'friends': result})


@friends_bp.route('/<username>', methods=['DELETE'])
@login_required
def unfriend(username):
    me = request.current_user['username']
    if not market_db.remove_friend(me, username):
        return jsonify({'ok': False, 'error': 'Not friends'}), 404
    return jsonify({'ok': True})


@friends_bp.route('/<username>/contact', methods=['GET'])
@login_required
def get_contact(username):
    me = request.current_user['username']
    if not market_db.are_friends(me, username):
        return jsonify({'ok': False, 'error': 'Not friends'}), 403
    u = market_db.db_get_user(username)
    if not u:
        return jsonify({'ok': False, 'error': 'User not found'}), 404
    if u.get('contact_hidden'):
        return jsonify({'ok': False, 'error': 'Contact is hidden'}), 403
    if not market_db.has_contact_access(me, username):
        return jsonify({'ok': False, 'error': 'No contact access. Send a contact request first.'}), 403
    return jsonify({
        'ok':          True,
        'phone':       u.get('phone', ''),
        'wechat':      u.get('wechat', ''),
        'address':     u.get('address', ''),
        'postal_code': u.get('postal_code', ''),
    })


@friends_bp.route('/<username>/contact/request', methods=['POST'])
@login_required
def request_contact(username):
    me = request.current_user['username']
    if not market_db.are_friends(me, username):
        return jsonify({'ok': False, 'error': 'Not friends'}), 403
    users = user_manager._load_users()
    _, u  = user_manager._find_user(users, username)
    if not u:
        return jsonify({'ok': False, 'error': 'User not found'}), 404
    if u.get('contact_hidden'):
        return jsonify({'ok': False, 'error': 'Contact is hidden'}), 400
    row = market_db.send_contact_request(me, username)
    if row is None:
        return jsonify({'ok': False, 'error': 'Request already sent or approved'}), 400

    from .friends_socket import notify_contact_request
    notify_contact_request(username, _with_sender_identity([dict(row)])[0])
    return jsonify({'ok': True, 'request': row}), 201


@friends_bp.route('/contact/requests', methods=['GET'])
@login_required
def get_contact_requests():
    """Pending contact requests I need to respond to."""
    me   = request.current_user['username']
    reqs = _with_sender_identity(market_db.get_contact_requests_received(me))
    return jsonify({'ok': True, 'requests': reqs})


@friends_bp.route('/contact/sent', methods=['GET'])
@login_required
def get_contact_sent():
    """Contact requests I have sent (all statuses)."""
    me = request.current_user['username']
    return jsonify({'ok': True, 'requests': market_db.get_contact_requests_sent(me)})


@friends_bp.route('/unread', methods=['GET'])
@login_required
def get_unread():
    me       = request.current_user['username']
    by_friend = market_db.get_unread_counts(me)
    return jsonify({'ok': True, 'total': sum(by_friend.values()), 'by_friend': by_friend})


@friends_bp.route('/notifications', methods=['GET'])
@login_required
def get_notifications():
    """Everything the badge and the banners need, in one round trip.

    The client asks for this when its socket connects and on a slow fallback
    timer. Socket events keep it current in between; this is what makes the
    state self-heal after a reconnect, rather than depending on having caught
    every event while the tab was asleep.
    """
    me        = request.current_user['username']
    by_friend = market_db.get_unread_counts(me)
    return jsonify({
        'ok': True,
        'unread': {'total': sum(by_friend.values()), 'by_friend': by_friend},
        'friend_requests':  _with_sender_identity(market_db.get_pending_requests(me)),
        'contact_requests': _with_sender_identity(market_db.get_contact_requests_received(me)),
    })


@friends_bp.route('/<username>/read', methods=['POST'])
@login_required
def mark_read(username):
    me = request.current_user['username']
    ua, ub   = sorted([me, username])
    room_key = f'{ua}:{ub}'
    market_db.mark_chat_read(me, room_key)
    return jsonify({'ok': True})


@friends_bp.route('/contact/shared', methods=['GET'])
@login_required
def get_contact_shared():
    """Approved contact requests I have shared (people who can currently see my contact)."""
    me   = request.current_user['username']
    reqs = market_db.get_contact_requests_approved(me)
    users = user_manager._load_users()
    for r in reqs:
        _, u = user_manager._find_user(users, r['from_user'])
        r['from_display'] = u.get('display_name', r['from_user']) if u else r['from_user']
        r['from_avatar']  = u.get('avatar_url') if u else None
    return jsonify({'ok': True, 'requests': reqs})


@friends_bp.route('/contact/requests/<int:req_id>', methods=['PUT'])
@login_required
def respond_contact(req_id):
    me     = request.current_user['username']
    action = (request.get_json() or {}).get('action')
    if action == 'revoke':
        ok = market_db.revoke_contact_access(req_id, me)
        if not ok:
            return jsonify({'ok': False, 'error': 'Request not found or not yours'}), 404
        return jsonify({'ok': True})
    if action not in ('approve', 'decline'):
        return jsonify({'ok': False, 'error': 'action must be approve, decline, or revoke'}), 400
    requester = market_db.respond_contact_request(req_id, me, accept=(action == 'approve'))
    if not requester:
        return jsonify({'ok': False, 'error': 'Request not found or not yours'}), 404

    from .friends_socket import notify_contact_resolved
    notify_contact_resolved(req_id, me, requester, action)
    return jsonify({'ok': True})


@friends_bp.route('/<username>/history', methods=['GET'])
@login_required
def get_history(username):
    me = request.current_user['username']
    return jsonify({'ok': True, 'messages': market_db.get_chat_history(me, username)})
