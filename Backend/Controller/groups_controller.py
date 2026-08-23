"""
groups_controller.py
Group system REST API: create groups, add/remove members, group chat.
Independent concept — membership is NOT tied to friendship (add by username).

Blueprint: /api/groups
"""

from flask import Blueprint, request, jsonify
from .auth_controller import login_required
from .user_manager import user_manager
from . import market_db

groups_bp = Blueprint('groups', __name__, url_prefix='/api/groups')


def _enrich_user(u: dict) -> dict:
    return {
        'username':     u.get('username'),
        'display_name': u.get('display_name', u.get('username')),
        'avatar_url':   u.get('avatar_url'),
    }


def _require_member(gid: str, username: str):
    """Return (member_dict, error_response) — error is None if OK."""
    if not market_db.get_group(gid):
        return None, (jsonify({'ok': False, 'error': 'Group not found'}), 404)
    if not market_db.is_group_member(gid, username):
        return None, (jsonify({'ok': False, 'error': 'Not a group member'}), 403)
    return True, None


@groups_bp.route('', methods=['GET'])
@login_required
def my_groups():
    me = request.current_user['username']
    groups = market_db.get_my_groups(me)
    # attach member count
    for g in groups:
        g['member_count'] = len(market_db.get_group_members(g['id']))
    return jsonify({'ok': True, 'groups': groups})


@groups_bp.route('', methods=['POST'])
@login_required
def create_group():
    me   = request.current_user['username']
    name = (request.get_json() or {}).get('name', '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'name required'}), 400
    if len(name) > 50:
        return jsonify({'ok': False, 'error': 'name too long (max 50)'}), 400
    g = market_db.create_group(name, me)
    if not g:
        return jsonify({'ok': False, 'error': 'Could not create group'}), 400
    g['member_count'] = 1
    return jsonify({'ok': True, 'group': g}), 201


@groups_bp.route('/<gid>', methods=['GET'])
@login_required
def group_detail(gid):
    me = request.current_user['username']
    _, err = _require_member(gid, me)
    if err:
        return err
    g = market_db.get_group(gid)
    members = market_db.get_group_members(gid)
    users = user_manager._load_users()
    for m in members:
        _, u = user_manager._find_user(users, m['username'])
        if u:
            m['display_name'] = u.get('display_name', u.get('username'))
            m['avatar_url']   = u.get('avatar_url')
        else:
            m['display_name'] = m['username']
            m['avatar_url']   = None
    g['members'] = members
    g['member_count'] = len(members)
    return jsonify({'ok': True, 'group': g})


@groups_bp.route('/<gid>', methods=['PUT'])
@login_required
def update_group(gid):
    me   = request.current_user['username']
    if not market_db.get_group(gid):
        return jsonify({'ok': False, 'error': 'Group not found'}), 404
    if not market_db.is_group_member(gid, me, role='owner'):
        return jsonify({'ok': False, 'error': 'Only the owner can rename'}), 403
    name = (request.get_json() or {}).get('name', '').strip()
    if not market_db.rename_group(gid, name):
        return jsonify({'ok': False, 'error': 'name required (max 50)'}), 400
    return jsonify({'ok': True, 'group': market_db.get_group(gid)})


@groups_bp.route('/<gid>', methods=['DELETE'])
@login_required
def delete_group(gid):
    me = request.current_user['username']
    if not market_db.get_group(gid):
        return jsonify({'ok': False, 'error': 'Group not found'}), 404
    if not market_db.is_group_member(gid, me, role='owner'):
        return jsonify({'ok': False, 'error': 'Only the owner can delete the group'}), 403
    market_db.delete_group(gid)
    return jsonify({'ok': True})


@groups_bp.route('/<gid>/members', methods=['POST'])
@login_required
def add_member(gid):
    me       = request.current_user['username']
    if not market_db.get_group(gid):
        return jsonify({'ok': False, 'error': 'Group not found'}), 404
    if not market_db.is_group_member(gid, me, role='owner'):
        return jsonify({'ok': False, 'error': 'Only the owner can add members'}), 403

    username = (request.get_json() or {}).get('username', '').strip()
    if not username:
        return jsonify({'ok': False, 'error': 'username required'}), 400
    if username == me:
        return jsonify({'ok': False, 'error': 'Cannot add yourself'}), 400

    users = user_manager._load_users()
    key, u = user_manager._find_user(users, username)
    if key is None or not u.get('is_active', True):
        return jsonify({'ok': False, 'error': 'User not found or inactive'}), 404

    if not market_db.add_group_member(gid, username):
        return jsonify({'ok': False, 'error': 'Already a member'}), 400
    return jsonify({'ok': True, 'member': _enrich_user(u)}), 201


@groups_bp.route('/<gid>/members/<username>', methods=['DELETE'])
@login_required
def remove_member(gid, username):
    me = request.current_user['username']
    if not market_db.get_group(gid):
        return jsonify({'ok': False, 'error': 'Group not found'}), 404

    # Either owner kicks someone, or a member leaves themselves
    is_owner  = market_db.is_group_member(gid, me, role='owner')
    is_self   = (username == me)
    if not (is_owner or is_self):
        return jsonify({'ok': False, 'error': 'Not allowed'}), 403
    if is_owner and username == me:
        return jsonify({'ok': False, 'error': 'Owner must delete the group to leave'}), 400

    if username != me and not market_db.is_group_member(gid, username):
        return jsonify({'ok': False, 'error': 'User is not a member'}), 404

    if not market_db.remove_group_member(gid, username):
        return jsonify({'ok': False, 'error': 'Owner cannot be removed; delete the group instead'}), 400
    return jsonify({'ok': True})


@groups_bp.route('/<gid>/messages', methods=['GET'])
@login_required
def get_messages(gid):
    me = request.current_user['username']
    _, err = _require_member(gid, me)
    if err:
        return err
    msgs = market_db.get_group_messages(gid)
    users = user_manager._load_users()
    for m in msgs:
        _, u = user_manager._find_user(users, m['sender'])
        m['sender_display'] = u.get('display_name', m['sender']) if u else m['sender']
        m['sender_avatar']  = u.get('avatar_url') if u else None
    return jsonify({'ok': True, 'messages': msgs})


@groups_bp.route('/<gid>/messages', methods=['POST'])
@login_required
def send_message(gid):
    me = request.current_user['username']
    _, err = _require_member(gid, me)
    if err:
        return err
    content = (request.get_json() or {}).get('content', '').strip()
    if not content:
        return jsonify({'ok': False, 'error': 'content required'}), 400
    if len(content) > 1000:
        return jsonify({'ok': False, 'error': 'content too long (max 1000)'}), 400
    m = market_db.post_group_message(gid, me, content)
    return jsonify({'ok': True, 'message': m}), 201
