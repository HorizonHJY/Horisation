# Backend/Controller/auth_controller.py
"""
用户认证控制器
处理登录、注册、会话管理等API接口
"""

import re
from flask import Blueprint, request, jsonify, session, redirect, url_for
from functools import wraps
from .user_manager import user_manager

# 创建认证蓝图
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


def _validate_password(password):
    """8+ chars, uppercase, lowercase, digit. Returns error string or None."""
    if len(password) < 8:
        return 'Password must be at least 8 characters'
    if not re.search(r'[A-Z]', password):
        return 'Password must contain at least one uppercase letter'
    if not re.search(r'[a-z]', password):
        return 'Password must contain at least one lowercase letter'
    if not re.search(r'\d', password):
        return 'Password must contain at least one number'
    return None

def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_token = session.get('session_token')
        if not session_token:
            return jsonify({'ok': False, 'error': 'Login required', 'code': 'LOGIN_REQUIRED'}), 401

        user_info = user_manager.validate_session(session_token)
        if not user_info:
            session.pop('session_token', None)
            return jsonify({'ok': False, 'error': 'Invalid session', 'code': 'INVALID_SESSION'}), 401

        # 将用户信息添加到request中
        request.current_user = user_info
        return f(*args, **kwargs)

    return decorated_function

def admin_required(f):
    """管理员权限验证装饰器"""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        user_info = request.current_user
        if not user_manager.check_permission(user_info['username'], 'admin'):
            return jsonify({'ok': False, 'error': 'Admin permission required', 'code': 'ADMIN_REQUIRED'}), 403

        return f(*args, **kwargs)

    return decorated_function

@auth_bp.route('/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'ok': False, 'error': 'Username and password required'}), 400

        # 认证用户
        success, user_info = user_manager.authenticate_user(username, password)

        if not success:
            return jsonify({'ok': False, 'error': 'Invalid username or password'}), 401

        # 创建会话
        session_token = user_manager.create_session(username)
        session['session_token'] = session_token

        return jsonify({
            'ok': True,
            'message': 'Login successful',
            'user': user_info,
            'session_token': session_token
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Login failed: {str(e)}'}), 500

@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """用户登出"""
    try:
        session_token = session.get('session_token')
        if session_token:
            user_manager.logout_user(session_token)
            session.pop('session_token', None)

        return jsonify({'ok': True, 'message': 'Logout successful'})

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Logout failed: {str(e)}'}), 500

@auth_bp.route('/register', methods=['POST'])
@admin_required  # 只有管理员可以注册新用户
def register():
    """用户注册（管理员功能）"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        username = data.get('username', '').strip()
        password = data.get('password', '')
        role = data.get('role', 'user')
        email = data.get('email', '').strip()
        display_name = data.get('display_name', '').strip()

        if not username or not password:
            return jsonify({'ok': False, 'error': 'Username and password required'}), 400

        pw_err = _validate_password(password)
        if pw_err:
            return jsonify({'ok': False, 'error': pw_err}), 400

        success, message = user_manager.create_user(
            username=username, password=password, role=role,
            email=email, display_name=display_name
        )

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({'ok': True, 'message': message})

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Registration failed: {str(e)}'}), 500


@auth_bp.route('/signup', methods=['POST'])
def signup():
    """Public self-registration — requires a valid invite code."""
    from .market_db import validate_invite_code
    data         = request.get_json() or {}
    username     = data.get('username', '').strip()
    password     = data.get('password', '')
    display_name = data.get('display_name', '').strip()
    invite_code  = data.get('invite_code', '').strip()

    if not username or not password or not invite_code:
        return jsonify({'ok': False, 'error': 'Username, password and invite code are required'}), 400
    pw_err = _validate_password(password)
    if pw_err:
        return jsonify({'ok': False, 'error': pw_err}), 400
    if not validate_invite_code(invite_code):
        return jsonify({'ok': False, 'error': 'Invalid or expired invite code'}), 403

    success, message = user_manager.create_user(
        username=username, password=password, role='user',
        display_name=display_name or username,
    )
    if not success:
        return jsonify({'ok': False, 'error': message}), 400

    # Auto-login
    session_token = user_manager.create_session(username)
    session['session_token'] = session_token
    _, user_info = user_manager.authenticate_user(username, password)
    return jsonify({'ok': True, 'user': user_info})


# ── Invite code management (horizon only) ─────────────────────────────────────

@auth_bp.route('/invite-codes', methods=['GET'])
@login_required
def list_invite_codes():
    if request.current_user['username'] != 'horizon':
        return jsonify({'ok': False, 'error': 'Forbidden'}), 403
    from .market_db import get_invite_codes
    return jsonify({'ok': True, 'codes': get_invite_codes()})


@auth_bp.route('/invite-codes', methods=['POST'])
@login_required
def create_invite_code():
    if request.current_user['username'] != 'horizon':
        return jsonify({'ok': False, 'error': 'Forbidden'}), 403
    from .market_db import create_invite_code as db_create_invite_code
    from datetime import datetime
    data       = request.get_json() or {}
    code       = data.get('code', '').strip()
    valid_from = data.get('valid_from', '')
    valid_to   = data.get('valid_to', '')
    if not code or not valid_from or not valid_to:
        return jsonify({'ok': False, 'error': 'code, valid_from and valid_to are required'}), 400
    try:
        vf = datetime.strptime(valid_from, '%Y-%m-%d')
        vt = datetime.strptime(valid_to,   '%Y-%m-%d').replace(hour=23, minute=59, second=59)
    except ValueError:
        return jsonify({'ok': False, 'error': 'Date format must be YYYY-MM-DD'}), 400
    if vt < vf:
        return jsonify({'ok': False, 'error': 'valid_to must be after valid_from'}), 400
    result = db_create_invite_code(code, vf, vt, request.current_user['username'])
    if result is None:
        return jsonify({'ok': False, 'error': 'Invite code already exists'}), 409
    return jsonify({'ok': True, 'invite_code': result})


@auth_bp.route('/invite-codes/<int:code_id>', methods=['DELETE'])
@login_required
def delete_invite_code(code_id):
    if request.current_user['username'] != 'horizon':
        return jsonify({'ok': False, 'error': 'Forbidden'}), 403
    from .market_db import delete_invite_code as db_delete_invite_code
    if not db_delete_invite_code(code_id):
        return jsonify({'ok': False, 'error': 'Not found'}), 404
    return jsonify({'ok': True})

@auth_bp.route('/profile', methods=['GET'])
@login_required
def get_profile():
    """获取当前用户信息"""
    try:
        user_info = request.current_user
        return jsonify({
            'ok': True,
            'user': user_info
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to get profile: {str(e)}'}), 500

@auth_bp.route('/check-session', methods=['GET'])
def check_session():
    """检查会话状态"""
    try:
        session_token = session.get('session_token')
        if not session_token:
            return jsonify({'ok': False, 'logged_in': False, 'error': 'No session'})

        user_info = user_manager.validate_session(session_token)
        if not user_info:
            session.pop('session_token', None)
            return jsonify({'ok': False, 'logged_in': False, 'error': 'Invalid session'})

        return jsonify({
            'ok': True,
            'logged_in': True,
            'user': user_info
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Session check failed: {str(e)}'}), 500

@auth_bp.route('/users', methods=['GET'])
@admin_required
def list_users():
    """列出所有用户（管理员功能）"""
    try:
        users = user_manager.list_users()
        return jsonify({
            'ok': True,
            'users': users
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to list users: {str(e)}'}), 500

@auth_bp.route('/users/<username>/role', methods=['PUT'])
@admin_required
def update_user_role(username):
    """更新用户角色（管理员功能）"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        new_role = data.get('role')
        if not new_role:
            return jsonify({'ok': False, 'error': 'Role required'}), 400

        success, message = user_manager.update_user_role(username, new_role)

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({'ok': True, 'message': message})

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to update role: {str(e)}'}), 500

@auth_bp.route('/users/<username>/status', methods=['PUT'])
@admin_required
def update_user_status(username):
    """更新用户状态（管理员功能）"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        is_active = data.get('is_active')
        if is_active is None:
            return jsonify({'ok': False, 'error': 'is_active required'}), 400

        if is_active:
            success, message = user_manager.activate_user(username)
        else:
            success, message = user_manager.deactivate_user(username)

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({'ok': True, 'message': message})

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to update status: {str(e)}'}), 500

@auth_bp.route('/profile', methods=['PUT'])
@login_required
def update_own_profile():
    """User updates their own display name, email, and/or contact_info."""
    data            = request.get_json() or {}
    display_name    = data.get('display_name', '').strip() or None
    email           = data.get('email', '').strip() or None
    contact_info    = data.get('contact_info')   # None = don't update; '' = clear
    contact_hidden  = data.get('contact_hidden') # None = don't update; bool = set
    wechat          = data.get('wechat')         # None = don't update; '' = clear
    phone           = data.get('phone')          # None = don't update; '' = clear
    address         = data.get('address')        # None = don't update; '' = clear
    postal_code     = data.get('postal_code')    # None = don't update; '' = clear
    if display_name is None and email is None and contact_info is None \
            and contact_hidden is None and wechat is None and phone is None \
            and address is None and postal_code is None:
        return jsonify({'ok': False, 'error': 'Nothing to update'}), 400
    username = request.current_user['username']
    success, message = user_manager.update_user_profile(
        username, display_name=display_name, email=email,
        contact_info=contact_info, contact_hidden=contact_hidden,
        wechat=wechat, phone=phone, address=address, postal_code=postal_code,
    )
    if not success:
        return jsonify({'ok': False, 'error': message}), 400
    return jsonify({'ok': True, 'message': message})


@auth_bp.route('/avatar', methods=['POST'])
@login_required
def upload_avatar():
    """User uploads their own avatar image."""
    from Backend.Controller.r2_manager import upload_avatar as r2_upload_avatar
    file = request.files.get('avatar')
    if not file or not file.filename:
        return jsonify({'ok': False, 'error': 'No file provided'}), 400
    import os
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png'):
        return jsonify({'ok': False, 'error': 'Only JPEG/PNG allowed'}), 400
    file.seek(0, 2)
    if file.tell() > 2 * 1024 * 1024:
        return jsonify({'ok': False, 'error': 'Avatar must be under 2MB'}), 400
    file.seek(0)
    username = request.current_user['username']
    try:
        _, avatar_url = r2_upload_avatar(file.stream, username, file.filename)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Upload failed: {str(e)}'}), 500
    user_manager.update_user_profile(username, avatar_url=avatar_url)
    return jsonify({'ok': True, 'avatar_url': avatar_url})


@auth_bp.route('/password', methods=['PUT'])
@login_required
def change_own_password():
    """User changes their own password (requires current password)."""
    data = request.get_json() or {}
    current  = data.get('current_password', '')
    new_pass = data.get('new_password', '')
    if not current or not new_pass:
        return jsonify({'ok': False, 'error': 'Current and new password required'}), 400
    pw_err = _validate_password(new_pass)
    if pw_err:
        return jsonify({'ok': False, 'error': pw_err}), 400
    username = request.current_user['username']
    ok, _ = user_manager.authenticate_user(username, current)
    if not ok:
        return jsonify({'ok': False, 'error': 'Current password is incorrect'}), 403
    success, message = user_manager.reset_user_password(username, new_pass)
    if not success:
        return jsonify({'ok': False, 'error': message}), 400
    # Invalidate all sessions so the user must log in again with the new password
    from Backend.Controller.market_db import db_delete_user_sessions
    db_delete_user_sessions(username)
    session.pop('session_token', None)
    return jsonify({'ok': True, 'message': 'Password changed. Please log in again with your new password.'})


@auth_bp.route('/users/<username>/profile', methods=['PUT'])
@admin_required
def update_user_profile(username):
    data = request.get_json() or {}
    display_name = data.get('display_name')
    email = data.get('email')
    success, message = user_manager.update_user_profile(username, display_name, email)
    if not success:
        return jsonify({'ok': False, 'error': message}), 400
    return jsonify({'ok': True, 'message': message})


@auth_bp.route('/users/<username>/password', methods=['PUT'])
@admin_required
def reset_user_password(username):
    data = request.get_json() or {}
    new_password = data.get('password', '')
    if not new_password:
        return jsonify({'ok': False, 'error': 'Password required'}), 400
    pw_err = _validate_password(new_password)
    if pw_err:
        return jsonify({'ok': False, 'error': pw_err}), 400
    success, message = user_manager.reset_user_password(username, new_password)
    if not success:
        return jsonify({'ok': False, 'error': message}), 400
    return jsonify({'ok': True, 'message': message})


@auth_bp.route('/users/<username>/public', methods=['GET'])
@login_required
def get_user_public(username):
    """Return public-safe profile info for any user (no email / contact / password)."""
    from Backend.Controller.market_db import db_get_user
    u = db_get_user(username)
    if not u:
        return jsonify({'ok': False, 'error': 'User not found'}), 404
    return jsonify({'ok': True, 'user': {
        'username':     u['username'],
        'display_name': u['display_name'],
        'avatar_url':   u['avatar_url'],
        'created_at':   u['created_at'],
    }})


@auth_bp.route('/users/<username>', methods=['DELETE'])
@admin_required
def delete_user(username):
    success, message = user_manager.delete_user(username)
    if not success:
        return jsonify({'ok': False, 'error': message}), 400
    return jsonify({'ok': True, 'message': message})


@auth_bp.route('/permissions/check', methods=['POST'])
@login_required
def check_permissions():
    """检查用户权限"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        user_info = request.current_user
        username = user_info['username']

        permission = data.get('permission')
        sector = data.get('sector')

        result = {}

        if permission:
            result['has_permission'] = user_manager.check_permission(username, permission)

        if sector:
            result['has_sector_access'] = user_manager.check_sector_access(username, sector)

        return jsonify({
            'ok': True,
            'permissions': result
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Permission check failed: {str(e)}'}), 500