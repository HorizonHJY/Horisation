# Backend/Controller/auth_controller.py
"""
用户认证控制器
处理登录、注册、会话管理等API接口
"""

from flask import Blueprint, request, jsonify, session, redirect, url_for
from functools import wraps
from .user_manager import user_manager

# 创建认证蓝图
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

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

        if len(password) < 6:
            return jsonify({'ok': False, 'error': 'Password must be at least 6 characters'}), 400

        # 创建用户
        success, message = user_manager.create_user(
            username=username,
            password=password,
            role=role,
            email=email,
            display_name=display_name
        )

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({'ok': True, 'message': message})

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Registration failed: {str(e)}'}), 500

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