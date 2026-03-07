# Backend/Controller/user_manager.py
"""
用户管理系统
处理用户认证、权限管理、会话管理
数据存储：SQLite via market_db
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from . import market_db


class UserManager:
    """用户管理类"""

    # 用户权限等级定义
    USER_ROLES = {
        'horizon': {
            'level': 100,
            'name': 'Horizon超级管理员',
            'sectors': ['all'],
            'permissions': ['admin', 'read', 'write', 'delete', 'user_manage']
        },
        'horizonadmin': {
            'level': 90,
            'name': 'Horizon管理员',
            'sectors': ['horizon', 'admin'],
            'permissions': ['admin', 'read', 'write', 'delete']
        },
        'vip1': {
            'level': 80,
            'name': 'VIP1用户',
            'sectors': ['vip', 'general'],
            'permissions': ['read', 'write']
        },
        'vip2': {
            'level': 70,
            'name': 'VIP2用户',
            'sectors': ['vip', 'general'],
            'permissions': ['read', 'write']
        },
        'vip3': {
            'level': 60,
            'name': 'VIP3用户',
            'sectors': ['vip', 'general'],
            'permissions': ['read', 'write']
        },
        'test': {
            'level': 50,
            'name': '测试用户',
            'sectors': ['general'],
            'permissions': ['read', 'write']
        },
        'user': {
            'level': 10,
            'name': '普通用户',
            'sectors': ['general'],
            'permissions': ['read']
        }
    }

    def __init__(self, data_dir: str = "_data"):
        self.data_dir  = data_dir
        self.notes_dir = os.path.join(data_dir, "notes")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.notes_dir, exist_ok=True)
        self._ensure_default_users()

    def _ensure_default_users(self):
        """Ensure default admin users exist in DB."""
        if not market_db.db_get_user('horizon'):
            market_db.db_create_user(
                'horizon', 'horizon', 'horizon',
                'horizon@horisation.com', 'Horizon Administrator'
            )
            print("✅ Created admin user: horizon/horizon")
        if not market_db.db_get_user('fanfan0315'):
            market_db.db_create_user(
                'fanfan0315', 'yyf', 'vip1',
                'fanfan0315@horisation.com', 'Fanfan0315'
            )
            print("✅ Created user: fanfan0315/yyf")

    # ── Compatibility shims (used by friends_controller.py) ───────────────────

    def _load_users(self) -> Dict:
        """Compatibility shim — returns {username: user_dict} from DB."""
        users = market_db.db_list_users()
        return {u['username']: u for u in users}

    def _find_user(self, users: Dict, username: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Find user by username field. Returns (key, user_dict)."""
        for key, user in users.items():
            if user.get('username') == username:
                return key, user
        return None, None

    # ── Auth ──────────────────────────────────────────────────────────────────

    def create_user(self, username: str, password: str, role: str,
                    email: str = "", display_name: str = "") -> Tuple[bool, str]:
        if role not in self.USER_ROLES:
            return False, f"Invalid role: {role}"
        if market_db.db_get_user(username):
            return False, "Username already exists"
        market_db.db_create_user(username, password, role, email, display_name or username)
        return True, f"User {username} created successfully"

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        u = market_db.db_get_user(username)
        if not u:
            return False, None
        if not u.get('is_active', True):
            return False, None
        if u.get('password') == password:
            return True, self._public_user_dict(u)
        return False, None

    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        market_db.db_cleanup_sessions()
        expires_at = datetime.utcnow() + timedelta(hours=24)
        market_db.db_create_session(token, username, expires_at)
        return token

    def validate_session(self, session_token: str) -> Optional[Dict]:
        sess = market_db.db_get_session(session_token)
        if not sess:
            return None
        expires_at = sess['expires_at']
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        # Compare as naive UTC datetimes
        if expires_at.tzinfo is not None:
            expires_at = expires_at.replace(tzinfo=None)
        if datetime.utcnow() > expires_at:
            market_db.db_delete_session(session_token)
            return None
        u = market_db.db_get_user(sess['username'])
        if u:
            result = self._public_user_dict(u)
            result['contact_info'] = u.get('contact_info', '')
            return result
        return None

    def logout_user(self, session_token: str) -> bool:
        sess = market_db.db_get_session(session_token)
        if not sess:
            return False
        market_db.db_delete_session(session_token)
        return True

    def check_permission(self, username: str, permission: str) -> bool:
        u = market_db.db_get_user(username)
        if not u:
            return False
        role = u.get('role', '')
        if role not in self.USER_ROLES:
            return False
        return permission in self.USER_ROLES[role]['permissions']

    def check_sector_access(self, username: str, sector: str) -> bool:
        u = market_db.db_get_user(username)
        if not u:
            return False
        role = u.get('role', '')
        if role not in self.USER_ROLES:
            return False
        role_sectors = self.USER_ROLES[role]['sectors']
        if 'all' in role_sectors:
            return True
        return sector in role_sectors

    def get_user_info(self, username: str) -> Optional[Dict]:
        u = market_db.db_get_user(username)
        if not u:
            return None
        return {
            'username':     u['username'],
            'role':         u['role'],
            'display_name': u['display_name'],
            'email':        u['email'],
            'created_at':   u['created_at'],
            'is_active':    u['is_active'],
            'role_info':    self.USER_ROLES.get(u['role'], {}),
        }

    def list_users(self) -> List[Dict]:
        return [{
            'username':     u['username'],
            'role':         u['role'],
            'display_name': u['display_name'],
            'email':        u['email'],
            'created_at':   u['created_at'],
            'is_active':    u['is_active'],
            'role_info':    self.USER_ROLES.get(u['role'], {}),
        } for u in market_db.db_list_users()]

    def search_users(self, query: str) -> List[Dict]:
        return market_db.db_search_users(query)

    def update_user_role(self, username: str, new_role: str) -> Tuple[bool, str]:
        if new_role not in self.USER_ROLES:
            return False, f"Invalid role: {new_role}"
        if not market_db.db_update_user(username, role=new_role):
            return False, "User not found"
        return True, f"User {username} role updated to {new_role}"

    def deactivate_user(self, username: str) -> Tuple[bool, str]:
        if not market_db.db_update_user(username, is_active=False):
            return False, "User not found"
        return True, f"User {username} deactivated"

    def activate_user(self, username: str) -> Tuple[bool, str]:
        if not market_db.db_update_user(username, is_active=True):
            return False, "User not found"
        return True, f"User {username} activated"

    def update_user_profile(self, username: str, display_name: str = None,
                            email: str = None, avatar_url: str = None,
                            contact_info: str = None,
                            contact_hidden: bool = None,
                            wechat: str = None,
                            phone: str = None) -> Tuple[bool, str]:
        fields = {}
        if display_name is not None:
            fields['display_name'] = display_name
        if email is not None:
            fields['email'] = email
        if avatar_url is not None:
            fields['avatar_url'] = avatar_url
        if contact_info is not None:
            fields['contact_info'] = contact_info
        if contact_hidden is not None:
            fields['contact_hidden'] = bool(contact_hidden)
        if wechat is not None:
            fields['wechat'] = wechat
        if phone is not None:
            fields['phone'] = phone
        if not market_db.db_update_user(username, **fields):
            return False, "User not found"
        return True, f"User {username} profile updated"

    def reset_user_password(self, username: str, new_password: str) -> Tuple[bool, str]:
        if len(new_password) < 6:
            return False, "Password must be at least 6 characters"
        if not market_db.db_update_user(username, password=new_password):
            return False, "User not found"
        return True, f"Password for {username} has been reset"

    def delete_user(self, username: str) -> Tuple[bool, str]:
        if username == 'horizon':
            return False, "Cannot delete the root admin account"
        if not market_db.db_delete_user(username):
            return False, "User not found"
        return True, f"User {username} deleted"

    def _public_user_dict(self, u: dict) -> dict:
        return {
            'username':     u['username'],
            'role':         u['role'],
            'display_name': u.get('display_name', u['username']),
            'email':        u.get('email', ''),
            'avatar_url':   u.get('avatar_url'),
            'role_info':    self.USER_ROLES.get(u['role'], {}),
        }


# 创建全局用户管理器实例
user_manager = UserManager()
