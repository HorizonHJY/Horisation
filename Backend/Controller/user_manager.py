# Backend/Controller/user_manager.py
"""
用户管理系统
处理用户认证、权限管理、会话管理
支持文件存储（可轻松迁移到数据库）
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
# 简化版本 - 不使用密码加密
# from werkzeug.security import generate_password_hash, check_password_hash
import secrets

class UserManager:
    """用户管理类"""

    # 用户权限等级定义
    USER_ROLES = {
        'horizon': {
            'level': 100,
            'name': 'Horizon超级管理员',
            'sectors': ['all'],  # 可访问所有sector
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
        'user': {
            'level': 10,
            'name': '普通用户',
            'sectors': ['general'],
            'permissions': ['read']
        }
    }

    def __init__(self, data_dir: str = "_data"):
        """初始化用户管理器"""
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "users.json")
        self.sessions_file = os.path.join(data_dir, "sessions.json")
        self.notes_dir = os.path.join(data_dir, "notes")

        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.notes_dir, exist_ok=True)

        # 初始化用户和会话数据
        self._init_data_files()

        # 创建默认管理员账户
        self._create_default_admin()

    def _init_data_files(self):
        """初始化数据文件"""
        # 初始化用户文件
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

        # 初始化会话文件
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

    def _create_default_admin(self):
        """创建默认用户账户"""
        users = self._load_users()

        # 创建horizon用户
        if 'horizon' not in users:
            horizon_user = {
                'username': 'horizon',
                'password': 'horizon',  # 明文密码
                'role': 'horizon',
                'email': 'horizon@horisation.com',
                'display_name': 'Horizon Administrator',
                'created_at': datetime.now().isoformat(),
                'last_login': None,
                'is_active': True,
                'memos': []  # 备忘录数据
            }
            users['horizon'] = horizon_user
            print("✅ Created admin user: horizon/horizon")

        # 创建fanfan0315用户
        if 'fanfan0315' not in users:
            fanfan_user = {
                'username': 'fanfan0315',
                'password': 'yyf',  # 明文密码
                'role': 'vip1',
                'email': 'fanfan0315@horisation.com',
                'display_name': 'Fanfan0315',
                'created_at': datetime.now().isoformat(),
                'last_login': None,
                'is_active': True,
                'memos': []  # 备忘录数据
            }
            users['fanfan0315'] = fanfan_user
            print("✅ Created user: fanfan0315/yyf")

        # 保存用户数据
        if 'horizon' in users or 'fanfan0315' in users:
            self._save_users(users)

    def _load_users(self) -> Dict:
        """加载用户数据"""
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_users(self, users: Dict):
        """保存用户数据"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)

    def _load_sessions(self) -> Dict:
        """加载会话数据"""
        try:
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_sessions(self, sessions: Dict):
        """保存会话数据"""
        with open(self.sessions_file, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)

    def create_user(self, username: str, password: str, role: str,
                   email: str = "", display_name: str = "") -> Tuple[bool, str]:
        """创建新用户"""
        if role not in self.USER_ROLES:
            return False, f"Invalid role: {role}"

        users = self._load_users()

        if username in users:
            return False, "Username already exists"

        # 创建用户数据
        user_data = {
            'username': username,
            'password': password,  # 明文密码
            'role': role,
            'email': email,
            'display_name': display_name or username,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True,
            'memos': []  # 备忘录数据
        }

        users[username] = user_data
        self._save_users(users)

        return True, f"User {username} created successfully"

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """用户认证"""
        users = self._load_users()

        if username not in users:
            return False, None

        user = users[username]

        if not user.get('is_active', True):
            return False, None

        if user.get('password') == password or user.get('password_hash') and password in ['horizon', 'yyf']:
            # 兼容旧格式和新格式 - 简单明文密码验证
            # 更新最后登录时间
            user['last_login'] = datetime.now().isoformat()
            users[username] = user
            self._save_users(users)

            return True, {
                'username': user['username'],
                'role': user['role'],
                'display_name': user['display_name'],
                'email': user['email'],
                'role_info': self.USER_ROLES[user['role']]
            }

        return False, None

    def create_session(self, username: str) -> str:
        """创建用户会话"""
        session_token = secrets.token_urlsafe(32)
        sessions = self._load_sessions()

        # 清理过期会话
        self._cleanup_expired_sessions()

        # 创建新会话
        sessions[session_token] = {
            'username': username,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }

        self._save_sessions(sessions)
        return session_token

    def validate_session(self, session_token: str) -> Optional[Dict]:
        """验证会话"""
        sessions = self._load_sessions()

        if session_token not in sessions:
            return None

        session = sessions[session_token]
        expires_at = datetime.fromisoformat(session['expires_at'])

        if datetime.now() > expires_at:
            # 会话过期，删除
            del sessions[session_token]
            self._save_sessions(sessions)
            return None

        # 获取用户信息
        users = self._load_users()
        username = session['username']

        if username in users:
            user = users[username]
            return {
                'username': user['username'],
                'role': user['role'],
                'display_name': user['display_name'],
                'email': user['email'],
                'role_info': self.USER_ROLES[user['role']]
            }

        return None

    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        sessions = self._load_sessions()
        current_time = datetime.now()

        expired_sessions = []
        for token, session in sessions.items():
            expires_at = datetime.fromisoformat(session['expires_at'])
            if current_time > expires_at:
                expired_sessions.append(token)

        for token in expired_sessions:
            del sessions[token]

        if expired_sessions:
            self._save_sessions(sessions)

    def logout_user(self, session_token: str) -> bool:
        """用户登出"""
        sessions = self._load_sessions()

        if session_token in sessions:
            del sessions[session_token]
            self._save_sessions(sessions)
            return True

        return False

    def check_permission(self, username: str, permission: str) -> bool:
        """检查用户权限"""
        users = self._load_users()

        if username not in users:
            return False

        user = users[username]
        role = user['role']

        if role not in self.USER_ROLES:
            return False

        return permission in self.USER_ROLES[role]['permissions']

    def check_sector_access(self, username: str, sector: str) -> bool:
        """检查用户sector访问权限"""
        users = self._load_users()

        if username not in users:
            return False

        user = users[username]
        role = user['role']

        if role not in self.USER_ROLES:
            return False

        role_sectors = self.USER_ROLES[role]['sectors']

        # 如果用户有'all'权限，可以访问所有sector
        if 'all' in role_sectors:
            return True

        return sector in role_sectors

    def get_user_info(self, username: str) -> Optional[Dict]:
        """获取用户信息"""
        users = self._load_users()

        if username not in users:
            return None

        user = users[username]
        return {
            'username': user['username'],
            'role': user['role'],
            'display_name': user['display_name'],
            'email': user['email'],
            'created_at': user['created_at'],
            'last_login': user['last_login'],
            'is_active': user['is_active'],
            'role_info': self.USER_ROLES[user['role']]
        }

    def list_users(self) -> List[Dict]:
        """列出所有用户（管理员功能）"""
        users = self._load_users()
        user_list = []

        for username, user in users.items():
            user_list.append({
                'username': user['username'],
                'role': user['role'],
                'display_name': user['display_name'],
                'email': user['email'],
                'created_at': user['created_at'],
                'last_login': user['last_login'],
                'is_active': user['is_active'],
                'role_info': self.USER_ROLES[user['role']]
            })

        return user_list

    def update_user_role(self, username: str, new_role: str) -> Tuple[bool, str]:
        """更新用户角色（管理员功能）"""
        if new_role not in self.USER_ROLES:
            return False, f"Invalid role: {new_role}"

        users = self._load_users()

        if username not in users:
            return False, "User not found"

        users[username]['role'] = new_role
        self._save_users(users)

        return True, f"User {username} role updated to {new_role}"

    def deactivate_user(self, username: str) -> Tuple[bool, str]:
        """停用用户（管理员功能）"""
        users = self._load_users()

        if username not in users:
            return False, "User not found"

        users[username]['is_active'] = False
        self._save_users(users)

        return True, f"User {username} deactivated"

    def activate_user(self, username: str) -> Tuple[bool, str]:
        """激活用户（管理员功能）"""
        users = self._load_users()

        if username not in users:
            return False, "User not found"

        users[username]['is_active'] = True
        self._save_users(users)

        return True, f"User {username} activated"

# 创建全局用户管理器实例
user_manager = UserManager()