# app.py
"""
Horisation Flask 应用主入口
CSV/Excel 数据分析与金融建模 Web 应用
"""

import os
from flask import Flask, render_template, session, redirect, url_for, g
from functools import wraps

# 导入 Blueprint
from Backend.Controller.csvcontroller import bp as csv_bp
from Backend.Controller.auth_controller import auth_bp
from Backend.Controller.notes_controller import notes_bp
from Backend.Controller.memos_controller import memos_bp

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'Template')
STATIC_DIR = os.path.join(BASE_DIR, 'Static')
UPLOAD_DIR = os.path.join(BASE_DIR, '_uploads')

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("=" * 60)
print("Horisation Application Starting...")
print("=" * 60)
print(f"Template Directory: {TEMPLATE_DIR} (exists: {os.path.exists(TEMPLATE_DIR)})")
print(f"Static Directory: {STATIC_DIR} (exists: {os.path.exists(STATIC_DIR)})")
print(f"Upload Directory: {UPLOAD_DIR} (exists: {os.path.exists(UPLOAD_DIR)})")
print("=" * 60)

# 创建 Flask 应用
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# 应用配置
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['SECRET_KEY'] = 'horisation-secret-key-2024'  # 会话密钥

# 注册 Blueprint
app.register_blueprint(csv_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(notes_bp)
app.register_blueprint(memos_bp)
print("Registered Blueprints: csv_api, auth, notes, memos")

# 导入用户管理器
from Backend.Controller.user_manager import user_manager

# ==================== 全局上下文处理器 ====================

@app.before_request
def load_logged_in_user():
    """在每个请求前加载当前用户信息"""
    session_token = session.get('session_token')
    if session_token:
        user_info = user_manager.validate_session(session_token)
        if user_info:
            g.current_user = user_info
        else:
            session.pop('session_token', None)
            g.current_user = None
    else:
        g.current_user = None

@app.context_processor
def inject_user():
    """向模板注入用户信息"""
    return {'current_user': g.get('current_user')}

# 装饰器：需要登录
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.current_user is None:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== 路由定义 ====================

@app.route('/')
def home():
    """主页 - 需要登录"""
    if not g.current_user:
        return redirect(url_for('login'))
    return render_template('Home.html', active_page='home')

@app.route('/login')
def login():
    """登录页面"""
    if g.current_user:
        return redirect(url_for('home'))
    return render_template('auth/login.html')

@app.route('/register')
def register():
    """注册页面"""
    # 只有管理员可以注册新用户
    if not g.current_user or not user_manager.check_permission(g.current_user['username'], 'admin'):
        return redirect(url_for('home'))
    return render_template('auth/register.html', roles=user_manager.USER_ROLES)

@app.route('/profile')
@login_required
def profile():
    """用户个人资料"""
    return render_template('auth/profile.html', active_page='profile')

@app.route('/admin/users')
@login_required
def admin_users():
    """用户管理页面（管理员）"""
    if not user_manager.check_permission(g.current_user['username'], 'admin'):
        return redirect(url_for('home'))
    users = user_manager.list_users()
    return render_template('auth/admin_users.html', users=users, roles=user_manager.USER_ROLES, active_page='admin')

@app.route('/csv')
@login_required
def csv():
    """CSV 工作区"""
    # 检查用户是否有权限访问
    if not user_manager.check_sector_access(g.current_user['username'], 'general'):
        return redirect(url_for('home'))
    return render_template('CSV.html', active_page='csv')

@app.route('/hormemo')
@login_required
def hormemo():
    """备忘录页面 - 按用户隔离"""
    return render_template('hormemo.html', active_page='hormemo')

@app.route('/notes')
@login_required
def notes():
    """私人笔记页面"""
    return render_template('notes/notes.html', active_page='notes')

@app.route('/limit')
@login_required
def limit():
    """限额跟踪页面"""
    # 检查用户是否有权限访问
    if not user_manager.check_sector_access(g.current_user['username'], 'general'):
        return redirect(url_for('home'))
    return render_template('limit.html', active_page='limit')


# ==================== 错误处理 ====================

@app.errorhandler(413)
def request_entity_too_large(error):
    """文件过大错误处理"""
    from flask import jsonify
    return jsonify({'ok': False, 'error': 'File too large (max 100MB)'}), 413


@app.errorhandler(404)
def not_found(error):
    """404 错误处理"""
    return render_template('Home.html', active_page='home'), 404


@app.errorhandler(500)
def internal_error(error):
    """500 错误处理"""
    from flask import jsonify
    return jsonify({'ok': False, 'error': 'Internal server error'}), 500


# ==================== 启动应用 ====================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Server running at: http://localhost:5000")
    print("CSV Workspace: http://localhost:5000/csv")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
