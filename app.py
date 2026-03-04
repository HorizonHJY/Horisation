# app.py
"""
Horisation Flask Application
API-only backend. All UI is served by the React SPA (frontend/dist/).
"""

import os
from flask import Flask, g, session, send_from_directory, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

# Import Blueprints
from Backend.Controller.csvcontroller import bp as csv_bp
from Backend.Controller.auth_controller import auth_bp
from Backend.Controller.notes_controller import notes_bp
from Backend.Controller.memos_controller import memos_bp
from Backend.Controller.market_controller import market_bp
from Backend.Controller.feedback_controller import feedback_bp
from Backend.Controller.friends_controller import friends_bp
from Backend.Controller.market_db import init_db
from Backend.Controller.socketio_instance import socketio
import Backend.Controller.game_controller as _game_ctrl    # registers socket events
import Backend.Controller.friends_socket  as _friends_sock  # registers socket events
from Backend.Controller.game_controller import game_bp

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'frontend', 'dist')
UPLOAD_DIR = os.path.join(BASE_DIR, '_uploads')

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Flask app
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH']      = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER']           = UPLOAD_DIR
app.config['SECRET_KEY']              = 'horisation-secret-key-2024'
app.config['SESSION_COOKIE_SECURE']   = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Trust one layer of reverse proxy (Nginx / Cloudflare)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Initialise SocketIO with Redis message queue
socketio.init_app(
    app,
    message_queue='redis://',
    async_mode='eventlet',
    cors_allowed_origins='*',
)

# Register API blueprints
app.register_blueprint(csv_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(notes_bp)
app.register_blueprint(memos_bp)
app.register_blueprint(market_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(friends_bp)
app.register_blueprint(game_bp)

# Initialise database
init_db()

# Import user manager for session validation
from Backend.Controller.user_manager import user_manager

# ── Session middleware ────────────────────────────────────────────
@app.before_request
def load_logged_in_user():
    token = session.get('session_token')
    if token:
        user_info = user_manager.validate_session(token)
        if user_info:
            g.current_user = user_info
        else:
            session.pop('session_token', None)
            g.current_user = None
    else:
        g.current_user = None

# ── Serve React SPA ──────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    asset_path = os.path.join(STATIC_DIR, path)
    if path and os.path.exists(asset_path) and os.path.isfile(asset_path):
        return send_from_directory(STATIC_DIR, path)
    return send_from_directory(STATIC_DIR, 'index.html')

# ── Error handlers ───────────────────────────────────────────────
@app.errorhandler(413)
def request_entity_too_large(_):
    return jsonify({'ok': False, 'error': 'File too large (max 100MB)'}), 413

@app.errorhandler(500)
def internal_error(_):
    return jsonify({'ok': False, 'error': 'Internal server error'}), 500

# ── Dev entry point ──────────────────────────────────────────────
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
