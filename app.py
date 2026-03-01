# app.py
"""
Horisation Flask Application
API-only backend. All UI is served by the React SPA (frontend/dist/).
"""

import os
from flask import Flask, g, session, send_from_directory, jsonify
from functools import wraps

# Import Blueprints
from Backend.Controller.csvcontroller import bp as csv_bp
from Backend.Controller.auth_controller import auth_bp
from Backend.Controller.notes_controller import notes_bp
from Backend.Controller.memos_controller import memos_bp

# Paths
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = os.path.join(BASE_DIR, 'frontend', 'dist')
UPLOAD_DIR   = os.path.join(BASE_DIR, '_uploads')

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Flask app
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024   # 100 MB
app.config['UPLOAD_FOLDER']      = UPLOAD_DIR
app.config['SECRET_KEY']         = 'horisation-secret-key-2024'

# Register API blueprints
app.register_blueprint(csv_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(notes_bp)
app.register_blueprint(memos_bp)

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
    """Serve React build for all non-API routes."""
    # Serve static assets (JS, CSS, images) directly
    asset_path = os.path.join(STATIC_DIR, path)
    if path and os.path.exists(asset_path) and os.path.isfile(asset_path):
        return send_from_directory(STATIC_DIR, path)
    # Fall back to index.html for client-side routing
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
    app.run(debug=True, host='0.0.0.0', port=5000)
