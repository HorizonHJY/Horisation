# app.py
"""
Horisation Flask Application
API-only backend. All UI is served by the React SPA (frontend/dist/).
"""

import os
import html
import json
from flask import Flask, g, session, send_from_directory, jsonify, request, Response
from werkzeug.middleware.proxy_fix import ProxyFix

# Initialise database FIRST — blueprints import user_manager which needs tables
from Backend.Controller.market_db import init_db
init_db()

from Backend.Controller.travel_db import init_travel_db
init_travel_db()

# Import Blueprints (user_manager is instantiated during these imports)
from Backend.Controller.auth_controller import auth_bp
from Backend.Controller.notes_controller import notes_bp
from Backend.Controller.memos_controller import memos_bp
from Backend.Controller.market_controller import market_bp
from Backend.Controller.feedback_controller import feedback_bp
from Backend.Controller.friends_controller import friends_bp
from Backend.Controller.groups_controller import groups_bp
from Backend.Controller.socketio_instance import socketio
import Backend.Controller.game_controller as _game_ctrl    # registers socket events
import Backend.Controller.friends_socket  as _friends_sock  # registers socket events
from Backend.Controller.game_controller import game_bp
from Backend.Controller.travel_controller import travel_bp
from Backend.Controller.bill_db import init_bill_db
init_bill_db()

from Backend.Controller.market_task_db import init_task_db
init_task_db()

from Backend.Controller.tarot_db import init_tarot_db
init_tarot_db()

from Backend.Controller.bazi_db import init_bazi_db
init_bazi_db()

from Backend.Controller.bazi_compat_db import init_bazi_compat_db
init_bazi_compat_db()

from Backend.Controller.mbti_db import init_mbti_db
init_mbti_db()

from Backend.Controller.feedback_db import init_feedback_db
init_feedback_db()

from Backend.Controller.xp_db import init_xp_db
init_xp_db()

# The AI layer owns its own table (ai_usage). See Doc/ai_service.md.
from Backend.Service.ai import init_ai_db
init_ai_db()

from Backend.Controller.bill_controller import bill_bp
from Backend.Controller.market_task_controller import task_bp
from Backend.Controller.weather_controller import weather_bp
from Backend.Controller.tarot_controller import tarot_bp
from Backend.Controller.bazi_controller import bazi_bp
from Backend.Controller.bazi_compat_controller import bazi_compat_bp
from Backend.Controller.mbti_controller import mbti_bp
from Backend.Controller.reading_feedback_controller import reading_feedback_bp
from Backend.Controller.fx_controller import fx_rate_bp, fx_posts_bp

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
_local = os.environ.get('LOCAL_DEV') == '1'
# Surfaced to the client so the UI can mark itself as a local instance. Local
# and production both answer on localhost during development, and telling them
# apart by eye is otherwise impossible.
app.config['LOCAL_DEV']               = _local
app.config['SESSION_COOKIE_SECURE']   = not _local
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Trust one layer of reverse proxy (Nginx / Cloudflare)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# SocketIO: threading mode locally (no Redis/eventlet needed), eventlet+Redis in prod
if _local:
    socketio.init_app(app, async_mode='threading', cors_allowed_origins='*')
else:
    socketio.init_app(app, message_queue='redis://', async_mode='eventlet', cors_allowed_origins='*')

# Register API blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(notes_bp)
app.register_blueprint(memos_bp)
app.register_blueprint(market_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(friends_bp)
app.register_blueprint(groups_bp)
app.register_blueprint(game_bp)
app.register_blueprint(travel_bp)
app.register_blueprint(bill_bp)
app.register_blueprint(task_bp)
app.register_blueprint(weather_bp)
app.register_blueprint(tarot_bp)
app.register_blueprint(bazi_bp)
app.register_blueprint(bazi_compat_bp)
app.register_blueprint(mbti_bp)
app.register_blueprint(reading_feedback_bp)
app.register_blueprint(fx_rate_bp)
app.register_blueprint(fx_posts_bp)

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

# ── Open Graph share card for market listings ────────────────────
# Crawlers (WhatsApp, WeChat, Twitter, Facebook, Telegram, Slack, Discord,
# LinkedIn…) fetch share URLs server-side and don't execute JS, so we need
# the og:* tags in the initial HTML. Humans hitting the same URL get the
# SPA, which already handles the /market/l/:listingId route client-side.
_OG_BOT_SIGNATURES = (
    'whatsapp', 'twitterbot', 'facebookexternalhit', 'facebot',
    'linkedinbot', 'slackbot', 'telegrambot', 'discordbot', 'pinterest',
    'skypeuripreview', 'embedly', 'quora link preview', 'redditbot',
    'applebot', 'bingbot', 'googlebot', 'yandexbot', 'duckduckbot',
    'mmsocial', 'wechat', 'micromessenger', 'mastodon', 'bluesky',
    'okhttp', 'ia_archiver', 'bitlybot', 'vkshare',
)

def _is_share_crawler() -> bool:
    ua = (request.headers.get('User-Agent') or '').lower()
    return any(sig in ua for sig in _OG_BOT_SIGNATURES)


@app.route('/market/l/<listing_id>')
def market_share_card(listing_id: str):
    from Backend.Controller.market_db import get_listing
    listing = get_listing(listing_id)
    if not listing:
        body = (
            '<!doctype html><html lang="en"><head>'
            '<meta charset="utf-8">'
            '<title>Listing not found · Arch Bay</title>'
            '</head><body style="font-family:sans-serif;padding:40px;text-align:center">'
            '<h1>Listing not found</h1>'
            '<p>This item may have been removed.</p>'
            '<p><a href="/market">Back to Arch Bay</a></p>'
            '</body></html>'
        )
        return Response(body, status=404, mimetype='text/html; charset=utf-8')

    # Humans: hand straight to the SPA, which routes /market/l/:listingId.
    if not _is_share_crawler():
        return send_from_directory(STATIC_DIR, 'index.html')

    price = listing.get('price')
    try:
        price_str = f"${int(price)}" if price is not None and float(price).is_integer() else f"${price}"
    except (TypeError, ValueError):
        price_str = ''
    title_raw = (listing.get('title') or 'Listing').strip()
    title = f"{title_raw} · {price_str}" if price_str else title_raw
    desc = (listing.get('description') or '').replace('\r', ' ').replace('\n', ' ').strip()
    if len(desc) > 120:
        desc = desc[:120].rstrip() + '…'
    if not desc:
        desc = f'For sale on Arch Bay{" — " + price_str if price_str else ""}.'

    images = listing.get('images') or []
    image_url = images[0].get('url') if images else None

    page_url = f"{request.scheme}://{request.host}/market/l/{listing_id}"
    spa_target = f"/market/l/{listing_id}"

    e = html.escape
    tags = [
        f'<meta property="og:title" content="{e(title)}">',
        f'<meta property="og:description" content="{e(desc)}">',
        f'<meta property="og:url" content="{e(page_url)}">',
        '<meta property="og:type" content="product">',
        '<meta property="og:site_name" content="Arch Bay">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{e(title)}">',
        f'<meta name="twitter:description" content="{e(desc)}">',
    ]
    if image_url:
        tags.append(f'<meta property="og:image" content="{e(image_url)}">')
        tags.append(f'<meta name="twitter:image" content="{e(image_url)}">')
    if price_str:
        tags.append(f'<meta property="product:price:amount" content="{e(str(price))}">')

    body = (
        '<!doctype html><html lang="en"><head>'
        '<meta charset="utf-8">'
        f'<title>{e(title)}</title>'
        + ''.join(tags) +
        f'<meta http-equiv="refresh" content="0;url={e(spa_target)}">'
        '</head><body>'
        f'<p>Redirecting to <a href="{e(spa_target)}">{e(title)}</a>…</p>'
        f'<script>window.location.replace({json.dumps(spa_target)});</script>'
        '</body></html>'
    )
    resp = Response(body, mimetype='text/html; charset=utf-8')
    resp.headers['Cache-Control'] = 'public, max-age=300'
    return resp


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

# ── Dev entry point ────────────────────────────────
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
