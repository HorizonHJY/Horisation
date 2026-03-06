# Horisation — Change Log

---

## 2026-03-06

### User & Session Migration — JSON → SQLite
- `users.json` and `sessions.json` fully retired; replaced by `user` and `session` tables in `market.db`
- `User` model: auto-increment integer PK, all existing profile fields + `contact_hidden` flag
- `UserSession` model: token PK, username, created_at, expires_at
- `_migrate_from_json()` in `init_db()`: one-time migration on first startup, renames source file to `.migrated` after completion
- `user_manager.py` fully rewritten — same public API, all JSON ops replaced with SQLAlchemy helpers from `market_db.py`
- `db_search_users(q)`: case-insensitive ilike search on username + display_name, limit 20

### Friend System
- New models in `market_db.py`: `FriendRequest`, `Friendship`, `PrivateChatMessage`
- New blueprint `friends_controller.py` (`/api/friends/*`):
  - `GET /users?q=` — search users (min 2 chars)
  - `POST /requests` — send friend request (with optional message)
  - `GET /requests/pending` — incoming requests
  - `GET /requests/sent` — sent requests
  - `PUT /requests/<id>` — accept / reject
  - `GET /list` — friend list (with display name + avatar)
  - `DELETE /<username>` — unfriend
  - `GET /<username>/contact` — view contact info (now requires contact request approval)
  - `GET /<username>/history` — private chat history
- New `friends_socket.py`: real-time friend request + accept notifications via Socket.IO
- New `Friends.jsx` page (`/friends`):
  - **Friends tab**: list with online indicator, Chat / Request Contact / Unfriend buttons
  - **Requests tab**: Friend Requests section + Contact Requests section (Approve / Decline)
  - **Add tab**: live user search (2+ chars), Add / Pending / Friends status per result
  - Real-time private chat with message history (Socket.IO room keyed on sorted usernames)
  - Auto-open chat when navigated from Market with `location.state.openChat`

### Contact Request System
- New `ContactRequest` model: tracks per-user contact sharing permissions (`pending / approved / declined`)
- `contact_hidden` boolean field on `User` — prevents friends from sending contact requests
- New endpoints under `/api/friends/`:
  - `POST /<username>/contact/request` — request to see someone's contact info
  - `GET /contact/requests` — incoming contact requests I need to respond to
  - `GET /contact/sent` — contact requests I have sent
  - `PUT /contact/requests/<id>` — approve or decline (action: `approve` / `decline`)
- `GET /<username>/contact` updated: now checks `has_contact_access()` before returning info
- **Profile page**: contact info section updated with description; new "Hide my contact" toggle switch
- **Friends page**: contact status per friend shown inline (Request Contact / Pending / Contact (green) / Hidden)

### Market Improvements
- **Browse tab filters out own listings** — you no longer see your own items when browsing
- **Tab order changed**: Browse → My Listings → Post Item
- **Seller info on cards**: seller avatar + display name shown; clicking opens seller profile modal
- **Seller profile modal**: shows seller's avatar, name, all active listings as mini-cards + Reach Out / Add Friend button
- **"Reach Out" button** replaces "I'm Interested":
  - Already friends → navigates to `/friends` with `location.state { openChat, initialMessage }` pre-filling chat input with `"嗨！我看到你发布的《title》，想聊一聊 😊"`
  - Not friends → sends friend request with auto-message
- New API `GET /api/market/user/<username>` — returns active listings for any user
- Listing response enriched with `seller_display` and `seller_avatar` (batch-fetched from User table, no N+1)
- `original_price` field added to `Listing` model

### Login Page Redesign
- New `FlowerCanvas.jsx` component — watercolor petal growth animation (canvas-based, ResizeObserver, unique SVG filter IDs per instance, `origin` prop)
- Login page fully redesigned: full-screen fixed canvas background (flowers from bottom-right), frosted-glass login card, logo + "Arch Bay" title above card, bottom-left St. Louis tagline in Chinese
- **Responsive** (`<= 600px`): logo shrinks to 200×120, content centres, tagline hidden to save space
- CSS classes (`login-page-overlay`, `login-logo`, `login-form-wrap`, `login-tagline`) moved to `index.css` for clean media query control

### Online Gomoku (五子棋)
- New page `/fun/online-gomoku` — real-time multiplayer Five in a Row via Socket.IO
- `GameRoom` model in `market_db.py` (host, player2, board JSON, current_turn, winner, win_cells)
- `game_controller.py`: Socket.IO events for room create/join/move/leave/reset
- `OnlineGomoku.jsx`: lobby list + in-game board UI

### Profile & Permissions
- Permissions section hidden for VIP1 (level ≤ 60) and `user` role — only shown to VIP2+ and admins

### Local Dev Script
- `scripts/dev.bat` — launches Flask + Vite in separate windows
- `scripts/_flask_local.bat` — sets `LOCAL_DEV=1` before starting Flask (uses threading mode, no Redis/eventlet required)
- `app.py`: `LOCAL_DEV=1` switches SocketIO to `async_mode='threading'` and disables `SESSION_COOKIE_SECURE`

### Deploy
- `scripts/deploy.sh`: removed users.json / sessions.json backup block (data now lives in `market.db` which is gitignored and never touched by `git reset --hard`)

---

## 2026-03-02

### Marketplace Feature (二手交易平台)
- New page `/market` — Browse, Post Listing, My Listings tabs
- Image storage: Cloudflare R2 (`horisation-market` bucket), up to 3 images per listing
- Listing metadata: SQLite via SQLAlchemy (`_data/market.db`), auto-created on startup
- New files:
  - `Backend/Controller/r2_manager.py` — R2 upload/delete via boto3
  - `Backend/Controller/market_db.py` — SQLAlchemy models (Listing, ListingImage)
  - `Backend/Controller/market_controller.py` — Blueprint `/api/market/*`
  - `frontend/src/pages/Market.jsx`
  - `Key/r2_config.json` (gitignored)
- Added `sqlalchemy>=2.0.0`, `boto3>=1.34.0` to `requirements.txt`
- Bug fix: Python 3.9 incompatible type hints (`dict | None` → `Optional[dict]`) causing 502 on server

### Message Board (留言板)
- New page `/feedback` — all users can post messages, delete own messages; admins can delete any
- Message model added to `market.db` (same SQLite DB)
- New files:
  - `Backend/Controller/feedback_controller.py` — Blueprint `/api/feedback/*`
  - `frontend/src/pages/Feedback.jsx`
- Features: relative time display, 500-char limit, newest-first order

### User Management Improvements
- Admin (`/admin`) can now: edit display name, edit email, reset password, delete user
- New backend endpoints: `PUT /api/auth/users/<username>/profile`, `PUT /api/auth/users/<username>/password`, `DELETE /api/auth/users/<username>`
- `horizon` account protected from deletion
- Bootstrap JS added to `index.html` (modals were broken without it)

### User Self-Service Profile
- `/profile` page: users can update their own display name, email, password, and avatar
- Avatar stored in R2 under `avatars/<username>.<ext>`, overwrites on re-upload
- New backend endpoints: `PUT /api/auth/profile`, `PUT /api/auth/password`, `POST /api/auth/avatar`
- `avatar_url` field added to user session info

### Navigation & UI
- Sidebar: added Community section (Market, Message Board)
- Sidebar: Log Out button fixed to bottom
- Logo: replaced Font Awesome horse-head icon with `logol.avif`
- Logo file placed in `frontend/public/` for Vite static serving
- Navigation restructured: Hormemo moved from Main to **Toolkit** section (visible to all roles); CSV Workspace and Data tools remain Toolkit but **horizon-only**
- User avatars now displayed in Message Board (fallback to initials if no avatar set)

### Mobile Sidebar (响应式侧边栏)
- On mobile (`< 768px`), sidebar is hidden off-screen by default (`translateX(-100%)`)
- Hamburger button (`☰`) added to topbar, visible only on mobile (`d-md-none`)
- Tapping hamburger slides sidebar in with CSS transition (`0.25s ease`)
- Dark overlay backdrop rendered behind open sidebar; tap overlay to close
- Navigating to any page auto-closes the sidebar
- Topbar and main content expand to full width on mobile (`left: 0`, `margin-left: 0`)
- Files changed: `Layout.jsx` (state management), `Topbar.jsx` (hamburger), `Sidebar.jsx` (isOpen/onClose props), `index.css` (media query)

### Server — Python Upgrade
- Upgraded server Python from 3.9 (EOL) to 3.11
- New venv: `/home/ec2-user/venv311/`
- Old venv (`~/venv`) deleted after confirming stability

### Deploy Script
- `scripts/deploy.sh` added to project (git-tracked, replaces ad-hoc server script)
- Server's `~/deploy.sh` now simply calls `bash ~/Horisation/scripts/deploy.sh`
- Script now includes pip install step for Python dependencies

### .gitignore Updates
- Added `_data/users.json`, `_data/sessions.json` — user data managed independently per environment
- Added `_data/market.db` — SQLite DB auto-created on startup, not committed

---

## 2026-03-01

### Deployment & Server Setup
- Deployed project to AWS EC2 (Amazon Linux 2023) with Gunicorn + Nginx
- Configured Nginx reverse proxy at `/etc/nginx/conf.d/horizonyhj.com.conf`
- Set up systemd service at `/etc/systemd/system/horisation.service`
- Gunicorn binary location: `/home/ec2-user/venv/bin/gunicorn` (venv is outside project root)
- SSL via Let's Encrypt + Cloudflare Full mode

### Requirements
- Created `requirements.txt` (flask, pandas, numpy, openpyxl, xlrd, pyarrow, gunicorn)

### Bug Fixes — Authentication
- Removed hardcoded auth backdoor: `password in ['horizon', 'yyf']`
- Fixed `authenticate_user` and all user lookup methods to search by `username` field (not dict key) using `_find_user()` helper
- Same fix applied to `memos_controller.py`
- Fixed `users.json` key/username inconsistencies

### Feature Removal
- Removed `/limit` route and `limit.html` page
- Removed `last_login` field (caused git conflicts since users.json was tracked)

### Frontend — React SPA Migration
- Migrated entire frontend from Flask/Jinja2 to React 18 + Vite
- Flask is now API-only; all routes under `/api/*`
- Deleted `Template/` and `Static/` directories

### React Pages Built
| Page | Route |
|------|-------|
| Login | `/login` |
| Home | `/home` |
| CSV Workspace | `/csv` |
| Hormemo | `/hormemo` |
| Profile | `/profile` |
| Admin Users | `/admin` |
| Under Development | `/under-development` |
| Gomoku | `/fun/gomoku` |

### Bug Fix — Server Login Failure
- Root cause: Flask saw requests as HTTP (internal), refused to set Secure Cookie
- Fix: Added `ProxyFix`, `SESSION_COOKIE_SECURE`, `SESSION_COOKIE_HTTPONLY`, `SESSION_COOKIE_SAMESITE`

### Documentation
- Created `Doc/project_intro.md`, `Doc/server.md`, `Doc/log.md`, `Doc/data_storage.md`

---

## Deploy Checklist
```bash
# Local — push changes
git add -A && git commit -m "..." && git push

# Server — one command
bash ~/deploy.sh
```
