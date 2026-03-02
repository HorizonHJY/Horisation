# Horisation — Change Log

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
- Navigation restructured: Toolkit section (CSV, Data Analysis, Data Handling, Data Visualisation) now **horizon-only**; other roles only see Main, Community, For Fun
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
