# Horisation — Change Log

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
- Removed hardcoded auth backdoor: `password in ['horizon', 'yyf']` fallback was allowing any user with a `password_hash` field to login with those passwords
- Fixed `authenticate_user` and all user lookup methods to search by `username` **field** (not dict key) using new `_find_user(users, username)` helper
- Same fix applied to `memos_controller.py`
- Fixed `users.json` key/username inconsistencies (e.g. key `"sammy"` with `username: "Sammy"`)

### Feature Removal
- Removed `/limit` route and `limit.html` page
- Removed `last_login` field entirely: no longer written on login, not stored in JSON, not shown in UI
  - Reason: caused git conflicts on every login since `_data/users.json` was tracked in git

### Frontend — React SPA Migration
- Migrated entire frontend from Flask/Jinja2 to React 18 + Vite
- Flask is now API-only; all routes under `/api/*`
- React app lives in `frontend/`, built to `frontend/dist/`
- Flask serves `frontend/dist/index.html` as catch-all for non-API routes
- Deleted `Template/` and `Static/` directories (no longer needed)

### React Pages Built
| Page | Route | Notes |
|------|-------|-------|
| Login | `/login` | Two-panel layout, session-based auth |
| Home | `/home` | Feature grid dashboard |
| CSV Workspace | `/csv` | File upload, preview, summary |
| Hormemo | `/hormemo` | Full CRUD memos via `/api/memos/` |
| Profile | `/profile` | User info display |
| Admin Users | `/admin` | Role management, create/activate users |
| Under Development | `/under-development` | Placeholder for unfinished sections |
| Gomoku | `/fun/gomoku` | Local 2-player Five in a Row, 15×15 board |

### Navigation (English only)
- Removed: Settings, Help, Ad-Hoc / Limit
- Added: For Fun section with Gomoku
- Under Development redirects: Data Analysis, Data Handling, Data Visualisation, Notes

### Documentation
- Created `Doc/project_intro.md` — full English project overview
- Created `Doc/server.md` — server config reference
- Created `Doc/log.md` — this file

### Config
- Created `.gitignore` (excludes `node_modules/`, `dist/`, `_uploads/`, `Key/`, `__pycache__/`)
- Project is now pure English (all Chinese text removed from code, templates, comments)

### Bug Fix — Server Login Failure (Session Cookie + ProxyFix)
- **症状**: 本地登录正常，服务器（Nginx + Gunicorn + Cloudflare + HTTPS）登录失败
- **根因**: Flask 从 Gunicorn 收到请求时看到的是 HTTP（内网），不知道用户实际走的是 HTTPS，因此拒绝设置 Secure Cookie，导致 Set-Cookie 根本没有发出去
- **修复** (`app.py`):
  - 加入 `ProxyFix(x_proto=1)` — 让 Flask 读取 `X-Forwarded-Proto: https` 请求头，正确识别 HTTPS 环境
  - `SESSION_COOKIE_SECURE = True` — Cookie 仅 HTTPS 传输
  - `SESSION_COOKIE_HTTPONLY = True` — 禁止 JS 读取 Cookie
  - `SESSION_COOKIE_SAMESITE = 'Lax'` — Cloudflare 代理下跨域 Cookie 正常传递

---

## Deploy Checklist (for future reference)
```bash
# Local
cd frontend && npm run build
git add -A && git commit -m "..." && git push

# Server
cd /home/ec2-user/Horisation
git pull
cd frontend && npm install && npm run build
cd .. && sudo systemctl restart horisation
```
