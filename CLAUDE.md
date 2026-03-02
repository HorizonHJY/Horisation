# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Horisation is a private web platform for close friends. It provides personal tools, a community marketplace, a message board, memos, games, and user management. Access is invitation-only.

**Live site:** https://horizonyhj.com

---

## Architecture

```
Browser → Cloudflare → Nginx → Gunicorn (port 8000) → Flask (API only)
                             → React SPA (frontend/dist/)
```

- Flask is **API-only**. All routes are under `/api/*`.
- React 18 + Vite handles all UI. Built to `frontend/dist/`.
- Flask serves `frontend/dist/index.html` as a catch-all for non-API routes.

---

## Key Files

### Backend
| File | Purpose |
|------|---------|
| `app.py` | Flask entry point — registers blueprints, ProxyFix, session config, React catch-all |
| `Backend/Controller/auth_controller.py` | `/api/auth/*` — login, logout, session, user CRUD |
| `Backend/Controller/user_manager.py` | JSON-based user/session storage, role checks |
| `Backend/Controller/csvcontroller.py` | `/api/csv/*` — upload, preview, summary |
| `Backend/Controller/memos_controller.py` | `/api/memos/*` — per-user memo CRUD |
| `Backend/Controller/notes_controller.py` | `/api/notes/*` — per-user notes |
| `Backend/Controller/market_controller.py` | `/api/market/*` — marketplace listings |
| `Backend/Controller/feedback_controller.py` | `/api/feedback/*` — message board |
| `Backend/Controller/market_db.py` | SQLAlchemy models: Listing, ListingImage, Message |
| `Backend/Controller/r2_manager.py` | Cloudflare R2 upload/delete via boto3 |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/App.jsx` | Router, AuthContext, PrivateRoute |
| `frontend/src/api.js` | Fetch wrapper (`credentials: include`) |
| `frontend/src/components/Sidebar.jsx` | Navigation sidebar with logout |
| `frontend/src/pages/` | All page components |

### Data
| Path | Contents | Git tracked? |
|------|----------|-------------|
| `_data/users.json` | User accounts + memos | No (gitignored) |
| `_data/sessions.json` | Active session tokens | No (gitignored) |
| `_data/market.db` | SQLite: listings, images, messages | No (gitignored) |
| `_data/notes/` | Per-user note JSON files | Yes |
| `Key/r2_config.json` | Cloudflare R2 credentials | No (gitignored) |

---

## Development

### Local dev (recommended)
```bash
# Windows
scripts\dev.bat        # starts Flask + Vite, access http://localhost:5173

# Or manually
python app.py          # Flask on :5000
cd frontend && npm run dev   # Vite on :5173 (proxies /api to :5000)
```

### Local production test
```bash
scripts\build-run.bat  # npm build → Flask serves dist/ on :5000
```

---

## Deployment

```bash
# Server (one command)
bash ~/deploy.sh       # calls scripts/deploy.sh in the project

# scripts/deploy.sh does:
# 1. git fetch + reset --hard origin/main
# 2. pip install -r requirements.txt
# 3. npm install + npm run build
# 4. sudo systemctl restart horisation
```

### Server details
- EC2: Amazon Linux 2023
- Python: 3.11 at `/home/ec2-user/venv311/`
- Service: `/etc/systemd/system/horisation.service`
- Nginx config: `/etc/nginx/conf.d/horizonyhj.com.conf`
- R2 config: `/home/ec2-user/Horisation/Key/r2_config.json` (manual, never in git)

---

## API Endpoints

### Auth `/api/auth/`
| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| POST | `/login` | Public | Login |
| POST | `/logout` | login | Logout |
| GET | `/check-session` | Public | Check session |
| GET | `/profile` | login | Get own profile |
| PUT | `/profile` | login | Update display name / email |
| PUT | `/password` | login | Change own password |
| POST | `/avatar` | login | Upload avatar to R2 |
| POST | `/register` | admin | Create user |
| GET | `/users` | admin | List users |
| PUT | `/users/<u>/role` | admin | Change role |
| PUT | `/users/<u>/status` | admin | Activate/deactivate |
| PUT | `/users/<u>/profile` | admin | Edit name/email |
| PUT | `/users/<u>/password` | admin | Reset password |
| DELETE | `/users/<u>` | admin | Delete user |

### CSV `/api/csv/`
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/preview` | Preview first N rows |
| POST | `/summary` | Full file statistics |

### Memos `/api/memos/`
All login required. Full CRUD + `/complete`, `/statistics`.

### Market `/api/market/`
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/listings` | All active listings |
| POST | `/listings` | Create listing (multipart, up to 3 images) |
| GET | `/listings/<id>` | Single listing |
| PUT | `/listings/<id>` | Edit (seller only) |
| DELETE | `/listings/<id>` | Delete + R2 cleanup (seller only) |
| POST | `/listings/<id>/sold` | Mark as sold |
| GET | `/my` | Current user's listings |

### Feedback `/api/feedback/`
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/messages` | Get all messages (latest 200) |
| POST | `/messages` | Post message (max 500 chars) |
| DELETE | `/messages/<id>` | Delete own message (admin: any) |

---

## Patterns & Conventions

### Auth decorators
```python
@login_required      # checks session, sets request.current_user
@admin_required      # stacks login_required + admin permission check
```

### User lookup
```python
# Always use _find_user() — dict key ≠ username field
users = user_manager._load_users()
key, user = user_manager._find_user(users, username)
```

### API response format
```python
# Success
return jsonify({'ok': True, ...})
# Error
return jsonify({'ok': False, 'error': 'message'}), 4xx
```

### Frontend API calls
```javascript
api.get('/api/...')
api.post('/api/...', body)
api.put('/api/...', body)
api.delete('/api/...')
api.upload('/api/...', formData)   // for multipart
```

---

## User Roles

| Role | Level | Key Permissions |
|------|-------|----------------|
| `horizon` | 100 | admin, read, write, delete, user_manage — cannot be deleted |
| `horizonadmin` | 90 | admin, read, write, delete |
| `vip1/2/3` | 60–80 | read, write |
| `user` | 10 | read |

---

## Known Limitations / Future Work
- Passwords stored in plaintext → needs bcrypt
- `users.json` not thread-safe under concurrent writes → migrate to PostgreSQL
- Memos stored inside user objects → should be a separate DB table
- No CI/CD pipeline yet
