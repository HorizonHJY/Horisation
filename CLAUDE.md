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

------

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
| `Backend/Controller/market_db.py` | SQLAlchemy models + helpers: User, UserSession, Listing, ListingImage, Category, Message, MessageLike, Memo, GameRoom, friends/groups tables. Also owns the additive column migrations (`_migrate_columns`, `_migrate_category_labels`). |
| `Backend/Controller/groups_controller.py` | `/api/groups/*` — 群组：建组/拉人/群聊（独立于好友） |
| `Backend/Controller/friends_controller.py` | `/api/friends/*` — search, requests, private chat, contact-sharing approval |
| `Backend/Controller/friends_socket.py` | **The app's only `connect`/`disconnect` handler** plus private chat and the push helpers. python-socketio keys handlers by event name, so a second `@socketio.on('connect')` anywhere would silently replace this one — put per-connection work here instead. |
| `Backend/Controller/socketio_instance.py` | Shared `socketio` object and the sid→user registry that the single connect handler fills (`user_for_sid`) |
| `Backend/Controller/game_controller.py` | `/api/game/*` + Socket.IO events: online Gomoku rooms and moves |
| `Backend/Controller/travel_controller.py` / `travel_db.py` | `/api/travel/*` — multi-day itinerary planner, shareable 6-char plan id |
| `Backend/Controller/bill_controller.py` / `bill_db.py` | `/api/bill/*` — bill splitting, shareable 6-char bill id |
| `Backend/Controller/market_task_controller.py` / `market_task_db.py` | `/api/market/tasks/*` — bounty/task board |
| `Backend/Controller/weather_controller.py` | `/api/weather` — Open-Meteo current weather for St. Louis, 10-min in-memory cache |
| `Backend/Controller/r2_manager.py` | Cloudflare R2 upload/delete via boto3 |
| `Backend/Controller/tarot_controller.py` | `/api/tarot/*` — three-card spread. The shuffle runs here, not in the browser: a reading you can re-roll from devtools is not a reading. |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/App.jsx` | Router, AuthContext, ThemeContext, UnreadContext, PrivateRoute / FeatureRoute |
| `frontend/src/api.js` | Fetch wrapper (`credentials: include`) |
| `frontend/src/features.js` | Per-role feature flags (`canAccess`) — the frontend half of role gating |
| `frontend/src/index.css` | Global design system: tokens, focus ring, badge pairs, market card, dark theme |
| `frontend/src/components/Sidebar.jsx` | Navigation sidebar with logout |
| `frontend/src/components/Modal.jsx` | Shared modal shell (Escape, focus trap, scroll lock, `aria-modal`) + `ConfirmDialog`. **Use this for anything that covers the page — never a bare div, never `window.confirm`.** |
| `frontend/src/components/EnvRibbon.jsx` | Marks a non-production instance; renders nothing in production |
| `frontend/src/components/SocketProvider.jsx` | The app's single Socket.IO connection, alive for the whole session. `useSocket()` / `useSocketEvent()`. **Pages attach and detach handlers; a page must never call `socket.disconnect()`** — it would cut off notifications and chat everywhere. |
| `frontend/src/pages/` | All page components |

### Data
| Path | Contents | Git tracked? |
|------|----------|-------------|
| `_data/market.db` | SQLite — **all** structured data: users, sessions, listings, images, categories, memos, messages, friends, groups, games, travel, bills, tasks | No (gitignored) |
| `_data/notes/` | Per-user note JSON files | Yes |
| `Backend/data/tarot_deck.json` | 78 Rider–Waite–Smith cards: id, name, arcana, image filename, Waite's 1911 upright text | Yes |
| `frontend/public/tarot/*.jpg` | RWS card scans, 78 files (~7.6 MB) from `metabismuth/tarot-json` (MIT); the deck itself is US public domain | Yes |
| `_data/users.json.migrated` | Pre-March-2026 JSON store, migrated into SQLite and renamed | Yes (inert) |
| `Key/r2_config.json` | Cloudflare R2 credentials | No (gitignored) |
| `PRODUCT.md` | Confirmed product record (users, positioning, brand, principles) used by the `impeccable` design skill | Yes |
| `.impeccable/` | Design-detector config + critique snapshots (`hook.cache.json` is gitignored) | Yes, except the cache |

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
| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| GET | `/categories` | login | Active categories for forms (`slug`, `label`, `label_zh`, `order`, `active`, `icon`) |
| GET | `/categories/all` | admin | All categories incl. inactive |
| POST | `/categories` | admin | Create category (`label` = English, optional `label_zh`) |
| PUT | `/categories/<slug>` | admin | Update label / label_zh / order / active / icon |
| DELETE | `/categories/<slug>` | admin | Delete category |
| GET | `/listings` | login | All active listings |
| POST | `/listings` | login | Create listing (multipart, up to 3 images) |
| GET | `/listings/<id>` | login | Single listing. Increments `view_count` unless `?track=0` |
| PUT | `/listings/<id>` | login | Edit (seller only) |
| DELETE | `/listings/<id>` | login | Delete + R2 cleanup (seller only) |
| POST | `/listings/<id>/sold` | login | Mark as sold |
| POST | `/listings/<id>/restore` | login | Restore sold → active |
| GET | `/my` | login | Current user's listings |

### Feedback `/api/feedback/`
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/messages` | Get all messages (latest 200) |
| POST | `/messages` | Post message (max 500 chars) |
| DELETE | `/messages/<id>` | Delete own message (admin: any) |

### Tarot `/api/tarot/`
Login required. Shown only to `horizon` — but that gate lives in `features.js`
and `FeatureRoute`, like every other gated feature here; the API itself answers
any signed-in user. Nothing sensitive sits behind it, so this follows the
existing convention rather than inventing a one-off server-side check.

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/deck` | All 78 cards — the fan is laid out from this |
| POST | `/draw` | Three distinct cards, one per position. `secrets.randbelow`, never `random`. The response carries **only** the drawn cards, so the rest of the deck order never leaves the server. |

Upright only — the deck file carries no reversed meanings.

### Friends `/api/friends/` — notification surface
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/notifications` | **Snapshot the client asks for on every socket connect and on a 5-min fallback timer**: unread counts, pending friend requests, pending contact requests, all with sender identity attached. This is what makes state self-heal after a reconnect. |
| GET | `/unread` | Unread message counts only. Superseded by `/notifications`; kept but unused by the SPA. |
| GET | `/requests/pending` | Pending friend requests. Same — superseded, still served. |

**Socket events** (all pushed to the `user_<username>` room):
`friend_request_incoming`, `contact_request_incoming`, `contact_request_resolved`
(sent to **both** parties), `friend_accepted`, `chat_message`, `chat_error`,
`online_list`, and the `trade_intent_*` family. Anything pushed that the client
also fetches must carry the **same shape** — enrich it through
`_with_sender_identity` — or a pushed item renders differently from a fetched one.

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

Defined in `user_manager.py` `USER_ROLES`. Frontend feature visibility is gated a second
time by `frontend/src/features.js` — a role can pass the backend check and still not see
the entry point.

| Role | Level | Key Permissions |
|------|-------|----------------|
| `horizon` | 100 | admin, read, write, delete, user_manage — cannot be deleted |
| `admin` | 90 | admin, read, write, delete |
| `svip` | 70 | read, write |
| `vip` | 60 | read, write |
| `user` | 10 | read |

---

## Known Limitations / Future Work
- **Passwords stored in plaintext** → needs bcrypt. Anyone with `market.db` (including the
  admin "Download DB" button) has every user's password in the clear. Highest-value fix.
- `SECRET_KEY` hardcoded in `app.py` → should come from the environment
- SQLite single-writer under concurrent writes → PostgreSQL when scale warrants
- CI/CD exists: `.github/workflows/deploy.yml` SSHs to EC2 and runs `~/deploy.sh` on push to `main`

---

## Git Commit (OVERRIDE — Windows environment)

**Never use `git commit` directly.** PyCharm holds file locks on Windows, causing `index file corrupt` and `HEAD.lock` errors. Always use git plumbing:

```bash
# 1. Stage only the target files
rm -f .git/index
GIT_INDEX_FILE=/tmp/commit_index git read-tree HEAD
GIT_INDEX_FILE=/tmp/commit_index git add <file1> <file2> ...
TREE=$(GIT_INDEX_FILE=/tmp/commit_index git write-tree)

# 2. Create the commit object and advance the branch
HEAD_SHA=$(cat .git/refs/heads/main)
COMMIT=$(git commit-tree $TREE -p $HEAD_SHA -m "type(scope): message")
echo $COMMIT > .git/refs/heads/main

# 3. ALWAYS restore the default index afterward
rm -f .git/index && git read-tree HEAD
```

Skipping step 3 leaves `.git/index` missing — PyCharm then shows every file as "new", making it look like all files were removed and re-added on the next commit.
