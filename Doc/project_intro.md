# Horisation — Project Introduction

Last updated: 2026-09-20

---

## Overview

Horisation is a personal web platform built for private use among close friends.
It serves as a central hub for tools, community features, games, and anything the owner
wants to host and share. The name is a blend of "Horizon" and "Isation", representing
an ever-expanding personal space.

Access is **invitation-only**. Self-registration is available but requires a valid invite code issued by the owner.

---

## Target Users

| Role | Level | Description |
|------|-------|-------------|
| `horizon` | 100 | Super-admin (owner), full access, cannot be deleted |
| `admin` | 90 | Admin, manage users and content |
| `svip` | 70 | Trusted friends, extra features |
| `vip` | 60 | Trusted friends |
| `user` | 10 | General users, read-only access |

Canonical definition lives in `user_manager.py` `USER_ROLES`. Feature visibility is gated a
second time in the frontend by `frontend/src/features.js`.

---

## Tech Stack

### Backend
| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Framework | Flask + Flask-SocketIO |
| WSGI Server | Gunicorn (`-w 1` for SocketIO, eventlet worker) |
| Data Storage | SQLite via SQLAlchemy (`_data/market.db`) — all data including users and sessions |
| Image Storage | Cloudflare R2 (S3-compatible object storage) |
| Auth | Session-based (server-side cookies + Werkzeug ProxyFix) |
| Real-time | Socket.IO + Redis message queue (eventlet on server, threading locally) |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React 18 |
| Build Tool | Vite |
| Routing | React Router v6 |
| Styling | Bootstrap 5 |
| API Calls | Native fetch (credentials: include) |

### Infrastructure
| Component | Technology |
|-----------|------------|
| Cloud | AWS EC2 (Amazon Linux 2023, t-series) |
| Reverse Proxy | Nginx |
| DNS / CDN | Cloudflare |
| SSL | Let's Encrypt (Certbot) + Cloudflare Full mode |
| Python Env | `/home/ec2-user/venv311/` |

---

## Architecture

```
Browser
  └── Cloudflare (DNS + CDN + SSL termination)
        └── Nginx (HTTPS 443 → HTTP internal)
              ├── /api/*  → Gunicorn (port 8000) → Flask (API only)
              └── /*      → React SPA (frontend/dist/)
```

Flask is strictly API-only. React handles all UI and routing client-side.
Images are stored in Cloudflare R2; only the public URL is kept in the database.

---

## Features

### Currently Available

| Feature | Route | Roles | Description |
|---------|-------|-------|-------------|
| Home | `/home` | All | Personal dashboard with feature overview |
| Hormemo | `/hormemo` | All | Personal memo / task tracker (CRUD, priority, tags) |
| Market | `/market` | All | Second-hand trading — browse, post listings with images, Reach Out to sellers, seller profile modal. Categories (slug/label/order/active) stored in DB and configurable via System Management. Currency: USD ($). |
| Message Board | `/feedback` | All | Community message board, all users can post |
| Friends | `/friends` | All | Friend system: search, add, private chat, contact sharing with approval flow |
| Groups | `/groups` | All | Build a group by username, independent of the friend graph; group chat. See `Doc/groups.md` |
| Tasks | `/tasks` | All | Bounty / task board |
| Tarot | `/tarot` | All | 78-card Rider–Waite–Smith deck: shuffle, choose your own three, past / present / future, tap a card to look closer, then an optional AI reading of the three together (DeepSeek; user 1 / vip 3 / admin ∞ per Chicago day) with a 1–5 fit rating. Server-side shuffle; the AI reads the spread the server drew, by id. `horizon`-only 09-08 → 09-10, open to everyone since |
| Profile | `/profile` | All | Update display name, email, password, avatar, contact info (with hide toggle) |
| Gomoku (Online) | `/fun/gomoku-online` | horizon, admin, svip | Real-time multiplayer Five in a Row via Socket.IO |
| Travel Planner | `/travel` | horizon, admin, svip, vip | Multi-day itinerary planner, shareable 6-char plan id |
| Bill Split | `/bill-split` | horizon, admin, svip, vip | Bill splitting, shareable 6-char bill id |
| Admin | `/admin` | admin permission | User management (create, edit, reset password, delete, role) + invite code management (horizon only) |
| System Management | `/admin/system` | admin permission | Manage market category list (slug, label, order, active toggle) |
| CSV Workspace | `/csv` | horizon (sidebar only) | Upload, preview, and summarise CSV / Excel files. The **sidebar entry** is horizon-only; the route itself is not role-gated, so the URL works for any signed-in user |

> `Gomoku.jsx` (local 2-player) exists under `pages/fun/` but has **no route and no
> nav entry** — it is currently unreachable. It used to be listed here as
> `/fun/gomoku`, which never existed.
>
> Roles are `horizon` / `admin` / `svip` / `vip` / `user` (see `user_manager.py`).
> An earlier version of this table used `horizonadmin` and `vip3`, which are not
> real roles.

### Under Development

Nothing is currently in this state. Data Analysis, Data Handling, Data Visualisation
and Notes were listed here pointing at `/under-development`; that route and those nav
entries no longer exist. Data visualisation is still wanted — it lives in
`Doc/todo.md` under Feature Ideas, which is where planned work belongs.

`notes_controller.py` still serves `/api/notes/*` (11 routes) but no frontend calls it;
see `Doc/todo.md`.

---

## Project Structure

```
Horisation/
├── app.py                            # Flask entry point (API + React catch-all + SocketIO init)
├── requirements.txt
├── deploy.sh → scripts/deploy.sh     # Server deploy entry point
├── PRODUCT.md                        # Confirmed product record (impeccable design skill)
├── .impeccable/                      # Design detector config + critique snapshots
├── scripts/
│   ├── deploy.sh                     # Full deploy: pull → pip → npm build → restart
│   ├── dev.bat                       # Windows local dev: Flask + Vite
│   ├── _resolve_python.bat           # Locates a usable interpreter (HORISATION_PYTHON overrides)
│   ├── _flask_local.bat              # Sets LOCAL_DEV=1, starts Flask (threading mode)
│   └── build-run.bat                 # Windows local production test
├── _data/                            # Runtime data (gitignored except notes/)
│   ├── market.db                     # SQLite: ALL data — users, sessions, listings, friends, chat, memos
│   └── notes/                        # Per-user note files (git tracked)
├── Backend/
│   └── Controller/
│       ├── auth_controller.py        # /api/auth/* — login, register, profile, avatar
│       ├── csvcontroller.py          # /api/csv/*
│       ├── memos_controller.py       # /api/memos/*
│       ├── notes_controller.py       # /api/notes/*
│       ├── market_controller.py      # /api/market/*
│       ├── feedback_controller.py    # /api/feedback/*
│       ├── friends_controller.py     # /api/friends/* — search, requests, friends, contact, chat history
│       ├── friends_socket.py         # Socket.IO events: friend notifications, private chat
│       ├── game_controller.py        # Socket.IO events: online Gomoku rooms and moves
│       ├── user_manager.py           # User / session management (SQLite via market_db)
│       ├── market_db.py              # All SQLAlchemy models + helpers
│       └── r2_manager.py             # Cloudflare R2 upload/delete
├── frontend/
│   ├── public/
│   │   ├── logo.png                  # Login page logo
│   │   └── logol.avif                # Sidebar logo
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                   # Router + Auth/Theme/Unread contexts
│       ├── api.js                    # Fetch wrapper
│       ├── features.js               # Per-role feature flags
│       ├── index.css                 # Design tokens, focus ring, badges, responsive rules
│       ├── components/
│       │   ├── Sidebar.jsx
│       │   ├── Topbar.jsx
│       │   ├── Layout.jsx
│       │   ├── Modal.jsx             # Shared modal shell + ConfirmDialog (Escape, focus trap, scroll lock)
│       │   ├── EnvRibbon.jsx         # LOCAL marker; renders nothing in production
│       │   ├── SocketProvider.jsx    # The app's single Socket.IO connection (session-scoped)
│       │   ├── WeatherGreeting.jsx   # Home page greeting + Open-Meteo weather
│       │   ├── FlowerCanvas.jsx      # Watercolor petal animation (canvas, SVG filter)
│       │   └── HandLoader.jsx        # Loading spinner
│       └── pages/
│           ├── Login.jsx             # Full-screen flower animation + frosted-glass card; link to Register
│           ├── Register.jsx          # Public self-registration (invite code required)
│           ├── Home.jsx
│           ├── CSV.jsx
│           ├── Hormemo.jsx
│           ├── Market.jsx            # Browse / My Listings / Post Item; seller modal; Reach Out
│           ├── Feedback.jsx
│           ├── Friends.jsx           # Friends list, search, private chat, contact requests
│           ├── Profile.jsx
│           ├── AdminUsers.jsx
│           ├── UnderDevelopment.jsx
│           └── fun/
│               ├── Gomoku.jsx        # Local 2-player
│               └── OnlineGomoku.jsx  # Real-time multiplayer (Socket.IO)
├── Key/
│   └── r2_config.json                # R2 credentials (gitignored)
└── Doc/
    ├── project_intro.md              # This file
    ├── server.md                     # Server configuration reference
    ├── data_storage.md               # Data storage architecture
    └── log.md                        # Change log
```

---

## UI / Responsiveness

- **Desktop (≥ 768px)** — since 2026-09-20 the sidebar is **closed by default** and nothing
  of it is on screen but the ☰ button at the top-left of the topbar. Click it (or press
  `[`) and the sidebar docks at 240px with the content moving over; the `«` in its header
  closes it. No hover zone, no icon rail, no breadcrumb. The choice is remembered per
  browser in `localStorage['archbay.nav']`. Opening and closing snap rather than animate —
  a deliberate click gets a deliberate result, and a 240px reflow of the whole page is not
  worth animating.
  (History: a hover-revealed 14px "spine" with a breadcrumb shipped 2026-09-11 and was
  rejected by the owner nine days later. The button pattern is what ChatGPT does.)
- **Mobile (< 768px)**: unchanged — the same ☰ opens a slide-in drawer with a scrim;
  navigating anywhere closes it.
- The whole map lives in `frontend/src/nav.js`; the sidebar renders it. Items are gated
  **per item** (`canAccess(role, item.feature)`); an item with no `feature` key is open
  to everyone:
  - All members: Main (Home), Community (Market, Tasks, Message Board, Friends, Groups),
    For Fun → Tarot, Toolkit → Memo
  - `horizon` / `admin` / `svip` additionally see: For Fun → Online Gomoku
  - `vip` and above additionally see: Toolkit → Travel Planner, Bill Split
  - `horizon` additionally sees: Toolkit → CSV Workspace
  - Users with `admin` permission: the Admin section
  - A section only renders if at least one of its items is visible

---

## Roadmap

See `Doc/todo.md` for the full prioritised list. Key items:

- [ ] Password hashing (bcrypt) — currently plaintext
- [ ] Listing image re-upload in Edit flow
- [ ] Avalon board game
- [ ] Data visualisation tools
- [ ] Migrate SQLite → PostgreSQL for concurrent write safety
- [x] Group messaging / group chat — done 2026-08-22
- [x] CI/CD pipeline (GitHub Actions → EC2) — done, `.github/workflows/deploy.yml`

> **Note (2026-09-10):** the Features table and the sidebar section above were rebuilt
> from the code and are current. The **Project Structure** tree further up is not — it
> still predates Travel Planner, Bill Split, Tasks, the weather endpoint and Tarot.
> `CLAUDE.md` has the current backend file map; treat code as truth where the two disagree.
