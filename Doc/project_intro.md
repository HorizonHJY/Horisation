# Horisation — Project Introduction

Last updated: 2026-03-06

---

## Overview

Horisation is a personal web platform built for private use among close friends.
It serves as a central hub for tools, community features, games, and anything the owner
wants to host and share. The name is a blend of "Horizon" and "Isation", representing
an ever-expanding personal space.

Access is **invitation-only**. No public registration.

---

## Target Users

| Role | Description |
|------|-------------|
| `horizon` | Super-admin (owner), full access, cannot be deleted |
| `horizonadmin` | Admin, manage users and content |
| `vip1 / vip2 / vip3` | Trusted friends, access to all features |
| `user` | General users, read-only access |

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
| Market | `/market` | All | Second-hand trading — browse, post listings with images, Reach Out to sellers, seller profile modal |
| Message Board | `/feedback` | All | Community message board, all users can post |
| Friends | `/friends` | All | Friend system: search, add, private chat, contact sharing with approval flow |
| Profile | `/profile` | All | Update display name, email, password, avatar, contact info (with hide toggle) |
| Gomoku (Local) | `/fun/gomoku` | All | Local 2-player Five in a Row, 15×15 board |
| Gomoku (Online) | `/fun/online-gomoku` | All | Real-time multiplayer Five in a Row via Socket.IO |
| Admin | `/admin` | admin+ | User management (create, edit, reset password, delete, role) |
| CSV Workspace | `/csv` | horizon only | Upload, preview, and summarise CSV / Excel files |

### Under Development

| Feature | Route |
|---------|-------|
| Data Analysis | `/under-development` |
| Data Handling | `/under-development` |
| Data Visualisation | `/under-development` |
| Notes | `/under-development` |

---

## Project Structure

```
Horisation/
├── app.py                            # Flask entry point (API + React catch-all + SocketIO init)
├── requirements.txt
├── deploy.sh → scripts/deploy.sh     # Server deploy entry point
├── scripts/
│   ├── deploy.sh                     # Full deploy: pull → pip → npm build → restart
│   ├── dev.bat                       # Windows local dev: Flask + Vite
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
│       ├── App.jsx                   # Router + AuthContext
│       ├── api.js                    # Fetch wrapper
│       ├── index.css                 # Global styles + responsive rules
│       ├── components/
│       │   ├── Sidebar.jsx
│       │   ├── Topbar.jsx
│       │   ├── Layout.jsx
│       │   ├── FlowerCanvas.jsx      # Watercolor petal animation (canvas, SVG filter)
│       │   └── HandLoader.jsx        # Loading spinner
│       └── pages/
│           ├── Login.jsx             # Full-screen flower animation + frosted-glass card
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

- **Desktop**: Fixed sidebar (240px) + fixed topbar; main content offset accordingly
- **Mobile (< 768px)**: Sidebar hidden off-screen; hamburger button (`☰`) in topbar opens a slide-in drawer with a dark overlay backdrop. Navigating to any page auto-closes the drawer.
- Sidebar navigation is role-gated:
  - All users: Main (Home), Community, For Fun, Toolkit (Hormemo)
  - `horizon` additionally sees: Toolkit → CSV Workspace, Data Analysis, Data Handling, Data Visualisation
  - Users with `admin` permission: Admin section

---

## Roadmap

- [ ] Password hashing (bcrypt) — currently plaintext
- [ ] Group messaging / group chat
- [ ] Avalon board game
- [ ] More games in For Fun section
- [ ] Data visualisation tools
- [ ] CI/CD pipeline (GitHub Actions → EC2)
- [ ] Migrate SQLite → PostgreSQL for concurrent write safety
