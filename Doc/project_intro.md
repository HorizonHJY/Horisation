# Horisation — Project Introduction

Last updated: 2026-03-03

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
| Framework | Flask |
| WSGI Server | Gunicorn (4 workers) |
| User Storage | JSON files (`_data/users.json`) |
| Listing / Message Storage | SQLite via SQLAlchemy (`_data/market.db`) |
| Image Storage | Cloudflare R2 (S3-compatible object storage) |
| Auth | Session-based (server-side cookies + Werkzeug ProxyFix) |

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
| Market | `/market` | All | Second-hand trading — post listings with images, price, contact |
| Message Board | `/feedback` | All | Community message board, all users can post |
| Profile | `/profile` | All | Update display name, email, password, avatar |
| Gomoku | `/fun/gomoku` | All | Local 2-player Five in a Row, 15×15 board |
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
├── app.py                            # Flask entry point (API + React catch-all)
├── requirements.txt
├── deploy.sh → scripts/deploy.sh     # Server deploy entry point
├── scripts/
│   ├── deploy.sh                     # Full deploy: pull → pip → npm build → restart
│   ├── dev.bat                       # Windows local dev: Flask + Vite
│   └── build-run.bat                 # Windows local production test
├── _data/                            # Runtime data (gitignored except notes/)
│   ├── users.json                    # User accounts + memos (gitignored)
│   ├── sessions.json                 # Active sessions (gitignored)
│   ├── market.db                     # SQLite: listings + images + messages (gitignored)
│   └── notes/                        # Per-user note files
├── Backend/
│   └── Controller/
│       ├── auth_controller.py        # /api/auth/*
│       ├── csvcontroller.py          # /api/csv/*
│       ├── memos_controller.py       # /api/memos/*
│       ├── notes_controller.py       # /api/notes/*
│       ├── market_controller.py      # /api/market/*
│       ├── feedback_controller.py    # /api/feedback/*
│       ├── user_manager.py           # User / session management (JSON)
│       ├── market_db.py              # SQLAlchemy models: Listing, ListingImage, Message
│       └── r2_manager.py             # Cloudflare R2 upload/delete
├── frontend/
│   ├── public/
│   │   └── logol.avif                # Logo
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                   # Router + AuthContext
│       ├── api.js                    # Fetch wrapper
│       ├── components/
│       │   ├── Sidebar.jsx
│       │   ├── Topbar.jsx
│       │   └── Layout.jsx
│       └── pages/
│           ├── Login.jsx
│           ├── Home.jsx
│           ├── CSV.jsx
│           ├── Hormemo.jsx
│           ├── Market.jsx
│           ├── Feedback.jsx
│           ├── Profile.jsx
│           ├── AdminUsers.jsx
│           ├── UnderDevelopment.jsx
│           └── fun/
│               └── Gomoku.jsx
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

- [ ] Migrate user storage (users.json) to PostgreSQL
- [ ] Password hashing (bcrypt)
- [ ] More games in For Fun section
- [ ] Data visualisation tools
- [ ] CI/CD pipeline (GitHub Actions → EC2)
