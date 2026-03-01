# Horisation — Project Introduction

## Overview

Horisation is a personal web platform built for private use among close friends.
It serves as a central hub for tools, utilities, notes, games, and anything the owner
wants to host and share — with no particular vertical focus.
The name is a blend of "Horizon" and "Isation", representing an ever-expanding personal space.

---

## Purpose

- A personal home base on the internet, accessible from anywhere
- A place to share lightweight tools and utilities with friends
- Storage for daily notes, memos, and personal records
- A playground for small web games and experiments
- Anything the owner wants — no restrictions on content type

---

## Target Users

| Role | Description |
|------|-------------|
| `horizon` | Super-admin (owner), full access |
| `vip1` | Trusted friends, access to most features |
| `user` | General users, read-only access |

Access is invitation-only. No public registration.

---

## Tech Stack

### Backend
| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Framework | Flask |
| WSGI Server | Gunicorn (4 workers) |
| User Storage | JSON files (planned migration to PostgreSQL) |
| Auth | Session-based (server-side cookies) |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React 18 |
| Build Tool | Vite |
| Routing | React Router v6 |
| Styling | Bootstrap 5 |
| API Calls | Native fetch |

### Infrastructure
| Component | Technology |
|-----------|------------|
| Cloud | AWS EC2 (Amazon Linux 2023) |
| Reverse Proxy | Nginx |
| DNS / CDN | Cloudflare |
| SSL | Let's Encrypt (Certbot) |

---

## Architecture

```
Browser
  └── Cloudflare (DNS + SSL)
        └── Nginx (HTTPS 443)
              ├── /api/*   → Gunicorn → Flask (API only)
              └── /*       → React SPA (static build)
```

Flask is strictly API-only. React handles all UI and routing client-side.

---

## Features

### Currently Available
| Feature | Route | Description |
|---------|-------|-------------|
| Home | `/home` | Personal dashboard |
| Hormemo | `/hormemo` | Personal memo / task tracker |
| CSV Workspace | `/csv` | Upload, preview, and summarise CSV/Excel files |
| Gomoku | `/fun/gomoku` | Local 2-player Gomoku (Five in a Row) game |

### Under Development
| Feature | Route | Notes |
|---------|-------|-------|
| Data Analysis | `/data-analysis` | Planned |
| Data Handling | `/data-handling` | Planned |
| Data Visualisation | `/data-visualisation` | Planned |
| Private Notes | `/notes` | Planned |

---

## Project Structure

```
Horisation/
├── app.py                        # Flask entry point (API only)
├── requirements.txt
├── _data/                        # JSON storage (users, sessions)
├── Backend/
│   └── Controller/
│       ├── auth_controller.py    # /api/auth/*
│       ├── csvcontroller.py      # /api/csv/*
│       ├── notes_controller.py   # /api/notes/*
│       ├── memos_controller.py   # /api/memos/*
│       └── user_manager.py       # User / session management
├── frontend/                     # React SPA
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx               # Router + Auth context
│       ├── api.js                # Fetch wrapper
│       ├── components/           # Sidebar, Topbar, Layout
│       └── pages/                # All page components
└── Doc/
    ├── project_intro.md          # This file
    └── server.md                 # Server configuration
```

---

## Roadmap

- [ ] Migrate user storage to PostgreSQL
- [ ] Add more games to the For Fun section
- [ ] Data visualisation tools
- [ ] CI/CD pipeline (GitHub Actions → EC2)
- [ ] Monitoring (CloudWatch)
- [ ] Dockerise the application
