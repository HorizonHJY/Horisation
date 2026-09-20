# Horisation

A private web platform for close friends — tools, community, and games in one place.

**Live:** https://horizonyhj.com · Access is invitation-only.

---

## Features

| Feature | Description |
|---------|-------------|
| **Market** | Second-hand trading — listings with photos, categories, and a two-step deal flow (interest → seller accepts → buyer confirms received) |
| **Tasks** | Bounty board — post a job, someone picks it up |
| **Message Board** | Threaded replies and likes |
| **Friends** | Requests, private chat, and separately-approved contact sharing |
| **Groups** | Group chat, independent of the friend graph |
| **Tarot** | 78-card Rider–Waite–Smith deck: shuffle, choose your own three, past / present / future, tap a card to look closer. Shuffle happens server-side |
| **Online Gomoku** | Real-time multiplayer Five in a Row over Socket.IO |
| **Memo** | Personal memo & task tracker with priorities and tags |
| **Travel Planner** | Multi-day itineraries with a shareable plan id |
| **Bill Split** | Split a bill, shareable by id |
| **CSV Workspace** | Upload, preview, and summarise CSV / Excel files |
| **Profile / Admin** | Profile editing; user management and market categories for admins |

Some features are limited by role — see `frontend/src/features.js`, which is the
only place that decides. A feature with no entry there is open to every member.

---

## Stack

**Backend:** Python 3.11 · Flask · Gunicorn · SQLite (SQLAlchemy) · Cloudflare R2

**Frontend:** React 18 · Vite · React Router v6 · Bootstrap 5

**Infrastructure:** AWS EC2 (Amazon Linux 2023) · Nginx · Cloudflare · Let's Encrypt

---

## Local Development

```bash
# Install Python deps
pip install -r requirements.txt

# Windows — starts Flask + Vite in one step
scripts\dev.bat

# Or manually
python app.py                  # API on :5000
cd frontend && npm run dev     # UI on :5173
```

Open http://localhost:5173

---

## Deploy to Server

```bash
bash ~/deploy.sh
```

Pulls latest code, installs deps, builds frontend, restarts service.

---

## Project Docs

| File | Contents |
|------|----------|
| `Doc/project_intro.md` | Full architecture & feature overview |
| `Doc/data_storage.md` | How and where data is stored |
| `Doc/server.md` | Server configuration reference |
| `Doc/ai_service.md` | AI service layer design (reviewed; tarot reading is the first feature) |
| `Doc/log.md` | Change log |
| `CLAUDE.md` | Guide for AI-assisted development |

---

## Related repositories

| Repo | What it is |
|------|-----------|
| [HorizonHJY/Horizon_MCP](https://github.com/HorizonHJY/Horizon_MCP) | MCP server that lets an agent (OpenClaw, over WhatsApp) query this site's database **read-only** — status, table listing, on-demand CSV/JSON export. Runs next to OpenClaw on the Mac mini and pulls a snapshot of `market.db` over SSH; never exports `user` or `session`. Separate repo so it deploys with a `git pull` and never touches this one. |
