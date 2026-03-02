# Horisation

A private web platform for close friends — tools, community, and games in one place.

**Live:** https://horizonyhj.com · Access is invitation-only.

---

## Features

| Feature | Description |
|---------|-------------|
| **Hormemo** | Personal memo & task tracker with priorities and tags |
| **CSV Workspace** | Upload, preview, and summarise CSV / Excel files |
| **Market** | Second-hand trading — post listings with photos, price, contact |
| **Message Board** | Community message board for all users |
| **Gomoku** | Local 2-player Five in a Row (15×15) |
| **Profile** | Update display name, email, password, avatar |
| **Admin** | User management — create, edit, reset password, delete |

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
| `Doc/log.md` | Change log |
| `CLAUDE.md` | Guide for AI-assisted development |
