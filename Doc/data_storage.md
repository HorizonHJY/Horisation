# Horisation — Data Storage Reference

Last updated: 2026-03-01

---

## Overview

The project uses **three different storage layers** depending on the nature of the data:

| Layer | Technology | Location | Used For |
|-------|-----------|----------|----------|
| JSON files | Plain JSON | `_data/` | Users, sessions, notes |
| Relational DB | SQLite (SQLAlchemy) | `_data/market.db` | Market listings & images metadata |
| Object Storage | Cloudflare R2 | Cloud bucket | Market listing images (actual files) |

---

## 1. JSON File Storage (`_data/`)

Simple file-based storage. Each read/write loads or saves the entire file.

### `_data/users.json`
Stores all user accounts. Dict key is the user's registration key (may differ from `username` field — use `_find_user()` to look up by username).

```json
{
  "horizon": {
    "username": "horizon",
    "password": "horizon",
    "role": "horizon",
    "email": "horizon@horisation.com",
    "display_name": "Horizon Administrator",
    "created_at": "2025-10-12T16:51:56.646191",
    "is_active": true,
    "memos": [
      {
        "id": "uuid",
        "content": "...",
        "type": "general | todo | reminder | idea",
        "tags": [],
        "priority": "low | normal | high",
        "status": "active | completed",
        "created_at": "ISO8601",
        "updated_at": "ISO8601",
        "due_date": null,
        "completed_at": null
      }
    ]
  }
}
```

**Managed by:** `Backend/Controller/user_manager.py`
- `_load_users()` / `_save_users()`
- `_find_user(users, username)` — search by `username` field, not dict key

> ⚠️ Memos are stored **inside** the user object (denormalised). This works for small datasets but means loading all users just to read one user's memos.

---

### `_data/sessions.json`
Active login session tokens. Expires after 24 hours.

```json
{
  "token_string": {
    "username": "horizon",
    "created_at": "ISO8601",
    "expires_at": "ISO8601"
  }
}
```

**Managed by:** `Backend/Controller/user_manager.py`
- `create_session(username)` → token
- `validate_session(token)` → user info or None

---

### `_data/notes/<username>_notes.json`
Per-user notes files. One file per user, created on first note.

```json
[
  {
    "id": "uuid",
    "title": "...",
    "content": "...",
    "created_at": "ISO8601",
    "updated_at": "ISO8601"
  }
]
```

**Managed by:** `Backend/Controller/notes_manager.py`

---

## 2. SQLite Database (`_data/market.db`)

Used for the Market feature. Created automatically on app startup via `init_db()`.
Built with **SQLAlchemy ORM** — can be migrated to PostgreSQL by changing one line (the engine URL).

### Table: `listings`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | Primary key |
| seller_username | TEXT | References username in users.json |
| title | TEXT | Max 100 chars |
| description | TEXT | Full text |
| price | REAL | Float, non-negative |
| category | TEXT | electronics / clothing / books / furniture / other |
| contact | TEXT | WeChat / phone, shown publicly |
| status | TEXT | `active` / `sold` / `removed` |
| created_at | DATETIME | UTC |
| updated_at | DATETIME | UTC, auto-updates |

### Table: `listing_images`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | Primary key |
| listing_id | TEXT (UUID) | FK → listings.id (cascade delete) |
| r2_url | TEXT | Public URL served to browser |
| r2_key | TEXT | Object key in R2 bucket (for deletion) |
| display_order | INTEGER | 0, 1, 2 |

**Managed by:** `Backend/Controller/market_db.py`

> Migration to PostgreSQL: change engine URL in `market_db.py`:
> ```python
> # SQLite (current)
> create_engine('sqlite:///_data/market.db')
>
> # PostgreSQL (future)
> create_engine('postgresql://user:password@host/dbname')
> ```

---

## 3. Cloudflare R2 (Object Storage)

Used to store actual image files for market listings.
Accessed via `boto3` with an S3-compatible API.

**Bucket:** `horisation-market`
**Config file:** `Key/r2_config.json` (gitignored — never committed)

### Object Key Format
```
listings/<listing_uuid>/<image_uuid>.jpg
```

### Flow
```
User uploads image
  → Flask validates (JPEG/PNG, max 5MB, max 3 per listing)
  → r2_manager.upload_image() puts file to R2
  → R2 returns public URL
  → URL + key stored in listing_images table
  → Browser loads image directly from R2 (no Flask involvement)

User deletes listing
  → market_db.delete_listing() returns list of r2_keys
  → r2_manager.delete_image() removes each file from R2
  → DB rows deleted via cascade
```

**Managed by:** `Backend/Controller/r2_manager.py`
- `upload_image(file_obj, filename)` → `(r2_key, public_url)`
- `delete_image(r2_key)` → `bool`

---

## File Layout Summary

```
_data/
├── users.json          ← all users + their memos (JSON)
├── sessions.json       ← active session tokens (JSON)
├── market.db           ← market listings + image metadata (SQLite)
└── notes/
    ├── horizon_notes.json
    └── <username>_notes.json

_uploads/               ← temporary CSV uploads (not persisted)

Key/
└── r2_config.json      ← R2 credentials (gitignored)

Cloudflare R2 bucket: horisation-market
└── listings/<id>/<img>.jpg   ← actual image files
```

---

## Limitations & Future Improvements

| Current | Limitation | Future Fix |
|---------|-----------|------------|
| users.json | No concurrent write safety | Move to PostgreSQL |
| Memos inside users.json | Must load all users to query memos | Separate `memos` table in DB |
| Plain text passwords | Security risk | bcrypt hashing |
| SQLite | Single-writer, file-based | Migrate to PostgreSQL (one-line change) |
