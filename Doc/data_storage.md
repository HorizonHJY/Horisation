# Horisation — Data Storage Reference

Last updated: 2026-03-06

---

## Overview

All persistent data lives in a single **SQLite database** (`_data/market.db`) managed by SQLAlchemy ORM.
Actual image files are stored in **Cloudflare R2** object storage.

| Layer | Technology | Location | Used For |
|-------|-----------|----------|----------|
| Relational DB | SQLite (SQLAlchemy) | `_data/market.db` | All structured data |
| Object Storage | Cloudflare R2 | Cloud bucket | Image files (listings + avatars) |
| JSON files | Plain JSON | `_data/notes/` | Per-user notes (git tracked) |

> **Previous state**: `users.json` and `sessions.json` were used before March 2026.
> They were migrated to SQLite on first startup and renamed to `.migrated`.

---

## SQLite Database — `_data/market.db`

Created automatically on app startup via `init_db()` in `market_db.py`.
Built with SQLAlchemy ORM — can migrate to PostgreSQL by changing the engine URL (one line).

### Table: `user`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER | Auto-increment PK |
| username | TEXT | Unique, indexed |
| password | TEXT | Plaintext (⚠️ needs bcrypt) |
| role | TEXT | horizon / horizonadmin / vip1 / vip2 / vip3 / user |
| email | TEXT | Optional |
| display_name | TEXT | Shown in UI |
| is_active | BOOLEAN | Deactivated users cannot log in |
| avatar_url | TEXT | R2 public URL |
| contact_info | TEXT | WeChat / phone / handle — shared on request |
| contact_hidden | BOOLEAN | When true, friends cannot send contact requests |
| created_at | DATETIME | UTC |

**Managed by:** `market_db.py` helpers (`db_create_user`, `db_get_user`, `db_update_user`, etc.)
via `user_manager.py` public API.

---

### Table: `session`

| Column | Type | Notes |
|--------|------|-------|
| token | TEXT | PK (64-char random hex) |
| username | TEXT | Indexed |
| created_at | DATETIME | UTC |
| expires_at | DATETIME | UTC — 24h from login |

---

### Table: `listings`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| seller_username | TEXT | Indexed |
| title | TEXT | Max 100 chars |
| description | TEXT | Full text |
| price | REAL | Selling price (non-negative) |
| original_price | REAL | Optional; shown with strikethrough if > price |
| category | TEXT | electronics / clothing / books / furniture / other |
| contact | TEXT | Legacy field (kept for compatibility) |
| status | TEXT | `active` / `sold` / `removed` |
| created_at | DATETIME | UTC |
| updated_at | DATETIME | UTC, auto-updates |

---

### Table: `listing_images`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| listing_id | TEXT (UUID) | FK → listings.id (cascade delete) |
| r2_url | TEXT | Public URL served to browser |
| r2_key | TEXT | Object key in R2 (for deletion) |
| display_order | INTEGER | 0, 1, 2 |

---

### Table: `messages` (Message Board)

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| username | TEXT | Author |
| display_name | TEXT | Snapshot at post time |
| content | TEXT | Max 500 chars |
| created_at | DATETIME | UTC |

---

### Table: `memos`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| username | TEXT | Indexed |
| content | TEXT | Memo body |
| type | TEXT | general / todo / reminder / idea |
| priority | TEXT | low / normal / high |
| status | TEXT | active / completed |
| tags | TEXT | JSON array |
| due_date | TEXT | Optional date string |
| completed_at | DATETIME | Set when status → completed |
| created_at / updated_at | DATETIME | UTC |

---

### Table: `game_rooms` (Online Gomoku)

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| name | TEXT | Room display name |
| host | TEXT | Player 1 username (black stones) |
| player2 | TEXT | Player 2 username (white stones), nullable |
| status | TEXT | waiting / playing / finished |
| board | TEXT | JSON array of 225 cells (15×15) |
| current_turn | TEXT | Username whose turn it is |
| winner | TEXT | Username of winner, nullable |
| win_cells | TEXT | JSON array of winning cell indices |
| created_at | DATETIME | UTC |

---

### Table: `friend_requests`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| from_user | TEXT | Indexed |
| to_user | TEXT | Indexed |
| status | TEXT | pending / accepted / rejected |
| message | TEXT | Optional note sent with request |
| created_at / updated_at | DATETIME | UTC |

---

### Table: `friendships`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| user_a | TEXT | Lexicographically smaller username |
| user_b | TEXT | Lexicographically larger username |
| created_at | DATETIME | UTC |

> `user_a <= user_b` always — use `_friend_pair(a, b)` helper to normalise before querying.

---

### Table: `private_chat_messages`

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT (UUID) | PK |
| room_key | TEXT | `"{user_a}:{user_b}"` (sorted), indexed |
| sender | TEXT | Sender username |
| content | TEXT | Max 1000 chars |
| created_at | DATETIME | UTC |

---

### Table: `contact_requests`

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER | Auto-increment PK |
| from_user | TEXT | Requester, indexed |
| to_user | TEXT | Target (whose contact is requested), indexed |
| status | TEXT | pending / approved / declined |
| created_at | DATETIME | UTC |

> Friends can request to see another friend's `contact_info`.
> The target must `approve` before the info is revealed.
> `contact_hidden = true` on User prevents any requests from being sent.

---

## Cloudflare R2 (Object Storage)

Used to store image files. Accessed via `boto3` (S3-compatible API).

**Bucket:** `horisation-market`
**Config file:** `Key/r2_config.json` (gitignored — never committed)

### Object Key Format
```
listings/<listing_uuid>/<image_uuid>.jpg   ← listing images
avatars/<username>.jpg                     ← user avatars
```

### Flow — Listing Image
```
User uploads image
  → Flask validates (JPEG/PNG, max 5MB, max 3 per listing)
  → r2_manager.upload_image() → R2
  → URL + key stored in listing_images table
  → Browser loads image directly from R2 CDN

User deletes listing
  → market_db.delete_listing() returns r2_keys list
  → r2_manager.delete_image() removes each from R2
  → DB rows deleted via cascade
```

**Managed by:** `Backend/Controller/r2_manager.py`

---

## Notes Storage (`_data/notes/`)

Per-user JSON files, one file per user, git-tracked.

```
_data/notes/
├── horizon_notes.json
└── <username>_notes.json
```

Each file is a JSON array of note objects: `{ id, title, content, created_at, updated_at }`.

**Managed by:** `Backend/Controller/notes_controller.py`

---

## File Layout Summary

```
_data/
├── market.db            ← all structured data (SQLite, gitignored)
└── notes/               ← per-user note JSON files (git tracked)

Key/
└── r2_config.json       ← R2 credentials (gitignored)

Cloudflare R2 bucket: horisation-market
├── listings/<id>/<img>.jpg
└── avatars/<username>.jpg
```

---

## PostgreSQL Migration Path

Change one line in `market_db.py`:

```python
# Current (SQLite)
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)

# Future (PostgreSQL)
engine = create_engine('postgresql://user:password@host/dbname', echo=False)
```

All models and helpers are ORM-based and require no further changes.

---

## Known Limitations

| Issue | Impact | Fix |
|-------|--------|-----|
| Plaintext passwords | Security risk | bcrypt hashing |
| SQLite single-writer | Concurrent writes may fail under load | Migrate to PostgreSQL |
| No soft delete on users | Deleted user data lost | Add `deleted_at` timestamp |
