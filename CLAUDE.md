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
| `Backend/Controller/market_db.py` | SQLAlchemy models + helpers: User, UserSession, Listing, ListingImage, Category, Message, MessageLike, Memo, GameRoom, friends/groups tables, TradeIntent, WantedListing (求购 posts: buyer posts, sellers DM, buyer marks fulfilled). Also owns the additive column migrations (`_migrate_columns`, `_migrate_category_labels`). |
| `Backend/Controller/groups_controller.py` | `/api/groups/*` — 群组：建组/拉人/群聊（独立于好友） |
| `Backend/Controller/friends_controller.py` | `/api/friends/*` — search, requests, private chat, contact-sharing approval, and `/conversations` (every DM room with last line + unread, **non-friends included** — the Chats list) |
| `Backend/Controller/friends_socket.py` | **The app's only `connect`/`disconnect` handler** plus private chat and the push helpers. python-socketio keys handlers by event name, so a second `@socketio.on('connect')` anywhere would silently replace this one — put per-connection work here instead. |
| `Backend/Controller/socketio_instance.py` | Shared `socketio` object and the sid→user registry that the single connect handler fills (`user_for_sid`) |
| `Backend/Controller/game_controller.py` | `/api/game/*` + Socket.IO events: online Gomoku rooms and moves |
| `Backend/Controller/travel_controller.py` / `travel_db.py` | `/api/travel/*` — multi-day itinerary planner, shareable 6-char plan id |
| `Backend/Controller/bill_controller.py` / `bill_db.py` | `/api/bill/*` — bill splitting, shareable 6-char bill id |
| `Backend/Controller/market_task_controller.py` / `market_task_db.py` | `/api/market/tasks/*` — bounty/task board |
| `Backend/Controller/weather_controller.py` | `/api/weather` — current weather + `tomorrow` block (Open-Meteo, St. Louis, 10-min cache). `/api/weather/history` — rolling 30-day daily temps `[{date, hi_c, hi_f, lo_c, lo_f, wmo, is_forecast, is_today}, …]` (one call with `past_days=23&forecast_days=7`, 1-hour cache). Both units always, after a 2026-09-20 mislabel incident. |
| `Backend/Controller/r2_manager.py` | Cloudflare R2 upload/delete via boto3 |
| `Backend/Controller/tarot_controller.py` | `/api/tarot/*` — three-card spread. The shuffle runs here, not in the browser: a reading you can re-roll from devtools is not a reading. **The card the reader touches is the card they get** (since 2026-10-09): `/draw` shuffles and hides the order, `/pick` turns over the card at the touched seat. Also the AI reading (`/reading`), rating and history routes — thin: quota, retries and the model call all live in the AI layer. |
| `Backend/Controller/tarot_db.py` | `tarot_readings` table: one row per `/draw` (the hidden shuffled order in `deck_order_json`; the spread grows one card per `/pick`), filled in by `/reading` (question, model text, parsed JSON) and by the rating. Owner-only lookups. **Business data; not the AI layer's table.** |
| `Backend/Service/bazi/` | **八字排盘 — arithmetic only.** `chart.py` builds the whole chart (four pillars, ten gods, hidden stems, 星运/自坐, 空亡, 纳音, common 神煞, 大运/小运/起运) from `lunar_python` plus true solar time (longitude + equation of time + China's 1986–91 DST). Checked against 问真八字. `cities.py` is the name → longitude table. `compat.py` is 合盘 arithmetic (day pillar 合冲, 生肖, 五行互补, 十神, score). The model never computes a pillar or a relation. |
| `Backend/Controller/bazi_controller.py` / `bazi_db.py` | `/api/bazi/*`, the `bazi_readings` table (birth data + chart + reading + the reader's 准/不准 marks) and `bazi_followups` (questions about a chart). Owner-only. **Delete is a hide** (`deleted_at`): the rows stay for analysis, by Horizon's decision 2026-10-12, and the page says so. The reading runs **in the background** (`socketio.start_background_task`) and the page polls — it takes up to 2 min, longer than nginx holds a request. |
| `Backend/Controller/bazi_compat_controller.py` / `bazi_compat_db.py` | `/api/bazi/compat/*`, the `bazi_compat_readings` + `bazi_compat_followups` tables. Owner-only; same claim/poll/hide pattern as `bazi_db`. Takes two existing `bazi_readings` ids (left = 本命, right = 对象), stores a snapshot of both charts + the pre-computed `compat` dict, then runs the AI reading in the background. |
| `Backend/Service/mbti/` + `Backend/Controller/mbti_controller.py` + `mbti_db.py` | MBTI test. In-house 20 / 60 question bank (`questions.py`), 16-type cards (`types.py`), scoring is deterministic — no AI call, no quota. `mbti_results` stores one row per test (users may re-take); `visibility` is per-row (`private` / `friends` / `public`). `/api/mbti/user/<username>` honours visibility, using `market_db.are_friends` to see if the viewer is a friend. |
| `Backend/Service/ai/scope.py` + `prompts/scope.md` | **What the tarot and BaZi readers will answer.** Edit the .md, not the code: 只回答 / 不回答 / 拒答话术 go into both system prompts; 拦截词 / 危机词 are checked by the server **before** any model call (no tokens, no quota) — tarot answers 422, a BaZi follow-up is stored as answered by `screen:blocked` / `screen:crisis`. Restart to apply edits. |
| `Backend/Service/ai/` | **The AI layer.** `service.py` is the one entry point (`ai.run(feature, user, role, quota_key, payload)` → `AIResult`); `client.py` the vendor-neutral HTTP call (DeepSeek default, Anthropic supported — raw `requests`, never a vendor SDK, because of eventlet); `quota.py` per-role limits + global daily cap; `usage.py` the `ai_usage` table; `prompts/` versioned prompt builders/parsers; `features/` the registry. Design and the reasons in `Doc/ai_service.md`; tests in `tests/test_ai_service.py`. |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/App.jsx` | Router, AuthContext (incl. guest mode + `requireAuth`), ThemeContext, UnreadContext, PrivateRoute / PageRoute / FeatureRoute, the global `SignUpPrompt` |
| `frontend/src/api.js` | Fetch wrapper (`credentials: include`). Never throws; every result has `ok`. On a non-2xx response the **whole error body rides along** (`retryable`, `error_kind`, `quota`…), not just `error`. |
| `frontend/src/features.js` | Per-role feature flags (`canAccess`) — the frontend half of role gating |
| `frontend/src/nav.js` | **The site map, in one place**: `NAV_SECTIONS`, which the sidebar renders. Add a page here. Icons are **Lucide React components** (`import { Home, Store, … } from 'lucide-react'`), passed as the `icon` field. The sidebar renders them as `<Icon size={16} strokeWidth={1.75} />`. |
| `frontend/src/index.css` | Global design system: tokens, focus ring, badge pairs, market card, dark theme |
| `frontend/src/components/Layout.jsx` | The shell. Desktop: the sidebar is **closed by default and toggled by a button** — the ☰ at the top-left opens it (docked, content moves over), the `«` in its header closes it; `[` does the same. Remembered in `localStorage['archbay.nav']` as `open`/`closed`. No hover, no rail, no breadcrumb — a menu you asked for or no menu (the ChatGPT pattern; the owner rejected a hover-revealed spine on 2026-09-20). Phone: hamburger + drawer, unchanged. |
| `frontend/src/components/Sidebar.jsx` | The navigation panel itself, rendered from `nav.js`. Takes `hidden` and sets `inert` when off-screen so Tab cannot walk into an invisible menu. |
| `frontend/src/components/Topbar.jsx` | The ☰ menu button (both form factors; hidden on desktop while the sidebar is open), theme toggle, user menu |
| `frontend/src/components/Modal.jsx` | Shared modal shell (Escape, focus trap, scroll lock, `aria-modal`) + `ConfirmDialog`. **Use this for anything that covers the page — never a bare div, never `window.confirm`.** It portals out of its caller's subtree, so a surface with its own palette must redeclare its variables (and scrollbar theming) on the dialog — pass `backdropClassName` for that. |
| `frontend/src/components/EnvRibbon.jsx` | Marks a non-production instance; renders nothing in production |
| `frontend/src/components/SocketProvider.jsx` | The app's single Socket.IO connection, alive for the whole session. `useSocket()` / `useSocketEvent()`. **Pages attach and detach handlers; a page must never call `socket.disconnect()`** — it would cut off notifications and chat everywhere. |
| `frontend/src/components/WeatherGreeting.jsx` | Home-page greeting. Four rows (today where/what, today numbers, tomorrow where/what, tomorrow HIGH/LOW). "See 30-day trend" button lazy-loads `WeatherChart`. |
| `frontend/src/components/WeatherChart.jsx` | 30-day temperature line chart. Uses `react-plotly.js` + `plotly.js-basic-dist` (basic variant to keep the chunk down). Double line: High red / Low blue; history is solid+round markers, forecast dashed+diamond markers, with a "today" dotted vertical. Tooltip shows both °F and °C. Lazy-loaded by `WeatherGreeting`. |
| `frontend/src/pages/Friends.jsx` | Really the messages page. Desktop: two panes, Teams-style — left rail with Chats / Friends / Requests / Add, right pane the open conversation; phone: one at a time with a back arrow. **Chats is the default** and lists every DM including non-friends (Market Reach Out). Contact sharing is one text chip (View contact / Request contact / Requested · waiting / Contact hidden); everything else about a person is behind `···` as a full sentence. Rewritten 2026-09-21 — the old row had five unlabelled icons. |
| `frontend/src/pages/Bazi.jsx` | `/bazi` — a 算盘 / 合盘 tab shell. 算盘 (default): form → chart (问真-style table, five-element bar, 大运 track) → AI reading laid out as a seal (判词), four slips (用喜忌闲), luck-pillar cards, a 流年 thread and a 准/不准 checklist (saved on the server), then a follow-up chat (追问). A 历史命盘 list reopens or hides past charts. 合盘 is in a separate file. Its own rice-paper palette under `.bazi`, like `.tarot`. |
| `frontend/src/pages/BaziCompat.jsx` | `/bazi` tab 合盘 — pick two of your own charts (left = 本命, right = 对象), click 合盘, see the compat result: double mini-chart row, score seal + verdict + chemistry, day-pillar card with 合冲 tint, five-element compare bars with 互补 chips, ten-god 4-pillar cards, 生肖, strengths / cautions / advice, 验证 checkmarks, 追问. Phone: pair stacks, mini-chart pillars go 2x2, columns collapse. 合盘历史 is a separate list under this tab. |
| `frontend/src/pages/Mbti.jsx` | `/mbti` — three-state page: intro (pick 20 / 60, show last result if any), quiz (progress bar + one statement + two choices, back button, cancel), result (XL purple-gradient badge + nickname + description + four dimension bars + strengths / blind-spots + match chips + visibility radio + retake). Visibility change saves immediately via `PATCH /api/mbti/me/visibility`. |
| `frontend/src/pages/TarotHistory.jsx` | `/tarot/history` — your interpreted spreads, newest first, 20 per page (*Older* pages back by `read_at`). Each row is a `<details>`: three thumbnails, date, rating, question; opens to the full reading. Unread draws are not listed. Linked from the Tarot page title row and back. |
| `frontend/src/components/TarotReading.jsx` | The "整体解读" section under the three cards: optional question → `POST /api/tarot/reading` → summary / three positions / one next step → 1–5 fit rating. Reads `retryable` off the error body to decide whether to offer *Try again*. |
| `frontend/src/pages/Market.jsx` | The marketplace page: Browse / **Wanted** / My Listings / Post Item tabs, filters in the query string, and the `ListingCard` / `ListingDetailModal` / `SellerModal` surfaces. A sold listing renders with a grey mask + rotated `SOLD` stamp over its image (`.market-card__img--sold`, overlay is `pointer-events:none` so the card stays one click target). **Browse returns `status='active'` only** — the sold mask shows on My Listings and a seller's public page, never in Browse (deliberate, per PRODUCT.md 2026-09-28). |
| `frontend/src/pages/MarketWanted.jsx` | Market → Wanted sub-panel. Buyers post what they're looking for; sellers DM them (via `navigate('/friends',{state:{openChat,initialMessage}})`, same mechanism as Market's Reach Out). Three sub-tabs (Browse / My / Post). Open to guests for browse; posting / fulfilling / deleting requires login. Fulfilled cards render dimmed with a line-through title. |
| `frontend/src/pages/` | All page components |

### Data
| Path | Contents | Git tracked? |
|------|----------|-------------|
| `_data/market.db` | SQLite — **all** structured data: users, sessions, listings, images, categories, memos, messages, friends, groups, games, travel, bills, tasks, tarot readings, AI usage | No (gitignored) |
| `_data/notes/` | Per-user note JSON files | Yes |
| `Backend/data/tarot_deck.json` | 78 Rider–Waite–Smith cards: id, name, arcana, image filename, Waite's 1911 upright text, plus `name_zh` / `keywords_zh`. The Chinese layer is **authored in `scripts/tarot_add_zh.mjs`** (no open Chinese dataset exists) — edit it there and re-run, never in the JSON by hand. | Yes |
| `frontend/public/tarot/*.jpg` | RWS card scans, 78 files (~7.6 MB) from `metabismuth/tarot-json` (MIT); the deck itself is US public domain | Yes |
| `_data/users.json.migrated` | Pre-March-2026 JSON store, migrated into SQLite and renamed | Yes (inert) |
| `Key/r2_config.json` | Cloudflare R2 credentials | No (gitignored) |
| `Key/ai_config.json` | AI provider + key (`{provider, api_key, model}`; template at `ai_config.example.json`). Env vars `AI_PROVIDER` / `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY` / `AI_MODEL` win over the file. `AI_ENABLED=0` is the kill switch. | No (gitignored) |
| `Doc/ai_service.md` | **Reviewed design** for the AI layer (`Backend/Service/ai/`): contract, quota keys, timezone, the eventlet constraint, tarot reading flow, history/rating tables. Read it before touching anything AI. | Yes |
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
- AI config: `/home/ec2-user/Horisation/Key/ai_config.json` (manual, never in git; copy `ai_config.example.json`). Without it every `/reading` answers 503 "config" and nothing else on the site is affected.

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
| GET | `/wanted` | optional | Open wanted posts (reverse of listings — buyer posts, sellers reach out). Optional `?category=`. Guests can browse. |
| POST | `/wanted` | login | Create a wanted post `{title, description, category, contact?, price_min?, price_max?, condition='any'|'new'|'used'}` |
| GET | `/wanted/mine` | login | All your wanted posts (open / fulfilled / removed), newest first |
| GET | `/wanted/<id>` | optional | Single wanted post |
| PUT | `/wanted/<id>` | login | Edit (buyer only) |
| POST | `/wanted/<id>/fulfilled` | login | Buyer marks "I found it" — drops out of the open feed |
| POST | `/wanted/<id>/reopen` | login | Undo fulfill |
| DELETE | `/wanted/<id>` | login | Delete |

Reach Out on a wanted post opens a DM with the buyer (same `navigate('/friends',{state:{openChat,initialMessage}})` as listings' Reach Out). No trade_intent is created — a wanted post has no listing to reserve, and the buyer/seller roles are swapped.

### Feedback `/api/feedback/`
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/messages` | Get all messages (latest 200) |
| POST | `/messages` | Post message (max 500 chars) |
| DELETE | `/messages/<id>` | Delete own message (admin: any) |

### Tarot `/api/tarot/`
Login required, and open to every member — it carries no entry in `features.js`
and no `FeatureRoute`, the same as Market, Tasks, Message Board, Friends and
Groups. It was `['horizon']` only while it was being built (2026-09-08 to
09-10). Do not "re-gate" it by listing all five roles in `FEATURES`; a flag
naming every role is a gate that does nothing.

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/deck` | All 78 cards — the fan is laid out from this |
| POST | `/draw` | Shuffles all 78 with `secrets.SystemRandom`, never `random`, stores the order in a new `tarot_readings` row and returns **only** `reading_id`. No card leaves the server here. |
| POST | `/pick` | Body `{reading_id, seat}`; `seat` is the card's index in `/deck` order (0–77), i.e. its place in the fan. Returns the card the server shuffled into that seat, in the next empty position (past → present → future), plus `done`. 409 for a seat already taken, a fourth card, or a lost race (the write is compare-and-swap on `spread_json`). |
| POST | `/reading` | Body `{reading_id, question?}`. The AI reads the spread **the server drew** — nothing about the cards travels up. 404 if the id is not yours; 409 until all three cards are picked. Idempotent: an already-read spread returns the stored reading without a model call or a quota charge. Errors carry `error_kind`, `retryable` and (for quota) `quota`; status 429 quota, 503 global cap / disabled / no key, 502 model failure. |
| POST | `/readings/<id>/rating` | `{rating: 1–5, note?}` — how well it fit. Only after a reading exists. Overwritable. |
| GET | `/readings` | Own interpreted readings, newest first (`limit` ≤ 50, `before` ISO cursor = last row's `read_at`). Used by `/tarot/history`. |

Upright only — the deck file carries no reversed meanings.

**AI reading.** Quota is per person per Chicago day: `user` 1, `vip`/`svip` 3, `admin`/`horizon` unlimited;
a site-wide cap of 100 ok calls per 24h and `AI_ENABLED=0` sit above that. A failed call is logged but
not charged; one automatic retry on timeout / 5xx. The model answers in JSON (`past`, `present`,
`future`, `summary`, `next_step`); if it breaks the schema the raw text is kept, shown whole, and still
charged. Prompt text lives in `Backend/Service/ai/prompts/tarot.py` and is versioned (`tarot-v1`);
bump `PROMPT_VERSION` whenever the wording changes so ratings stay comparable across versions.

The flow since 2026-09-12: *Start* (after the on-screen instruction to hold a question and
say it three times) → shuffle → take three; **each card turns over the moment it lands** and
its reading appears below at once. There is no separate reveal step.

`/draw` is called the moment the shuffle animation starts, before the reader has
touched anything; the card they then pick takes the next card in that order.
Which physical card you touch in a shuffled deck carries no information, so this
is as true as a real reading — but it means **the pick is theatre and the draw is
not**, and the two must not be conflated when changing this page.

### BaZi `/api/bazi/`
Login required, open to every member. Listed under Tarot in For Fun.

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/cities` | Birth places with longitude (for the datalist) |
| POST | `/chart` | `{date, time, place, longitude?, utc_offset?=8, gender}` → `{reading_id, chart}`. `time` is clock time; the server applies true solar time. Unknown `place` needs `longitude`. Writes a `bazi_readings` row. |
| GET | `/latest` | Your most recent chart + reading state, so the page reopens where you left it |
| GET | `/readings` | Your charts, newest first, one line each (pillars, birth, place, verdict) |
| POST | `/reading` | `{reading_id}` → starts the AI reading in the background, returns at once with `item.state` (`pending`/`done`). A done reading is returned free. Feature `bazi`: no per-person limit, 100/day site cap, 120 s read timeout, 3,500 tokens. |
| GET | `/reading/<id>` (= `/readings/<id>`) | Poll / open one. `item`: chart, `state` (`none` / `pending` / `done` / `error`), reading, `verify`, `followups`. A pending older than 4 min counts as lost and can be restarted. |
| PUT | `/readings/<id>/verify` | `{marks: {"0": "yes", "2": "no"}}` — the reader's 准/不准, whole set replaced |
| POST | `/readings/<id>/ask` | `{question}` (≤ 300 chars) → a follow-up answered in the background (feature `bazi_ask`: no per-person limit, 500/day site cap, 60 s, plain text). The model gets the chart, the reading, the marks and the last 8 Q&A. 409 before the reading exists. |
| GET | `/followups/<id>` | Poll one follow-up |
| DELETE | `/readings/<id>` | Hide one chart from you (row kept) |
| DELETE | `/mine` | Hide every chart of yours (rows kept) |

The prompt (`Backend/Service/ai/prompts/bazi.py`, `bazi-v1`) is Horizon's own persona and four asks (2026-10-12), plus: the chart is handed over finished and must not be recomputed, and the answer is fixed-shape JSON so the page can lay it out.

### BaZi 合盘 `/api/bazi/compat/`
Login required. Operates on two of the caller's own `bazi_readings`. Left = 本命, right = 对象.

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/` (= `/api/bazi/compat`) | `{chart_a_id, chart_b_id}` → `{reading_id, item}`. Loads both charts, runs `compat.build_compat()` (day pillar 合冲, 生肖, 五行互补, 十神, score 0–100), stores a snapshot in `bazi_compat_readings`. 404 if either chart is not yours. |
| POST | `/reading` | `{reading_id}` → starts the AI reading in the background. Feature `bazi_compat`: no per-person limit, 60/day site cap, 120 s, 3,500 tokens. Idempotent via `claim()`. |
| GET | `/reading/<id>` (= `/readings/<id>`) | Poll / open one. `item`: `chart_a`, `chart_b`, `compat`, `reading`, `state`, `verify`, `followups`. |
| GET | `/readings` | Your compat history, newest first (score, both 八字, both birth lines, verdict). |
| PUT | `/readings/<id>/verify` | `{marks: {"0": "yes"}}` — your 准/不准 on the compat verify items. |
| POST | `/readings/<id>/ask` | `{question}` (≤ 300) → follow-up on this pair (feature `bazi_compat_ask`: 300/day site cap, 60 s, text). 409 before the compat reading exists. Scope-screened. |
| GET | `/followups/<id>` | Poll one follow-up. |
| DELETE | `/readings/<id>` | Hide one compat (row kept). |
| DELETE | `/mine` | Hide all your compats (rows kept). |

Prompt `Backend/Service/ai/prompts/bazi_compat.py`, `bazi-compat-v1`. Model sees both finished charts, the pre-computed relations and the gender pair (异性情缘盘 / 同性友情合作盘). JSON shape: `verdict / level / chemistry / day_pillar / complement / ten_god / zodiac / strengths / cautions / advice / verify / summary`.

### MBTI `/api/mbti/`
Deterministic — no AI call. Users may re-take; latest row is "current".

| Method | Route | Auth | Description |
|--------|-------|------|-------------|
| GET | `/questions?mode=short\|full` | optional | 20 or 60 statements, each with two choices and the pole each choice votes for |
| GET | `/types` | optional | The 16-type reference card (nickname / tagline / description / strengths / blind spots / common matches) |
| POST | `/submit` | login | `{mode, answers:[0|1, ...]}` → scored type + `type_card`. Writes a new `mbti_results` row |
| GET | `/me` | login | Caller's latest + full history |
| GET | `/user/<username>` | optional | Someone else's latest. Visibility filter: `public` → everyone; `friends` → viewer must be a friend; `private` → owner only. Null `result` means "not set or not visible to you". |
| PATCH | `/me/visibility` | login | `{visibility: 'private'|'friends'|'public'}` on the latest row |
| DELETE | `/me` | login | Wipe every result for the caller |

### Friends `/api/friends/` — notification surface
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/conversations` | **The Chats list.** Every DM room the caller is in — friends and non-friends — with the other person's identity, `is_friend`, `last_sender` / `last_content` / `last_at`, `unread`; newest first. Added 2026-09-21: before it, a non-friend's message raised the badge but had no row anywhere to open it from. |
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
@login_optional      # attaches request.current_user when signed in, else None —
                     #   for GETs a visitor may read (Market, Message Board,
                     #   Tasks, Tarot, BaZi). Writes stay @login_required.
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

## Related repositories

- **[HorizonHJY/Horizon_MCP](https://github.com/HorizonHJY/Horizon_MCP)** — MCP server giving an agent
  (OpenClaw on the owner's Mac mini, driven over WhatsApp) read-only access to this site's database:
  `db_status`, `list_tables`, `export_data`, `refresh_data`. It pulls a consistent snapshot of
  `_data/market.db` over SSH (`ssh archbay`, SQLite backup API) and refuses the `user` and `session`
  tables outright. **Nothing in it runs on the EC2 box** and it never writes. If a table is renamed
  here, that repo's tests still pass (they use a throwaway db) but its exports of that table stop
  working — tell whoever maintains it.

---

## Known Limitations / Future Work
- **Passwords stored in plaintext** → needs bcrypt. Anyone with `market.db` (including the
  admin "Download DB" button) has every user's password in the clear. Highest-value fix.
- `SECRET_KEY` hardcoded in `app.py` → should come from the environment
- SQLite single-writer under concurrent writes → PostgreSQL when scale warrants
- CI/CD exists: `.github/workflows/deploy.yml` SSHs to EC2 and runs `~/deploy.sh` on push to `main` — **except** pushes that only touch `*.md`, `Doc/**` or `.impeccable/**` (`paths-ignore`), which restart nothing

---

## Before Every Push — update the docs

Every push to `main` ships, so the docs ship with it. Before `git push`, update
whichever of these the change touched, and commit them in the same push:

| Doc | Update when |
|-----|-------------|
| `Doc/log.md` | **Always.** A dated entry at the top of *Latest Summary* (what changed, why, how it was verified, commit hash); bump *Last Updated*; add a row to the decision table if a choice was made. |
| `CLAUDE.md` | A route, file, table, column or convention changed — the *Key Files* and *API Endpoints* tables must match the code. |
| `PRODUCT.md` | Horizon confirmed a product decision ("Confirmed YYYY-MM-DD: …"). |
| `Doc/ai_service.md` | Anything under `Backend/Service/ai/` or the tarot flow changed. |
| Other `Doc/*.md` | The doc that owns the area (`groups.md`, `server.md`, `data_storage.md`, `mobile_ux_principles.md`, `dev_workflow.md`, `todo.md`). |

A PreToolUse hook (`.claude/hooks/docs-before-push.sh`) blocks a `git push` whose
commits change code but no `.md` file. It can only check that *some* doc moved,
not that the right one did — that part is on you. For a change that truly needs
no doc, put `[no-docs]` in a commit message.

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
