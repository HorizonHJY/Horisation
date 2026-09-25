# Local Dev & Delivery Workflow

How to run Horisation locally, prove a change with real evidence, and ship it.
Written 2026-09-24 after the Friends-mobile work, where three of these rules
each cost a wasted round.

## 0. Always start from the live tip

```bash
cd ~/Horisation
git fetch origin
git status -sb         # must say "## main...origin/main", not "behind N"
```

A screenshot taken from a stale checkout is worthless. On 2026-09-24 a whole
before/after comparison was run on a tree **10 commits behind** `origin/main`,
and the page in those images no longer existed. `git pull --ff-only origin main`
before you touch anything.

## 1. Run it locally

**Backend (Flask API + Socket.IO):**

```bash
cd ~/Horisation
# one-time: python3.11 -m venv .venv-dev && .venv-dev/bin/pip install -r requirements.txt
LOCAL_DEV=1 .venv-dev/bin/python /tmp/run_flask.py   # see below
```

`LOCAL_DEV=1` is **required**. Without it `app.py` takes the production branch
of `socketio.init_app` — `message_queue='redis://', async_mode='eventlet'` —
and dies with *"Redis requires a monkey patched socket library to work with
eventlet"* even though no Redis is installed locally. With it, Socket.IO runs
in `threading` mode.

`/tmp/run_flask.py` (port 5100 because macOS Control Center owns 5000):

```python
import sys, os
sys.path.insert(0, '/Users/horizon/Horisation')
os.environ['LOCAL_DEV'] = '1'
os.chdir('/Users/horizon/Horisation')
import app as A
A.socketio.run(A.app, debug=False, host='127.0.0.1', port=5100, allow_unsafe_werkzeug=True)
```

When new backend dependencies land (the AI layer added `requests`), reinstall:
`.venv-dev/bin/pip install -r requirements.txt`.

**Frontend (Vite):**

```bash
cd ~/Horisation/frontend
npm run dev -- --host 127.0.0.1     # --host 127.0.0.1 matters, see below
```

Vite binds **IPv6-only** by default, so `http://127.0.0.1:5173` is refused while
`http://[::1]:5173` works. `--host 127.0.0.1` makes IPv4 work, which the
screenshot tool below needs.

Point the dev proxy at 5100 in `frontend/vite.config.js` (`/api` and
`/socket.io` → `http://127.0.0.1:5100`). **This is a debug change — revert it
before committing** (see §4).

**Local accounts** (plaintext passwords in the dev DB — bcrypt migration is
still on `todo.md`): `horizon` / `horizon`, `testuser` / `test`,
`fanfan0315` / `yyf`.

## 2. Screenshot it, with no new dependencies

The OpenClaw browser tool refuses `localhost`/private hosts (SSRF policy is a
protected config path). Instead drive the installed Chrome over CDP from Node
26's built-in `WebSocket` — no npm install:

```bash
# headless Chrome on a private debug port
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --user-data-dir=/tmp/chrome-shot-profile \
  --remote-debugging-port=19222 about:blank &
```

Then a small script that: attaches to a fresh page target, sets
`Emulation.setDeviceMetricsOverride` (width/height/deviceScaleFactor/mobile),
turns on touch emulation for phone widths (so `@media (pointer: coarse)`
applies), injects the session cookie with `Network.setCookie`, navigates,
clicks the tab you need, and `Page.captureScreenshot`. Reference:
`/tmp/shot2.mjs` and `/tmp/measure2.mjs` (they measure element boxes too).

Gotchas that cost time:

- **The login cookie is named `session`**, not `session_token`. `POST
  /api/auth/login` sets a Flask session cookie; the token in the JSON body is
  for API calls, not the browser session.
- **Log in from Python, not the shell.** `curl -d '{"password":"..."}'` was
  intermittently mangled and returned 401; `--data-binary` or a
  `urllib` + `http.cookiejar` snippet is reliable.
- Log in **twice-ish**: a `401` in the log right after startup is often just a
  request racing the server; retry.
- The Friends page defaults to the **Chats** tab (often empty). Click the
  **Friends** tab before measuring the friend rows.
- Measure, don't eyeball: compare `scrollWidth` vs `clientWidth` for
  truncation, and `getBoundingClientRect()` for tap targets. A screenshot
  review is a sanity check, not proof.

## 3. Verify before you claim

Check every surface the change touches — **phone (390), tablet (768), desktop
(1280/1440)**. The project's rule is that all three must be good and the
**desktop experience must be the best**.

For a UI fix, "done" means: build passes, and the specific defect is shown
gone by measurement on each surface.

## 4. Delivery checklist (the standard close-out)

Once the change is approved:

1. **Revert local debug edits** — restore `frontend/vite.config.js` (proxy port
   back to 5000).
2. **Stage only the files this change touches.** Never commit the workspace
   litter (`package.json`, root `node_modules/`, `*.bak`, `business-card.html`).
3. **`npm run build`** in `frontend/` — must pass.
4. **Write the commit message as *why*, not *what*.**
5. **`git push origin main`** — this triggers the GitHub Actions deploy to EC2
   (`.github/workflows/deploy.yml`, `paths-ignore` for docs-only pushes).
6. **Report**: commit hash, screenshots for each surface, deploy note, and
   whether any leftover dev processes should be killed.

Note on naming: Horizon says "master", but this repo's branch is **`main`**
(remote `origin/main`).

## 5. Cross-checking with Claude

When a design decision is wanted, Claude Code CLI can be asked for options, and
can write the code for review:

```bash
claude -p "$(cat /tmp/spec.md)" --model opus --permission-mode acceptEdits
```

Claude runs **read-only tools only** by default in this setup — `npm` and `git`
are denied at the permission layer, so it cannot run the build or the diff. That
is the reviewer's job: run `npm run build`, the screenshots, and the
measurements yourself, and don't take its word for build status.
