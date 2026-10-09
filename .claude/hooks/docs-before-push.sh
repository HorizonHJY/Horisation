#!/bin/bash
# PreToolUse hook (Bash): stop a `git push` from this repo when the commits
# about to go out change code but no .md file. See "Before Every Push" in
# CLAUDE.md for which docs to update.
#
# Bypass for a change that truly needs no doc: put [no-docs] in a commit
# message in the range being pushed.
#
# Registered twice: in this repo's .claude/settings.json (sessions started in
# the repo) and in ~/.claude/settings.json with HOOK_SCOPE=user (sessions
# started elsewhere, e.g. OpenClaw from $HOME). The user copy steps aside when
# the project copy is already loaded, so the check never runs twice.

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

if [ "$HOOK_SCOPE" = "user" ] && [ "$CLAUDE_PROJECT_DIR" = "$REPO" ]; then
  exit 0
fi

INPUT="$(cat)"
read -r CMD_IS_PUSH CMD_TOUCHES_REPO <<<"$(REPO="$REPO" python3 -c '
import json, os, re, sys
d = json.loads(sys.stdin.read() or "{}")
cmd = (d.get("tool_input") or {}).get("command", "")
cwd = d.get("cwd", "")
repo = os.environ["REPO"]
push = bool(re.search(r"\bgit\b[^|;&\n]*\bpush\b", cmd))
here = cwd == repo or cwd.startswith(repo + "/") or repo in cmd or os.path.basename(repo) in cmd
print(int(push), int(here))
' <<<"$INPUT")"

[ "$CMD_IS_PUSH" = "1" ] && [ "$CMD_TOUCHES_REPO" = "1" ] || exit 0

UP="$(git -C "$REPO" rev-parse --verify -q '@{u}' 2>/dev/null || echo origin/main)"
FILES="$(git -C "$REPO" diff --name-only "$UP"..HEAD 2>/dev/null)"
[ -n "$FILES" ] || exit 0

if git -C "$REPO" log --format=%B "$UP"..HEAD | grep -q '\[no-docs\]'; then
  exit 0
fi

CODE="$(echo "$FILES" | grep -v -E '\.md$|^Doc/|^\.impeccable/|^\.claude/')"
DOCS="$(echo "$FILES" | grep -E '\.md$')"

if [ -n "$CODE" ] && [ -z "$DOCS" ]; then
  {
    echo "Push blocked: these commits change code but no .md file."
    echo "Update the docs first (CLAUDE.md → \"Before Every Push\"):"
    echo "  - Doc/log.md — always: Latest Summary + decision row if a choice was made"
    echo "  - CLAUDE.md — if routes, files, tables or conventions changed"
    echo "  - PRODUCT.md — if a product decision was confirmed"
    echo "  - Doc/ai_service.md — if anything under Backend/Service/ai/ or tarot changed"
    echo "Commit the docs, then push again. If no doc truly applies, add [no-docs] to a commit message."
    echo "Code files in this push:"
    echo "$CODE" | sed 's/^/  /'
  } >&2
  exit 2
fi
exit 0
