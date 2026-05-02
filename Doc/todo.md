# Horisation — To-Do List

Last updated: 2026-05-02

---

## In Progress / Near-term

| Priority | Item | Notes |
|----------|------|-------|
| High | Password hashing (bcrypt) | Currently stored plaintext — security risk |
| Medium | Listing image re-upload in Edit | When editing a listing, allow replacing/removing images; requires R2 delete + multipart PUT |
| Medium | Market listing search / filter | Filter by category, price range, keyword |

---

## Feature Ideas

| Item | Notes |
|------|-------|
| Group messaging | Group chat rooms for multiple friends |
| Avalon board game | Social deduction game |
| More games | Expand "For Fun" section |
| Data visualisation tools | Charts/graphs in CSV Workspace |
| CI/CD pipeline | GitHub Actions → auto-deploy to EC2 |
| Push notifications | Browser push for new messages / friend requests |

---

## Technical Debt

| Item | Notes |
|------|-------|
| SQLite → PostgreSQL | Better concurrent write safety; low priority for current scale |
| Thread-safety audit | SQLite is fine for now, but review under higher load |

---

## Decided Against (for now)

| Item | Reason |
|------|--------|
| Chinese / bilingual UI | Maintenance overhead too high; all users read English fine |
| Listing image re-upload | Low frequency use case; delete + re-post is simpler workaround |
