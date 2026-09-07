# Horisation — To-Do List

Last updated: 2026-09-06

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
| Avalon board game | Social deduction game |
| More games | Expand "For Fun" section |
| Data visualisation tools | Charts/graphs in CSV Workspace |
| Push notifications | Browser push for new messages / friend requests |

## Recently Done

| Item | Date | Notes |
|------|------|-------|
| Group messaging（群组） | 2026-08-22 | 独立建组+按用户名拉人+群聊, `/api/groups`, 见 `Doc/groups.md` |
| Brand rename → Arch Bay | 2026-09-06 | 可见文案 'Horisation'→'Arch Bay'，提交 f0fc7f6；仓库目录未改名 |
| Marketplace 意向成单流 | 2026-09-06 | trade_intents + listing reserved/sold + 两段式成交（待 A/B 实测 & merge main） |

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
