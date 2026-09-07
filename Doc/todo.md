# Horisation — To-Do List

Last updated: 2026-09-07

---

## In Progress / Near-term

| Priority | Item | Notes |
|----------|------|-------|
| High | Password hashing (bcrypt) | Currently stored plaintext. Anyone with `market.db` — including via the admin "Download DB" button — has every user's password in the clear. Also move `SECRET_KEY` out of `app.py`. |
| High | Login 401 has no visible feedback? | During the 2026-09-07 session a user entered a wrong password 5 times and reported "nothing happens". Verify `Login.jsx` surfaces the 401; same defect class as the Market posting flow. |
| Medium | Apply the Market fixes to `Tasks.jsx` | Tasks carries a byte-identical `useToast`, `.search` block, `radio-inputs` header and toast markup, and has already drifted in language and labelling. Extract a shared `ModuleShell` so they cannot drift again. |
| Medium | Register page tagline overlaps the form at ≤600px | The rule hiding `.login-tagline` is `@media (max-width:1024px) and (min-width:601px)`, so 375px falls outside it |
| Medium | Listing image re-upload in Edit | When editing a listing, allow replacing/removing images; requires R2 delete + multipart PUT |
| Low | `side-tab` detector findings | `Hormemo.jsx:181` is a true positive by rule definition (4px coloured left border on a card) though the colour encodes memo priority. `Feedback.jsx:40/344` judged false positives — blockquote left rules on reply quotes. Decide: fix, ignore, or leave. |

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
| Market 设计审查整改 | 2026-09-07 | `/impeccable critique` 14/40 → 5 条 Priority Issues 全部整改。键盘可达、AA 对比度、可寻址 listing 路由、共享 Modal、分类双语。见 `Doc/log.md` |
| `app.py` 提交态截断修复 | 2026-09-07 | 自 `0e91080` 起缺 `__main__` 块，本地 dev 完全跑不起来 |
| 本地/生产环境标识 | 2026-09-07 | 非生产实例显示 LOCAL 带子 + 标题前缀，见 Pattern 12 |
| CI/CD (GitHub Actions → EC2) | — | `.github/workflows/deploy.yml`，push 到 main 自动部署 |

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
| Full bilingual UI (every string in two languages) | Maintenance overhead too high |

> **Superseded 2026-09-07.** The old entry read "Chinese / bilingual UI — all users read English
> fine", but the code had drifted the other way: `Market.jsx` alone carried ~2,168 Chinese
> characters holding load-bearing meaning (category labels, delivery labels, the photo size
> limit, the pre-filled chat message, the whole export dialog), which an English-reading member
> of the circle could not use. The rule confirmed with the owner and recorded in `PRODUCT.md` is
> **English is the interface language; Chinese appears only as a deliberate accent** — a subtitle
> or a smaller companion to an English label, never the only carrier of meaning. Full duplicate
> translation of every string is still not wanted; that is what this row now means.

> **Listing image re-upload** was listed here *and* under In Progress. It stays in In Progress:
> the workaround (delete + re-post) also destroys the listing's view count and its URL, which
> now matters because listings are shareable.
