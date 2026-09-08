# Horisation — To-Do List

Last updated: 2026-09-08

---

## In Progress / Near-term

| Priority | Item | Notes |
|----------|------|-------|
| **Next** | Tarot v2 | v1 shipped 2026-09-08 (see below). Open for a later pass: reversed cards, a saved reading history, other spreads (Celtic Cross), and card names in Chinese — `tarot_deck.json` has no `name_zh`, so positions are bilingual but card names are English only. |
| High | Password hashing (bcrypt) | Currently stored plaintext. Anyone with `market.db` — including via the admin "Download DB" button — has every user's password in the clear. Also move `SECRET_KEY` out of `app.py`. |
| High | Login 401 has no visible feedback? | During the 2026-09-07 session a user entered a wrong password 5 times and reported "nothing happens". Verify `Login.jsx` surfaces the 401; same defect class as the Market posting flow. |
| Medium | Apply the Market fixes to `Tasks.jsx` | Tasks carries a byte-identical `useToast`, `.search` block, `radio-inputs` header and toast markup, and has already drifted in language and labelling. Extract a shared `ModuleShell` so they cannot drift again. |
| Medium | Register page tagline overlaps the form at ≤600px | The rule hiding `.login-tagline` is `@media (max-width:1024px) and (min-width:601px)`, so 375px falls outside it |
| Medium | Listing image re-upload in Edit | When editing a listing, allow replacing/removing images; requires R2 delete + multipart PUT |
| Low | `side-tab` detector findings | `Hormemo.jsx:181` is a true positive by rule definition (4px coloured left border on a card) though the colour encodes memo priority. `Feedback.jsx:40/344` judged false positives — blockquote left rules on reply quotes. Decide: fix, ignore, or leave. |

---

## Tarot section — what shipped (v1, 2026-09-08)

Every decision the sketch below left open, and how it went:

| Open question | Settled as |
|---|---|
| Where the art lives | **Bundled** in `frontend/public/tarot/` — 78 JPEGs, ~7.6 MB, from `metabismuth/tarot-json` (MIT). R2 was rejected: a manual upload step in deploy is a worse tax than 7.6 MB in a repo nobody clones twice. |
| Card text | Waite's *Pictorial Key to the Tarot* (1911), public domain, taken from `ekelen/tarot-api` and merged onto the images by card name. Two names needed aliases to match: `Strength`→`Fortitude`, `Judgement`→`Last Judgment`. 78/78 matched. |
| Reversed cards | **No** — owner's call. `tarot_deck.json` carries upright text only. |
| Fan geometry | Rotating 78 cards about one distant origin makes a *wheel* that swallows the spread. Rewritten as horizontal spread + a parabola for the lift + a small tilt (`fanStyle` in `Tarot.jsx`). |
| Shuffle | Server-side, `secrets.randbelow`, and the response carries only the three drawn cards. |

Verified: 78/78 deck rows point at a real image file; draws never repeat a card
within a spread; every card can come up; the distribution passes chi-square
(χ²=73.0, 77 df, well under the 124.8 threshold) over 1800 drawn slots.

## Tarot section — design sketch (agreed 2026-09-07, superseded by the table above)

**Scope.** One route, `/tarot`, gated to `horizon` in `features.js` the same way
`onlineGomoku` and `travelPlanner` already are. Classic three-card spread only:
past, present, future. No account history, no sharing, no persistence in v1.

**The deck.** 78 cards: 22 Major Arcana plus 56 Minor (four suits x ace–ten,
page, knight, queen, king). Card art is the Rider–Waite–Smith deck, which is
**public domain in the US** (published 1909, US copyright expired). Getting the
images right is most of the work — 78 files, and they must be served from
somewhere the app already trusts. Two options, decide before starting:
- Bundle them in `frontend/public/tarot/` — simple, versioned with the code,
  adds a few MB to the repo.
- Upload to the existing Cloudflare R2 bucket — keeps the repo small, matches
  how listing images already work, but adds a manual upload step to deploy.

**The animation the owner wants.** A fan of face-down cards arcing across the
screen, then three slots that fill and flip. Notes:
- 78 individually animated DOM nodes is a lot. Position with one CSS transform
  per card (rotate + translate off a shared origin) and animate `transform` and
  `opacity` only, so it stays on the compositor.
- The flip is a `rotateY` on a container with two absolutely-positioned faces
  and `backface-visibility: hidden`.
- `prefers-reduced-motion` must land the cards in place without the arc — the
  global rule in `index.css` shortens durations, which is not enough on its own
  for an animation that *is* the interaction.
- Preload only the three drawn faces; the other 75 backs are one shared image.

**Shuffle must be server-side.** `Math.random()` in the browser is fine for a
toy, but the draw is the whole point, so put it behind an endpoint
(`POST /api/tarot/draw` → three cards plus upright/reversed) so the result is
not inspectable or re-rollable from devtools. Reversed cards are half the
readings; decide whether v1 includes them.

**Interface language.** English primary with a Chinese accent, per PRODUCT.md —
card names read `The Star 星星`, positions read `Past 过去`.

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
| 塔罗牌 section v1 | 2026-09-08 | `/tarot`，仅 `horizon` 可见。78 张 RWS 牌扇形展开 → 三张牌阵翻牌 → 正位释义。服务端 `secrets` 洗牌。无逆位、无历史记录。 |
| Group messaging（群组） | 2026-08-22 | 独立建组+按用户名拉人+群聊, `/api/groups`, 见 `Doc/groups.md` |
| Market 设计审查整改 | 2026-09-07 | `/impeccable critique` 14/40 → 5 条 Priority Issues 全部整改。键盘可达、AA 对比度、可寻址 listing 路由、共享 Modal、分类双语。见 `Doc/log.md` |
| `app.py` 提交态截断修复 | 2026-09-07 | 自 `0e91080` 起缺 `__main__` 块，本地 dev 完全跑不起来 |
| 本地/生产环境标识 | 2026-09-07 | 非生产实例显示 LOCAL 带子 + 标题前缀，见 Pattern 12 |
| 全局实时通知 | 2026-09-07 | 一条 session 级 socket + `/api/friends/notifications` 快照；聊天窗口内可直接回复联系方式请求；红点含待处理请求 |
| 单一 connect 处理器 | 2026-09-07 | 修好 game_controller 被覆盖导致联机五子棋一直拿不到用户的问题，见 Pattern 13 |
| CI/CD (GitHub Actions → EC2) | — | `.github/workflows/deploy.yml`，push 到 main 自动部署 |
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
