# Horisation — To-Do List

Last updated: 2026-09-20

---

## In Progress / Near-term

| Priority | Item | Notes |
|----------|------|-------|
| High | Certificate auto-renewal is broken | Expired 2026-09-11 and took the site down (526) for ~3h; renewed by hand. Find out why `certbot.timer` stopped (cause never established; port 80 does answer, with a 301 — read `journalctl -u certbot`, see `Doc/server.md`) and fix it before ~2026-12-10, or add an expiry check to `scripts/deploy.sh` so a deploy warns inside 14 days. |
| High | Password hashing (bcrypt) | Currently stored plaintext. Anyone with `market.db` — including via the admin "Download DB" button — has every user's password in the clear. Also move `SECRET_KEY` out of `app.py`. |
| High | Login 401 has no visible feedback? | During the 2026-09-07 session a user entered a wrong password 5 times and reported "nothing happens". Verify `Login.jsx` surfaces the 401; same defect class as the Market posting flow. |
| Medium | Tarot P1: history page | `GET /api/tarot/readings` already serves it; the page is not written. v1 deck + spread (09-08); v2 choose your own three (09-09); v3 tap to look closer (09-10); v4 Start-first copy, reveal-as-you-pick, Chinese names + keywords (09-12); v5 AI reading + rating (09-20). Still open: reversed cards, other spreads (Celtic Cross), Chinese rendering of Waite's long text (only keywords are bilingual). |

| Medium | Apply the Market fixes to `Tasks.jsx` | Tasks carries a byte-identical `useToast`, `.search` block, `radio-inputs` header and toast markup, and has already drifted in language and labelling. Extract a shared `ModuleShell` so they cannot drift again. |
| Medium | Register page tagline overlaps the form at ≤600px | The rule hiding `.login-tagline` is `@media (max-width:1024px) and (min-width:601px)`, so 375px falls outside it |
| Medium | Listing image re-upload in Edit | When editing a listing, allow replacing/removing images; requires R2 delete + multipart PUT |
| Low | `side-tab` detector findings | `Hormemo.jsx:181` is a true positive by rule definition (4px coloured left border on a card) though the colour encodes memo priority. `Feedback.jsx:40/344` judged false positives — blockquote left rules on reply quotes. Decide: fix, ignore, or leave. |
| Low | Read the first `ai_usage` rows | Key is on the server and real readings work (2026-09-20 evening). Worth one look at latency / tokens / cost after a few days of use, and at the first ratings. |

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
| 首页天气：华氏 + 摄氏，顺手修了单位错标 | 2026-09-20 | Open-Meteo 默认回摄氏度，后端没传 `temperature_unit`，前端却写 °F——"27°F Overcast" 其实是 27°C。现在后端两种单位都算好返回（key 里带单位），首页显示 `78°F / 26°C`。 |
| Travel Planner 对全员开放 | 2026-09-20 | 同塔罗牌的做法：删 `features.js` 的 `travelPlanner`、路由的 `FeatureRoute`、侧边栏条目的 `feature`。后端 `/api/travel/*` 本来就只查登录和所有权，没有角色门。Bill Split 仍是 vip+。 |
| My Listings 加 "Copy my page link" | 2026-09-20 | Export as image 旁边多一个按钮，复制 `/u/<username>` 公开页链接，方便发群里。 |
| 塔罗牌 v5：AI 整体解读 + 打分 | 2026-09-20 | 三张牌下方新增"整体解读"：可选写问题 → DeepSeek 按 `tarot-v1` prompt 回 JSON（过去/现在/未来/总结/明天一件小事）→ 1–5 贴合度打分。后端新建 AI 层 `Backend/Service/ai/`（统一 `ai.run`、厂商无关 `requests` 调用、按角色配额 user 1 / vip 3 / admin ∞ 每 Chicago 日、全站 100/24h、`AI_ENABLED` 总开关、失败不扣次数并自动重试一次、`ai_usage` 记账）和 `tarot_readings` 表（每次抽牌一行，解读与打分填入，将来做训练集）。14 个单测。顺手修了 `api.js` 把错误 body 的细节字段吞掉的问题。上线还差 key。 |
| 纯文档 push 不再重启生产 | 2026-09-20 | `deploy.yml` 加 `paths-ignore`（`**.md`、`Doc/**`、`.impeccable/**`）。之前每次改文档都会重启 gunicorn，撞上过一次 502。 |
| 导出长图缩略图空白 | 2026-09-20 | 两个根因叠加：R2 桶没有 CORS 策略（控制台加了）；以及 Chrome 把 Market 页面普通加载的无 CORS 头响应缓存在同一个 URL 下，导出时跨域请求撞上它被拒。后者靠代码修：导出用 `?export=1` + `crossOrigin` 单独取一份。用真 R2 图片跑真 html2canvas 验证过缩略图像素方差 81（灰块≈0）。见 `Doc/server.md`。 |
| 侧边栏改为按钮开合 | 2026-09-20 | 书脊 + 悬停 + 面包屑上线九天后被否掉（"不喜欢，路径别显示，不要悬停"）。改成 ChatGPT 式：默认收起、左上角 ☰ 打开并停靠、面板内 « 关闭、`[` 同效、记住。开合不做动画。手机端不变。 |
| 塔罗牌 v4：中文牌义 + 边抽边翻 | 2026-09-12 | 开始前的提示改为"闭上眼，默念问题三次，再点 Start"；按钮 Shuffle the deck → Start；每张牌落位即翻面、释义即时出现，不再攒到最后一起翻。78 张牌加 `name_zh` + `keywords_zh`（`scripts/tarot_add_zh.mjs`，自写，无开源中文数据集）。 |
| 侧边栏收进书脊 | 2026-09-11 | 桌面端默认只剩 14px 书脊（弧标 + 分区点 + 未读红灯），靠边或 `[` 滑出，可钉住并记住。顶栏加面包屑。站点地图抽成 `nav.js`。手机端不变。 |
| 塔罗牌对全员开放 | 2026-09-10 | 删掉 `features.js` 的 `tarot` 项、路由的 `FeatureRoute`、侧边栏条目的 `feature`。仍需登录（站点本身是邀请制）。 |
| 塔罗牌 v3：点开细看 | 2026-09-10 | 翻开的牌位变成按钮，点开后卡片从牌位原地放大（约等于扫描件原尺寸，不做无意义的超采样），指针移动时倾斜并带高光，可翻到背面，三张之间用按钮或 ←/→ 来回翻。走共享 `Modal`（Esc / focus trap / 滚动锁）。 |
| 塔罗牌 v2：自己抽牌 | 2026-09-09 | 先提示心里想一个问题 → 洗牌动画（两次交切后展开成 2/3 行）→ 鼠标划过牌抬起、邻牌让位 → 自己点三张，牌飞入牌位 → 依次翻面 → 牌堆收拢让位给释义。窄屏点空处自动取最近的牌；键盘方向键 + Enter 可走完全程。 |
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
