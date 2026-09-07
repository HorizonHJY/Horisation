# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are a small, invitation-only circle of close friends around St. Louis. They arrive on
phones as often as on desktop, in short opportunistic sessions rather than long work blocks — checking
what friends posted, answering a message, putting up something to sell, settling a shared bill.

The circle is community-first: people come to be connected to each other, not to operate a tool.
Tools exist because they give the group something concrete to do together. Success for this product
is measured by liveliness — whether the circle keeps showing up, posting, and replying.

Roles are real and enforced, not decorative: `horizon` (owner, level 100), `admin` (90), `svip` (70),
`vip` (60), `user` (10). Higher roles unlock whole features, so the same page can differ per visitor.

## Product Purpose

Arch Bay gives one private group of friends a single place to do everything they already do
together — trade second-hand goods, split bills, plan trips, post bounties, chat one-to-one and in
groups, leave messages on a shared board, and keep personal notes.

## Positioning

The unreplaceable property is **one place for all of it, over one shared friend graph**. The
alternative is a WeChat group plus a resale app plus a document tool plus a bill splitter — four
products that do not know about each other and do not share an identity. Here, the person selling a
couch, the person owed $12 from dinner, and the person in the group chat are the same account, with
the same avatar and the same profile, one click apart.

Design weight therefore belongs on **continuity across modules and on navigation efficiency**: a
visitor should never lose their place or their sense of who they are dealing with when moving between
Market, Friends, Groups, Tasks, Bill Split, and Travel.

## Operating Context

- Access is invitation-only. Self-registration exists but requires an invite code issued by the owner.
- Used on both desktop and mobile browsers. iOS Safari is a first-class target and has historically
  been the source of real layout defects (see `Doc/mobile_ux_principles.md`).
- Breakpoints in active use: ≤600px phones, 601–1024px tablets, >1024px desktop.
- Sidebar (240px) plus topbar (60px) on desktop; the sidebar collapses to a hamburger drawer on mobile.
- Real-time surfaces (private chat, online Gomoku) run over Socket.IO; group chat currently polls at 3s.

## Capabilities and Constraints

Shipped surfaces: Home (weather greeting + quick access), Market (listings with images, categories,
delivery options, view counts), Tasks (bounty board), Message Board (threaded replies + likes),
Friends (requests, private chat, separately-approved contact sharing), Groups (independent of the
friend graph), Online Gomoku, Memo, Travel Planner, Bill Split, CSV Workspace, Profile, Admin, and
System Management.

Technical constraints future work must respect:

- Flask is strictly API-only under `/api/*`; React owns all UI and routing.
- Bootstrap 5, Font Awesome 6, and Google Fonts load from CDNs in `frontend/index.html`.
- Images live in Cloudflare R2; the database stores only public URLs.
- All structured data is in one SQLite file (`_data/market.db`) via SQLAlchemy.
- Feature visibility is gated twice: backend role permissions and the frontend `FEATURES` map in
  `frontend/src/features.js`.

Explicitly undecided: dark mode exists but is gated to `horizon` only, so it is not yet a committed
product surface for anyone else.

Confirmed 2026-09-07: the marketplace uses a two-step trade flow rather than a bare
active/sold toggle. A buyer expresses interest, the seller accepts (the listing becomes
reserved and every other pending interest is auto-declined), and the buyer confirms receipt
to complete the sale. Its interface strings follow the language rule below: English primary,
Chinese accent.

Confirmed 2026-09-07: exporting one's own listings as a shareable image, for posting outside the
circle (e.g. WeChat Moments), is a real product need and not a leftover. It must work on iOS Safari,
which is where it is used most.

## Brand Commitments

- **Name in the interface: Arch Bay.** Settled 2026-09-07. The name has flipped twice — the
  repository, the domain (horizonyhj.com) and this document's earlier drafts all said
  "Horisation", and commit `0e195b1` renamed Arch Bay → Horisation before `f0fc7f6` renamed it
  back. The visible product is **Arch Bay**; "Horisation" survives only as the repository and
  host name, which are not user-facing and are not worth churning. Anything a member reads
  says Arch Bay. The CSS class carrying the wordmark is deliberately named `brand-wordmark`
  rather than after either name, so the next rename does not have to touch the stylesheet.
- **Mark:** an arch, drawn as inline SVG (a nod to the St. Louis Gateway Arch).
- **Tagline in use:** "St. Louis private harbor."
- **Type:** Playfair Display and Cinzel for display, Inter for body, Noto Serif SC for Chinese, all
  already committed in `index.html` and tokenized in `index.css`.
- **Language:** English is the interface language. Chinese appears only as deliberate accent — subtitles
  and atmosphere, as on the Home page — never as the primary carrier of meaning. Chinese that carries
  meaning the English does not is a defect.

## Evidence on Hand

- Live production deployment at https://horizonyhj.com.
- Real user accounts, listings, messages, and chat history in production; none of it is in git.
- `Doc/` carries genuine project history: `log.md` (dated decisions and root-caused bug patterns),
  `data_storage.md`, `project_intro.md`, `server.md`, `groups.md`, `mobile_ux_principles.md`, `todo.md`.
- Several `Doc/` files have drifted behind the code (stale role names, missing modules, "no CI/CD"
  when GitHub Actions is live). Treat code as truth where the two disagree.
- No testimonials, metrics, press, pricing, or customer evidence exist. Do not fabricate any.

## Product Principles

1. **The circle is the product.** Anything that makes the group feel more present to each other
   outranks anything that makes a single tool more capable.
2. **One identity, everywhere.** A person's account, avatar, and profile must read the same in Market,
   Friends, Groups, Tasks, and Bill Split. Divergence between modules is a defect.
3. **Phone-first reality, desktop-first layout.** The layout is built for a sidebar, but the traffic
   arrives on phones. Every surface must survive 375px and iOS Safari.
4. **Closed membership, permeable reach.** Joining stays invitation-only, and the design assumes a
   level of trust no public marketplace can — no engagement bait, no stranger-defense patterns.
   Reach, however, is deliberately not sealed: a member may carry a listing outward as a shareable
   image to find a buyer beyond the circle. Outward sharing of one's own content is a confirmed
   product need; inward growth mechanics are still refused.
5. **English interface, Chinese warmth.** Accent, never obligation.

## Accessibility & Inclusion

No formal standard has been adopted. One accessibility commitment is already recorded in project
history: the accent color was chosen to hold WCAG AA (4.5:1) contrast. Treat AA as the working floor
until the owner says otherwise.
