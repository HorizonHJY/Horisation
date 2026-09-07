---
target: Market.jsx
total_score: 14
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 2
target_identity: "file:C:\\Users\\Horiz\\PycharmProjects\\Horisation\\.claude\\worktrees\\understand-project-code-44929c\\frontend\\src\\pages\\Market.jsx"
target_fingerprint: "sha256:a537492a7a63f9b4df543c63fc54c44a4c33a527f50981d0fea1b1b87a0f67a3"
target_path: "C:\\Users\\Horiz\\PycharmProjects\\Horisation\\.claude\\worktrees\\understand-project-code-44929c\\frontend\\src\\pages\\Market.jsx"
timestamp: 2026-09-07T03-37-22Z
slug: frontend-src-pages-market-jsx
---
Method: dual-agent (A: a479d61 design review · B: ab3d05c detector+browser)

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|---|---|---|
| 1 | Visibility of System Status | 2 | loadBrowse `if (listRes.ok)` has no else — a failed fetch renders the "be the first to post" empty state |
| 2 | Match System / Real World | 1 | Card badge renders raw slug ("Furniture"); filter pill renders label (家具). Same field, two names, two languages, same screen |
| 3 | User Control and Freedom | 1 | Every return to Browse wipes search + both filters; four modals with no Escape and no URL |
| 4 | Consistency and Standards | 1 | Native window.confirm for the only irreversible action; iOS Safari shows "horizonyhj.com says:" |
| 5 | Error Prevention | 1 | Create form has no `required` (edit modal does); toast-only validation; backdrop click silently discards draft |
| 6 | Recognition Rather Than Recall | 2 | Delete confirm names nothing, and the modal that launched it has already closed |
| 7 | Flexibility and Efficiency | 2 | Multi-select filters and visibilitychange refresh are good; no sort, no keyboard path, no shareable link |
| 8 | Aesthetic and Minimalist Design | 2 | Four uncoordinated badge color systems on one card, all inline-hardcoded, none from tokens |
| 9 | Error Recovery | 1 | handleExportImage swallows every failure into console.error — button spins, stops, nothing happens |
| 10 | Help and Documentation | 1 | The only guidance on the page is a Chinese parenthetical inside an English label |

Total: 14/40 (all ten heuristics applicable — Operate surface with form, destructive actions, money decision).

Three root causes account for roughly half the deductions: (a) inline hardcoded styles bypassing the token system, (b) view state in component state instead of the URL, (c) an unmanaged English/Chinese split.

## Design Specificity Verdict

Category-interchangeable. The skeleton is the generic resale template and nothing in it knows it serves ~11 friends in St. Louis who already have each other's phone numbers.

Two sharpest pieces of evidence, both against the product's own claimed positioning:

1. Tapping a seller destroys their identity. openSellerModal (L669-680) discards seller_avatar/seller_display that the listing already carries, rebuilds from friendsMap, and falls back to {display_name: username, avatar_url: null} for non-friends. Violates PRODUCT.md Principle 2 ("One identity, everywhere") in the exact interaction the positioning depends on.
2. No listing has a URL. detailListing is component state (L592). A product whose thesis is "one shared friend graph, one click apart" cannot send a friend a link to an item. Meanwhile 110 lines (L913-1041) implement export-to-PNG for WeChat Moments — an outward growth mechanic Principle 4 disavows. The investment is inverted.

Also: the export image is headed "Arch Bay" (L943), the retired brand, on the one artifact that leaves the product. index.css ships .page-title/.page-subtitle (Playfair + Noto Serif SC) and zero pages use them.

### Deterministic scan

detect --json frontend/src/pages/Market.jsx -> [] exit 0 (Market.jsx itself: ZERO rule hits)
detect --json frontend/src -> 8 findings, exit 2, all severity=warning category=slop

| Rule | Count | Location |
|---|---|---|
| bounce-easing | 5 | index.css L879/895/905/934/963 — same cubic-bezier(.65,1.35,.5,1) |
| side-tab | 3 | Feedback.jsx L40/L344, Hormemo.jsx L181 |

False positives: Feedback.jsx L40/L344 are blockquote left-rules on reply-quote blocks, not card accents — rule-intent mismatch. The 5 bounce-easing hits are one decision (a single skeuomorphic toggle component), not five defects; overshoot on a snapping physical lever is defensible. Hormemo.jsx:181 is a true positive by rule definition but the color is semantic (memo priority), not decorative.

The disagreement is itself the finding: the 61 rules target visual slop (gradients, bounce, card-in-card, decorative accents). Market's problems are a11y, state management, identity continuity, and bilingual governance — which deterministic rules cannot see. Market.jsx scoring [] is not cleanliness; it is a ruler that does not measure this.

### Visual overlays

NOT OBTAINED. Assessment B successfully started Flask + Vite and rendered the SPA, but /market sits behind auth and the sub-agent correctly declined to enter a password (even the local dev seed account). No screenshots, console output, or overlay evidence exists for the Market page; every judgment above is source-derived. Nothing was fabricated.

Unauthenticated findings (not Market defects): FlowerCanvas.jsx:78-80 throws InvalidStateError "drawImage ... width or height of 0" on every load of Login/Home/Register (prevW/prevH still 0 when ResizeObserver fires). PrivateRoute passes state={{from: location}} but Login.jsx:22 hardcodes navigate('/home') and never reads it — deep-link destination silently discarded.

## Overall Impression

Not a badly built page — a page built twice and finished neither time. Real craft underneath (.market-card is the repo's only proper design-system component; visibilitychange refresh and sessionStorage view dedup are phone-aware engineering; Set-based multi-select is the right model), covered by 1,377 lines of inline hardcoding, four badge palettes, an export feature using the wrong brand name, and an unmanaged bilingual layer.

Biggest single opportunity: make a listing a first-class addressable object (/market/l/:id). That one move fixes "can't send it to a friend," "lose your place on return," "delete confirm names nothing," and "no idea which item you were on after the seller modal," and lets the Message button carry a real link instead of a Chinese sentence.

## What's Working

1. .market-card is genuine design-system work (index.css L584-617): scopes local vars (--mc-font/--mc-muted/--mc-border/--mc-bg) mapped to global tokens, so the dark-theme override (L994-999) only remaps four locals — no selector duplication, no !important. Equal-height cards with bottom-aligned actions for free.
2. The visibilitychange refresh (L624-632) is correct phone-first thinking for the recorded "short opportunistic sessions," with tabRef avoiding a stale-closure re-subscribe. Paired with the sessionStorage viewed_<id> guard (L783-792) that stops self-inflated view counts.
3. Set-based multi-select filters with size===0 meaning "all" (L577-578, L859-860) — no artificial "All" pill to sync, no invalid states. And the filtered-empty state (L1181-1187, offers Clear) is correctly distinct from the genuinely-empty state.

## Priority Issues

### [P0] Grid unreachable by keyboard; page fails its own stated AA floor

Every path into a listing is a <div onClick>: market-card__img (L236), __title (L244), __seller (L290), detail-modal seller row (L442), SellerAvatar fallback (L104-111). None focusable, no role/tabIndex, no Enter/Space. A keyboard or screen-reader user cannot open any listing — but can Tab to Delete on their own.

Contrast (independently recomputed, all fail AA 4.5:1):
card description #8a8a8a/#fff = 3.45 · view count #aaa/#fff = 2.32 · struck price #999/#fff = 2.85 · 自提 badge #888/#f0f0f0 = 3.11 · 包邮 badge #27ae60/#e8f8f0 = 2.62 · avatar initials #fff/#6b9cdb = 2.84 · export price #e74c3c/#fff = 3.82
Plus 0.62rem (~9.9px) delivery badges and view counts against the project's own recorded 14px floor.

PRODUCT.md sets AA as the working floor and records that the accent was chosen to hold it; the page hardcodes seven greys bypassing tokens. #6b9cdb at 2.84:1 carries the identity mark.

Fix: (a) one focusable <button>/<a> per card wrapping image+title+seller, :focus-visible ring on --accent. (b) Delete hardcoded greys; descriptions to --text-secondary (#555, 7.46:1); darken --text-muted to ~#6e6e6e (4.6:1). (c) Raise .62rem badges to .72rem min. (d) Avatar fallback to --accent-hover (#5286c7).
Command: /impeccable audit -> /impeccable polish

### [P1] Tapping a seller destroys their identity; seller modal is full of dead controls

openSellerModal (L669-680) rebuilds the seller from friendsMap and drops avatar/display name for non-friends. SellerModal renders with currentUser={null} and all callbacks stubbed to () => {} (L553-558) — every green "Message" button is inert, images and titles are cursor:pointer and inert. Loading passes listings={[]} (L1057), so "No active listings" flashes before items arrive.

Emotional sequence at the trust-building moment: identity degrades -> "this person has nothing" -> a grid of dead controls.

Fix: (a) friendsMap[username] ?? {username, display_name: listing.seller_display || username, avatar_url: listing.seller_avatar}; better, fetch the canonical profile so Market/Friends//u/:username share one source. (b) Pass real currentUser/onDetail/onReachOut, or render a read-only card variant with no buttons — never inert buttons. (c) Render HandLoader while sellerLoading, never the empty state.
Command: /impeccable harden

### [P1] Posting: errors invisible on phone, success hides the item you just posted

Errors: handleCreate (L710-713) reports every validation failure via a toast pinned top-0 end-0 (L907) — off-screen when submit is at the bottom of a scrolled mobile form, auto-dismissing in 2.8s. No required attributes (the edit modal has them), no inline errors, no scroll-to-error, no role="alert", no env(safe-area-inset-top) so it sits under the notch — the exact defect class Doc/mobile_ux_principles.md section 2 warns about.
Success: setTab('browse') (L729), where browseListing (L853) filters out your own listings. The new item is invisible; a sole seller lands on "No listings yet. Be the first to post!"

Fix: (a) required + aria-invalid + inline field message + scrollIntoView on first invalid field. (b) role="alert" and top: calc(0.75rem + env(safe-area-inset-top, 0px)); bottom-center on <=600px. (c) On success setTab('mylistings') and briefly highlight the new card. (d) Price inputs to col-12 col-sm-6 — they currently do not stack at 375px.
Command: /impeccable harden -> /impeccable adapt

### [P2] All view state discarded on every navigation; no listing has a URL

L618-621 resets searchQuery + both filters on every entry to Browse, including returning from My Listings. detailListing/sellerModal are component state. handleReachOut (L826) navigates fully out to /friends. Four modals with no Escape, no focus trap, no role="dialog", no body scroll-lock — on iOS Safari the background scrolls behind them.

PRODUCT.md puts the design weight here explicitly: "a visitor should never lose their place... when moving between Market, Friends, Groups." This page loses it at every boundary, and the missing listing URL blocks pasting an item into the group chat.

Fix: (a) Route detail and seller as /market/l/:id and /market/u/:username, rendered as modals over the grid. (b) Mirror filters into the query string; stop the blanket reset. (c) Shared useModal hook (Escape, focus trap, aria-modal, scroll-lock) for all four modals and Tasks.
Command: /impeccable shape

### [P2] Unmanaged bilingual split, stale brand, flagship export broken on the primary platform

~2,168 Chinese characters carry load-bearing meaning: category labels (衣服/家具/厨具/美妆, with Electronics in English, L12), delivery labels, price badges (包邮, 含$X配送费), filter axis labels (分类/配送), the photo constraint spec (L1321-1323), the injected chat message (L823), and the entire export modal (L920-1034). Home.jsx by contrast has one Chinese subtitle — the pattern PRODUCT.md describes as correct.

Export modal additionally: headed "Arch Bay" (L943); fixed width:420 preview inside a ~343px modal-dialog at 375px, overflowing; driven by link.click() on a data: URL (L841-844) which does not download on iOS Safari, with console.error as the only failure handling (L846).

PRODUCT.md: "Chinese that carries meaning the English does not is a defect."

Fix: (a) Bilingual category/delivery pairs {slug, label_en, label_zh}, English primary with Chinese accent; fix Electronics -> 电子产品 in the fallback list. (b) Card badge renders categories.find(c => c.slug === l.category)?.label (L250, L403) so card and pill say the same word — export code L1000 already does this correctly. (c) Export modal to English with Chinese subtitle; Arch Bay -> Horisation + arch mark. (d) On iOS surface the canvas as an <img> with press-and-hold to save; toast on failure. (e) Raise with the owner whether an outward-broadcast export belongs in a product whose principle is "Private, not public."
Command: /impeccable clarify

## Persona Red Flags

Keyboard / screen-reader user: search__button has onClick={() => {}} (L1103) — a focusable no-op. After 8 filter pills, the grid contains nothing tabbable. They can delete their own listing but cannot open anyone else's. No focus trap in modals; toasts have no role="alert" so "Listing posted!" and "Title is required." are both silent. L1340 img has alt=""; L385 thumbnails have no alt.

Motor-impaired / large-thumb phone user: .market-card__btn is height:30px (index.css L672), 14px under the iOS 44px minimum. At 375px each card is ~127px of content width; the owner action row (Mark Sold / Edit / Delete, white-space:nowrap) wraps into three stacked 30px bars with overflowing text. Delete sits directly below Mark Sold, distinguished only by color. A misfire hits a native window.confirm that does not name the item; a second misfire destroys the listing and its R2 images. The remove-photo x (L1345-1352) is ~18x16px on a 90px thumbnail corner.

Chinese-speaking friend on iPhone selling a desk before moving out (the core user): price inputs do not stack at 375px (~150px each, labels wrapping to three lines). Taps Post with price blank — toast fires off-screen above a form scrolled to the bottom, 2.8s. Nothing appears to happen. Taps again. Lands on Browse after success and her desk is not there. Goes to My Listings, taps the long-image export — the 420px preview clips inside the ~343px modal. Taps the image export — spinner runs, stops, nothing downloads, no error. Both things she opened the app to do have failed.

English-only friend in the same circle (interface language is English per PRODUCT.md): the filter row is labeled 分类 and 配送 with all-Chinese pills except one "Electronics", while the cards say "Furniture" and "Clothing" — she cannot connect any pill to any card. Every card carries 仅自提 or 可自提或配送; she cannot tell which need a car. She taps Message, is thrown to /friends, and finds her own input pre-filled with a Chinese sentence and emoji she did not write and cannot read, one tap from being sent under her name.

## Minor Observations

- reachOutStatus is dead end-to-end: computed L901/L1060/L1213, threaded into three components, read by none. Two extra API calls per browse load (/api/friends/list, /api/friends/requests/sent) build a map with no consumer. Fossil of a removed friend-gated messaging flow.
- Prices are never formatted — no toFixed(2) anywhere; price + delivery_fee on Float columns can produce $25.000000000000004.
- ListingDetailModal hardcodes a different visual language: border 2px solid #323232, boxShadow 6px 6px #323232 (L359) — neo-brutalist inside a soft-cream --radius-lg / --shadow-sm system.
- alert-dismissible with no dismiss button (L907), copied verbatim into Tasks.jsx L622.
- useToast leaks and races: setTimeout never cleared on unmount (L29); two toasts in quick succession means the first timer clears the second.
- market-card__img:hover translateY(-3px) (index.css L616) lifts the image well out of the card leaving a 3px gap; :hover sticks after tap on touch.
- aspect-ratio 3/4 + object-fit contain letterboxes nearly every real photo against cream.
- Dark mode breaks on the hardcoded literals (#f0f0f0, #e8f0fe, #e8f8f0, #999, #888) against --bg-surface #1e293b. Gated to horizon today — who is the person reviewing this.
- html2canvas is a full import on every Market mount (L6) for a feature most users never open; should be React.lazy / dynamic import().
- Five pages, five container conventions: Market container-fluid py-4 inside .page-content padding 32px 36px; Friends py-3; Feedback py-4 + maxWidth 720; TravelPlanner maxWidth 960.
- hasOriginal computed and unused (L231, L352); .market-card__footer (index.css L658-660) defined and unreferenced.

## Questions to Consider

1. If the whole point is one shared friend graph, why can't I send a friend a link to a listing? The page pre-writes a Chinese sentence containing an item's title because it has no way to reference the item itself.
2. Eleven friends do not need a marketplace. Do they need a marketplace, or a shelf? Search + 8 multi-select filters + sold/restore lifecycle + view counts is the IA of a catalog with thousands of items. What is the real N?
3. browseListing hides your own listings — design decision or accident? It creates the worst moment in the flow and makes the empty state lie.
4. What is the delivery-fee system actually for, between friends in one city? Three modes, a fee field, four price-rendering branches, dual stacked prices, five badge variants — the largest single source of card noise.
5. The export-to-PNG feature is the most designed thing on this page and it points outward, to WeChat Moments, against a stated "Private, not public" principle. Is it a leftover, or is the product record out of date? One of the two documents is wrong.
6. Market.jsx and Tasks.jsx contain a byte-identical useToast, .search block, radio-inputs header, category-pill strip and toast markup — and have already drifted in language and labeling. What is the smallest shared ModuleShell that would make these modules structurally incapable of drifting?

## Out of design scope but blocking

app.py is truncated in the committed blob (HEAD 4776 bytes), ending mid-comment at "# -- Dev entry point ----" plus a U+FFFD replacement character, with no `if __name__ == '__main__':` block (grep count 0). python app.py imports everything, seeds the DB, and exits 0 without binding a port. scripts/dev.bat and _flask_local.bat cannot work on this branch (they also hardcode a no-longer-existing D:\Anaconda\envs\Horisation). This is the failure mode recorded as Pattern 5 in Doc/log.md.
