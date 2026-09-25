# Mobile UX Principles for Horisation

## Why This Matters

Horisation is used on both desktop and mobile (via browser). Every UI component must work well on both.

## Key Lessons Learned

### 1. Avoid `position: fixed` + `overflow: hidden` on Mobile

**Problem:** Wrapping the page in a `fixed` container with `overflow: hidden` blocks native touch scrolling on iOS Safari. The user can't scroll past content that overflows the viewport.

**Fix:** 
- Use `position: fixed` only for **background/decorative elements** (canvas, images)
- Keep the **main content** in normal document flow so the browser handles scrolling natively
- Set `min-height: 100vh` on the content wrapper if you need it to fill the screen

### 2. Safe Area Insets (`env(safe-area-inset-*)`)

**Problem:** iPhone notch and home indicator overlap with page content (especially bottom buttons).

**Fix:** Use CSS `env()` functions to account for physical screen cutouts:
```css
padding-bottom: calc(2rem + env(safe-area-inset-bottom, 16px));
padding-top: calc(1rem + env(safe-area-inset-top, 0px));
```

### 3. `clamp()` for Responsive Sizing

Use `clamp(MIN, PREFERRED, MAX)` for font sizes, spacing, and widths instead of fixed values:
```css
font-size: clamp(0.85rem, 1.4vw, 1rem);
padding: clamp(2rem, 5vh, 4rem);
```

This eliminates the need for many media queries.

### 4. Media Queries that Actually Matter

For this project, these breakpoints cover all bases:
- **≤600px** — phones
- **601-1024px** — tablets
- **>1024px** — desktop

Mobile-specific adjustments (≤600px):
- Center-align content (`align-items: center`)
- Reduce logo/image sizes
- Hide decorative content (taglines) that takes up space

### 5. Check on Real Devices

Desktop DevTools mobile emulation is good, but not perfect. Safari on iOS handles `position: fixed`, `overflow`, and `-webkit-overflow-scrolling` differently. Always test on a real iPhone if possible.

### 6. A Control Next to a Name Must Not Be Wider Than the Name

A list row is **avatar · name · … · controls**, and `flex: 1` on the name only
works if the controls have a real width budget. A four-word chip in the Friends
rail measured **135px**; the rail is **340px** and the display name needed
**85px**, so every name longer than ~7 characters truncated — at 768px, 1280px,
even 1440px. The breakpoint was never the problem; the chip was.

Rules that follow:

- **Phone (≤767px):** a low-frequency action like "Request contact" does not
  earn space in the row. Move it into the `···` menu and leave the row as
  avatar · name · username · `···`.
- **Tablet / desktop:** if there is room, show the words; if the column is
  narrow (this rail is), drop the label to an icon and put the words in
  `title` + `aria-label`. The name is the primary content, the button is not.
- **Widen the primary surface.** Desktop is the roomiest surface, so give the
  list a little more of it (380px rail from 1280px) rather than shrinking the
  row's content.

Verify with `scrollWidth <= clientWidth` on the name element, not by eye.

### 7. Touch Targets: 44px, and Grow the Box Not the Glyph

This project's token is `--tap-min: 44px`. A `.fr-chip` rendered **26px** tall
and a `···` button **32×32px** — both under it, both caught in the same audit.

```css
@media (pointer: coarse) {
  .fr-chip:not(.fr-chip--xs) { min-height: var(--tap-min); }
  .fr-more__btn { width: auto; height: auto;
                  min-width: var(--tap-min); min-height: var(--tap-min); }
}
```

Two cautions:

- Gate it on `pointer: coarse`, **not** a width breakpoint — a touch laptop is
  wide and still needs 44px, and a mouse user does not need the bigger square.
- Grow the **hit area**, not the icon. `min-width/height` on the button keeps
  the glyph a small square; setting the icon bigger would look wrong.
- Exclude labels that merely sit inside a bigger target (`.fr-chip--xs` is the
  "Not friends" word inside a conversation button — 44px there wrecks the row).

### 8. Icon-only Needs Words Elsewhere

When a label collapses to an icon, the words must survive in `title` and
`aria-label`. On a chip the words also stay in the `···` menu, so the action is
never reachable only through a guess.

## Checklist for New Pages

- [ ] Does the page work in a mobile viewport (375×812)?
- [ ] Can the user scroll to see all content?
- [ ] Are buttons/links tappable (not hidden behind safe areas)?
- [ ] Does the layout not break between phone and desktop sizes?
- [ ] Are fonts readable at small sizes (min 14px for body text)?
- [ ] Does any display name in a list row survive without an ellipsis
      (`scrollWidth <= clientWidth`)?
- [ ] Are all interactive controls ≥ 44px on a coarse pointer?
- [ ] Did you check phone, tablet **and** desktop — with desktop the best of the
      three?
