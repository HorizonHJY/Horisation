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

## Checklist for New Pages

- [ ] Does the page work in a mobile viewport (375×812)?
- [ ] Can the user scroll to see all content?
- [ ] Are buttons/links tappable (not hidden behind safe areas)?
- [ ] Does the layout not break between phone and desktop sizes?
- [ ] Are fonts readable at small sizes (min 14px for body text)?
