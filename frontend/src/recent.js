import { NAV_SECTIONS } from './nav'

/* Recently visited — the sidebar's short memory of where you have been.

   Pure localStorage, no server call: the list is a convenience for this
   browser, not a fact about the account. Three entries, most recent first,
   deduped by path, and never a page the current viewer cannot open (a guest
   who signed out must not keep seeing Memo).

   The nav labels are the source of truth for what a path is *called*, so a
   route is stored as a path and rendered by looking its label up here; a path
   we no longer recognise is simply dropped rather than shown as a raw URL. */

export const RECENT_KEY = 'archbay.recent'
const MAX = 3

// path -> label, from the one site map. Home is the one page in NAV_SECTIONS
// with no explicit `to` in older data, so guard for it.
const LABELS = (() => {
  const map = {}
  for (const section of NAV_SECTIONS) {
    for (const item of section.items) {
      if (item.to) map[item.to] = item.label
    }
  }
  return map
})()

/* Paths a viewer may legitimately be shown. The sidebar already filters items
   by role; this keeps the memory from resurrecting an entry that a guest or a
   downgraded role can no longer reach. */
export function labelFor(path) {
  return LABELS[path] || null
}

export function readRecent() {
  try {
    const raw = JSON.parse(localStorage.getItem(RECENT_KEY) || '[]')
    if (!Array.isArray(raw)) return []
    return raw.filter(p => typeof p === 'string' && labelFor(p))
  } catch {
    return []
  }
}

export function recordVisit(path) {
  const label = labelFor(path)
  if (!label) return
  try {
    const next = [path, ...readRecent().filter(p => p !== path)].slice(0, MAX)
    localStorage.setItem(RECENT_KEY, JSON.stringify(next))
    // Let the sidebar (already mounted) pick the change up without a reload.
    window.dispatchEvent(new CustomEvent('archbay:recent'))
  } catch { /* private mode: the memory just does not persist */ }
}
