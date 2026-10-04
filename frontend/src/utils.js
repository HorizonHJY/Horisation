/**
 * Returns an error string if the password fails requirements, or null if valid.
 * Requirements: 8+ chars, uppercase, lowercase, digit.
 */
export function validatePassword(pw) {
  if (pw.length < 8)        return 'Password must be at least 8 characters.'
  if (!/[A-Z]/.test(pw))    return 'Must contain at least one uppercase letter.'
  if (!/[a-z]/.test(pw))    return 'Must contain at least one lowercase letter.'
  if (!/\d/.test(pw))       return 'Must contain at least one number.'
  return null
}

// ── Shared date/time formatting ────────────────────────────────────────────
// The app is pinned to the circle's timezone and locale. Using explicit
// Intl.* calls (rather than bare toLocale*String) keeps format and timezone
// consistent across every page instead of drifting per call site.
const CT_TZ   = 'America/Chicago'
export const APP_TZ = CT_TZ

const DATE_MED = new Intl.DateTimeFormat('en-US', { timeZone: CT_TZ, month: 'short', day: 'numeric' })
const DATE_YMD = new Intl.DateTimeFormat('en-US', { timeZone: CT_TZ, year: 'numeric', month: '2-digit', day: '2-digit' })
const DATE_JOIN = new Intl.DateTimeFormat('en-US', { timeZone: CT_TZ, month: 'long', year: 'numeric' })
const DATE_LONG = new Intl.DateTimeFormat('en-US', { timeZone: CT_TZ, year: 'numeric', month: 'long', day: 'numeric' })
const TIME_HM   = new Intl.DateTimeFormat('en-GB', { timeZone: CT_TZ, hour: '2-digit', minute: '2-digit', hour12: false })

export const fmtDateMed  = (iso) => DATE_MED.format(new Date(iso))
export const fmtDateYmd  = (iso) => DATE_YMD.format(new Date(iso))
export const fmtDateJoin = (iso) => DATE_JOIN.format(new Date(iso))
export const fmtDateLong = (iso) => DATE_LONG.format(new Date(iso))
export const fmtTimeHM   = (iso) => TIME_HM.format(new Date(iso))
