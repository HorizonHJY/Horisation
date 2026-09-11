/**
 * The site's map, in one place.
 *
 * Three things read it and must agree: the sidebar (renders it), the spine
 * (one dot per section, the current one lit), and the topbar breadcrumb
 * (names where you are once the sidebar is tucked away). Keeping it here
 * means adding a page is one edit, not three that can disagree.
 *
 * `feature` gates an item through features.js; an item without one is open
 * to every member. Sections and items with nothing visible are not rendered.
 */

export const NAV_SECTIONS = [
  {
    key: 'main', title: 'Main',
    items: [
      { to: '/home', icon: 'fa-home', label: 'Home' },
    ],
  },
  {
    key: 'community', title: 'Community',
    items: [
      { to: '/market',   icon: 'fa-store',        label: 'Market' },
      { to: '/tasks',    icon: 'fa-bullhorn',     label: 'Tasks' },
      { to: '/feedback', icon: 'fa-comments',     label: 'Message Board' },
      { to: '/friends',  icon: 'fa-user-friends', label: 'Friends' },
      { to: '/groups',   icon: 'fa-users',        label: 'Groups' },
    ],
  },
  {
    key: 'fun', title: 'For Fun',
    items: [
      { to: '/fun/gomoku-online', icon: 'fa-globe', label: 'Online Gomoku', feature: 'onlineGomoku' },
      { to: '/tarot',             icon: 'fa-moon',  label: 'Tarot' },
    ],
  },
  {
    key: 'toolkit', title: 'Toolkit',
    items: [
      { to: '/hormemo',    icon: 'fa-clipboard-list', label: 'Memo' },
      { to: '/travel',     icon: 'fa-route',          label: 'Travel Planner', feature: 'travelPlanner' },
      { to: '/bill-split', icon: 'fa-receipt',        label: 'Bill Split',     feature: 'billSplit' },
      { to: '/csv',        icon: 'fa-file-csv',       label: 'CSV Workspace',  horizonOnly: true },
    ],
  },
]

/* Pages reachable without a sidebar entry, so the breadcrumb still has a
   name for them. Order matters: longer prefixes first. */
const EXTRA = [
  { prefix: '/admin/system', section: 'Admin',   label: 'System' },
  { prefix: '/admin',        section: 'Admin',   label: 'User Management' },
  { prefix: '/profile',      section: 'Account', label: 'Profile' },
  { prefix: '/u/',           section: 'Community', label: 'Profile' },
]

/** Where a path lives: { section, label } or null. */
export function locate(pathname) {
  for (const s of NAV_SECTIONS) {
    for (const item of s.items) {
      if (pathname === item.to || pathname.startsWith(item.to + '/')) {
        return { section: s.title, sectionKey: s.key, label: item.label }
      }
    }
  }
  const extra = EXTRA.find(e => pathname.startsWith(e.prefix))
  return extra ? { section: extra.section, sectionKey: null, label: extra.label } : null
}
