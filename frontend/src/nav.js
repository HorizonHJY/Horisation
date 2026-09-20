/**
 * The site's map, in one place.
 *
 * The sidebar renders it. It lives here rather than inside Sidebar.jsx so
 * that anything else that ever needs the map reads the same one.
 *
 * `feature` gates an item through features.js; an item without one is open
 * to every member. `horizonOnly` is the one entry that is not a feature flag
 * but a role check. Sections with nothing visible are not rendered.
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
      { to: '/travel',     icon: 'fa-route',          label: 'Travel Planner' },
      { to: '/bill-split', icon: 'fa-receipt',        label: 'Bill Split',     feature: 'billSplit' },
      { to: '/csv',        icon: 'fa-file-csv',       label: 'CSV Workspace',  horizonOnly: true },
    ],
  },
]
