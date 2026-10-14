/**
 * The site's map, in one place.
 *
 * The sidebar renders it. It lives here rather than inside Sidebar.jsx so
 * that anything else that ever needs the map reads the same one.
 *
 * `feature` gates an item through features.js; an item without one is open
 * to every member. Sections with nothing visible are not rendered.
 */

import {
  Home,
  Store,
  Megaphone,
  MessageSquare,
  UserRound,
  Users,
  Globe,
  Moon,
  Sparkles,
  ClipboardList,
  Route,
  Receipt,
} from 'lucide-react'

export const NAV_SECTIONS = [
  {
    key: 'main', title: 'Main',
    items: [
      { to: '/home', icon: Home, label: 'Home' },
    ],
  },
  {
    key: 'community', title: 'Community',
    items: [
      { to: '/market',   icon: Store,         label: 'Market' },
      { to: '/tasks',    icon: Megaphone,     label: 'Tasks' },
      { to: '/feedback', icon: MessageSquare, label: 'Message Board' },
      { to: '/friends',  icon: UserRound,     label: 'Friends' },
      { to: '/groups',   icon: Users,         label: 'Groups' },
    ],
  },
  {
    key: 'fun', title: 'For Fun',
    items: [
      { to: '/fun/gomoku-online', icon: Globe,    label: 'Online Gomoku', feature: 'onlineGomoku' },
      { to: '/tarot',             icon: Moon,     label: 'Tarot' },
      { to: '/bazi',              icon: Sparkles, label: 'BaZi 八字' },
    ],
  },
  {
    key: 'toolkit', title: 'Toolkit',
    items: [
      { to: '/hormemo',    icon: ClipboardList, label: 'Memo' },
      { to: '/travel',     icon: Route,         label: 'Travel Planner' },
      { to: '/bill-split', icon: Receipt,       label: 'Bill Split',     feature: 'billSplit' },
    ],
  },
]
