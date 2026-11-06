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
  KeyRound,
  Banknote,
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
  Brain,
  BookUser,
} from 'lucide-react'

/* `guest` says whether a signed-out visitor can open this page. The sidebar
   drops guest-only hidden items; the public server endpoints behind them
   decide what, if anything, a guest actually gets back. Private pages are
   also blocked at the route level, so this field only tightens the sidebar;
   it is not a security gate. */
export const NAV_SECTIONS = [
  {
    key: 'main', title: 'Main',
    items: [
      { to: '/home',  icon: Home,     label: 'Home',  guest: true },
      /* Personal profile page. Private until it opens up to every member;
         Sidebar only renders it when the signed-in username matches. */
      { to: '/about', icon: BookUser, label: 'About', ownerOnly: 'horizon' },
    ],
  },
  {
    key: 'community', title: 'Community',
    items: [
      { to: '/market',   icon: Store,         label: 'Market',        guest: true },
      { to: '/sublets',  icon: KeyRound,      label: 'Sublets',       guest: true },
      { to: '/fx',       icon: Banknote,      label: 'FX',            guest: true },
      { to: '/tasks',    icon: Megaphone,     label: 'Tasks',         guest: true },
      { to: '/feedback', icon: MessageSquare, label: 'Message Board', guest: true },
      { to: '/friends',  icon: UserRound,     label: 'Friends' },
      { to: '/groups',   icon: Users,         label: 'Groups' },
    ],
  },
  {
    key: 'fun', title: 'For Fun',
    items: [
      { to: '/fun/gomoku-online', icon: Globe,    label: 'Online Gomoku', feature: 'onlineGomoku' },
      { to: '/tarot',             icon: Moon,     label: 'Tarot',         guest: true },
      { to: '/bazi',              icon: Sparkles, label: 'BaZi',          guest: true },
      { to: '/mbti',              icon: Brain,    label: 'MBTI',          guest: true },
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
