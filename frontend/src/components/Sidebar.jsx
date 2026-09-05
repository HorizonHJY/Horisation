import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth, useUnread, useFeature } from '../App'

const NAV_MAIN = [
  { to: '/home', icon: 'fa-home', label: 'Home' },
]

const NAV_COMMUNITY = [
  { to: '/market',   icon: 'fa-store',        label: 'Market' },
  { to: '/tasks',    icon: 'fa-bullhorn',     label: 'Tasks' },
  { to: '/feedback', icon: 'fa-comments',     label: 'Message Board' },
  { to: '/friends',  icon: 'fa-user-friends', label: 'Friends' },
  { to: '/groups',   icon: 'fa-users',       label: 'Groups' },
]

const NAV_FUN = [
  { to: '/fun/gomoku-online', icon: 'fa-globe', label: 'Online Gomoku', feature: 'onlineGomoku' },
]

const NAV_TOOLKIT_BASE = [
  { to: '/hormemo',    icon: 'fa-clipboard-list', label: 'Memo' },
  { to: '/travel',     icon: 'fa-route',           label: 'Travel Planner', feature: 'travelPlanner' },
  { to: '/bill-split', icon: 'fa-receipt',         label: 'Bill Split',     feature: 'billSplit' },
]

const NAV_TOOLKIT_HORIZON = [
  { to: '/csv', icon: 'fa-file-csv', label: 'CSV Workspace' },
]

export default function Sidebar({ isOpen, onClose }) {
  const { user, logout } = useAuth()
  const { total: unreadTotal } = useUnread()
  const navigate = useNavigate()
  const canGomoku = useFeature('onlineGomoku')
  const canTravel = useFeature('travelPlanner')
  const canBill   = useFeature('billSplit')

  const isAdmin   = user?.role_info?.permissions?.includes('admin')
  const isHorizon = user?.role === 'horizon'

  const visibleFun     = NAV_FUN.filter(item => !item.feature || canGomoku)
  const visibleToolkit = NAV_TOOLKIT_BASE.filter(item => {
    if (item.feature === 'travelPlanner') return canTravel
    if (item.feature === 'billSplit')     return canBill
    return true
  })

  const nav = [
    { section: 'Main',      items: NAV_MAIN },
    { section: 'Community', items: NAV_COMMUNITY },
    ...(visibleFun.length > 0 ? [{ section: 'For Fun', items: visibleFun }] : []),
    { section: 'Toolkit',   items: isHorizon ? [...visibleToolkit, ...NAV_TOOLKIT_HORIZON] : visibleToolkit },
  ]

  function handleLogout() {
    logout()
    navigate('/login')
    onClose?.()
  }

  return (
    <div className={`sidebar d-flex flex-column${isOpen ? ' sidebar-open' : ''}`} style={{ height: '100vh' }}>
      <div className="logo">
        <svg viewBox="0 0 160 130" fill="none" xmlns="http://www.w3.org/2000/svg"
             style={{ height: 40, width: 'auto' }}>
          <path d="M14 118 C14 118 14 34 80 16 C146 34 146 118 146 118 L128 118 C128 118 128 50 80 34 C32 50 32 118 32 118 Z" fill="currentColor"/>
          <path d="M48 118 C48 92 112 92 112 118 Z" fill="currentColor"/>
          <ellipse cx="80" cy="120" rx="66" ry="6" fill="currentColor" opacity="0.15"/>
        </svg>
        <span className="arch-bay-text" style={{ fontSize: '1.25rem' }}>Horisation</span>
      </div>

      <div className="sidebar-nav">
        {nav.map(({ section, items }) => (
          <div className="nav-section" key={section}>
            <div className="nav-title">{section}</div>
            {items.map(({ to, icon, label }) => (
              <NavLink
                key={label}
                to={to}
                onClick={onClose}
                className={({ isActive }) => `nav-item${isActive && to !== '/under-development' ? ' active' : ''}`}
              >
                <i className={`fas ${icon}`} />
                <span>{label}</span>
                {to === '/friends' && unreadTotal > 0 && (
                  <span className="badge bg-danger ms-auto" style={{ fontSize: '.65rem', minWidth: 18 }}>
                    {unreadTotal > 99 ? '99+' : unreadTotal}
                  </span>
                )}
              </NavLink>
            ))}
          </div>
        ))}

        {isAdmin && (
          <div className="nav-section">
            <div className="nav-title">Admin</div>
            <NavLink to="/admin" onClick={onClose} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
              <i className="fas fa-users-cog" />
              <span>User Management</span>
            </NavLink>
            <NavLink to="/admin/system" onClick={onClose} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
              <i className="fas fa-cogs" />
              <span>System</span>
            </NavLink>
          </div>
        )}
      </div>

      <div className="nav-section" style={{ marginTop: 'auto' }}>
        <button className="nav-item w-100 border-0 bg-transparent text-start" onClick={handleLogout}>
          <i className="fas fa-sign-out-alt" />
          <span>Log Out</span>
        </button>
      </div>
    </div>
  )
}
