import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth, useUnread, useFeature } from '../App'

const NAV_MAIN = [
  { to: '/home', icon: 'fa-home', label: 'Home' },
]

const NAV_COMMUNITY = [
  { to: '/market',   icon: 'fa-store',        label: 'Market' },
  { to: '/feedback', icon: 'fa-comments',     label: 'Message Board' },
  { to: '/friends',  icon: 'fa-user-friends', label: 'Friends' },
]

const NAV_FUN = [
  { to: '/fun/gomoku-online', icon: 'fa-globe', label: 'Online Gomoku', feature: 'onlineGomoku' },
]

const NAV_TOOLKIT_BASE = [
  { to: '/hormemo', icon: 'fa-clipboard-list', label: 'Hormemo' },
  { to: '/travel',  icon: 'fa-route',          label: 'Travel Planner' },
]

const NAV_TOOLKIT_HORIZON = [
  { to: '/csv', icon: 'fa-file-csv', label: 'CSV Workspace' },
]

export default function Sidebar({ isOpen, onClose }) {
  const { user, logout } = useAuth()
  const { total: unreadTotal } = useUnread()
  const navigate = useNavigate()
  const canGomoku = useFeature('onlineGomoku')

  const isAdmin   = user?.role_info?.permissions?.includes('admin')
  const isHorizon = user?.role === 'horizon'

  const visibleFun = NAV_FUN.filter(item => !item.feature || canGomoku)

  const nav = [
    { section: 'Main',      items: NAV_MAIN },
    { section: 'Community', items: NAV_COMMUNITY },
    ...(visibleFun.length > 0 ? [{ section: 'For Fun', items: visibleFun }] : []),
    { section: 'Toolkit',   items: isHorizon ? [...NAV_TOOLKIT_BASE, ...NAV_TOOLKIT_HORIZON] : NAV_TOOLKIT_BASE },
  ]

  function handleLogout() {
    logout()
    navigate('/login')
    onClose?.()
  }

  return (
    <div className={`sidebar d-flex flex-column${isOpen ? ' sidebar-open' : ''}`} style={{ height: '100vh' }}>
      <div className="logo">
        <img src="/logo.png" alt="Arch Bay" style={{ height: 44, width: 44, objectFit: 'contain' }} />
        <span className="arch-bay-text" style={{ fontSize: '1.25rem' }}>Arch Bay</span>
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
