import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../App'

const NAV_MAIN = [
  { to: '/home', icon: 'fa-home', label: 'Home' },
]

const NAV_COMMUNITY = [
  { to: '/market',   icon: 'fa-store',    label: 'Market' },
  { to: '/feedback', icon: 'fa-comments', label: 'Message Board' },
]

const NAV_FUN = [
  { to: '/fun/gomoku',        icon: 'fa-chess-board', label: 'Gomoku' },
  { to: '/fun/gomoku-online', icon: 'fa-globe',       label: 'Online Gomoku' },
]

const NAV_TOOLKIT_BASE = [
  { to: '/hormemo', icon: 'fa-clipboard-list', label: 'Hormemo' },
]

const NAV_TOOLKIT_HORIZON = [
  { to: '/csv',               icon: 'fa-file-csv',   label: 'CSV Workspace' },
  { to: '/under-development', icon: 'fa-chart-bar',  label: 'Data Analysis' },
  { to: '/under-development', icon: 'fa-database',   label: 'Data Handling' },
  { to: '/under-development', icon: 'fa-chart-line', label: 'Data Visualisation' },
]

export default function Sidebar({ isOpen, onClose }) {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const isAdmin   = user?.role_info?.permissions?.includes('admin')
  const isHorizon = user?.role === 'horizon'

  const nav = [
    { section: 'Main',      items: NAV_MAIN },
    { section: 'Community', items: NAV_COMMUNITY },
    { section: 'For Fun',   items: NAV_FUN },
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
        <img src="/logol.avif" alt="Horisation" style={{ height: 32, width: 32, objectFit: 'contain' }} />
        <span>Horisation</span>
      </div>

      <div className="flex-grow-1">
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
