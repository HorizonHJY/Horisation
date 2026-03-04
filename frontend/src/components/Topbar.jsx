import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth, useTheme } from '../App'
import { api } from '../api'

export default function Topbar({ onMenuClick }) {
  const { user, logout } = useAuth()
  const { isDark, toggleTheme } = useTheme()
  const navigate = useNavigate()

  const handleLogout = async () => {
    await api.post('/api/auth/logout')
    logout()
    navigate('/login')
  }

  if (!user) return null

  return (
    <div className="topbar">
      <button className="hamburger-btn d-md-none me-auto" onClick={onMenuClick}>
        <i className="fas fa-bars" />
      </button>
      <div className="toggle-container me-3" style={{ fontSize: '13px' }}>
        <input
          type="checkbox"
          className="toggle-input"
          checked={isDark}
          onChange={toggleTheme}
          aria-label="Toggle dark mode"
        />
        <div className="toggle-handle-wrapper">
          <div className="toggle-handle">
            <div className="toggle-handle-knob" />
            <div className="toggle-handle-bar-wrapper">
              <div className="toggle-handle-bar" />
            </div>
          </div>
        </div>
        <div className="toggle-base">
          <div className="toggle-base-inside" />
        </div>
      </div>

      <div className="dropdown">
        <div
          className="d-flex align-items-center gap-2"
          data-bs-toggle="dropdown"
          style={{ cursor: 'pointer' }}
        >
          <span className="user-name">{user.display_name}</span>
          {user.avatar_url ? (
            <img
              src={user.avatar_url}
              alt={user.display_name}
              className="user-avatar"
              style={{ objectFit: 'cover' }}
            />
          ) : (
            <div className="user-avatar">
              {user.display_name?.[0]?.toUpperCase()}
            </div>
          )}
        </div>
        <ul className="dropdown-menu dropdown-menu-end">
          <li><h6 className="dropdown-header">{user.role_info?.name}</h6></li>
          <li>
            <button className="dropdown-item" onClick={() => navigate('/profile')}>
              <i className="fas fa-user me-2" />Profile
            </button>
          </li>
          <li><hr className="dropdown-divider" /></li>
          <li>
            <button className="dropdown-item text-danger" onClick={handleLogout}>
              <i className="fas fa-sign-out-alt me-2" />Logout
            </button>
          </li>
        </ul>
      </div>
    </div>
  )
}
