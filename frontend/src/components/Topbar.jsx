import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../App'
import { api } from '../api'

export default function Topbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = async () => {
    await api.post('/api/auth/logout')
    logout()
    navigate('/login')
  }

  if (!user) return null

  return (
    <div className="topbar">
      <div className="dropdown">
        <div
          className="d-flex align-items-center gap-2"
          data-bs-toggle="dropdown"
          style={{ cursor: 'pointer' }}
        >
          <span className="user-name">{user.display_name}</span>
          <div className="user-avatar">
            {user.display_name?.[0]?.toUpperCase()}
          </div>
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
