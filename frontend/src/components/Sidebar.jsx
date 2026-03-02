import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../App'

const NAV = [
  {
    section: 'Main',
    items: [
      { to: '/home',    icon: 'fa-home',           label: 'Home' },
      { to: '/hormemo', icon: 'fa-clipboard-list',  label: 'Hormemo' },
    ]
  },
  {
    section: 'Toolkit',
    items: [
      { to: '/csv',                  icon: 'fa-file-csv',    label: 'CSV Workspace' },
      { to: '/under-development',    icon: 'fa-chart-bar',   label: 'Data Analysis' },
      { to: '/under-development',    icon: 'fa-database',    label: 'Data Handling' },
      { to: '/under-development',    icon: 'fa-chart-line',  label: 'Data Visualisation' },
    ]
  },
  {
    section: 'Community',
    items: [
      { to: '/market', icon: 'fa-store', label: 'Market' },
    ]
  },
  {
    section: 'For Fun',
    items: [
      { to: '/fun/gomoku', icon: 'fa-chess-board', label: 'Gomoku' },
    ]
  },
]

export default function Sidebar() {
  const { user } = useAuth()

  const isAdmin = user?.role_info?.permissions?.includes('admin')

  return (
    <div className="sidebar">
      <div className="logo">
        <i className="fas fa-horse-head" />
        <span>Horisation</span>
      </div>

      {NAV.map(({ section, items }) => (
        <div className="nav-section" key={section}>
          <div className="nav-title">{section}</div>
          {items.map(({ to, icon, label }) => (
            <NavLink
              key={label}
              to={to}
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
          <NavLink to="/admin" className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <i className="fas fa-users-cog" />
            <span>User Management</span>
          </NavLink>
        </div>
      )}
    </div>
  )
}
