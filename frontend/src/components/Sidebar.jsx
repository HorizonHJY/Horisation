import React, { forwardRef } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth, useNotifications } from '../App'
import { canAccess } from '../features'
import { NAV_SECTIONS } from '../nav'

export function ArchMark({ className, style }) {
  return (
    <svg viewBox="0 0 160 130" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} style={style} aria-hidden="true">
      <path d="M14 118 C14 118 14 34 80 16 C146 34 146 118 146 118 L128 118 C128 118 128 50 80 34 C32 50 32 118 32 118 Z" fill="currentColor"/>
      <path d="M48 118 C48 92 112 92 112 118 Z" fill="currentColor"/>
      <ellipse cx="80" cy="120" rx="66" ry="6" fill="currentColor" opacity="0.15"/>
    </svg>
  )
}

/**
 * The navigation panel. On a phone it is the slide-in drawer it always was.
 * On a desktop it is either docked (open) or gone (closed); the button in
 * its header closes it, the one in the topbar opens it. See Layout.
 */
const Sidebar = forwardRef(function Sidebar(
  { isOpen, onClose, docked, onCollapse, hidden },
  ref,
) {
  const { user, logout } = useAuth()
  // Messages you have not read plus people waiting on an answer from you.
  const { badgeTotal } = useNotifications()
  const navigate = useNavigate()

  const isAdmin   = user?.role_info?.permissions?.includes('admin')
  const isHorizon = user?.role === 'horizon'

  /* Each item is gated on its own flag. `canAccess` reads the same FEATURES
     map the routes do, so the sidebar and FeatureRoute cannot disagree. */
  const allowed = (item) =>
    (!item.feature || canAccess(user?.role, item.feature))
    && (!item.horizonOnly || isHorizon)

  const sections = NAV_SECTIONS
    .map(s => ({ ...s, items: s.items.filter(allowed) }))
    .filter(s => s.items.length > 0)

  function handleLogout() {
    logout()
    navigate('/login')
    onClose?.()
  }

  return (
    <div
      ref={ref}
      className={`sidebar d-flex flex-column${isOpen ? ' sidebar-open' : ''}`}
      style={{ height: '100vh' }}
      aria-label="Navigation"
      // Off-screen on a desktop, its links must not be in the tab order —
      // or Tab walks through a menu nobody can see. React 18 wants the
      // attribute as an empty string, not a boolean.
      {...(hidden ? { inert: '' } : {})}
    >
      <div className="logo">
        <ArchMark style={{ height: 40, width: 'auto' }} />
        <span className="brand-wordmark" style={{ fontSize: '1.25rem' }}>Arch Bay</span>
        {/* Desktop only: the way to close it lives on the thing being closed. */}
        {docked && (
          <button
            type="button"
            className="sidebar-collapse"
            onClick={onCollapse}
            aria-label="Close the menu"
            title="Close  [ "
          >
            <i className="fas fa-angles-left" aria-hidden="true" />
          </button>
        )}
      </div>

      <div className="sidebar-nav">
        {sections.map(({ key, title, items }) => (
          <div className="nav-section" key={key}>
            <div className="nav-title">{title}</div>
            {items.map(({ to, icon, label }) => (
              <NavLink
                key={label}
                to={to}
                onClick={onClose}
                className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
              >
                <i className={`fas ${icon}`} />
                <span>{label}</span>
                {to === '/friends' && badgeTotal > 0 && (
                  <span className="badge bg-danger ms-auto" style={{ fontSize: '.65rem', minWidth: 18 }}>
                    {badgeTotal > 99 ? '99+' : badgeTotal}
                  </span>
                )}
              </NavLink>
            ))}
          </div>
        ))}

        {isAdmin && (
          <div className="nav-section">
            <div className="nav-title">Admin</div>
            <NavLink to="/admin" end onClick={onClose} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
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
})

export default Sidebar
