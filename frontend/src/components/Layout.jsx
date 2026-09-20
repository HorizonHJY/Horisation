import React, { useCallback, useEffect, useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import Topbar from './Topbar'

/**
 * The shell.
 *
 * Desktop: the sidebar is closed by default and opens on a button — the one
 * at the top-left of the page when it is closed, the one in the sidebar's
 * own header when it is open. Open, it docks and the content moves over;
 * closed, nothing of it remains. No hover, no rail: a menu you asked for or
 * no menu. `[` does the same from the keyboard. The choice is remembered.
 *
 * Phone: the hamburger and the slide-in drawer, unchanged.
 */

const DESKTOP = '(min-width: 768px)'
const NAV_KEY = 'archbay.nav'          // 'open' | 'closed'

function readOpen() {
  try { return localStorage.getItem(NAV_KEY) === 'open' } catch { return false }
}

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [open, setOpen] = useState(readOpen)
  const [isDesktop, setIsDesktop] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(DESKTOP).matches)
  const location = useLocation()

  useEffect(() => {
    const mq = window.matchMedia(DESKTOP)
    const apply = () => setIsDesktop(mq.matches)
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [])

  // Moving between pages closes the phone drawer. A docked desktop sidebar
  // stays: you opened it on purpose, and navigating is what it is for.
  useEffect(() => { setMobileOpen(false) }, [location.pathname])

  const toggle = useCallback(() => {
    if (!isDesktop) { setMobileOpen(o => !o); return }
    setOpen(o => {
      const next = !o
      try { localStorage.setItem(NAV_KEY, next ? 'open' : 'closed') } catch { /* private mode */ }
      return next
    })
  }, [isDesktop])

  useEffect(() => {
    if (!isDesktop) return
    const onKey = (e) => {
      if (e.target.closest?.('input, textarea, select, [contenteditable="true"]')) return
      if (e.key === '[' && !e.metaKey && !e.ctrlKey && !e.altKey) { e.preventDefault(); toggle() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [isDesktop, toggle])

  const sidebarVisible = isDesktop ? open : mobileOpen

  return (
    <div className="shell" data-nav={open ? 'open' : 'closed'}>
      <Sidebar
        isOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
        docked={isDesktop && open}
        onCollapse={toggle}
        hidden={!sidebarVisible}
      />
      {mobileOpen && (
        <div className="sidebar-overlay" onClick={() => setMobileOpen(false)} />
      )}
      <Topbar onMenuClick={toggle} menuButtonVisible={!isDesktop || !open} />
      <div className="main-content">
        <div className="page-content">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
