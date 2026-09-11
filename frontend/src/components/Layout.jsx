import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import Sidebar, { ArchMark } from './Sidebar'
import Topbar from './Topbar'
import { useNotifications } from '../App'
import { NAV_SECTIONS, locate } from '../nav'

/**
 * The shell.
 *
 * On a desktop the navigation no longer lives in a fixed 240px column. It
 * tucks into a thin spine on the left edge — the arch, a dot per section,
 * a lantern if someone is waiting on you — and slides over the page when
 * you go to the edge or press `[`. A pin docks it for anyone who preferred
 * the column; that choice is remembered per browser.
 *
 * On a phone nothing changed: hamburger, drawer, scrim.
 */

const DESKTOP = '(min-width: 768px)'
const PIN_KEY = 'archbay.nav'          // 'pinned' | 'spine'
const LEAVE_GRACE_MS = 180             // slack between panel and spine, so the
                                       // boundary between them never flickers

function readPinned() {
  try { return localStorage.getItem(PIN_KEY) === 'pinned' } catch { return false }
}

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [pinned, setPinned] = useState(readPinned)
  const [revealed, setRevealed] = useState(false)
  const [isDesktop, setIsDesktop] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(DESKTOP).matches)

  const location = useLocation()
  const { badgeTotal } = useNotifications()
  const sidebarRef = useRef(null)
  const leaveTimer = useRef(0)
  const returnFocusRef = useRef(null)
  const focusOnRevealRef = useRef(false)

  useEffect(() => {
    const mq = window.matchMedia(DESKTOP)
    const apply = () => setIsDesktop(mq.matches)
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [])

  // Moving between pages closes whatever was open, on either form factor.
  useEffect(() => { setMobileOpen(false); setRevealed(false) }, [location.pathname])

  const togglePin = useCallback(() => {
    setPinned(p => {
      const next = !p
      try { localStorage.setItem(PIN_KEY, next ? 'pinned' : 'spine') } catch { /* private mode */ }
      return next
    })
    setRevealed(false)
  }, [])

  const clearLeave = () => { if (leaveTimer.current) { clearTimeout(leaveTimer.current); leaveTimer.current = 0 } }
  const reveal = useCallback(() => { clearLeave(); if (!pinned) setRevealed(true) }, [pinned])
  const scheduleHide = useCallback(() => {
    if (pinned) return
    clearLeave()
    leaveTimer.current = setTimeout(() => setRevealed(false), LEAVE_GRACE_MS)
  }, [pinned])
  useEffect(() => () => clearLeave(), [])

  /* `[` opens the menu for the keyboard and moves focus into it; Esc puts
     both back. Skipped while typing anywhere. */
  useEffect(() => {
    if (!isDesktop || pinned) return
    const onKey = (e) => {
      if (e.target.closest?.('input, textarea, select, [contenteditable="true"]')) return
      if (e.key === '[' && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault()
        setRevealed(r => {
          if (!r) {
            returnFocusRef.current = document.activeElement
            focusOnRevealRef.current = true
          }
          return !r
        })
      } else if (e.key === 'Escape') {
        setRevealed(r => {
          if (r) returnFocusRef.current?.focus?.()
          return false
        })
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [isDesktop, pinned])

  /* Focus can only move in once the panel is no longer inert, and inert is
     cleared by the render that `revealed` triggers — so wait for that commit
     rather than racing it with a frame callback. */
  useEffect(() => {
    if (!revealed || !focusOnRevealRef.current) return
    focusOnRevealRef.current = false
    sidebarRef.current?.querySelector('a, button')?.focus()
  }, [revealed])

  const here = locate(location.pathname)
  const navMode = pinned ? 'pinned' : 'spine'
  // Out of sight on a desktop unless revealed or pinned; on a phone, unless the drawer is open.
  const sidebarHidden = isDesktop ? !(pinned || revealed) : !mobileOpen

  return (
    <div className={`shell${revealed ? ' is-revealed' : ''}`} data-nav={navMode}>
      <Sidebar
        ref={sidebarRef}
        isOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
        pinned={pinned}
        onTogglePin={togglePin}
        hidden={sidebarHidden}
        onPointerEnter={isDesktop ? clearLeave : undefined}
        onPointerLeave={isDesktop ? scheduleHide : undefined}
      />
      {mobileOpen && (
        <div className="sidebar-overlay" onClick={() => setMobileOpen(false)} />
      )}

      {/* Desktop only (CSS). The strip you can see is the spine; the strip
          that reacts is wider, so you do not have to aim at 14px. */}
      <div className="spine-hot" aria-hidden="true" onPointerEnter={reveal} />
      <div className="spine" aria-hidden="true">
        <ArchMark className="spine__arch" />
        <div className="spine__dots">
          {NAV_SECTIONS.filter(s => s.key !== 'main').map(s => (
            <span
              key={s.key}
              className={`spine__dot${here?.sectionKey === s.key ? ' is-here' : ''}`}
            />
          ))}
        </div>
        {badgeTotal > 0 && <span className="spine__lantern" />}
        <span className="spine__foot"><i className="fas fa-chevron-right" /></span>
      </div>

      {/* The page dims a touch under the revealed panel, so it reads as behind. */}
      <div
        className="spine-scrim"
        aria-hidden="true"
        onPointerEnter={scheduleHide}
        onClick={() => { clearLeave(); setRevealed(false) }}
      />

      <Topbar onMenuClick={() => setMobileOpen(o => !o)} crumb={here} />
      <div className="main-content">
        <div className="page-content">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
