import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { api, setLoginRequiredHandler } from './api'

import Layout from './components/Layout'
import HandLoader from './components/HandLoader'
import EnvRibbon from './components/EnvRibbon'
import SocketProvider, { useSocket } from './components/SocketProvider'
import { canAccess } from './features'
import Login from './pages/Login'
import Register from './pages/Register'
import Home from './pages/Home'
import Hormemo from './pages/Hormemo'
import Profile from './pages/Profile'
import AdminUsers from './pages/AdminUsers'
import SystemManagement from './pages/SystemManagement'
import OnlineGomoku from './pages/fun/OnlineGomoku'
import Market from './pages/Market'
import Feedback from './pages/Feedback'
import Friends from './pages/Friends'
import Groups from './pages/Groups'
import UserProfile from './pages/UserProfile'
import TravelPlanner from './pages/TravelPlanner'
import BillSplit from './pages/BillSplit'
import Tasks from './pages/Tasks'
import Tarot from './pages/Tarot'
import TarotHistory from './pages/TarotHistory'
import Bazi from './pages/Bazi'
import Mbti from './pages/Mbti'

// ── Theme Context ────────────────────────────────────────────────
export const ThemeContext = createContext(null)
export const useTheme = () => useContext(ThemeContext)

function ThemeProvider({ children }) {
  const [isDark, setIsDark] = useState(() => localStorage.getItem('theme') === 'dark')

  useEffect(() => {
    const t = isDark ? 'dark' : 'light'
    document.documentElement.setAttribute('data-theme', t)
    document.documentElement.setAttribute('data-bs-theme', t)
    localStorage.setItem('theme', t)
    // Keep the browser UI (address bar / status bar) in step with the page,
    // otherwise dark pages keep a stark white chrome. Values match --bg above.
    const meta = document.querySelector('meta[name="theme-color"]')
    if (meta) meta.setAttribute('content', isDark ? '#0f172a' : '#ffffff')
  }, [isDark])

  const toggleTheme = () => setIsDark(d => !d)

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

// ── Notifications Context ────────────────────────────────────────
/* Everything that can put a badge or a banner in front of you: unread private
   messages, pending friend requests, and pending contact requests. It lives
   above the router so it stays correct on every page, is fed by the global
   socket, and re-syncs from one snapshot endpoint whenever that socket
   connects — a reconnect repairs the state instead of leaving it stale. */
export const NotificationsContext = createContext(null)
export const useNotifications = () => useContext(NotificationsContext)


/* Snapshot fallback. Socket events are the fast path; this only exists to
   catch anything missed while the tab was suspended or offline, so it is
   deliberately slow — it used to run every 30s and was the *only* path. */
const SNAPSHOT_INTERVAL_MS = 5 * 60 * 1000

function NotificationsProvider({ children }) {
  const { user } = useAuth()
  const { socket } = useSocket()
  const [unreadMap, setUnreadMap]           = useState({})
  const [friendRequests, setFriendRequests] = useState([])
  const [contactRequests, setContactRequests] = useState([])

  const refresh = useCallback(async () => {
    if (!user) return
    const d = await api.get('/api/friends/notifications')
    if (!d.ok) return
    setUnreadMap(d.unread?.by_friend || {})
    setFriendRequests(d.friend_requests || [])
    setContactRequests(d.contact_requests || [])
  }, [user])

  useEffect(() => {
    if (!user) {
      setUnreadMap({}); setFriendRequests([]); setContactRequests([])
      return
    }
    refresh()
    const id = setInterval(refresh, SNAPSHOT_INTERVAL_MS)

    /* Browsers throttle and eventually suspend sockets in a background tab —
       phones especially — so events can be missed while you are away and the
       badge goes quietly stale. Re-syncing the moment the tab comes back is
       what stops that needing a manual refresh; the slow timer alone would
       leave it wrong for minutes. */
    const onVisible = () => { if (document.visibilityState === 'visible') refresh() }
    document.addEventListener('visibilitychange', onVisible)
    window.addEventListener('focus', onVisible)

    return () => {
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVisible)
      window.removeEventListener('focus', onVisible)
    }
  }, [user, refresh])

  // Live updates. Registered here rather than on a page so they arrive
  // wherever the user happens to be.
  useEffect(() => {
    if (!socket || !user) return

    const onConnect = () => refresh()   // re-sync after any (re)connect
    // io() starts connecting immediately, so the first 'connect' can fire
    // before this listener exists. Catch that case explicitly.
    if (socket.connected) refresh()
    const onFriendRequest  = (req) => setFriendRequests(prev =>
      prev.some(r => r.id === req.id) ? prev : [req, ...prev])
    const onContactRequest = (req) => setContactRequests(prev =>
      prev.some(r => r.id === req.id) ? prev : [req, ...prev])
    const onContactResolved = ({ id }) => setContactRequests(prev =>
      prev.filter(r => r.id !== id))
    const onChatMessage = (msg) => {
      if (!msg || msg.sender === user.username) return
      // The open conversation clears itself; anything else becomes a badge.
      if (window.__hzActiveChatWith === msg.sender) return
      setUnreadMap(prev => ({ ...prev, [msg.sender]: (prev[msg.sender] || 0) + 1 }))
    }
    const onFriendAccepted = () => refresh()

    socket.on('connect', onConnect)
    socket.on('friend_request_incoming', onFriendRequest)
    socket.on('contact_request_incoming', onContactRequest)
    socket.on('contact_request_resolved', onContactResolved)
    socket.on('chat_message', onChatMessage)
    socket.on('friend_accepted', onFriendAccepted)

    return () => {
      socket.off('connect', onConnect)
      socket.off('friend_request_incoming', onFriendRequest)
      socket.off('contact_request_incoming', onContactRequest)
      socket.off('contact_request_resolved', onContactResolved)
      socket.off('chat_message', onChatMessage)
      socket.off('friend_accepted', onFriendAccepted)
    }
  }, [socket, user, refresh])

  const clearUnread = useCallback((username) => {
    setUnreadMap(prev => { const n = { ...prev }; delete n[username]; return n })
  }, [])

  const bumpUnread = useCallback((username) => {
    setUnreadMap(prev => ({ ...prev, [username]: (prev[username] || 0) + 1 }))
  }, [])

  /** Drop a contact request locally, right after answering it. */
  const dismissContactRequest = useCallback((id) => {
    setContactRequests(prev => prev.filter(r => r.id !== id))
  }, [])

  const dismissFriendRequest = useCallback((id) => {
    setFriendRequests(prev => prev.filter(r => r.id !== id))
  }, [])

  /** The pending contact request from one person, if there is one. */
  const contactRequestFrom = useCallback(
    (username) => contactRequests.find(r => r.from_user === username) || null,
    [contactRequests])

  const total = Object.values(unreadMap).reduce((a, b) => a + b, 0)
  // What the sidebar shows: messages you have not read plus people waiting on
  // an answer from you. A pending request is as actionable as a message.
  const badgeTotal = total + friendRequests.length + contactRequests.length

  return (
    <NotificationsContext.Provider value={{
      unreadMap, total, badgeTotal,
      friendRequests, contactRequests,
      refresh, clearUnread, bumpUnread,
      dismissContactRequest, dismissFriendRequest, contactRequestFrom,
    }}>
      {children}
    </NotificationsContext.Provider>
  )
}

// ── Auth Context ────────────────────────────────────────────────
export const AuthContext = createContext(null)
export const useAuth = () => useContext(AuthContext)

export function useFeature(feature) {
  const { user } = useContext(AuthContext) ?? {}
  return canAccess(user?.role, feature)
}

/* A guest is a visitor who said "continue as guest" at /login. They can see
   the public pages — Home, Market, Message Board, Tasks, Tarot, BaZi — and
   anything else the server returns to a logged-out request. Trying to use
   something gated (post, like, chat, cast a chart) pops the sign-up prompt.
   The flag lives in localStorage so a reload stays in guest mode. */
const GUEST_KEY = 'archbay.guest'

function AuthProvider({ children }) {
  const [user, setUser]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [guest, setGuest]     = useState(() => {
    try { return localStorage.getItem(GUEST_KEY) === '1' } catch { return false }
  })
  const [signUpPrompt, setSignUpPrompt] = useState(null) // { action: '...' } | null
  // Reported by check-session whether or not anyone is signed in, so the
  // local-instance marker is already correct on the login page.
  const [backendLocal, setBackendLocal] = useState(false)

  useEffect(() => {
    api.get('/api/auth/check-session')
      .then(data => {
        if (data.ok && data.logged_in) {
          setUser(data.user)
          // Signing in clears any lingering guest flag.
          try { localStorage.removeItem(GUEST_KEY) } catch { /* private mode */ }
          setGuest(false)
        }
        setBackendLocal(Boolean(data.local_dev))
      })
      .finally(() => setLoading(false))
  }, [])

  const login = (userData) => {
    setUser(userData)
    try { localStorage.removeItem(GUEST_KEY) } catch { /* private mode */ }
    setGuest(false)
  }
  const logout = () => {
    api.post('/api/auth/logout').finally(() => setUser(null))
  }
  const continueAsGuest = useCallback(() => {
    try { localStorage.setItem(GUEST_KEY, '1') } catch { /* private mode */ }
    setGuest(true)
  }, [])
  /* `action` is a short noun the modal names in its sentence
     ("to post, you need an account"); it is UI copy, not a permission check. */
  const requireAuth = useCallback((action) => {
    if (user) return true
    setSignUpPrompt({ action: action || null })
    return false
  }, [user])
  const dismissSignUpPrompt = useCallback(() => setSignUpPrompt(null), [])

  // Any 401 LOGIN_REQUIRED from the API pops the sign-up prompt. The handler
  // is a bare pop, not a bounce — the visitor stays on the page and can see
  // what they were reading before they decided to interact. A signed-in user
  // whose session simply expired lands here too; the modal still names the
  // right next step for them.
  useEffect(() => {
    setLoginRequiredHandler(() => setSignUpPrompt({ action: null }))
    return () => setLoginRequiredHandler(null)
  }, [])

  if (loading) return <HandLoader fullPage />

  return (
    <AuthContext.Provider value={{
      user, login, logout,
      guest, continueAsGuest,
      requireAuth, signUpPrompt, dismissSignUpPrompt,
    }}>
      <EnvRibbon backendLocal={backendLocal} />
      {children}
    </AuthContext.Provider>
  )
}

// ── Route Guards ─────────────────────────────────────────────────
/* The private pages — Memo, Travel, Bill Split, Friends, Groups, Profile,
   admin. A guest here bounces to /login. */
function PrivateRoute({ children }) {
  const { user } = useAuth()
  const location = useLocation()
  if (!user) return <Navigate to="/login" state={{ from: location }} replace />
  return children
}

/* The public pages — Home, Market, Message Board, Tasks, Tarot, BaZi — plus
   the shell around them. A signed-in user or a guest both get in; a bare
   visitor who has not yet picked goes to /login so they can decide. */
function PageRoute({ children }) {
  const { user, guest } = useAuth()
  const location = useLocation()
  if (!user && !guest) return <Navigate to="/login" state={{ from: location }} replace />
  return children
}

function PublicOnlyRoute({ children }) {
  const { user } = useAuth()
  if (user) return <Navigate to="/home" replace />
  return children
}

function FeatureRoute({ feature, children }) {
  const allowed = useFeature(feature)
  if (!allowed) return <Navigate to="/home" replace />
  return children
}

/* Both of these need the signed-in user, so they sit inside AuthProvider.
   The socket is created only while someone is signed in, and notifications
   sit inside it because they are fed by its events. */
function SessionProviders({ children }) {
  const { user } = useAuth()
  return (
    <SocketProvider enabled={Boolean(user)}>
      <NotificationsProvider>{children}</NotificationsProvider>
    </SocketProvider>
  )
}

/* Pops when a guest tries to use something gated. Lives above the router so
   it never leaves with navigation; the text names what the visitor was
   trying to do, so the ask reads like a next step rather than a wall. */
function SignUpPrompt() {
  const { signUpPrompt, dismissSignUpPrompt } = useAuth()
  const navigate = useNavigate()
  if (!signUpPrompt) return null
  const action = signUpPrompt.action
  const go = (path) => { dismissSignUpPrompt(); navigate(path) }
  return (
    <div
      className="guest-prompt-backdrop"
      role="presentation"
      onClick={dismissSignUpPrompt}
    >
      <div
        className="guest-prompt"
        role="dialog"
        aria-modal="true"
        aria-labelledby="guest-prompt-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 id="guest-prompt-title" className="guest-prompt__title">Join Arch Bay</h3>
        <p className="guest-prompt__body">
          {action
            ? <>加入账号后就能 <strong>{action}</strong>。注册只需要一个邀请码。</>
            : <>这个功能要先注册账号才能用。邀请制,一分钟搞定。</>}
        </p>
        <div className="guest-prompt__actions">
          <button type="button" className="btn btn-primary" onClick={() => go('/register')}>
            注册
          </button>
          <button type="button" className="btn btn-outline-secondary" onClick={() => go('/login')}>
            已有账号,登录
          </button>
          <button type="button" className="btn btn-link" onClick={dismissSignUpPrompt}>
            再看看
          </button>
        </div>
      </div>
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────
export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
      <AuthProvider>
      <SignUpPrompt />
      <SessionProviders>
        <Routes>
          <Route path="/login"    element={<PublicOnlyRoute><Login /></PublicOnlyRoute>} />
          <Route path="/register" element={<PublicOnlyRoute><Register /></PublicOnlyRoute>} />

          <Route element={<PageRoute><Layout /></PageRoute>}>
            <Route index element={<Navigate to="/home" replace />} />
            {/* Public: a guest can see these; a signed-in user owns the extras */}
            <Route path="/home"              element={<Home />} />
            {/* A listing and a seller are addressable, so they can be pasted
                into a group chat and survive a reload. All three render the
                same page; Market opens the matching overlay from the params. */}
            <Route path="/market"                    element={<Market />} />
            <Route path="/market/l/:listingId"       element={<Market />} />
            <Route path="/market/u/:sellerUsername"  element={<Market />} />
            <Route path="/feedback"          element={<Feedback />} />
            <Route path="/tasks"             element={<Tasks />} />
            <Route path="/tarot"             element={<Tarot />} />
            <Route path="/bazi"              element={<Bazi />} />
            <Route path="/mbti"              element={<Mbti />} />

            {/* Members only: these use the real auth guard */}
            <Route path="/hormemo"           element={<PrivateRoute><Hormemo /></PrivateRoute>} />
            <Route path="/profile"           element={<PrivateRoute><Profile /></PrivateRoute>} />
            <Route path="/admin"             element={<PrivateRoute><AdminUsers /></PrivateRoute>} />
            <Route path="/admin/system"      element={<PrivateRoute><SystemManagement /></PrivateRoute>} />
            <Route path="/fun/gomoku-online" element={<PrivateRoute><FeatureRoute feature="onlineGomoku"><OnlineGomoku /></FeatureRoute></PrivateRoute>} />
            <Route path="/friends"           element={<PrivateRoute><Friends /></PrivateRoute>} />
            <Route path="/groups"            element={<PrivateRoute><Groups /></PrivateRoute>} />
            <Route path="/u/:username"       element={<PrivateRoute><UserProfile /></PrivateRoute>} />
            <Route path="/travel"            element={<PrivateRoute><TravelPlanner /></PrivateRoute>} />
            <Route path="/bill-split"        element={<PrivateRoute><FeatureRoute feature="billSplit"><BillSplit /></FeatureRoute></PrivateRoute>} />
            <Route path="/tarot/history"     element={<PrivateRoute><TarotHistory /></PrivateRoute>} />
          </Route>

          <Route path="*" element={<Navigate to="/home" replace />} />
        </Routes>
      </SessionProviders>
      </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  )
}
