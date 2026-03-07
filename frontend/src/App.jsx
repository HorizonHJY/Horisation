import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { api } from './api'

import Layout from './components/Layout'
import HandLoader from './components/HandLoader'
import { canAccess } from './features'
import Login from './pages/Login'
import Home from './pages/Home'
import CSV from './pages/CSV'
import Hormemo from './pages/Hormemo'
import Profile from './pages/Profile'
import AdminUsers from './pages/AdminUsers'
import OnlineGomoku from './pages/fun/OnlineGomoku'
import Market from './pages/Market'
import Feedback from './pages/Feedback'
import Friends from './pages/Friends'

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
  }, [isDark])

  const toggleTheme = () => setIsDark(d => !d)

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

// ── Unread Context ───────────────────────────────────────────────
export const UnreadContext = createContext(null)
export const useUnread = () => useContext(UnreadContext)

function UnreadProvider({ children }) {
  const { user } = useAuth()
  const [unreadMap, setUnreadMap] = useState({})   // { username: count }
  const intervalRef = useRef(null)

  const refresh = useCallback(async () => {
    if (!user) return
    const d = await api.get('/api/friends/unread')
    if (d.ok) setUnreadMap(d.by_friend)
  }, [user])

  useEffect(() => {
    if (!user) { setUnreadMap({}); return }
    refresh()
    intervalRef.current = setInterval(refresh, 30000)
    return () => clearInterval(intervalRef.current)
  }, [user, refresh])

  const clearUnread = useCallback((username) => {
    setUnreadMap(prev => { const n = { ...prev }; delete n[username]; return n })
  }, [])

  const bumpUnread = useCallback((username) => {
    setUnreadMap(prev => ({ ...prev, [username]: (prev[username] || 0) + 1 }))
  }, [])

  const total = Object.values(unreadMap).reduce((a, b) => a + b, 0)

  return (
    <UnreadContext.Provider value={{ unreadMap, total, refresh, clearUnread, bumpUnread }}>
      {children}
    </UnreadContext.Provider>
  )
}

// ── Auth Context ────────────────────────────────────────────────
export const AuthContext = createContext(null)
export const useAuth = () => useContext(AuthContext)

/** Check if the current user can access a feature flag. */
export function useFeature(feature) {
  const { user } = useContext(AuthContext) ?? {}
  return canAccess(user?.role, feature)
}

function AuthProvider({ children }) {
  const [user, setUser]       = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/api/auth/check-session')
      .then(data => {
        if (data.ok && data.logged_in) setUser(data.user)
      })
      .finally(() => setLoading(false))
  }, [])

  const login  = (userData) => setUser(userData)
  const logout = () => {
    api.post('/api/auth/logout').finally(() => setUser(null))
  }

  if (loading) return <HandLoader fullPage />

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

// ── Route Guards ─────────────────────────────────────────────────
function PrivateRoute({ children }) {
  const { user } = useAuth()
  const location = useLocation()
  if (!user) return <Navigate to="/login" state={{ from: location }} replace />
  return children
}

function PublicOnlyRoute({ children }) {
  const { user } = useAuth()
  if (user) return <Navigate to="/home" replace />
  return children
}

// ── App ───────────────────────────────────────────────────────────
export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
      <AuthProvider>
      <UnreadProvider>
        <Routes>
          {/* Public */}
          <Route path="/login" element={<PublicOnlyRoute><Login /></PublicOnlyRoute>} />

          {/* Protected — all inside Layout */}
          <Route element={<PrivateRoute><Layout /></PrivateRoute>}>
            <Route index element={<Navigate to="/home" replace />} />
            <Route path="/home"              element={<Home />} />
            <Route path="/csv"               element={<CSV />} />
            <Route path="/hormemo"           element={<Hormemo />} />
            <Route path="/profile"           element={<Profile />} />
            <Route path="/admin"             element={<AdminUsers />} />
            <Route path="/fun/gomoku-online" element={<OnlineGomoku />} />
            <Route path="/market"            element={<Market />} />
            <Route path="/feedback"          element={<Feedback />} />
            <Route path="/friends"           element={<Friends />} />
          </Route>

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/home" replace />} />
        </Routes>
      </UnreadProvider>
      </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  )
}
