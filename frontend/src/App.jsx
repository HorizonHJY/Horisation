import React, { createContext, useContext, useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { api } from './api'

import Layout from './components/Layout'
import Login from './pages/Login'
import Home from './pages/Home'
import CSV from './pages/CSV'
import Hormemo from './pages/Hormemo'
import Profile from './pages/Profile'
import AdminUsers from './pages/AdminUsers'
import UnderDevelopment from './pages/UnderDevelopment'
import Gomoku from './pages/fun/Gomoku'

// ── Auth Context ────────────────────────────────────────────────
export const AuthContext = createContext(null)
export const useAuth = () => useContext(AuthContext)

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

  if (loading) return (
    <div className="d-flex justify-content-center align-items-center vh-100">
      <div className="spinner-border text-primary" />
    </div>
  )

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
      <AuthProvider>
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
            <Route path="/under-development" element={<UnderDevelopment />} />
            <Route path="/fun/gomoku"        element={<Gomoku />} />
          </Route>

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/home" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
