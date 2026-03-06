import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../App'
import { api } from '../api'
import FlowerCanvas from '../components/FlowerCanvas'

export default function Login() {
  const [form, setForm]   = useState({ username: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate  = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const data = await api.post('/api/auth/login', form)
      if (data.ok) {
        login(data.user)
        navigate('/home')
      } else {
        setError(data.error || 'Invalid username or password')
      }
    } catch {
      setError('Connection error. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-vh-100 d-flex" data-bs-theme="light" data-theme="light">
      {/* Left panel — petal animation */}
      <div className="d-none d-lg-block" style={{ width: '45%', flexShrink: 0 }}>
        <FlowerCanvas>
          <img
            src="/logo.png"
            alt="Arch Bay"
            style={{ width: 96, height: 96, objectFit: 'contain', marginBottom: '1.25rem', opacity: 0.9 }}
          />
          <h1
            style={{
              fontFamily: "'Playfair Display', serif",
              fontWeight: 600,
              fontSize: '3rem',
              color: '#1a1a1a',
              margin: '0 0 0.75rem',
              letterSpacing: '-0.02em',
              textAlign: 'center',
            }}
          >
            Arch Bay
          </h1>
          <p
            style={{
              fontFamily: "'Playfair Display', serif",
              fontStyle: 'italic',
              fontSize: '1rem',
              color: '#3a3a3a',
              opacity: 0.75,
              textAlign: 'center',
              maxWidth: 260,
              lineHeight: 1.6,
              margin: 0,
            }}
          >
            A private space for friends —<br />trade, chat, and stay connected.
          </p>
        </FlowerCanvas>
      </div>

      {/* Right panel — login form */}
      <div className="flex-grow-1 d-flex justify-content-center align-items-center p-4" style={{ background: '#f4f6f9' }}>
        <div style={{ width: '100%', maxWidth: 400 }}>
          <div className="text-center mb-5">
            <img src="/logo.png" alt="Arch Bay" className="d-lg-none mb-3" style={{ width: 72, height: 72, objectFit: 'contain' }} />
            <h2 className="fw-bold">Welcome back</h2>
            <p className="text-muted">Sign in to your account</p>
          </div>

          {error && (
            <div className="alert alert-danger py-2 small">{error}</div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="mb-3">
              <label className="form-label fw-semibold">Username</label>
              <div className="login-group">
                <i className="fas fa-user login-icon" />
                <input
                  className="login-input"
                  placeholder="Enter username"
                  value={form.username}
                  onChange={e => setForm(f => ({ ...f, username: e.target.value }))}
                  autoComplete="username"
                  autoFocus
                  required
                />
              </div>
            </div>
            <div className="mb-4">
              <label className="form-label fw-semibold">Password</label>
              <div className="login-group">
                <i className="fas fa-lock login-icon" />
                <input
                  type="password"
                  className="login-input"
                  placeholder="Enter password"
                  value={form.password}
                  onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                  autoComplete="current-password"
                  required
                />
              </div>
            </div>
            <button
              type="submit"
              className="btn btn-primary w-100 btn-lg"
              disabled={loading}
            >
              {loading ? <span className="spinner-border spinner-border-sm" /> : 'Sign In'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
