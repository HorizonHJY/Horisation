import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../App'
import { api } from '../api'
import FlowerCanvas from '../components/FlowerCanvas'

export default function Login() {
  const [form, setForm]       = useState({ username: '', password: '' })
  const [error, setError]     = useState('')
  const [loading, setLoading] = useState(false)
  const { login }  = useAuth()
  const navigate   = useNavigate()

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
    <div style={{ position: 'fixed', inset: 0, overflow: 'hidden' }}>

      {/* Canvas background */}
      <div style={{ position: 'absolute', inset: 0 }}>
        <FlowerCanvas origin="right" />
      </div>

      {/* ── Centre: logo + form ───────────────────────────────────────────── */}
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        paddingTop: '8vh',          /* nudge content slightly below true centre */
        pointerEvents: 'none',
      }}>
        {/* Logo + title */}
        <img
          src="/logo.png"
          alt="Arch Bay"
          style={{ width: 999, height:300, objectFit: 'contain', opacity: 0.92, marginBottom: '0.1rem' }}
        />
        <h1 style={{
          fontFamily: "'Playfair Display', serif",
          fontWeight: 800,
          fontSize: 'clamp(2rem, 4vw, 2.8rem)',
          color: '#1a1a1a',
          letterSpacing: '-0.02em',
          margin: '0 0 2.2rem',
        }}>
          Arch Bay
        </h1>

        {/* Login form card */}
        <div style={{
          pointerEvents: 'auto',
          width: '100%', maxWidth: 600,
          padding: '0 1rem',
        }}>
          <div style={{
            background: 'rgba(255,255,255,0.80)',
            backdropFilter: 'blur(14px)',
            WebkitBackdropFilter: 'blur(14px)',
            borderRadius: 16,
            padding: '2rem 2rem 1.75rem',
            boxShadow: '0 8px 32px rgba(0,0,0,0.10)',
            border: '1px solid rgba(255,255,255,0.65)',
          }}>
            <h2 style={{ fontWeight: 700, marginBottom: 4, fontSize: '1.3rem' }}>Welcome back</h2>
            <p style={{ color: '#666', marginBottom: '1.4rem', fontSize: '.875rem' }}>
              Sign in to your account
            </p>

            {error && (
              <div className="alert alert-danger py-2 small mb-3">{error}</div>
            )}

            <form onSubmit={handleSubmit}>
              <div className="mb-3">
                <label className="form-label fw-semibold" style={{ fontSize: '.875rem' }}>Username</label>
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
                <label className="form-label fw-semibold" style={{ fontSize: '.875rem' }}>Password</label>
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
                className="btn btn-primary w-100"
                style={{ height: 44, fontSize: '1rem' }}
                disabled={loading}
              >
                {loading ? <span className="spinner-border spinner-border-sm" /> : 'Sign In'}
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* ── Bottom-left: tagline ──────────────────────────────────────────── */}
      <div style={{
        position: 'absolute',
        bottom: 'clamp(1.5rem, 3vw, 2.5rem)',
        left:   'clamp(1.5rem, 3vw, 2.5rem)',
        pointerEvents: 'none',
        maxWidth: 360,
      }}>
        <p style={{
          fontFamily: "'Playfair Display', serif",
          fontWeight: 600,
          fontSize: 'clamp(1.3rem, 2.2vw, 1.8rem)',
          color: '#1a1a1a',
          opacity: 0.82,
          lineHeight: 1.3,
          margin: '0 0 0.5rem',
          letterSpacing: '-0.01em',
        }}>
          St. Louis's private harbor.
        </p>
        <p style={{
          fontFamily: "'Playfair Display', serif",
          fontStyle: 'italic',
          fontSize: 'clamp(0.85rem, 1.4vw, 1rem)',
          color: '#3a3a3a',
          opacity: 0.62,
          lineHeight: 1.7,
          margin: 0,
        }}>
          Connect with friends,<br />
          discover great deals,<br />
          stay close.
        </p>
      </div>
    </div>
  )
}
