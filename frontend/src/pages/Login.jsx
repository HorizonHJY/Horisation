import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../App'
import { api } from '../api'
import FlowerCanvas from '../components/FlowerCanvas'

export default function Login() {
  const [form, setForm] = useState({ username: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

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
    /* Outer shell - fixed fullscreen background */
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: '#f5f4f0' }}>

      {/* Canvas background - behind everything */}
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 0, pointerEvents: 'none' }}>
        <FlowerCanvas origin="right" />
      </div>

      {/* Scrollable container - full width, scrollbar on the very right */}
      <div
        style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          zIndex: 2,
          overflowY: 'auto',
          WebkitOverflowScrolling: 'touch',
        }}
      >
        {/* Content wrapper - flex column, min-height to push footer down when needed */}
        <div
          className="login-page-overlay"
          style={{
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100%',
            width: '100%',
            maxWidth: '650px',
            padding: 'clamp(0.8rem, 3vw, 2rem)',
            boxSizing: 'border-box',
          }}
        >
          {/* Top spacer */}
          <div style={{ flex: '0 1 min(1.5rem, 3vh)', minHeight: '0.5rem' }} />

          {/* Logo + Title */}
          <div
            className="login-header"
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}
          >
            <svg viewBox="0 0 160 130" fill="none" xmlns="http://www.w3.org/2000/svg"
                 style={{ width: 'clamp(64px, 10vw, 96px)', height: 'auto', marginBottom: 'clamp(0.5rem, 2vh, 1.25rem)', color: '#2d2d2d' }}>
              <path d="M14 118 C14 118 14 34 80 16 C146 34 146 118 146 118 L128 118 C128 118 128 50 80 34 C32 50 32 118 32 118 Z" fill="currentColor"/>
              <path d="M48 118 C48 92 112 92 112 118 Z" fill="currentColor"/>
              <ellipse cx="80" cy="120" rx="66" ry="6" fill="currentColor" opacity="0.15"/>
            </svg>
            <h1 style={{
              fontFamily: "'Playfair Display', serif",
              fontWeight: 800,
              fontSize: 'clamp(1.8rem, 4vw, 2.8rem)',
              color: '#1a1a1a',
              letterSpacing: '-0.02em',
              margin: '0 0 clamp(1rem, 4vh, 2rem)',
            }}>
              Horisation
            </h1>
          </div>

          {/* Form Card */}
          <div className="login-form-wrap" style={{ width: '100%', margin: '0 auto', maxWidth: 600 }}>
            <div style={{
              background: 'rgba(255,255,255,0.80)',
              backdropFilter: 'blur(14px)',
              WebkitBackdropFilter: 'blur(14px)',
              borderRadius: 16,
              padding: 'clamp(1rem, 4vw, 2rem) clamp(1rem, 4vw, 2rem) clamp(0.8rem, 3vw, 1.75rem)',
              boxShadow: '0 8px 32px rgba(0,0,0,0.10)',
              border: '1px solid rgba(255,255,255,0.65)',
            }}>
              <h2 style={{ fontWeight: 700, marginBottom: 4, fontSize: '1.3rem', textAlign: 'center' }}>Welcome back</h2>
              <p style={{ color: '#666', marginBottom: '1.4rem', fontSize: '.875rem', textAlign: 'center' }}>
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

              <p className="text-center mt-3 mb-0" style={{ fontSize: '.85rem', color: '#666' }}>
                New here?{' '}
                <Link to="/register" style={{ color: '#6b9cdb', fontWeight: 600 }}>Register with invite code</Link>
              </p>
            </div>
          </div>

          {/* Dynamic spacer */}
          <div style={{ flex: '1 1 0', minHeight: '1rem' }} />

          {/* Tagline / Footer */}
          <div style={{
            marginTop: 'auto',
            width: '100%',
            maxWidth: 500,
            pointerEvents: 'none',
            paddingBottom: 'clamp(0.5rem, 2vh, 1rem)',
          }}>
            <p style={{
              fontFamily: "'Playfair Display', serif",
              fontWeight: 600,
              fontSize: 'clamp(1rem, 2.2vw, 1.8rem)',
              color: '#1a1a1a',
              opacity: 0.82,
              lineHeight: 1.3,
              margin: '0 0 0.5rem',
              letterSpacing: '-0.01em',
            }}>
              St. Louis private harbor.
            </p>
            <p style={{
              fontFamily: "'Playfair Display', serif",
              fontStyle: 'italic',
              fontSize: 'clamp(0.75rem, 1.4vw, 1rem)',
              color: '#3a3a3a',
              opacity: 0.62,
              lineHeight: 1.7,
              margin: 0,
            }}>
              {'欢迎来到圣路易斯 让我们把村里的生活变得丰富多彩一些吧'} <br />
              {'希望你们在这里能买到心仪的物品 延续物品的生命'}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
