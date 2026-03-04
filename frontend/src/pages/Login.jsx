import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../App'
import { api } from '../api'

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
    <div className="min-vh-100 d-flex" style={{ background: '#f4f6f9' }}>
      {/* Left panel */}
      <div
        className="d-none d-lg-flex flex-column justify-content-center align-items-center text-white p-5"
        style={{ width: '45%', background: 'linear-gradient(135deg, #1e2a3a 0%, #3a7bd5 100%)' }}
      >
        <img src="/logo.png" alt="Arch Bay" style={{ width: 120, height: 120, objectFit: 'contain', marginBottom: '1.5rem' }} />
        <h1 className="fw-bold mb-3">Arch Bay</h1>
        <p className="text-center opacity-75" style={{ maxWidth: 320 }}>
          A private space for friends — trade, chat, and stay connected.
        </p>
        <div className="mt-5 d-flex flex-column gap-3" style={{ maxWidth: 280 }}>
          {[
            ['fa-store',          'Market',         'Buy and sell second-hand items'],
            ['fa-clipboard-list', 'Hormemo',         'Personal memo and task tracker'],
            ['fa-user-friends',   'Friends',         'Private chat and contact sharing'],
            ['fa-comments',       'Message Board',   'Share updates with everyone'],
            ['fa-globe',          'Online Gomoku',   'Play Five in a Row with friends'],
          ].map(([icon, title, desc]) => (
            <div key={title} className="d-flex gap-3 align-items-start">
              <i className={`fas ${icon} mt-1`} style={{ width: 20, opacity: .8 }} />
              <div>
                <div className="fw-semibold">{title}</div>
                <div style={{ fontSize: '.82rem', opacity: .65 }}>{desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right panel */}
      <div className="flex-grow-1 d-flex justify-content-center align-items-center p-4">
        <div style={{ width: '100%', maxWidth: 400 }}>
          <div className="text-center mb-5">
            <img src="/logo.png" alt="Arch Bay" className="d-lg-none mb-3" style={{ width: 48, height: 48, objectFit: 'contain' }} />
            <h2 className="fw-bold">Welcome back</h2>
            <p className="text-muted">Sign in to your account</p>
          </div>

          {error && (
            <div className="alert alert-danger py-2 small">{error}</div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="mb-3">
              <label className="form-label fw-semibold">Username</label>
              <input
                className="form-control form-control-lg"
                placeholder="Enter username"
                value={form.username}
                onChange={e => setForm(f => ({ ...f, username: e.target.value }))}
                autoComplete="username"
                autoFocus
                required
              />
            </div>
            <div className="mb-4">
              <label className="form-label fw-semibold">Password</label>
              <input
                type="password"
                className="form-control form-control-lg"
                placeholder="Enter password"
                value={form.password}
                onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                autoComplete="current-password"
                required
              />
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
