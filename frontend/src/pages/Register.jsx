import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../App'
import { api } from '../api'
import FlowerCanvas from '../components/FlowerCanvas'
import { validatePassword } from '../utils'

export default function Register() {
  const [form, setForm] = useState({
    username: '', display_name: '', password: '', confirm: '', invite_code: '',
  })
  const [error, setError]     = useState('')
  const [loading, setLoading] = useState(false)
  const { login }  = useAuth()
  const navigate   = useNavigate()

  const set = (key) => (e) => setForm(f => ({ ...f, [key]: e.target.value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (form.password !== form.confirm) return setError('Passwords do not match.')
    const pwErr = validatePassword(form.password)
    if (pwErr) return setError(pwErr)
    setLoading(true)
    try {
      const data = await api.post('/api/auth/signup', {
        username:     form.username.trim(),
        display_name: form.display_name.trim(),
        password:     form.password,
        invite_code:  form.invite_code.trim(),
      })
      if (data.ok) {
        login(data.user)
        navigate('/home')
      } else {
        setError(data.error || 'Registration failed.')
      }
    } catch {
      setError('Connection error. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* Fixed background canvas */}
      <div style={{ position: 'fixed', inset: 0, zIndex: 0 }}>
        <FlowerCanvas origin="right" />
      </div>

      {/* Scrollable content */}
      <div className="login-page-overlay" style={{
        position: 'relative',
        zIndex: 1,
        minHeight: '100vh',
        width: '100%',
        maxWidth: 650,
        padding: 'clamp(1.2rem, 4vh, 2rem)',
        paddingBottom: 'calc(clamp(1.2rem, 4vh, 2rem) + env(safe-area-inset-bottom, 16px))',
        boxSizing: 'border-box',
        pointerEvents: 'none',
      }}>
        {/* Top spacer */}
        <div style={{ flex: '0 1 min(1.5rem, 3vh)', minHeight: '0.5rem' }} />

        <div style={{
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', width: '100%',
          pointerEvents: 'auto',
        }}>
          <svg viewBox="0 0 160 130" fill="none" xmlns="http://www.w3.org/2000/svg"
               style={{ width: 'clamp(64px, 10vw, 96px)', height: 'auto', marginBottom: 'clamp(0.5rem, 2vh, 1.25rem)' }}>
            <path d="M14 118 C14 118 14 34 80 16 C146 34 146 118 146 118 L128 118 C128 118 128 50 80 34 C32 50 32 118 32 118 Z" fill="#2d2d2d"/>
            <path d="M48 118 C48 92 112 92 112 118 Z" fill="#2d2d2d"/>
            <ellipse cx="80" cy="120" rx="66" ry="6" fill="#2d2d2d" opacity="0.15"/>
          </svg>
          <h1 style={{
            fontFamily: "'Playfair Display', serif",
            fontWeight: 800, fontSize: 'clamp(1.6rem, 4vw, 2.8rem)',
            color: '#1a1a1a', letterSpacing: '-0.02em', margin: '0 0 clamp(1rem, 4vh, 2rem)',
          }}>
            Arch Bay
          </h1>
        </div>

        <div style={{ pointerEvents: 'auto', width: '100%', maxWidth: 600 }}>
          <div style={{
            background: 'rgba(255,255,255,0.80)',
            backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
            borderRadius: 16, padding: 'clamp(1rem, 4vw, 2rem) clamp(1rem, 4vw, 2rem) clamp(0.8rem, 3vw, 1.75rem)',
            boxShadow: '0 8px 32px rgba(0,0,0,0.10)',
            border: '1px solid rgba(255,255,255,0.65)',
          }}>
            <h2 style={{ fontWeight: 700, marginBottom: 4, fontSize: '1.3rem', textAlign: 'center' }}>Create Account</h2>
            <p style={{ color: '#666', marginBottom: '1.4rem', fontSize: '.875rem', textAlign: 'center' }}>
              You need an invite code to join.
            </p>

            {error && <div className="alert alert-danger py-2 small mb-3">{error}</div>}

            <form onSubmit={handleSubmit}>
              {[
                { key: 'username',     label: 'Username',     type: 'text',     icon: 'fa-user',        placeholder: 'Choose a username' },
                { key: 'display_name', label: 'Display Name', type: 'text',     icon: 'fa-id-badge',    placeholder: 'Your name (optional)' },
                { key: 'password',     label: 'Password',     type: 'password', icon: 'fa-lock',        placeholder: 'Min 8 chars, A-Z, a-z, 0-9' },
                { key: 'confirm',      label: 'Confirm',      type: 'password', icon: 'fa-lock',        placeholder: 'Confirm password' },
                { key: 'invite_code',  label: 'Invite Code',  type: 'text',     icon: 'fa-ticket-alt',  placeholder: 'Enter invite code' },
              ].map(({ key, label, type, icon, placeholder }) => (
                <div className="mb-3" key={key}>
                  <label className="form-label fw-semibold" style={{ fontSize: '.875rem' }}>{label}</label>
                  <div className="login-group">
                    <i className={`fas ${icon} login-icon`} />
                    <input
                      type={type}
                      className="login-input"
                      placeholder={placeholder}
                      value={form[key]}
                      onChange={set(key)}
                      required={key !== 'display_name'}
                      autoComplete={type === 'password' ? 'new-password' : key}
                    />
                  </div>
                </div>
              ))}

              <button
                type="submit"
                className="btn btn-primary w-100 mt-1"
                style={{ height: 44, fontSize: '1rem' }}
                disabled={loading}
              >
                {loading ? <span className="spinner-border spinner-border-sm" /> : 'Create Account'}
              </button>
            </form>

            <p className="text-center mt-3 mb-0" style={{ fontSize: '.85rem', color: '#666' }}>
              Already have an account?{' '}
              <Link to="/login" style={{ color: '#6b9cdb', fontWeight: 600 }}>Sign in</Link>
            </p>
          </div>
        </div>
      </div>

      <div className="login-tagline" style={{ position: 'fixed', pointerEvents: 'none', zIndex: 1 }}>
        <p style={{
          fontFamily: "'Playfair Display', serif", fontWeight: 600,
          fontSize: 'clamp(1.3rem, 2.2vw, 1.8rem)', color: '#1a1a1a',
          opacity: 0.82, lineHeight: 1.3, margin: '0 0 0.5rem', letterSpacing: '-0.01em',
        }}>
          St. Louis private harbor.
        </p>
        <p style={{
          fontFamily: "'Playfair Display', serif", fontStyle: 'italic',
          fontSize: 'clamp(0.85rem, 1.4vw, 1rem)', color: '#3a3a3a',
          opacity: 0.62, lineHeight: 1.7, margin: 0,
        }}>
          欢迎来到圣路易斯 让我们把村里的生活变得丰富多彩一些吧 <br />
          希望你们在这里能买到心仪的物品 延续物品的生命<br />
        </p>
      </div>
    </>
  )
}
