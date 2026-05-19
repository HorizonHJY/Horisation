import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
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

            fontStyle: 'italic',
            fontSize: 'clamp(0.85rem, 1.4vw, 1rem)',
            color: '#3a3a3a',
            opacity: 0.62,
            lineHeight: 1.7,
            margin: 0,
          }}>
            {'欢迎来到圣路易斯 让我们把村里的生活变得丰富多彩一些吧'} <br />
            {'希望你们在这里能买到心仪的物品 延续物品的生命'}
          </p>
        </div>

  return (
    /* Outer shell */
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: '#f5f4f0' }}>

      {/* Canvas background — behind everything */}
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 0, pointerEvents: 'none' }}>
        <FlowerCanvas origin="right" />
      </div>

      {/* ── 核心修改点 1：新增一个全屏宽度的滚动容器 ── */}
      {/* 因为它占据了 right: 0，所以滚动条会被推到屏幕最右侧 */}
      <div
        style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          zIndex: 2,
          overflowY: 'auto', // 滚动条挂载在这个全屏容器上
          WebkitOverflowScrolling: 'touch',
        }}
      >

        {/* ── 核心修改点 2：原来的表单容器变为内部子容器，只负责内容排版 ── */}
        <div
          className="login-page-overlay"
          style={{
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100%', // 确保即使内容不多，也能撑满全屏高度，把文字推到底
            width: '100%',
            maxWidth: '650px', // 依旧限制表单在左侧 650px 范围内
            padding: 'clamp(1.5rem, 3vw, 2.5rem)',
            boxSizing: 'border-box',
          }}
        >
          {/* 顶部弹性占位 */}
          <div style={{ flex: '0 1 5vh', minHeight: '1rem' }} />

          {/* Logo + title */}
          <div
            className="login-header"
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}
          >
            <img
              src="/logo.png"
              alt="Arch Bay"
              className="login-logo"
              style={{ objectFit: 'contain', opacity: 0.92 }}
            />
            <h1 style={{
              fontFamily: "'Playfair Display', serif",
              fontWeight: 800,
              fontSize: 'clamp(2rem, 4vw, 2.8rem)',
              color: '#1a1a1a',
              letterSpacing: '-0.02em',
              margin: '0 0 2.5rem',
            }}>
              Arch Bay
            </h1>
          </div>

          {/* Form card */}
          <div className="login-form-wrap" style={{ width: '100%', margin: '0 auto', maxWidth: 600 }}>
            <div style={{
              background: 'rgba(255,255,255,0.80)',
              backdropFilter: 'blur(14px)',
              WebkitBackdropFilter: 'blur(14px)',
              borderRadius: 16,
              padding: '2rem 2rem 1.75rem',
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

          {/* 弹性推挤层（把文字推到底部） */}
          <div style={{ flex: '1 1 4rem' }}></div>

          {/* Tagline */}
          <div style={{
            width: '100%',
            maxWidth: 500,
            pointerEvents: 'none',
            paddingBottom: '1rem',
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
              St. Louis private harbor.
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
              {'欢迎来到圣路易斯 让我们把村里的生活变得丰富多彩一些吧'} <br />
              {'希望你们在这里能买到心仪的物品 延续物品的生命'}
            </p>
          </div>

        </div>
      </div>
    </div>
  )
}