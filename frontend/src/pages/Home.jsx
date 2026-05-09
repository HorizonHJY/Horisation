import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../App'
import FlowerCanvas from '../components/FlowerCanvas'

const FEATURES = [
  { icon: 'fa-clipboard-list', color: '#6b9cdb', title: 'Hormemo',        desc: 'Personal memo and task tracker.',        to: '/hormemo'  },
  { icon: 'fa-store',          color: '#27ae60', title: 'Market',          desc: 'Browse and post second-hand listings.',  to: '/market'   },
  { icon: 'fa-comments',       color: '#d97706', title: 'Message Board',   desc: 'Chat and share with everyone.',          to: '/feedback' },
]

export default function Home() {
  const { user } = useAuth()
  const navigate = useNavigate()

  return (
    <div style={{ position: 'relative', minHeight: 'calc(100vh - 60px - 64px)' }}>
      {/* Canvas background — subtle, sits behind content */}
      <div style={{
        position: 'fixed',
        right: 0, bottom: 0,
        width: '60vw', height: '70vh',
        pointerEvents: 'none',
        opacity: 0.45,
        zIndex: 0,
      }}>
        <FlowerCanvas origin="right" />
      </div>

      {/* Foreground content */}
      <div style={{ position: 'relative', zIndex: 1 }}>
        <div className="hero-block">
          <h2 className="hero-title">Hello, {user?.display_name}</h2>
          <p className="hero-sub">欢迎来村里看看!</p>
        </div>

        <h5 style={{
          fontFamily: "'Playfair Display', serif",
          fontWeight: 700,
          fontSize: '1.3rem',
          letterSpacing: '-0.01em',
          color: 'var(--text-primary)',
          marginBottom: 4,
        }}>
          Quick Access
        </h5>
        <p style={{
          fontFamily: "'Noto Serif SC', serif",
          fontStyle: 'italic',
          color: 'var(--text-muted)',
          fontSize: '.92rem',
          marginBottom: 18,
        }}>
          常用功能 — 一键直达
        </p>

        <div className="row g-3">
          {FEATURES.map(({ icon, color, title, desc, to }) => (
            <div key={title} className="col-12 col-sm-6 col-lg-4">
              <div
                className="card h-100"
                style={{ cursor: 'pointer', transition: 'border-color .15s, transform .15s' }}
                onClick={() => navigate(to)}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = 'var(--border-medium)'
                  e.currentTarget.style.transform   = 'translateY(-2px)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = ''
                  e.currentTarget.style.transform   = ''
                }}
              >
                <div className="card-body p-4">
                  <div className="d-flex align-items-center gap-3 mb-2">
                    <div
                      className="rounded-circle d-flex align-items-center justify-content-center"
                      style={{ width: 42, height: 42, background: color + '1a', flexShrink: 0 }}
                    >
                      <i className={`fas ${icon}`} style={{ color }} />
                    </div>
                    <div style={{
                      fontFamily: "'Playfair Display', serif",
                      fontWeight: 700,
                      fontSize: '1.1rem',
                      letterSpacing: '-0.01em',
                    }}>
                      {title}
                    </div>
                  </div>
                  <p className="text-muted mb-0" style={{ fontSize: '.88rem', lineHeight: 1.55 }}>
                    {desc}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
