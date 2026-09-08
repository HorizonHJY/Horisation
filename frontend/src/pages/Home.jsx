import React from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../App'
import WeatherGreeting from '../components/WeatherGreeting'

const FEATURES = [
  { icon: 'fa-clipboard-list', color: '#6b9cdb', title: 'Memo',        desc: 'Personal memo and task tracker.',        to: '/hormemo'  },
  { icon: 'fa-store',          color: '#27ae60', title: 'Market',          desc: 'Browse and post second-hand listings.',  to: '/market'   },
  { icon: 'fa-comments',       color: '#d97706', title: 'Message Board',   desc: 'Chat and share with everyone.',          to: '/feedback' },
]

// The flower canvas belongs to the sign-in screens. Home is a working
// dashboard and reads better without a decorative layer behind it, so the
// relative/z-index wrappers that existed only to stack content above that
// canvas are gone too — nothing here needs a stacking context any more.
export default function Home() {
  const { user } = useAuth()

  return (
    <div>
      <WeatherGreeting name={user?.display_name} />

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
            <Link
              to={to}
              className="card card-hover h-100 text-decoration-none"
              style={{ cursor: 'pointer', background: 'var(--bg-surface)' }}
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
                    color: 'var(--text-primary)',
                  }}>
                    {title}
                  </div>
                </div>
                <p className="text-muted mb-0" style={{ fontSize: '.88rem', lineHeight: 1.55 }}>
                  {desc}
                </p>
              </div>
            </Link>
          </div>
        ))}
      </div>
    </div>
  )
}
