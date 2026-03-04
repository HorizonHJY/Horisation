import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../App'

const FEATURES = [
  { icon: 'fa-clipboard-list', color: '#3a7bd5', title: 'Hormemo',       desc: 'Personal memo and task tracker.',        to: '/hormemo'  },
  { icon: 'fa-store',          color: '#27ae60', title: 'Market',         desc: 'Browse and post second-hand listings.',  to: '/market'   },
  { icon: 'fa-comments',       color: '#e67e22', title: 'Message Board',  desc: 'Chat and share with everyone.',          to: '/feedback' },
]

export default function Home() {
  const { user } = useAuth()
  const navigate = useNavigate()

  return (
    <>
      <div
        className="rounded-3 text-white mb-4 p-4 p-md-5"
        style={{ background: 'linear-gradient(135deg, #1e2a3a 0%, #3a7bd5 100%)' }}
      >
        <h2 className="fw-bold mb-0">Hello, {user?.display_name} 👋</h2>
      </div>

      <h5 className="fw-semibold mb-3 text-muted">Quick Access</h5>
      <div className="row g-3">
        {FEATURES.map(({ icon, color, title, desc, to }) => (
          <div key={title} className="col-12 col-sm-6 col-lg-4">
            <div className="card h-100" style={{ cursor: 'pointer' }} onClick={() => navigate(to)}>
              <div className="card-body p-4">
                <div className="d-flex align-items-center gap-3 mb-2">
                  <div
                    className="rounded-circle d-flex align-items-center justify-content-center"
                    style={{ width: 42, height: 42, background: color + '1a', flexShrink: 0 }}
                  >
                    <i className={`fas ${icon}`} style={{ color }} />
                  </div>
                  <div className="fw-semibold">{title}</div>
                </div>
                <p className="text-muted mb-0" style={{ fontSize: '.88rem' }}>{desc}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  )
}
