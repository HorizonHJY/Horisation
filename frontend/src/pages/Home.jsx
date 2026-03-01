import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../App'

const FEATURES = [
  { icon: 'fa-clipboard-list', color: '#3a7bd5', title: 'Hormemo',       desc: 'Personal memo and task tracker.',   to: '/hormemo',           ready: true  },
  { icon: 'fa-file-csv',       color: '#27ae60', title: 'CSV Workspace', desc: 'Upload, preview and analyse data.', to: '/csv',               ready: true  },
  { icon: 'fa-chess-board',    color: '#8e44ad', title: 'Gomoku',        desc: 'Local 2-player Five in a Row.',     to: '/fun/gomoku',        ready: true  },
  { icon: 'fa-chart-bar',      color: '#e67e22', title: 'Data Analysis', desc: 'Advanced analytics tools.',         to: '/under-development', ready: false },
  { icon: 'fa-chart-line',     color: '#e74c3c', title: 'Visualisation', desc: 'Charts and dashboards.',            to: '/under-development', ready: false },
  { icon: 'fa-sticky-note',    color: '#16a085', title: 'Notes',         desc: 'Private markdown notebook.',        to: '/under-development', ready: false },
]

export default function Home() {
  const { user } = useAuth()
  const navigate = useNavigate()

  return (
    <>
      {/* Hero */}
      <div
        className="rounded-3 text-white mb-4 p-4 p-md-5"
        style={{ background: 'linear-gradient(135deg, #1e2a3a 0%, #3a7bd5 100%)' }}
      >
        <h2 className="fw-bold mb-1">
          Hello, {user?.display_name} 👋
        </h2>
        <p className="mb-3 opacity-75">Welcome to your personal platform. What are you working on today?</p>
        <button
          className="btn btn-light btn-sm fw-semibold"
          onClick={() => navigate('/hormemo')}
        >
          <i className="fas fa-clipboard-list me-2" />Open Hormemo
        </button>
      </div>

      {/* Feature grid */}
      <h5 className="fw-semibold mb-3 text-muted">Available Features</h5>
      <div className="row g-3">
        {FEATURES.map(({ icon, color, title, desc, to, ready }) => (
          <div key={title} className="col-12 col-sm-6 col-xl-4">
            <div
              className={`card h-100 ${ready ? '' : 'opacity-75'}`}
              style={{ cursor: ready ? 'pointer' : 'default' }}
              onClick={() => navigate(to)}
            >
              <div className="card-body p-4">
                <div className="d-flex align-items-center gap-3 mb-2">
                  <div
                    className="rounded-circle d-flex align-items-center justify-content-center"
                    style={{ width: 42, height: 42, background: color + '1a' }}
                  >
                    <i className={`fas ${icon}`} style={{ color }} />
                  </div>
                  <div>
                    <div className="fw-semibold">{title}</div>
                    {!ready && (
                      <span className="badge bg-warning text-dark" style={{ fontSize: '.65rem' }}>
                        Under Development
                      </span>
                    )}
                  </div>
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
