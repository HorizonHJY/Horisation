import React from 'react'
import { useAuth } from '../App'

export default function Profile() {
  const { user } = useAuth()
  if (!user) return null

  const roleColors = {
    horizon: 'role-horizon', horizonadmin: 'role-horizonadmin',
    vip1: 'role-vip1', vip2: 'role-vip2', vip3: 'role-vip3', user: 'role-user',
  }

  return (
    <>
      <h4 className="fw-bold mb-4">My Profile</h4>
      <div className="row g-4">
        {/* Avatar card */}
        <div className="col-12 col-md-4">
          <div className="card text-center p-4">
            <div
              className="rounded-circle mx-auto mb-3 d-flex align-items-center justify-content-center"
              style={{ width: 80, height: 80, background: '#3a7bd5', color: '#fff', fontSize: '2rem', fontWeight: 700 }}
            >
              {user.display_name?.[0]?.toUpperCase()}
            </div>
            <h5 className="fw-bold mb-1">{user.display_name}</h5>
            <span className={`role-badge ${roleColors[user.role] ?? 'role-user'}`}>
              {user.role_info?.name ?? user.role}
            </span>
            <div className="mt-3 text-muted small">
              Permission level: {user.role_info?.level}
            </div>
          </div>
        </div>

        {/* Info card */}
        <div className="col-12 col-md-8">
          <div className="card p-4">
            <h6 className="fw-semibold mb-3 text-muted text-uppercase" style={{ fontSize: '.75rem', letterSpacing: '.08em' }}>
              Account Details
            </h6>
            {[
              ['Username',   user.username],
              ['Email',      user.email || '—'],
              ['Role',       user.role],
            ].map(([label, value]) => (
              <div key={label} className="d-flex justify-content-between py-2 border-bottom">
                <span className="text-muted">{label}</span>
                <span className="fw-semibold">{value}</span>
              </div>
            ))}

            <h6 className="fw-semibold mt-4 mb-2 text-muted text-uppercase" style={{ fontSize: '.75rem', letterSpacing: '.08em' }}>
              Permissions
            </h6>
            <div className="d-flex flex-wrap gap-2">
              {user.role_info?.permissions?.map(p => (
                <span key={p} className="badge bg-primary bg-opacity-10 text-primary fw-normal">
                  {p}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
