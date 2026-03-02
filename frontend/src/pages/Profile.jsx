import React, { useState } from 'react'
import { useAuth } from '../App'
import { api } from '../api'

const ROLE_COLORS = {
  horizon: 'role-horizon', horizonadmin: 'role-horizonadmin',
  vip1: 'role-vip1', vip2: 'role-vip2', vip3: 'role-vip3', user: 'role-user',
}

export default function Profile() {
  const { user, login } = useAuth()

  const [nameForm, setNameForm]   = useState({ display_name: user?.display_name ?? '' })
  const [passForm, setPassForm]   = useState({ current_password: '', new_password: '', confirm: '' })
  const [savingName, setSavingName] = useState(false)
  const [savingPass, setSavingPass] = useState(false)
  const [msg, setMsg]             = useState(null)

  if (!user) return null

  const flash = (text, type = 'success') => {
    setMsg({ text, type })
    setTimeout(() => setMsg(null), 3000)
  }

  const saveName = async (e) => {
    e.preventDefault()
    if (!nameForm.display_name.trim()) return flash('Display name cannot be empty.', 'danger')
    setSavingName(true)
    const d = await api.put('/api/auth/profile', { display_name: nameForm.display_name.trim() })
    setSavingName(false)
    if (d.ok) {
      flash('Display name updated.')
      // Update AuthContext so topbar/sidebar reflect the change immediately
      login({ ...user, display_name: nameForm.display_name.trim() })
    } else {
      flash(d.error, 'danger')
    }
  }

  const savePassword = async (e) => {
    e.preventDefault()
    if (passForm.new_password !== passForm.confirm)
      return flash('New passwords do not match.', 'danger')
    if (passForm.new_password.length < 6)
      return flash('New password must be at least 6 characters.', 'danger')
    setSavingPass(true)
    const d = await api.put('/api/auth/password', {
      current_password: passForm.current_password,
      new_password:     passForm.new_password,
    })
    setSavingPass(false)
    if (d.ok) {
      flash('Password changed successfully.')
      setPassForm({ current_password: '', new_password: '', confirm: '' })
    } else {
      flash(d.error, 'danger')
    }
  }

  return (
    <>
      <h4 className="fw-bold mb-4">My Profile</h4>

      {msg && (
        <div className={`alert alert-${msg.type} py-2`}>{msg.text}</div>
      )}

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
            <span className={`role-badge ${ROLE_COLORS[user.role] ?? 'role-user'}`}>
              {user.role_info?.name ?? user.role}
            </span>
            <div className="mt-3 text-muted small">
              Permission level: {user.role_info?.level}
            </div>
          </div>
        </div>

        {/* Right column */}
        <div className="col-12 col-md-8 d-flex flex-column gap-4">

          {/* Account details */}
          <div className="card p-4">
            <h6 className="fw-semibold mb-3 text-muted text-uppercase" style={{ fontSize: '.75rem', letterSpacing: '.08em' }}>
              Account Details
            </h6>
            {[
              ['Username', user.username],
              ['Email',    user.email || '—'],
              ['Role',     user.role],
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
                <span key={p} className="badge bg-primary bg-opacity-10 text-primary fw-normal">{p}</span>
              ))}
            </div>
          </div>

          {/* Change display name */}
          <div className="card p-4">
            <h6 className="fw-semibold mb-3 text-muted text-uppercase" style={{ fontSize: '.75rem', letterSpacing: '.08em' }}>
              Change Display Name
            </h6>
            <form onSubmit={saveName} className="d-flex gap-2">
              <input
                type="text"
                className="form-control"
                value={nameForm.display_name}
                onChange={e => setNameForm({ display_name: e.target.value })}
                placeholder="New display name"
              />
              <button className="btn btn-primary flex-shrink-0" disabled={savingName}>
                {savingName ? <span className="spinner-border spinner-border-sm" /> : 'Save'}
              </button>
            </form>
          </div>

          {/* Change password */}
          <div className="card p-4">
            <h6 className="fw-semibold mb-3 text-muted text-uppercase" style={{ fontSize: '.75rem', letterSpacing: '.08em' }}>
              Change Password
            </h6>
            <form onSubmit={savePassword} className="d-flex flex-column gap-3">
              <input
                type="password"
                className="form-control"
                placeholder="Current password"
                value={passForm.current_password}
                onChange={e => setPassForm(f => ({ ...f, current_password: e.target.value }))}
                required
              />
              <input
                type="password"
                className="form-control"
                placeholder="New password (min 6 characters)"
                value={passForm.new_password}
                onChange={e => setPassForm(f => ({ ...f, new_password: e.target.value }))}
                required
              />
              <input
                type="password"
                className="form-control"
                placeholder="Confirm new password"
                value={passForm.confirm}
                onChange={e => setPassForm(f => ({ ...f, confirm: e.target.value }))}
                required
              />
              <button className="btn btn-primary" disabled={savingPass}>
                {savingPass ? <span className="spinner-border spinner-border-sm me-2" /> : null}
                Change Password
              </button>
            </form>
          </div>

        </div>
      </div>
    </>
  )
}
