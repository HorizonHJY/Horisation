import React, { useState, useRef } from 'react'
import { useAuth } from '../App'
import { api } from '../api'

const ROLE_COLORS = {
  horizon: 'role-horizon', horizonadmin: 'role-horizonadmin',
  vip1: 'role-vip1', vip2: 'role-vip2', vip3: 'role-vip3', user: 'role-user',
}

export default function Profile() {
  const { user, login } = useAuth()

  const [nameForm, setNameForm]     = useState({ display_name: user?.display_name ?? '', email: user?.email ?? '' })
  const [passForm, setPassForm]     = useState({ current_password: '', new_password: '', confirm: '' })
  const [contactForm, setContactForm]     = useState(user?.contact_info ?? '')
  const [contactHidden, setContactHidden] = useState(user?.contact_hidden ?? false)
  const [savingName, setSavingName] = useState(false)
  const [savingPass, setSavingPass] = useState(false)
  const [savingContact, setSavingContact] = useState(false)
  const [uploadingAvatar, setUploadingAvatar] = useState(false)
  const [avatarPreview, setAvatarPreview] = useState(null)
  const [msg, setMsg] = useState(null)
  const fileRef = useRef()

  if (!user) return null

  const flash = (text, type = 'success') => {
    setMsg({ text, type })
    setTimeout(() => setMsg(null), 3000)
  }

  // ── Display name + email ───────────────────────────────────────────
  const saveName = async (e) => {
    e.preventDefault()
    if (!nameForm.display_name.trim()) return flash('Display name cannot be empty.', 'danger')
    setSavingName(true)
    const d = await api.put('/api/auth/profile', {
      display_name: nameForm.display_name.trim(),
      email:        nameForm.email.trim(),
    })
    setSavingName(false)
    if (d.ok) {
      flash('Profile updated.')
      login({ ...user, display_name: nameForm.display_name.trim(), email: nameForm.email.trim() })
    } else {
      flash(d.error, 'danger')
    }
  }

  // ── Password ───────────────────────────────────────────────────────
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

  // ── Avatar ─────────────────────────────────────────────────────────
  const onAvatarPick = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setAvatarPreview(URL.createObjectURL(file))
  }

  const uploadAvatar = async () => {
    const file = fileRef.current?.files[0]
    if (!file) return
    setUploadingAvatar(true)
    const fd = new FormData()
    fd.append('avatar', file)
    const d = await api.upload('/api/auth/avatar', fd)
    setUploadingAvatar(false)
    if (d.ok) {
      flash('Avatar updated.')
      login({ ...user, avatar_url: d.avatar_url })
      setAvatarPreview(null)
      fileRef.current.value = ''
    } else {
      flash(d.error, 'danger')
    }
  }

  const currentAvatar = avatarPreview ?? user.avatar_url

  return (
    <>
      <h4 className="fw-bold mb-4">My Profile</h4>

      {msg && <div className={`alert alert-${msg.type} py-2`}>{msg.text}</div>}

      <div className="row g-4">
        {/* Avatar card */}
        <div className="col-12 col-md-4">
          <div className="card text-center p-4">
            {/* Avatar display */}
            {currentAvatar ? (
              <img
                src={currentAvatar}
                alt="avatar"
                className="rounded-circle mx-auto mb-3"
                style={{ width: 80, height: 80, objectFit: 'cover' }}
              />
            ) : (
              <div
                className="rounded-circle mx-auto mb-3 d-flex align-items-center justify-content-center"
                style={{ width: 80, height: 80, background: '#3a7bd5', color: '#fff', fontSize: '2rem', fontWeight: 700 }}
              >
                {user.display_name?.[0]?.toUpperCase()}
              </div>
            )}

            <h5 className="fw-bold mb-1">{user.display_name}</h5>
            <span className={`role-badge ${ROLE_COLORS[user.role] ?? 'role-user'}`}>
              {user.role_info?.name ?? user.role}
            </span>
            <div className="mt-3 text-muted small">Permission level: {user.role_info?.level}</div>

            {/* Avatar upload */}
            <div className="mt-3 d-flex flex-column gap-2">
              <input
                ref={fileRef}
                type="file"
                accept=".jpg,.jpeg,.png"
                className="form-control form-control-sm"
                onChange={onAvatarPick}
              />
              {avatarPreview && (
                <button
                  className="btn btn-sm btn-primary"
                  onClick={uploadAvatar}
                  disabled={uploadingAvatar}
                >
                  {uploadingAvatar
                    ? <span className="spinner-border spinner-border-sm" />
                    : 'Upload Avatar'}
                </button>
              )}
              <div className="text-muted" style={{ fontSize: '0.75rem' }}>JPEG/PNG, max 2MB</div>
            </div>
          </div>
        </div>

        {/* Right column */}
        <div className="col-12 col-md-8 d-flex flex-column gap-4">

          {/* Account details + edit name/email */}
          <div className="card p-4">
            <h6 className="fw-semibold mb-3 text-muted text-uppercase" style={{ fontSize: '.75rem', letterSpacing: '.08em' }}>
              Account Details
            </h6>
            <div className="d-flex justify-content-between py-2 border-bottom">
              <span className="text-muted">Username</span>
              <span className="fw-semibold">{user.username}</span>
            </div>
            <div className="d-flex justify-content-between py-2 border-bottom">
              <span className="text-muted">Role</span>
              <span className="fw-semibold">{user.role}</span>
            </div>

            <form onSubmit={saveName} className="mt-3 d-flex flex-column gap-3">
              <div>
                <label className="form-label fw-medium">Display Name</label>
                <input
                  type="text"
                  className="form-control"
                  value={nameForm.display_name}
                  onChange={e => setNameForm(f => ({ ...f, display_name: e.target.value }))}
                  required
                />
              </div>
              <div>
                <label className="form-label fw-medium">Email</label>
                <input
                  type="email"
                  className="form-control"
                  value={nameForm.email}
                  onChange={e => setNameForm(f => ({ ...f, email: e.target.value }))}
                  placeholder="email@example.com"
                />
              </div>
              <button className="btn btn-primary align-self-start" disabled={savingName}>
                {savingName ? <span className="spinner-border spinner-border-sm me-2" /> : null}
                Save Changes
              </button>
            </form>

            {(user.role_info?.level ?? 0) > 60 && (
              <>
                <h6 className="fw-semibold mt-4 mb-2 text-muted text-uppercase" style={{ fontSize: '.75rem', letterSpacing: '.08em' }}>
                  Permissions
                </h6>
                <div className="d-flex flex-wrap gap-2">
                  {user.role_info?.permissions?.map(p => (
                    <span key={p} className="badge bg-primary bg-opacity-10 text-primary fw-normal">{p}</span>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Contact info */}
          <div className="card p-4">
            <h6 className="fw-semibold mb-3 text-muted text-uppercase" style={{ fontSize: '.75rem', letterSpacing: '.08em' }}>
              Contact Info
            </h6>
            <p className="text-muted small mb-2">
              Friends can request to see your contact info. You must approve each request.
            </p>
            <form onSubmit={async (e) => {
              e.preventDefault()
              setSavingContact(true)
              const d = await api.put('/api/auth/profile', { contact_info: contactForm })
              setSavingContact(false)
              if (d.ok) flash('Contact info saved.')
              else flash(d.error, 'danger')
            }} className="d-flex gap-2 mb-3">
              <input
                type="text"
                className="form-control"
                placeholder="WeChat ID / phone / etc."
                value={contactForm}
                onChange={e => setContactForm(e.target.value)}
                maxLength={100}
              />
              <button className="btn btn-primary flex-shrink-0" disabled={savingContact}>
                {savingContact ? <span className="spinner-border spinner-border-sm" /> : 'Save'}
              </button>
            </form>
            <div className="form-check form-switch">
              <input
                className="form-check-input"
                type="checkbox"
                role="switch"
                id="hideContact"
                checked={contactHidden}
                onChange={async e => {
                  const hidden = e.target.checked
                  setContactHidden(hidden)
                  const d = await api.put('/api/auth/profile', { contact_hidden: hidden })
                  if (d.ok) flash(hidden ? 'Contact hidden from friends.' : 'Contact visible to friends.')
                  else { setContactHidden(!hidden); flash(d.error, 'danger') }
                }}
              />
              <label className="form-check-label small" htmlFor="hideContact">
                Hide my contact info (friends cannot request it while hidden)
              </label>
            </div>
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
                autoComplete="current-password"
                required
              />
              <input
                type="password"
                className="form-control"
                placeholder="New password (min 6 characters)"
                value={passForm.new_password}
                onChange={e => setPassForm(f => ({ ...f, new_password: e.target.value }))}
                autoComplete="new-password"
                required
              />
              <input
                type="password"
                className="form-control"
                placeholder="Confirm new password"
                value={passForm.confirm}
                onChange={e => setPassForm(f => ({ ...f, confirm: e.target.value }))}
                autoComplete="new-password"
                required
              />
              <button className="btn btn-primary align-self-start" disabled={savingPass}>
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
