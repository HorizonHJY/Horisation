import React, { useEffect, useLayoutEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../App'
import { api } from '../api'
import { validatePassword } from '../utils'

const ROLE_COLORS = {
  horizon: 'role-horizon', horizonadmin: 'role-horizonadmin',
  vip1: 'role-vip1', vip2: 'role-vip2', vip3: 'role-vip3', user: 'role-user',
}

export default function Profile() {
  const { user, login, logout } = useAuth()

  const [nameForm, setNameForm]     = useState({ display_name: user?.display_name ?? '', email: user?.email ?? '' })
  const [passForm, setPassForm]     = useState({ current_password: '', new_password: '', confirm: '' })
  const [contactForm, setContactForm]     = useState({ phone: user?.phone ?? '', wechat: user?.wechat ?? '', address: user?.address ?? '', postal_code: user?.postal_code ?? '' })
  const [savingName, setSavingName] = useState(false)
  const [savingPass, setSavingPass] = useState(false)
  const [showPass, setShowPass] = useState(false)
  const [badgeOpen, setBadgeOpen] = useState(false)
  const [uploadingAvatar, setUploadingAvatar] = useState(false)
  const [avatarPreview, setAvatarPreview] = useState(null)
  const [msg, setMsg] = useState(null)
  const fileRef = useRef()
  const [mbti, setMbti] = useState(null)
  const [mbtiState, setMbtiState] = useState('loading')  // loading | ok | empty | error
  const [savingWear, setSavingWear] = useState(false)
  const [wearError, setWearError] = useState(null)
  const badgeBtnRef = useRef(null)
  const badgePanelRef = useRef(null)

  const loadMbti = () => {
    setMbtiState('loading')
    api.get('/api/mbti/me')
      .then(d => {
        if (d.ok && d.result) {
          setMbti({ ...d.result, type_card: d.type_card })
          setMbtiState('ok')
        } else {
          setMbtiState('empty')
        }
      })
      .catch(() => setMbtiState('error'))
  }

  useEffect(() => { loadMbti() }, [])

  // Close the picker on outside click / Escape, and hand focus back to the button.
  useEffect(() => {
    if (!badgeOpen) return
    const onDown = (e) => {
      if (badgePanelRef.current?.contains(e.target)) return
      if (badgeBtnRef.current?.contains(e.target)) return
      setBadgeOpen(false)
    }
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setBadgeOpen(false)
        badgeBtnRef.current?.focus()
      }
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [badgeOpen])

  // Keep the panel inside the viewport. On a narrow screen the button sits far
  // enough right that a left-anchored 220px panel would run off the edge.
  useLayoutEffect(() => {
    const el = badgePanelRef.current
    if (!badgeOpen || !el) return
    el.style.transform = ''
    const r = el.getBoundingClientRect()
    const margin = 12
    const overRight = r.right - (window.innerWidth - margin)
    const overLeft = margin - r.left
    if (overRight > 0) el.style.transform = `translateX(${-overRight}px)`
    else if (overLeft > 0) el.style.transform = `translateX(${overLeft}px)`
  }, [badgeOpen, mbtiState])

  const VISIBILITY_LABEL = { public: 'Public', friends: 'Friends', private: 'Only me' }

  // The backend has three levels (private | friends | public). Picking one is
  // explicit — never derived from a two-state toggle, which used to overwrite
  // "friends" the first time somebody touched the badge.
  const setVisibility = async (v) => {
    if (!mbti || savingWear) return
    setWearError(null)
    if (v === mbti.visibility) {
      setBadgeOpen(false)
      return
    }
    const previous = mbti.visibility
    setSavingWear(true)
    setMbti(prev => ({ ...prev, visibility: v }))          // optimistic
    const d = await api.patch('/api/mbti/me/visibility', { visibility: v })
    setSavingWear(false)
    if (!d.ok) {
      setMbti(prev => ({ ...prev, visibility: previous })) // roll back
      setWearError(d.error || 'Could not save. Try again.')
      return
    }
    setBadgeOpen(false)
    flash('Badge visibility saved.', 'success')
  }

  if (!user) return null

  const flash = (text, type = 'success') => {
    setMsg({ text, type })
    setTimeout(() => setMsg(null), 3000)
  }

  // ── Display name + email ───────────────────────────────────────────
  const saveAccount = async (e) => {
    e.preventDefault()
    if (!nameForm.display_name.trim()) return flash('Display name cannot be empty.', 'danger')
    setSavingName(true)
    const d = await api.put('/api/auth/profile', {
      display_name: nameForm.display_name.trim(),
      email:        nameForm.email.trim(),
      phone:        contactForm.phone,
      wechat:       contactForm.wechat,
      address:      contactForm.address,
      postal_code:  contactForm.postal_code,
    })
    setSavingName(false)
    if (d.ok) {
      flash('Account info saved.')
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
    const pwErr = validatePassword(passForm.new_password)
    if (pwErr) return flash(pwErr, 'danger')
    setSavingPass(true)
    const d = await api.put('/api/auth/password', {
      current_password: passForm.current_password,
      new_password:     passForm.new_password,
    })
    setSavingPass(false)
    if (d.ok) {
      flash('Password changed. Logging you out…')
      setTimeout(() => logout(), 1500)
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
      <h4 className='fw-bold mb-4'>My Profile</h4>

      {msg && <div className={'alert alert-' + msg.type + ' py-2'}>{msg.text}</div>}

      {/* Header — avatar + name + MBTI, one row */}
      <div className='card p-4 mb-4'>
        <div className='d-flex align-items-center gap-3 flex-wrap'>
          <label className='profile-avatar' title='Change photo'>
            <input ref={fileRef} type='file' accept='.jpg,.jpeg,.png' style={{ display: 'none' }} onChange={onAvatarPick} />
            {currentAvatar ? (
              <img src={currentAvatar} alt='avatar' style={{ width: 80, height: 80, objectFit: 'cover' }} />
            ) : (
              <div className='d-flex align-items-center justify-content-center' style={{ width: 80, height: 80, background: '#6b9cdb', color: '#fff', fontSize: '2rem', fontWeight: 700 }}>
                {user.display_name?.[0]?.toUpperCase()}
              </div>
            )}
            <span className='profile-avatar__overlay'>Change</span>
          </label>

          <div className='flex-grow-1'>
            <h5 className='fw-bold mb-1'>{user.display_name}</h5>
            <div className='d-flex align-items-center gap-2 flex-wrap'>
              {mbti && mbti.visibility !== 'private' && (
                <span className='mbti-badge mbti-badge--sm' title={mbti.type_card?.nick || mbti.type}>{mbti.type}</span>
              )}
              <div className='position-relative'>
                <button
                  ref={badgeBtnRef}
                  type='button'
                  className='btn btn-sm btn-outline-secondary'
                  onClick={() => setBadgeOpen(v => !v)}
                  aria-expanded={badgeOpen}
                  aria-haspopup='menu'
                  aria-controls={badgeOpen ? 'badge-picker-menu' : undefined}
                >
                  <i className='fas fa-plus me-1' />Badges
                </button>
                {badgeOpen && (
                  <div className='badge-picker' id='badge-picker-menu' role='menu' aria-label='Badges' ref={badgePanelRef}>
                    {mbtiState === 'loading' && <div className='badge-picker__hint'>Loading…</div>}

                    {mbtiState === 'error' && (
                      <div className='badge-picker__hint'>
                        Could not load badges.{' '}
                        <button type='button' className='badge-picker__link' onClick={loadMbti}>Retry</button>
                      </div>
                    )}

                    {mbtiState === 'empty' && (
                      <div className='badge-picker__hint'>
                        No badges yet.{' '}
                        <Link to='/mbti' className='badge-picker__link' onClick={() => setBadgeOpen(false)}>Take the MBTI test</Link>
                      </div>
                    )}

                    {mbtiState === 'ok' && mbti && (
                      <>
                        <div className='badge-picker__title'>
                          <span className='mbti-badge mbti-badge--sm'>{mbti.type}</span>
                          <span>{mbti.type_card?.nick || mbti.type}</span>
                        </div>
                        <div className='badge-picker__label'>Who can see my MBTI</div>
                        {['public', 'friends', 'private'].map(v => (
                          <button
                            key={v}
                            type='button'
                            role='menuitemradio'
                            aria-checked={mbti.visibility === v}
                            className='badge-picker__item'
                            disabled={savingWear}
                            onClick={() => setVisibility(v)}
                          >
                            <i className={'fas ' + (mbti.visibility === v ? 'fa-check-square text-primary' : 'fa-square text-muted')} />
                            <span className='fw-medium'>{VISIBILITY_LABEL[v]}</span>
                          </button>
                        ))}
                        {wearError && <div className='badge-picker__error' role='alert'>{wearError}</div>}
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {avatarPreview && (
            <div className='d-flex gap-2'>
              <button className='btn btn-sm btn-primary' onClick={uploadAvatar} disabled={uploadingAvatar}>
                {uploadingAvatar ? <span className='spinner-border spinner-border-sm' /> : 'Save Avatar'}
              </button>
              <button className='btn btn-sm btn-outline-secondary' onClick={() => { setAvatarPreview(null); fileRef.current.value = '' }}>
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Forms — single column */}
      <div className='d-flex flex-column gap-4'>

        {/* Account info */}
        <div className='card p-4'>
          <h6 className='fw-semibold mb-3 text-muted'>Account Info</h6>
          <form onSubmit={saveAccount} className='d-flex flex-column gap-3'>
            <div className='row g-3'>
              <div className='col-sm-6'>
                <label className='form-label fw-medium'>Display Name</label>
                <input type='text' className='form-control' value={nameForm.display_name} onChange={e => setNameForm(f => ({ ...f, display_name: e.target.value }))} required />
              </div>
              <div className='col-sm-6'>
                <label className='form-label fw-medium'>Email</label>
                <input type='email' className='form-control' value={nameForm.email} onChange={e => setNameForm(f => ({ ...f, email: e.target.value }))} placeholder='email@example.com' />
              </div>
              <div className='col-sm-6'>
                <label className='form-label fw-medium'>Phone</label>
                <input type='text' className='form-control' placeholder='+86 138 0000 0000' value={contactForm.phone} onChange={e => setContactForm(f => ({ ...f, phone: e.target.value }))} maxLength={50} />
              </div>
              <div className='col-sm-6'>
                <label className='form-label fw-medium'>WeChat ID</label>
                <input type='text' className='form-control' placeholder='WeChat ID' value={contactForm.wechat} onChange={e => setContactForm(f => ({ ...f, wechat: e.target.value }))} maxLength={100} />
              </div>
              <div className='col-sm-8'>
                <label className='form-label fw-medium'>Address</label>
                <input type='text' className='form-control' placeholder='Street, City' value={contactForm.address} onChange={e => setContactForm(f => ({ ...f, address: e.target.value }))} maxLength={200} />
              </div>
              <div className='col-sm-4'>
                <label className='form-label fw-medium'>Postal Code</label>
                <input type='text' className='form-control' placeholder='000000' value={contactForm.postal_code} onChange={e => setContactForm(f => ({ ...f, postal_code: e.target.value }))} maxLength={20} />
              </div>
            </div>
            <button className='btn btn-primary align-self-start' disabled={savingName}>
              {savingName ? <span className='spinner-border spinner-border-sm me-2' /> : null}
              Save Changes
            </button>
          </form>
        </div>

        {/* Change password — collapsed */}
        <div className='card p-4'>
          <button type='button' className='btn btn-link p-0 text-decoration-none fw-semibold' onClick={() => setShowPass(v => !v)} aria-expanded={showPass}>
            <i className='fas fa-lock me-2' />Change Password
            <i className={'fas fa-chevron-' + (showPass ? 'up' : 'down') + ' ms-2'} />
          </button>
          {showPass && (
            <form onSubmit={savePassword} className='mt-3 d-flex flex-column gap-3'>
              <input type='password' className='form-control' placeholder='Current password' value={passForm.current_password} onChange={e => setPassForm(f => ({ ...f, current_password: e.target.value }))} autoComplete='current-password' required />
              <input type='password' className='form-control' placeholder='New password (min 8 chars, A-Z, a-z, 0-9)' value={passForm.new_password} onChange={e => setPassForm(f => ({ ...f, new_password: e.target.value }))} autoComplete='new-password' required />
              <input type='password' className='form-control' placeholder='Confirm new password' value={passForm.confirm} onChange={e => setPassForm(f => ({ ...f, confirm: e.target.value }))} autoComplete='new-password' required />
              <button className='btn btn-primary align-self-start' disabled={savingPass}>
                {savingPass ? <span className='spinner-border spinner-border-sm me-2' /> : null}
                Change Password
              </button>
            </form>
          )}
        </div>

      </div>
    </>
  )
}