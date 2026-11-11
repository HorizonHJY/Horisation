import React, { useEffect, useLayoutEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../App'
import { api } from '../api'
import { validatePassword } from '../utils'

// Keep this list in sync with Backend/Controller/auth_controller.py :: AVATAR_PRESETS
// and the SVG files in public/avatars/.
const AVATAR_PRESETS = [
  { id: 'fox',   label: 'Fox' },
  { id: 'cat',   label: 'Cat' },
  { id: 'bear',  label: 'Bear' },
  { id: 'bunny', label: 'Bunny' },
  { id: 'owl',   label: 'Owl' },
  { id: 'fish',  label: 'Fish' },
  { id: 'leaf',  label: 'Leaf' },
  { id: 'moon',  label: 'Moon' },
  { id: 'bloom', label: 'Bloom' },
  { id: 'star',  label: 'Star' },
]

/** The level bar under the profile header: total, a bar, and the gap to the
 *  next title. 满级 just says so — there is no bar to fill. */
function XpProgress({ xp }) {
  const full = xp.next_xp == null
  const pct  = Math.round((full ? 1 : xp.progress) * 100)
  const gap  = full ? 0 : Math.max(0, xp.next_xp - xp.xp)
  return (
    <div className='xp-card'>
      <div className='xp-card__row'>
        <span><span className='xp-card__total'>{xp.xp}</span> 经验</span>
        <span>{full ? '已满级' : `下一级 · ${xp.next_level_name} · ${xp.next_xp}`}</span>
      </div>
      <div
        className={'xp-bar' + (full ? ' xp-bar--full' : '')}
        role='progressbar'
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}
        aria-label={full ? '已满级' : `距离 ${xp.next_level_name} 还差 ${gap} 经验`}
      >
        <div className='xp-bar__fill' style={{ width: `${pct}%` }} />
      </div>
      <div className='xp-card__hint'>
        {full
          ? <>已满级 · 村里最高的称号了</>
          : <>距离下一级还差 <strong>{gap}</strong> 经验</>}
        {xp.streak_days > 0 && <> · 🔥 连续签到 {xp.streak_days} 天</>}
      </div>
    </div>
  )
}

const ROLE_COLORS = {
  horizon: 'role-horizon', horizonadmin: 'role-horizonadmin',
  vip1: 'role-vip1', vip2: 'role-vip2', vip3: 'role-vip3', user: 'role-user',
}

export default function Profile() {
  const { user, login, logout } = useAuth()

  const [nameForm, setNameForm]     = useState({ display_name: user?.display_name ?? '', email: user?.email ?? '', bio: user?.bio ?? '' })
  const [passForm, setPassForm]     = useState({ current_password: '', new_password: '', confirm: '' })
  const [contactForm, setContactForm]     = useState({ phone: user?.phone ?? '', wechat: user?.wechat ?? '', address: user?.address ?? '', postal_code: user?.postal_code ?? '' })
  const [savingName, setSavingName] = useState(false)
  const [savingPass, setSavingPass] = useState(false)
  const [showPass, setShowPass] = useState(false)
  const [badgeOpen, setBadgeOpen] = useState(false)
  const [uploadingAvatar, setUploadingAvatar] = useState(false)
  const [avatarPreview, setAvatarPreview] = useState(null)
  const [avatarOpen, setAvatarOpen] = useState(false)
  const [savingPreset, setSavingPreset] = useState(null)   // preset id while saving
  const [msg, setMsg] = useState(null)
  const fileRef = useRef()
  const avatarBtnRef = useRef(null)
  const avatarPanelRef = useRef(null)
  const contactSectionRef = useRef(null)
  const [mbti, setMbti] = useState(null)
  const [mbtiState, setMbtiState] = useState('loading')  // loading | ok | empty | error
  const [savingWear, setSavingWear] = useState(false)
  const [wearError, setWearError] = useState(null)
  const badgeBtnRef = useRef(null)
  const badgePanelRef = useRef(null)
  const [xp, setXp] = useState(null)   // GET /api/auth/xp; null until it lands or if it fails

  useEffect(() => {
    let cancelled = false
    api.get('/api/auth/xp').then(d => { if (!cancelled && d.ok) setXp(d) })
    return () => { cancelled = true }
  }, [])

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

  // Jumping here from the Friends page (#contact) should land the eye on
  // the phone / WeChat fields, not on the avatar at the top.
  useEffect(() => {
    if (typeof window === 'undefined') return
    if (window.location.hash !== '#contact') return
    const t = setTimeout(() => {
      contactSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 50)
    return () => clearTimeout(t)
  }, [])

  const contactEmpty = !(contactForm.phone || '').trim() && !(contactForm.wechat || '').trim()
  const scrollToContact = () => {
    contactSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    try { history.replaceState(null, '', '#contact') } catch { /* ignore */ }
  }

  // Close the avatar picker on outside click / Escape; hand focus back.
  useEffect(() => {
    if (!avatarOpen) return
    const onDown = (e) => {
      if (avatarPanelRef.current?.contains(e.target)) return
      if (avatarBtnRef.current?.contains(e.target)) return
      setAvatarOpen(false)
    }
    const onKey = (e) => {
      if (e.key === 'Escape') {
        setAvatarOpen(false)
        avatarBtnRef.current?.focus()
      }
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [avatarOpen])

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

  // Design decision: wearing a badge IS public. The column defaults to
  // 'friends' for rows created before this rule existed, and no UI ever let
  // anyone pick that value, so only 'public' counts as worn. That also keeps
  // what the owner sees in step with what visitors actually get.
  const worn = !!mbti && mbti.visibility === 'public'

  const toggleWear = async () => {
    if (!mbti || savingWear) return
    const next = worn ? 'private' : 'public'
    const previous = mbti.visibility
    setWearError(null)
    setSavingWear(true)
    setMbti(prev => ({ ...prev, visibility: next }))        // optimistic
    const d = await api.patch('/api/mbti/me/visibility', { visibility: next })
    setSavingWear(false)
    if (!d.ok) {
      setMbti(prev => ({ ...prev, visibility: previous }))  // roll back
      setWearError(d.error || 'Could not save. Try again.')
      return
    }
    setBadgeOpen(false)
    flash(worn ? 'Badge removed.' : 'Badge is now public.', 'success')
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
    const bioTrimmed = (nameForm.bio ?? '').slice(0, 120)
    const d = await api.put('/api/auth/profile', {
      display_name: nameForm.display_name.trim(),
      email:        nameForm.email.trim(),
      phone:        contactForm.phone,
      wechat:       contactForm.wechat,
      address:      contactForm.address,
      postal_code:  contactForm.postal_code,
      bio:          bioTrimmed,
    })
    setSavingName(false)
    if (d.ok) {
      flash('Account info saved.')
      login({
        ...user,
        display_name: nameForm.display_name.trim(),
        email:        nameForm.email.trim(),
        phone:        contactForm.phone,
        wechat:       contactForm.wechat,
        address:      contactForm.address,
        postal_code:  contactForm.postal_code,
        bio:          bioTrimmed,
      })
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
    setAvatarOpen(false)
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

  const pickPreset = async (preset) => {
    if (savingPreset) return
    setSavingPreset(preset)
    const d = await api.put('/api/auth/avatar/preset', { preset })
    setSavingPreset(null)
    if (d.ok) {
      flash('Avatar updated.')
      login({ ...user, avatar_url: d.avatar_url })
      setAvatarOpen(false)
      avatarBtnRef.current?.focus()
    } else {
      flash(d.error || 'Could not save avatar.', 'danger')
    }
  }

  const currentAvatar = avatarPreview ?? user.avatar_url

  return (
    <>
      <h4 className='fw-bold mb-4'>My Profile</h4>

      {msg && <div className={'alert alert-' + msg.type + ' py-2'}>{msg.text}</div>}

      {contactEmpty && (
        <div className='alert alert-warning d-flex align-items-center justify-content-between gap-3 py-2 mb-3'
             role='status' id='missing-contact-banner'>
          <span>
            <i className='fas fa-address-card me-2' aria-hidden='true' />
            Add a WeChat ID or phone number so friends can reach you after you approve a contact request.
          </span>
          <button type='button' className='btn btn-sm btn-primary flex-shrink-0' onClick={scrollToContact}>
            Add now
          </button>
        </div>
      )}

      {/* Header — avatar + name + MBTI, one row */}
      <div className='card p-4 mb-4'>
        <div className='d-flex align-items-center gap-3 flex-wrap'>
          <input ref={fileRef} type='file' accept='.jpg,.jpeg,.png' style={{ display: 'none' }} onChange={onAvatarPick} />
          <div className='position-relative'>
            <button
              ref={avatarBtnRef}
              type='button'
              className='profile-avatar profile-avatar--button'
              title='Change avatar'
              onClick={() => setAvatarOpen(v => !v)}
              aria-expanded={avatarOpen}
              aria-haspopup='dialog'
              aria-controls={avatarOpen ? 'avatar-picker-panel' : undefined}
            >
              {currentAvatar ? (
                <img src={currentAvatar} alt='avatar' style={{ width: 80, height: 80, objectFit: 'cover' }} />
              ) : (
                <div className='d-flex align-items-center justify-content-center' style={{ width: 80, height: 80, background: '#6b9cdb', color: '#fff', fontSize: '2rem', fontWeight: 700 }}>
                  {user.display_name?.[0]?.toUpperCase()}
                </div>
              )}
              <span className='profile-avatar__overlay'>Change</span>
            </button>
            {avatarOpen && (
              <div
                ref={avatarPanelRef}
                id='avatar-picker-panel'
                className='avatar-picker'
                role='dialog'
                aria-label='Pick an avatar'
              >
                <div className='avatar-picker__label'>Pick one</div>
                <div className='avatar-picker__grid'>
                  {AVATAR_PRESETS.map(p => {
                    const url = `/avatars/${p.id}.svg`
                    const isCurrent = user.avatar_url === url
                    const isSaving = savingPreset === p.id
                    return (
                      <button
                        key={p.id}
                        type='button'
                        className={'avatar-picker__item' + (isCurrent ? ' is-current' : '')}
                        onClick={() => pickPreset(p.id)}
                        disabled={!!savingPreset}
                        aria-pressed={isCurrent}
                        aria-label={p.label + (isCurrent ? ' (current)' : '')}
                        title={p.label}
                      >
                        <img src={url} alt='' width='48' height='48' />
                        {isSaving && <span className='avatar-picker__saving'>…</span>}
                      </button>
                    )
                  })}
                </div>
                <div className='avatar-picker__sep' />
                <button
                  type='button'
                  className='btn btn-sm btn-outline-secondary w-100'
                  onClick={() => { setAvatarOpen(false); fileRef.current?.click() }}
                >
                  <i className='fas fa-upload me-2' aria-hidden='true' />Upload photo
                </button>
              </div>
            )}
          </div>

          <div className='flex-grow-1'>
            <div className='d-flex align-items-center gap-2 flex-wrap mb-1'>
              <h5 className='fw-bold mb-0'>{user.display_name}</h5>
              {xp && (
                <span className='xp-level' title={`${xp.xp} 经验`}>
                  <span className='xp-level__lv'>Lv.{xp.level}</span>{xp.level_name}
                </span>
              )}
              {xp?.hidden_tag && (
                <span className='xp-native' title='注册满一年'>{xp.hidden_tag}</span>
              )}
            </div>
            <div className='d-flex align-items-center gap-2 flex-wrap'>
              {mbti && worn && (
                <button
                  type='button'
                  className='mbti-badge mbti-badge--sm mbti-badge--button'
                  title={mbti.type_card?.nick ? `${mbti.type_card.nick} · click to remove` : 'Click to remove'}
                  aria-label={`Remove ${mbti.type} badge`}
                  aria-pressed='true'
                  disabled={savingWear}
                  onClick={toggleWear}
                >
                  {mbti.type}
                  <i className='fas fa-xmark mbti-badge__x' aria-hidden='true' />
                </button>
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
                        <div className='badge-picker__label'>Tap a tag to wear or remove it</div>
                        <button
                          type='button'
                          role='menuitemcheckbox'
                          aria-checked={worn}
                          className='badge-picker__item'
                          disabled={savingWear}
                          onClick={toggleWear}
                        >
                          <i className={'fas ' + (worn ? 'fa-check-square text-primary' : 'fa-square text-muted')} />
                          <span className='mbti-badge mbti-badge--sm'>{mbti.type}</span>
                          <span className='fw-medium'>{mbti.type_card?.nick || mbti.type}</span>
                          <span className='badge-picker__state'>{savingWear ? 'Saving…' : worn ? 'Wearing' : 'Wear'}</span>
                        </button>
                        {wearError && <div className='badge-picker__error' role='alert'>{wearError}</div>}
                      </>
                    )}
                  </div>
                )}
              </div>
              {user.username && (
                <Link
                  to={`/u/${user.username}`}
                  className='btn btn-sm btn-outline-secondary'
                  title='Open your public profile'
                >
                  <i className='fas fa-arrow-up-right-from-square me-1' aria-hidden='true' />
                  View public profile
                </Link>
              )}
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

        {xp && <XpProgress xp={xp} />}
      </div>

      {/* Forms — single column */}
      <div className='d-flex flex-column gap-4'>

        {/* Account info */}
        <div className='card p-4' ref={contactSectionRef} id='contact'>
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
              <div className='col-12'>
                <label className='form-label fw-medium d-flex justify-content-between align-items-center'>
                  <span>Bio</span>
                  <span className='text-muted small'>{(nameForm.bio ?? '').length}/120</span>
                </label>
                <textarea
                  className='form-control'
                  rows={2}
                  maxLength={120}
                  placeholder='写一句话介绍自己'
                  value={nameForm.bio}
                  onChange={e => setNameForm(f => ({ ...f, bio: e.target.value }))}
                />
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