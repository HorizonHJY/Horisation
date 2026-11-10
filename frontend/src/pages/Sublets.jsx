import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'
import Modal, { ConfirmDialog } from '../components/Modal'
import HandLoader from '../components/HandLoader'
import { fmtDateMed } from '../utils'

/* Sublets — short-term apartment rentals.
   Three sub-tabs: Browse / My Sublets / Post.
   Images use the Market upload pipeline (multipart, max 3).
   The public feed hides any post whose end_date has passed. */

const LAYOUTS = [
  { value: 'studio', label: 'Studio',          zh: '开间' },
  { value: '1b1b',   label: '1 Bed 1 Bath',    zh: '一室一卫' },
  { value: '2b1b',   label: '2 Bed 1 Bath',    zh: '两室一卫' },
  { value: '2b2b',   label: '2 Bed 2 Bath',    zh: '两室两卫' },
  { value: '3b2b',   label: '3 Bed 2 Bath',    zh: '三室两卫' },
  { value: '3b3b',   label: '3 Bed 3 Bath',    zh: '三室三卫' },
  { value: '4b2b',   label: '4+ Bed',          zh: '四室及以上' },
  { value: 'other',  label: 'Other',           zh: '其他' },
]
const LAYOUT_BY = Object.fromEntries(LAYOUTS.map(o => [o.value, o]))

const FURNISHED = [
  { value: 'yes',     label: 'Furnished',     zh: '带家具' },
  { value: 'no',      label: 'Unfurnished',   zh: '不带家具' },
  { value: 'partial', label: 'Partial',       zh: '部分家具' },
]
const FURNISHED_BY = Object.fromEntries(FURNISHED.map(o => [o.value, o]))

const EMPTY_FORM = {
  apartment: '', rent: '', start_date: '', end_date: '',
  layout: '1b1b', area: '', furnished: 'no',
  description: '', contact: '',
}

const MAX_IMAGES = 3

function todayISO() {
  // Local-day ISO (YYYY-MM-DD). Using toISOString() would drift to UTC and
  // tell someone in Chicago at 23:00 that the min date is tomorrow.
  const d = new Date()
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

const money = (n) => `$${(Number(n) || 0).toFixed(2)}`

function OwnerAvatar({ username, displayName, avatarUrl, size = 28 }) {
  const style = { width: size, height: size, borderRadius: '50%', flexShrink: 0 }
  const name = displayName || username || '?'
  if (avatarUrl) return <img src={avatarUrl} alt="" style={{ ...style, objectFit: 'cover' }} />
  return (
    <span
      aria-hidden="true"
      style={{
        ...style, background: 'var(--accent-text, #6b9cdb)', color: '#fff',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontWeight: 700, fontSize: size * 0.42,
      }}
    >
      {name[0]?.toUpperCase() || '?'}
    </span>
  )
}

// ── Card ─────────────────────────────────────────────────────────
function SubletCard({ s, me, onOpen }) {
  const firstImg = s.images?.[0]?.url
  const isTaken  = s.status === 'taken'
  const layout   = LAYOUT_BY[s.layout] || LAYOUT_BY.other
  const furn     = FURNISHED_BY[s.furnished] || FURNISHED_BY.no

  return (
    <button
      type="button"
      className={`sl-card${isTaken ? ' sl-card--taken' : ''}`}
      onClick={() => onOpen(s)}
      aria-label={`Open sublet: ${s.apartment}${isTaken ? ' — taken' : ''}`}
    >
      <span className="sl-card__img">
        {firstImg
          ? <img src={firstImg} alt="" loading="lazy" />
          : <i className="fas fa-building sl-card__ph" aria-hidden="true" />
        }
        {isTaken && (
          <span className="sl-card__overlay" aria-hidden="true">
            <span className="sl-card__stamp">Taken<span className="zh">已转出</span></span>
          </span>
        )}
      </span>

      <span className="sl-card__body">
        <span className="sl-card__title" title={s.apartment}>{s.apartment}</span>
        <span className="sl-card__price">{money(s.rent)}<small>/mo</small></span>
        <span className="sl-card__dates">
          <i className="fas fa-calendar-days" aria-hidden="true" />
          {fmtDateMed(s.start_date)} – {fmtDateMed(s.end_date)}
        </span>
        <span className="sl-card__meta">
          <span className="sl-pill">{layout.label}</span>
          {s.area && <span className="sl-pill">{s.area}</span>}
          <span className="sl-pill sl-pill--soft">{furn.label}</span>
        </span>
      </span>
    </button>
  )
}

// ── Detail modal ─────────────────────────────────────────────────
function SubletDetailModal({ sublet, me, onClose, onEdit, onMarkTaken, onReopen, onDelete }) {
  const [imgIndex, setImgIndex] = useState(0)
  const isMine  = me && me === sublet.owner_username
  const isTaken = sublet.status === 'taken'
  const layout  = LAYOUT_BY[sublet.layout] || LAYOUT_BY.other
  const furn    = FURNISHED_BY[sublet.furnished] || FURNISHED_BY.no

  return (
    <Modal onClose={onClose} title={sublet.apartment} size="modal-lg">
      {({ titleId }) => (
        <>
          <div className="modal-header">
            <h5 className="modal-title fw-semibold" id={titleId}>
              {sublet.apartment}
            </h5>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose} />
          </div>

          <div className="modal-body">
            {sublet.images?.length > 0 && (
              <div className="mb-3">
                <div style={{ position: 'relative' }}>
                  <img
                    src={sublet.images[imgIndex].url}
                    alt={`${sublet.apartment} — photo ${imgIndex + 1} of ${sublet.images.length}`}
                    style={{
                      width: '100%', maxHeight: 380, objectFit: 'contain',
                      background: 'var(--bg-page)', borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-soft)', display: 'block',
                    }}
                  />
                </div>
                {sublet.images.length > 1 && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 8 }} role="group" aria-label="Photos">
                    {sublet.images.map((img, i) => (
                      <button
                        key={img.id}
                        type="button"
                        onClick={() => setImgIndex(i)}
                        aria-pressed={i === imgIndex}
                        aria-label={`Show photo ${i + 1}`}
                        style={{
                          padding: 0, lineHeight: 0, cursor: 'pointer', background: 'none',
                          border: i === imgIndex ? '2px solid var(--accent)' : '1px solid var(--border-medium)',
                          borderRadius: 'var(--radius-sm)',
                        }}
                      >
                        <img src={img.url} alt="" style={{ width: 58, height: 58, objectFit: 'cover', borderRadius: 3 }} />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="sl-detail__row">
              <span className="sl-detail__price">{money(sublet.rent)}<small>/mo</small></span>
              {isTaken && <span className="sl-pill sl-pill--warn">Taken 已转出</span>}
            </div>

            <div className="sl-detail__grid">
              <div><strong>开始日期</strong><span>{fmtDateMed(sublet.start_date)}</span></div>
              <div><strong>结束日期</strong><span>{fmtDateMed(sublet.end_date)}</span></div>
              <div><strong>房型</strong><span>{layout.label} · {layout.zh}</span></div>
              <div><strong>区域</strong><span>{sublet.area || '—'}</span></div>
              <div><strong>家具</strong><span>{furn.label} · {furn.zh}</span></div>
              <div><strong>状态</strong><span>{sublet.status === 'open' ? '可转' : '已转出'}</span></div>
            </div>

            {sublet.description && (
              <>
                <hr />
                <p className="sl-detail__desc">{sublet.description}</p>
              </>
            )}

            <hr />
            <div className="d-flex align-items-center gap-2">
              <OwnerAvatar
                username={sublet.owner_username}
                displayName={sublet.owner_display}
                avatarUrl={sublet.owner_avatar}
                size={34}
              />
              <div>
                <div style={{ fontWeight: 600 }}>{sublet.owner_display || sublet.owner_username}</div>
                <Link to={`/u/${sublet.owner_username}`} style={{ fontSize: '.8rem' }}>
                  @{sublet.owner_username}
                </Link>
              </div>
            </div>

            {sublet.contact && (
              <p className="small mt-2 mb-0" style={{ color: 'var(--text-muted)' }}>
                <i className="fas fa-phone me-1" aria-hidden="true" />
                联系方式: {sublet.contact}
              </p>
            )}
          </div>

          <div className="modal-footer" style={{ gap: 8 }}>
            {isMine ? (
              <>
                {!isTaken && (
                  <>
                    <button className="btn btn-outline-primary btn-sm" onClick={() => { onEdit(sublet); onClose() }}>
                      <i className="fas fa-pen me-1" aria-hidden="true" />编辑
                    </button>
                    <button className="btn btn-outline-success btn-sm" onClick={() => { onMarkTaken(sublet); onClose() }}>
                      <i className="fas fa-check me-1" aria-hidden="true" />标记已转出
                    </button>
                  </>
                )}
                {isTaken && (
                  <button className="btn btn-outline-secondary btn-sm" onClick={() => { onReopen(sublet); onClose() }}>
                    <i className="fas fa-rotate-left me-1" aria-hidden="true" />重新开放
                  </button>
                )}
                <button className="btn btn-outline-danger btn-sm" onClick={() => { onDelete(sublet) }}>
                  <i className="fas fa-trash me-1" aria-hidden="true" />删除
                </button>
              </>
            ) : (
              !isTaken && me && sublet.contact && (
                <span className="small" style={{ color: 'var(--text-muted)' }}>
                  使用上方联系方式联系发布者
                </span>
              )
            )}
            {!me && (
              <Link to="/login" className="btn btn-sm btn-primary">登录后联系</Link>
            )}
          </div>
        </>
      )}
    </Modal>
  )
}

// ── Post / Edit form ─────────────────────────────────────────────
function SubletForm({ initial, onSubmit, onCancel, busy, existingImages = [] }) {
  const [f, setF] = useState(initial || EMPTY_FORM)
  const [images, setImages]     = useState([])
  const [previews, setPreviews] = useState([])
  const [err, setErr] = useState('')
  const fileRef = useRef()

  const set = k => e => setF(p => ({ ...p, [k]: e.target.value }))

  const handleFileChange = (e) => {
    const incoming = Array.from(e.target.files)
    const incomingPreviews = incoming.map(x => URL.createObjectURL(x))
    const slots = MAX_IMAGES - existingImages.length
    setImages(prev => [...prev, ...incoming].slice(0, slots))
    setPreviews(prev => {
      const combined = [...prev, ...incomingPreviews]
      if (combined.length > slots) combined.slice(slots).forEach(u => URL.revokeObjectURL(u))
      return combined.slice(0, slots)
    })
    if (fileRef.current) fileRef.current.value = ''
  }

  const removeImage = (idx) => {
    URL.revokeObjectURL(previews[idx])
    setImages(prev => prev.filter((_, i) => i !== idx))
    setPreviews(prev => prev.filter((_, i) => i !== idx))
  }

  useEffect(() => () => {
    previews.forEach(u => URL.revokeObjectURL(u))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const submit = (e) => {
    e.preventDefault()
    setErr('')
    if (!f.apartment.trim()) return setErr('请填写公寓名称。')
    if (f.apartment.trim().length > 80) return setErr('公寓名称不能超过 80 个字。')
    const rentNum = Number(f.rent)
    if (!f.rent || isNaN(rentNum) || rentNum <= 0) return setErr('请填写一个大于 0 的月租金额。')
    if (!f.start_date) return setErr('请选择开始日期。')
    if (!f.end_date)   return setErr('请选择结束日期。')
    if (f.start_date > f.end_date) return setErr('结束日期不能早于开始日期。')
    if ((existingImages.length + images.length) > MAX_IMAGES) return setErr(`最多上传 ${MAX_IMAGES} 张图片。`)

    const fd = new FormData()
    fd.append('apartment', f.apartment.trim())
    fd.append('rent', String(rentNum))
    fd.append('start_date', f.start_date)
    fd.append('end_date', f.end_date)
    fd.append('layout', f.layout)
    fd.append('area', f.area.trim())
    fd.append('furnished', f.furnished)
    fd.append('description', f.description.trim())
    fd.append('contact', f.contact.trim())
    images.forEach(img => fd.append('images', img))
    onSubmit(fd)
  }

  const today = todayISO()

  return (
    <form className="sl-form" onSubmit={submit}>
      {err && <div className="alert alert-danger py-2 small" role="alert">{err}</div>}

      <div className="mb-3">
        <label className="form-label">公寓名称</label>
        <input className="form-control" required maxLength={80}
               value={f.apartment} onChange={set('apartment')}
               placeholder="e.g. Icon Chicago / 某某公寓" />
      </div>

      <div className="row g-3">
        <div className="col-6">
          <label className="form-label">月租金额 ($)</label>
          <input className="form-control" type="number" min="0" step="1" required
                 value={f.rent} onChange={set('rent')} placeholder="1500" />
        </div>
        <div className="col-6">
          <label className="form-label">区域</label>
          <input className="form-control" maxLength={100}
                 value={f.area} onChange={set('area')} placeholder="e.g. Loop / 中城" />
        </div>
      </div>

      <div className="row g-3 mt-1">
        <div className="col-6">
          <label className="form-label">开始日期</label>
          <input className="form-control" type="date" required
                 min={today}
                 value={f.start_date} onChange={set('start_date')} />
        </div>
        <div className="col-6">
          <label className="form-label">结束日期</label>
          <input className="form-control" type="date" required
                 min={f.start_date || today}
                 value={f.end_date} onChange={set('end_date')} />
        </div>
      </div>

      <div className="row g-3 mt-1">
        <div className="col-6">
          <label className="form-label">房型</label>
          <select className="form-select" value={f.layout} onChange={set('layout')}>
            {LAYOUTS.map(o => <option key={o.value} value={o.value}>{o.label} · {o.zh}</option>)}
          </select>
        </div>
        <div className="col-6">
          <label className="form-label">家具</label>
          <select className="form-select" value={f.furnished} onChange={set('furnished')}>
            {FURNISHED.map(o => <option key={o.value} value={o.value}>{o.label} · {o.zh}</option>)}
          </select>
        </div>
      </div>

      <div className="mb-3 mt-3">
        <label className="form-label">描述</label>
        <textarea className="form-control" rows={4}
                  value={f.description} onChange={set('description')}
                  placeholder="户型/配套/楼层/是否离地铁等,任何潜在租客要知道的信息。" />
      </div>

      <div className="mb-3">
        <label className="form-label">联系方式 <small className="text-muted">(微信/电话,可选)</small></label>
        <input className="form-control" maxLength={200}
               value={f.contact} onChange={set('contact')} placeholder="留一个方便联系的方式" />
      </div>

      <div className="mb-3">
        <label className="form-label">图片 <small className="text-muted">(最多 {MAX_IMAGES} 张)</small></label>
        <input ref={fileRef} type="file" accept="image/jpeg,image/png"
               multiple onChange={handleFileChange} className="form-control"
               disabled={(existingImages.length + images.length) >= MAX_IMAGES} />
        {(previews.length > 0 || existingImages.length > 0) && (
          <div className="sl-form__previews mt-2">
            {existingImages.map(img => (
              <div key={img.id} className="sl-form__preview">
                <img src={img.url} alt="" />
                <span className="sl-form__exist">已上传</span>
              </div>
            ))}
            {previews.map((src, i) => (
              <div key={i} className="sl-form__preview">
                <img src={src} alt="" />
                <button type="button" className="sl-form__remove"
                        onClick={() => removeImage(i)} aria-label="Remove image">×</button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="d-flex gap-2 justify-content-end">
        {onCancel && (
          <button type="button" className="btn btn-outline-secondary" onClick={onCancel} disabled={busy}>
            取消
          </button>
        )}
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? '保存中…' : '保存'}
        </button>
      </div>
    </form>
  )
}

// ── Page ─────────────────────────────────────────────────────────
export default function Sublets() {
  const { user } = useAuth()
  const me = user?.username

  const SUB_TABS = useMemo(() => [
    { key: 'browse', label: 'Browse' },
    user && { key: 'mine',   label: 'My Sublets' },
    user && { key: 'create', label: 'Post' },
  ].filter(Boolean), [user])

  const [subTab, setSubTab] = useState('browse')
  const [loading, setLoading] = useState(true)
  const [items, setItems] = useState([])
  const [myItems, setMyItems] = useState([])
  const [error, setError] = useState('')
  const [posting, setPosting] = useState(false)
  const [detail, setDetail] = useState(null)
  const [editing, setEditing] = useState(null)
  const [confirmDelete, setConfirmDelete] = useState(null)

  const loadBrowse = useCallback(async () => {
    setLoading(true); setError('')
    const d = await api.get('/api/market/sublets')
    if (d.ok) setItems(d.items || []); else setError(d.error || '加载失败')
    setLoading(false)
  }, [])

  const loadMine = useCallback(async () => {
    if (!user) return
    setLoading(true); setError('')
    const d = await api.get('/api/market/sublets/mine')
    if (d.ok) setMyItems(d.items || []); else setError(d.error || '加载失败')
    setLoading(false)
  }, [user])

  useEffect(() => {
    if (subTab === 'browse') loadBrowse()
    if (subTab === 'mine')   loadMine()
  }, [subTab, loadBrowse, loadMine])

  const submitNew = async (fd) => {
    setPosting(true); setError('')
    const d = await api.upload('/api/market/sublets', fd)
    setPosting(false)
    if (!d.ok) { setError(d.error || '发布失败'); return }
    setSubTab('mine'); loadMine()
  }

  const markTaken = async (s) => {
    const d = await api.post(`/api/market/sublets/${s.id}/taken`)
    if (!d.ok) { setError(d.error); return }
    if (subTab === 'browse') loadBrowse()
    loadMine()
  }

  const reopen = async (s) => {
    const d = await api.post(`/api/market/sublets/${s.id}/reopen`)
    if (!d.ok) { setError(d.error); return }
    if (subTab === 'browse') loadBrowse()
    loadMine()
  }

  const confirmThenDelete = (s) => { setDetail(null); setConfirmDelete(s) }
  const doDelete = async () => {
    const s = confirmDelete; setConfirmDelete(null)
    if (!s) return
    const d = await api.delete(`/api/market/sublets/${s.id}`)
    if (!d.ok) { setError(d.error); return }
    if (subTab === 'browse') loadBrowse()
    loadMine()
  }

  const list = subTab === 'mine' ? myItems : items

  return (
    <div className="sl-page container py-4">
      <header className="sl-header mb-3">
        <h2 className="sl-title">Sublets <small className="text-muted">转租</small></h2>
        <p className="sl-sub text-muted">
          短期转租:公寓名称、租金、起止日期、房型、区域、家具、描述。
        </p>
      </header>

      <div className="radio-inputs mb-4">
        {SUB_TABS.map(t => (
          <label className="radio" key={t.key}>
            <input type="radio" name="sl-subtab"
                   checked={subTab === t.key}
                   onChange={() => setSubTab(t.key)} />
            <span className="name">{t.label}</span>
          </label>
        ))}
      </div>

      {error && <div className="alert alert-danger py-2" role="alert">{error}</div>}

      {subTab === 'create' ? (
        <SubletForm
          onSubmit={submitNew}
          busy={posting}
          onCancel={() => setSubTab('browse')}
        />
      ) : loading ? (
        <div className="text-center py-5"><HandLoader /></div>
      ) : list.length === 0 ? (
        <div className="sl-empty text-center py-5">
          <i className="fas fa-building fa-2x mb-3 text-muted" aria-hidden="true" />
          <p className="text-muted mb-3">
            {subTab === 'mine' ? '你还没有发布任何转租。' : '暂无可转的房源。'}
          </p>
          {user && (
            <button className="btn btn-primary" onClick={() => setSubTab('create')}>
              发布转租
            </button>
          )}
          {!user && (
            <Link to="/login" className="btn btn-primary">登录后发布</Link>
          )}
        </div>
      ) : (
        <div className="sl-grid">
          {list.map(s => (
            <SubletCard key={s.id} s={s} me={me} onOpen={setDetail} />
          ))}
        </div>
      )}

      {detail && (
        <SubletDetailModal
          sublet={detail}
          me={me}
          onClose={() => setDetail(null)}
          onEdit={(s) => { setDetail(null); setEditing(s) }}
          onMarkTaken={markTaken}
          onReopen={reopen}
          onDelete={confirmThenDelete}
        />
      )}

      {editing && (
        <Modal onClose={() => setEditing(null)} title="编辑转租" size="modal-lg">
          {({ titleId }) => (
            <>
              <div className="modal-header">
                <h5 className="modal-title fw-semibold" id={titleId}>编辑转租</h5>
                <button type="button" className="btn-close" aria-label="Close" onClick={() => setEditing(null)} />
              </div>
              <div className="modal-body">
                <SubletEditForm
                  sublet={editing}
                  onCancel={() => setEditing(null)}
                  onSaved={() => { setEditing(null); loadMine(); if (subTab === 'browse') loadBrowse() }}
                  onError={setError}
                />
              </div>
            </>
          )}
        </Modal>
      )}

      {confirmDelete && (
        <ConfirmDialog
          title="删除这条转租?"
          itemName={confirmDelete.apartment}
          message="删除后无法恢复。"
          confirmLabel="删除"
          destructive
          onConfirm={doDelete}
          onClose={() => setConfirmDelete(null)}
        />
      )}
    </div>
  )
}

// Edit uses PUT + multipart; api.upload is POST, so do the fetch ourselves.
function SubletEditForm({ sublet, onCancel, onSaved, onError }) {
  const [busy, setBusy] = useState(false)
  const initial = {
    apartment: sublet.apartment,
    rent: String(sublet.rent),
    start_date: sublet.start_date || '',
    end_date:   sublet.end_date   || '',
    layout: sublet.layout,
    area:   sublet.area || '',
    furnished: sublet.furnished,
    description: sublet.description || '',
    contact: sublet.contact || '',
  }
  const submit = async (fd) => {
    setBusy(true)
    try {
      const res = await fetch(`/api/market/sublets/${sublet.id}`, {
        method: 'PUT', credentials: 'include', body: fd,
      })
      const text = await res.text()
      let body = {}
      try { body = JSON.parse(text) } catch { /* ignore */ }
      if (!res.ok || body.ok === false) {
        onError(body.error || `HTTP ${res.status}`)
        return
      }
      onSaved()
    } catch (e) {
      onError('网络错误。')
    } finally {
      setBusy(false)
    }
  }
  return (
    <SubletForm
      initial={initial}
      existingImages={sublet.images || []}
      busy={busy}
      onSubmit={submit}
      onCancel={onCancel}
    />
  )
}
