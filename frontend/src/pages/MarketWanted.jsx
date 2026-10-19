import React, { useState, useEffect, useCallback } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { fmtDateMed } from '../utils'
import { ConfirmDialog } from '../components/Modal'

/**
 * Market → Wanted sub-panel.
 *
 * The reverse of a listing: the buyer posts what they're looking for, sellers
 * see it in the Wanted feed and DM the buyer directly (via Friends). When the
 * buyer finds what they wanted, they mark it fulfilled and it drops out of the
 * open feed. Three sub-tabs: Browse / My Wanted / Post.
 *
 * Reach Out opens the Friends DM thread with the buyer — no trade_intent is
 * created (wanted posts don't reserve anything).
 */

const CONDITIONS = [
  { value: 'any',  label: 'Any condition', zh: '不限成色' },
  { value: 'new',  label: 'New',           zh: '全新' },
  { value: 'used', label: 'Used',          zh: '二手' },
]

const SUB_TABS = (user) => [
  { key: 'browse', label: 'Browse Wanted' },
  user && { key: 'mine',   label: 'My Wanted' },
  user && { key: 'create', label: 'Post Wanted' },
].filter(Boolean)

function priceRangeText(w) {
  const lo = w.price_min, hi = w.price_max
  if (lo == null && hi == null) return 'Open budget'
  if (lo != null && hi != null) return `$${lo} – $${hi}`
  if (lo != null) return `≥ $${lo}`
  return `≤ $${hi}`
}

function categoryLabel(categories, slug) {
  const c = categories.find(c => c.slug === slug)
  return c ? { label: c.label, zh: c.label_zh, icon: c.icon || 'fa-tag' }
           : { label: slug, zh: null, icon: 'fa-tag' }
}

// ── Card ──────────────────────────────────────────────────────────
function WantedCard({ w, categories, me, onReachOut, onFulfill, onReopen, onDelete }) {
  const cat = categoryLabel(categories, w.category)
  const isMine = me && me === w.buyer
  const fulfilled = w.status === 'fulfilled'

  return (
    <article className={`wanted-card${fulfilled ? ' wanted-card--fulfilled' : ''}`}>
      <header className="wanted-card__head">
        <h3 className="wanted-card__title">{w.title}</h3>
        <span className="badge-pill">
          <i className={`fas ${cat.icon} me-1`} aria-hidden="true" />
          {cat.label}{cat.zh && <small className="ms-1 text-muted">{cat.zh}</small>}
        </span>
      </header>

      <div className="wanted-card__meta">
        <span className="wanted-card__price">{priceRangeText(w)}</span>
        <span className="wanted-card__sep">·</span>
        <span>{CONDITIONS.find(c => c.value === w.condition)?.label || 'Any condition'}</span>
        <span className="wanted-card__sep">·</span>
        <Link to={`/u/${w.buyer}`} className="wanted-card__buyer">@{w.buyer}</Link>
        <span className="wanted-card__sep">·</span>
        <span className="text-muted small">{fmtDateMed(w.created_at)}</span>
      </div>

      <p className="wanted-card__desc">{w.description}</p>

      {w.contact && (
        <p className="wanted-card__contact small text-muted">
          <i className="fas fa-phone me-1" aria-hidden="true" />{w.contact}
        </p>
      )}

      <footer className="wanted-card__foot">
        {fulfilled && <span className="badge badge-success">Fulfilled</span>}
        {!isMine && !fulfilled && me && (
          <button className="btn btn-primary btn-sm" onClick={() => onReachOut(w)}>
            <i className="fas fa-paper-plane me-1" aria-hidden="true" />Reach Out
          </button>
        )}
        {!me && !fulfilled && (
          <Link to="/login" className="btn btn-outline-primary btn-sm">Sign in to Reach Out</Link>
        )}
        {isMine && !fulfilled && (
          <>
            <button className="btn btn-outline-success btn-sm" onClick={() => onFulfill(w)}>
              <i className="fas fa-check me-1" aria-hidden="true" />Mark found
            </button>
            <button className="btn btn-outline-danger btn-sm" onClick={() => onDelete(w)}>
              <i className="fas fa-trash me-1" aria-hidden="true" />Delete
            </button>
          </>
        )}
        {isMine && fulfilled && (
          <button className="btn btn-outline-secondary btn-sm" onClick={() => onReopen(w)}>
            <i className="fas fa-undo me-1" aria-hidden="true" />Reopen
          </button>
        )}
      </footer>
    </article>
  )
}

// ── Post form ─────────────────────────────────────────────────────
function WantedPostForm({ categories, initial, onSubmit, onCancel, busy }) {
  const [f, setF] = useState(initial || {
    title: '', description: '', category: categories[0]?.slug || 'other',
    contact: '', price_min: '', price_max: '', condition: 'any',
  })
  const set = k => e => setF(p => ({ ...p, [k]: e.target.value }))

  const submit = (e) => {
    e.preventDefault()
    onSubmit({
      title: f.title, description: f.description, category: f.category,
      contact: f.contact,
      price_min: f.price_min === '' ? null : f.price_min,
      price_max: f.price_max === '' ? null : f.price_max,
      condition: f.condition,
    })
  }

  return (
    <form className="wanted-form" onSubmit={submit}>
      <div className="mb-3">
        <label className="form-label">Title</label>
        <input className="form-control" required maxLength={100}
               value={f.title} onChange={set('title')}
               placeholder="e.g. Used IKEA office chair" />
      </div>

      <div className="row g-3">
        <div className="col-md-6">
          <label className="form-label">Category</label>
          <select className="form-select" value={f.category} onChange={set('category')}>
            {categories.map(c => (
              <option key={c.slug} value={c.slug}>
                {c.label}{c.label_zh ? ` · ${c.label_zh}` : ''}
              </option>
            ))}
          </select>
        </div>
        <div className="col-md-6">
          <label className="form-label">Condition</label>
          <select className="form-select" value={f.condition} onChange={set('condition')}>
            {CONDITIONS.map(c => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="row g-3 mt-1">
        <div className="col-6">
          <label className="form-label">Min $</label>
          <input className="form-control" type="number" step="0.01" min="0"
                 value={f.price_min} onChange={set('price_min')}
                 placeholder="optional" />
        </div>
        <div className="col-6">
          <label className="form-label">Max $</label>
          <input className="form-control" type="number" step="0.01" min="0"
                 value={f.price_max} onChange={set('price_max')}
                 placeholder="optional" />
        </div>
      </div>

      <div className="mb-3 mt-3">
        <label className="form-label">Description</label>
        <textarea className="form-control" rows={4} required
                  value={f.description} onChange={set('description')}
                  placeholder="Size, specs, any requirements. Where you'd pick it up." />
      </div>

      <div className="mb-3">
        <label className="form-label">Contact <small className="text-muted">(optional — DMs still work)</small></label>
        <input className="form-control" maxLength={200}
               value={f.contact} onChange={set('contact')}
               placeholder="WeChat / phone / anything you'd rather share here" />
      </div>

      <div className="d-flex gap-2 justify-content-end">
        {onCancel && (
          <button type="button" className="btn btn-outline-secondary" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
        )}
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? 'Posting…' : 'Post Wanted'}
        </button>
      </div>
    </form>
  )
}

// ── Main panel ────────────────────────────────────────────────────
export default function MarketWanted({ categories, user }) {
  const navigate = useNavigate()
  const [subTab, setSubTab] = useState('browse')
  const [loading, setLoading] = useState(true)
  const [items, setItems] = useState([])
  const [myItems, setMyItems] = useState([])
  const [error, setError] = useState('')
  const [posting, setPosting] = useState(false)
  const [categoryFilter, setCategoryFilter] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(null)

  const loadBrowse = useCallback(async () => {
    setLoading(true); setError('')
    const q = categoryFilter ? `?category=${encodeURIComponent(categoryFilter)}` : ''
    const d = await api.get(`/api/market/wanted${q}`)
    if (d.ok) setItems(d.items || []); else setError(d.error || 'Failed to load')
    setLoading(false)
  }, [categoryFilter])

  const loadMine = useCallback(async () => {
    if (!user) return
    setLoading(true); setError('')
    const d = await api.get('/api/market/wanted/mine')
    if (d.ok) setMyItems(d.items || []); else setError(d.error || 'Failed to load')
    setLoading(false)
  }, [user])

  useEffect(() => {
    if (subTab === 'browse') loadBrowse()
    if (subTab === 'mine') loadMine()
  }, [subTab, loadBrowse, loadMine])

  const reachOut = (w) => {
    // Open the DM with the buyer, same mechanism Market's "Reach Out" uses.
    // Message is in the user's own mouth: short, mentions the title, invites
    // the buyer to continue in chat.
    const initialMessage = `Hi! I saw you're looking for "${w.title}" — I might have one. Want to chat?`
    const chatPartner = { username: w.buyer, display_name: w.buyer, avatar_url: null }
    navigate('/friends', { state: { openChat: chatPartner, initialMessage } })
  }

  const fulfill = async (w) => {
    const d = await api.post(`/api/market/wanted/${w.id}/fulfilled`)
    if (!d.ok) { setError(d.error); return }
    loadMine(); if (subTab === 'browse') loadBrowse()
  }

  const reopen = async (w) => {
    const d = await api.post(`/api/market/wanted/${w.id}/reopen`)
    if (!d.ok) { setError(d.error); return }
    loadMine(); if (subTab === 'browse') loadBrowse()
  }

  const confirmThenDelete = (w) => setConfirmDelete(w)
  const doDelete = async () => {
    const w = confirmDelete; setConfirmDelete(null)
    if (!w) return
    const d = await api.delete(`/api/market/wanted/${w.id}`)
    if (!d.ok) { setError(d.error); return }
    loadMine(); if (subTab === 'browse') loadBrowse()
  }

  const submitNew = async (body) => {
    setPosting(true); setError('')
    const d = await api.post('/api/market/wanted', body)
    setPosting(false)
    if (!d.ok) { setError(d.error); return }
    setSubTab('mine'); loadMine()
  }

  const list = subTab === 'mine' ? myItems : items
  const me = user?.username

  return (
    <div className="market-wanted">
      <div className="radio-inputs mb-4">
        {SUB_TABS(user).map(t => (
          <label className="radio" key={t.key}>
            <input type="radio" name="wanted-subtab"
                   checked={subTab === t.key}
                   onChange={() => setSubTab(t.key)} />
            <span className="name">{t.label}</span>
          </label>
        ))}
      </div>

      {error && <div className="alert alert-danger py-2" role="alert">{error}</div>}

      {subTab === 'browse' && (
        <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }} className="mb-3">
          <div className="d-flex gap-2 align-items-center" style={{ width: 'max-content', paddingBottom: 2 }}
               role="group" aria-label="Filter by category">
            <span className="small me-1" style={{ whiteSpace: 'nowrap', color: 'var(--text-muted)' }}>Category</span>
            <button onClick={() => setCategoryFilter('')} aria-pressed={!categoryFilter}
                    className={`btn btn-sm ${!categoryFilter ? 'btn-primary' : 'btn-outline-secondary'}`}
                    style={{ borderRadius: 20, whiteSpace: 'nowrap' }}>All</button>
            {categories.map(cat => {
              const active = categoryFilter === cat.slug
              return (
                <button key={cat.slug}
                        onClick={() => setCategoryFilter(active ? '' : cat.slug)}
                        aria-pressed={active}
                        className={`btn btn-sm ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
                        style={{ borderRadius: 20, whiteSpace: 'nowrap' }}>
                  <i className={`fas ${cat.icon || 'fa-tag'} me-1`} aria-hidden="true" />{cat.label}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {subTab === 'create' ? (
        <WantedPostForm categories={categories}
                        onSubmit={submitNew} busy={posting}
                        onCancel={() => setSubTab('browse')} />
      ) : loading ? (
        <div className="text-center py-4"><div className="spinner-border" role="status" /></div>
      ) : list.length === 0 ? (
        <div className="empty-state py-5 text-center">
          <i className="fas fa-search fa-2x mb-3 text-muted" aria-hidden="true" />
          <p className="text-muted">
            {subTab === 'mine' ? "You haven't posted anything you're looking for yet."
                               : 'No one is looking for anything right now.'}
          </p>
          {user && (
            <button className="btn btn-primary" onClick={() => setSubTab('create')}>
              Post Wanted
            </button>
          )}
        </div>
      ) : (
        <div className="wanted-grid">
          {list.map(w => (
            <WantedCard key={w.id} w={w} categories={categories} me={me}
                        onReachOut={reachOut} onFulfill={fulfill}
                        onReopen={reopen} onDelete={confirmThenDelete} />
          ))}
        </div>
      )}

      {confirmDelete && (
        <ConfirmDialog
          title="Delete this wanted post?"
          itemName={confirmDelete.title}
          message="This cannot be undone."
          confirmLabel="Delete"
          destructive
          onConfirm={doDelete}
          onClose={() => setConfirmDelete(null)}
        />
      )}
    </div>
  )
}
