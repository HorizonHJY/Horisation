import React, { useState, useEffect, useRef } from 'react'
import { api } from '../api'
import { useAuth } from '../App'
import HandLoader from '../components/HandLoader'

const CATEGORIES = ['electronics', 'clothing', 'books', 'furniture', 'other']

const EMPTY_FORM = { title: '', description: '', price: '', original_price: '', category: 'electronics' }

// ── Toast ─────────────────────────────────────────────────────────────────────
function useToast() {
  const [toast, setToast] = useState(null)
  const show = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 2800)
  }
  return [toast, show]
}

// ── Listing Card ──────────────────────────────────────────────────────────────
function ListingCard({ listing, currentUser, onSold, onDelete, onInterested, interestedSet }) {
  const isMine      = listing.seller_username === currentUser
  const firstImg    = listing.images?.[0]?.url
  const isSold      = listing.status === 'sold'
  const hasOriginal = listing.original_price && listing.original_price > listing.price
  const interestStatus = interestedSet?.[listing.seller_username]  // 'sent'|'friends'|undefined

  return (
    <div className="market-card">
      {/* Image */}
      <div className="market-card__img">
        {firstImg
          ? <img src={firstImg} alt={listing.title} />
          : <i className="fas fa-image placeholder-icon" />
        }
      </div>

      {/* Title */}
      <div className="market-card__title" title={listing.title}>{listing.title}</div>

      {/* Category + sold badge */}
      <div className="market-card__meta">
        <span className="market-card__category">{listing.category}</span>
        {isSold && <span className="market-card__sold-badge">Sold</span>}
      </div>

      {/* Description */}
      <p className="market-card__desc">{listing.description}</p>

      {/* Footer: seller + price */}
      <div className="market-card__footer">
        <div className="market-card__seller">
          <i className="fas fa-user me-1" />{listing.seller_username}
          <div>{new Date(listing.created_at).toLocaleDateString()}</div>
        </div>
        {!isSold && (
          <div className="market-card__price">
            <span>¥{listing.price}</span>
            {hasOriginal && (
              <span style={{ fontSize: '.75rem', color: '#999', textDecoration: 'line-through', marginLeft: 6 }}>
                ¥{listing.original_price}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Owner actions */}
      {isMine && (
        <div className="market-card__action">
          {!isSold && (
            <button className="market-card__btn" onClick={() => onSold(listing.id)}>
              <i className="fas fa-check-circle" />Mark Sold
            </button>
          )}
          <button className="market-card__btn market-card__btn--danger" onClick={() => onDelete(listing.id)}>
            <i className="fas fa-trash" />Delete
          </button>
        </div>
      )}

      {/* Interested button for other users' active listings */}
      {!isMine && !isSold && (
        <div className="market-card__action">
          {interestStatus === 'friends' ? (
            <span className="badge bg-success px-3 py-2" style={{ fontSize: '.8rem' }}>
              <i className="fas fa-user-friends me-1" />Friends
            </span>
          ) : interestStatus === 'sent' ? (
            <span className="badge bg-warning text-dark px-3 py-2" style={{ fontSize: '.8rem' }}>
              <i className="fas fa-clock me-1" />Request Sent
            </span>
          ) : (
            <button
              className="market-card__btn"
              style={{ background: '#3a7bd5', color: '#fff', border: 'none' }}
              onClick={() => onInterested(listing)}
            >
              <i className="fas fa-star me-1" />I'm Interested
            </button>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function Market() {
  const { user } = useAuth()
  const [tab, setTab]               = useState('browse')
  const [listings, setListings]     = useState([])
  const [myListings, setMy]         = useState([])
  const [loading, setLoading]       = useState(false)
  const [form, setForm]             = useState(EMPTY_FORM)
  const [images, setImages]         = useState([])
  const [previews, setPreviews]     = useState([])
  const [submitting, setSub]        = useState(false)
  const [toast, showToast]          = useToast()
  // interestedSet: { [seller_username]: 'sent' | 'friends' }
  const [interestedSet, setInterested] = useState({})
  const fileRef = useRef()

  // Load listings on tab switch
  useEffect(() => {
    if (tab === 'browse')     loadBrowse()
    if (tab === 'mylistings') loadMine()
  }, [tab])

  async function loadBrowse() {
    setLoading(true)
    const [listRes, friendsRes, sentRes] = await Promise.all([
      api.get('/api/market/listings'),
      api.get('/api/friends/list'),
      api.get('/api/friends/requests/sent'),
    ])
    if (listRes.ok) setListings(listRes.listings)
    // Build interestedSet from friends + sent requests
    const map = {}
    if (friendsRes.ok)  friendsRes.friends.forEach(f => { map[f.username] = 'friends' })
    if (sentRes.ok)     sentRes.requests.filter(r => r.status === 'pending').forEach(r => { map[r.to_user] = 'sent' })
    setInterested(map)
    setLoading(false)
  }

  async function loadMine() {
    setLoading(true)
    const d = await api.get('/api/market/my')
    if (d.ok) setMy(d.listings)
    setLoading(false)
  }

  function handleFileChange(e) {
    const files = Array.from(e.target.files).slice(0, 3)
    setImages(files)
    setPreviews(files.map(f => URL.createObjectURL(f)))
  }

  function removeImage(idx) {
    const newFiles    = images.filter((_, i) => i !== idx)
    const newPreviews = previews.filter((_, i) => i !== idx)
    URL.revokeObjectURL(previews[idx])
    setImages(newFiles)
    setPreviews(newPreviews)
  }

  async function handleCreate(e) {
    e.preventDefault()
    if (!form.title.trim())       return showToast('Title is required.', 'danger')
    if (!form.description.trim()) return showToast('Description is required.', 'danger')
    if (!form.price || isNaN(form.price) || Number(form.price) < 0)
                                  return showToast('Enter a valid price.', 'danger')

    setSub(true)
    const fd = new FormData()
    Object.entries(form).forEach(([k, v]) => fd.append(k, v))
    images.forEach(img => fd.append('images', img))

    const d = await api.upload('/api/market/listings', fd)
    setSub(false)

    if (d.ok) {
      showToast('Listing posted!')
      setForm(EMPTY_FORM)
      setImages([])
      setPreviews([])
      if (fileRef.current) fileRef.current.value = ''
      setTab('browse')
    } else {
      showToast(d.error || 'Failed to post listing.', 'danger')
    }
  }

  async function handleSold(id) {
    const d = await api.post(`/api/market/listings/${id}/sold`)
    if (d.ok) {
      showToast('Marked as sold.')
      if (tab === 'browse')     loadBrowse()
      if (tab === 'mylistings') loadMine()
    } else {
      showToast(d.error, 'danger')
    }
  }

  async function handleDelete(id) {
    if (!window.confirm('Delete this listing?')) return
    const d = await api.delete(`/api/market/listings/${id}`)
    if (d.ok) {
      showToast('Listing deleted.')
      if (tab === 'browse')     loadBrowse()
      if (tab === 'mylistings') loadMine()
    } else {
      showToast(d.error, 'danger')
    }
  }

  async function handleInterested(listing) {
    const msg = `Hi! I'm interested in your listing "${listing.title}" and would love to connect. Want to add me as a friend?`
    const d = await api.post('/api/friends/requests', { to_user: listing.seller_username, message: msg })
    if (d.ok) {
      setInterested(prev => ({ ...prev, [listing.seller_username]: 'sent' }))
      showToast('Friend request sent!')
    } else if (d.error === 'Already friends') {
      setInterested(prev => ({ ...prev, [listing.seller_username]: 'friends' }))
    } else if (d.error === 'Request already pending') {
      setInterested(prev => ({ ...prev, [listing.seller_username]: 'sent' }))
    } else {
      showToast(d.error, 'danger')
    }
  }

  const displayList = tab === 'mylistings' ? myListings : listings

  return (
    <div className="container-fluid py-4">

      {/* Toast */}
      {toast && (
        <div className={`alert alert-${toast.type} alert-dismissible position-fixed top-0 end-0 m-3`}
             style={{ zIndex: 9999 }}>
          {toast.msg}
        </div>
      )}

      <div className="d-flex align-items-center mb-4">
        <i className="fas fa-store fa-lg me-2 text-primary" />
        <h4 className="mb-0 fw-bold">Market</h4>
        <span className="text-muted ms-2 small">Second-hand trading</span>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {[
          { key: 'browse',     label: 'Browse',      icon: 'fa-th-large' },
          { key: 'create',     label: 'Post Listing', icon: 'fa-plus-circle' },
          { key: 'mylistings', label: 'My Listings',  icon: 'fa-list-ul' },
        ].map(t => (
          <li className="nav-item" key={t.key}>
            <button
              className={`nav-link ${tab === t.key ? 'active' : ''}`}
              onClick={() => setTab(t.key)}
            >
              <i className={`fas ${t.icon} me-1`} />{t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Browse / My Listings ── */}
      {(tab === 'browse' || tab === 'mylistings') && (
        <>
          {loading ? (
            <div className="text-center py-5">
              <HandLoader />
            </div>
          ) : displayList.length === 0 ? (
            <div className="text-center py-5 text-muted">
              <i className="fas fa-box-open fa-3x mb-3" />
              <p>{tab === 'mylistings' ? "You haven't posted anything yet." : 'No listings yet. Be the first to post!'}</p>
              {tab === 'mylistings' && (
                <button className="btn btn-primary" onClick={() => setTab('create')}>
                  Post a Listing
                </button>
              )}
            </div>
          ) : (
            <div className="row row-cols-1 row-cols-sm-2 row-cols-lg-3 row-cols-xl-4 g-3">
              {displayList.map(l => (
                <div className="col" key={l.id}>
                  <ListingCard
                    listing={l}
                    currentUser={user.username}
                    onSold={handleSold}
                    onDelete={handleDelete}
                    onInterested={handleInterested}
                    interestedSet={interestedSet}
                  />
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ── Create Form ── */}
      {tab === 'create' && (
        <div className="row justify-content-center">
          <div className="col-lg-7">
            <div className="card shadow-sm">
              <div className="card-body p-4">
                <h5 className="card-title mb-4 fw-semibold">
                  <i className="fas fa-tag me-2 text-primary" />Post a Listing
                </h5>

                <form onSubmit={handleCreate}>
                  <div className="mb-3">
                    <label className="form-label fw-medium">Title <span className="text-danger">*</span></label>
                    <input
                      className="form-control"
                      maxLength={100}
                      placeholder="e.g. iPhone 13 128GB"
                      value={form.title}
                      onChange={e => setForm(f => ({ ...f, title: e.target.value }))}
                    />
                  </div>

                  <div className="mb-3">
                    <label className="form-label fw-medium">Description <span className="text-danger">*</span></label>
                    <textarea
                      className="form-control"
                      rows={4}
                      placeholder="Condition, reason for selling, included accessories..."
                      value={form.description}
                      onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                    />
                  </div>

                  <div className="row mb-3">
                    <div className="col">
                      <label className="form-label fw-medium">Original Price (¥) <span className="text-muted small">(optional)</span></label>
                      <input
                        type="number"
                        className="form-control"
                        min={0}
                        step="0.01"
                        placeholder="e.g. 5000.00"
                        value={form.original_price}
                        onChange={e => setForm(f => ({ ...f, original_price: e.target.value }))}
                      />
                    </div>
                    <div className="col">
                      <label className="form-label fw-medium">Selling Price (¥) <span className="text-danger">*</span></label>
                      <input
                        type="number"
                        className="form-control"
                        min={0}
                        step="0.01"
                        placeholder="0.00"
                        value={form.price}
                        onChange={e => setForm(f => ({ ...f, price: e.target.value }))}
                      />
                    </div>
                  </div>

                  <div className="row mb-3">
                    <div className="col">
                      <label className="form-label fw-medium">Category</label>
                      <select
                        className="form-select"
                        value={form.category}
                        onChange={e => setForm(f => ({ ...f, category: e.target.value }))}
                      >
                        {CATEGORIES.map(c => (
                          <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="mb-4">
                    <label className="form-label fw-medium">Photos <span className="text-muted">(up to 3, JPEG/PNG, max 5MB each)</span></label>
                    <input
                      ref={fileRef}
                      type="file"
                      className="form-control"
                      accept=".jpg,.jpeg,.png"
                      multiple
                      onChange={handleFileChange}
                    />
                    {previews.length > 0 && (
                      <div className="d-flex gap-2 mt-2 flex-wrap">
                        {previews.map((src, i) => (
                          <div key={i} className="position-relative">
                            <img
                              src={src}
                              alt=""
                              style={{ width: 90, height: 90, objectFit: 'cover', borderRadius: 6 }}
                            />
                            <button
                              type="button"
                              className="btn btn-sm btn-danger position-absolute top-0 end-0"
                              style={{ padding: '1px 5px', fontSize: '0.7rem' }}
                              onClick={() => removeImage(i)}
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <button
                    type="submit"
                    className="btn btn-primary w-100"
                    disabled={submitting}
                  >
                    {submitting
                      ? <><span className="spinner-border spinner-border-sm me-2" />Posting…</>
                      : <><i className="fas fa-paper-plane me-2" />Post Listing</>
                    }
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
