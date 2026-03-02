import React, { useState, useEffect, useRef } from 'react'
import { api } from '../api'
import { useAuth } from '../App'

const CATEGORIES = ['electronics', 'clothing', 'books', 'furniture', 'other']

const EMPTY_FORM = { title: '', description: '', price: '', category: 'electronics', contact: '' }

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
function ListingCard({ listing, currentUser, onSold, onDelete }) {
  const isMine = listing.seller_username === currentUser
  const firstImg = listing.images?.[0]?.url

  return (
    <div className="card h-100 shadow-sm">
      {firstImg ? (
        <img
          src={firstImg}
          alt={listing.title}
          className="card-img-top"
          style={{ height: 180, objectFit: 'cover' }}
        />
      ) : (
        <div
          className="card-img-top bg-light d-flex align-items-center justify-content-center"
          style={{ height: 180 }}
        >
          <i className="fas fa-image fa-3x text-muted" />
        </div>
      )}

      <div className="card-body d-flex flex-column">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <h6 className="card-title mb-0 fw-semibold">{listing.title}</h6>
          <span className={`badge ms-2 ${listing.status === 'sold' ? 'bg-secondary' : 'bg-success'}`}>
            {listing.status === 'sold' ? 'Sold' : `¥${listing.price}`}
          </span>
        </div>

        <span className="badge bg-primary-subtle text-primary-emphasis mb-2" style={{ width: 'fit-content' }}>
          {listing.category}
        </span>

        <p className="card-text text-muted small flex-grow-1" style={{ whiteSpace: 'pre-wrap' }}>
          {listing.description.length > 100
            ? listing.description.slice(0, 100) + '…'
            : listing.description}
        </p>

        <div className="mt-auto pt-2 border-top small text-muted">
          <div><i className="fas fa-user me-1" />{listing.seller_username}</div>
          <div><i className="fas fa-phone me-1" />{listing.contact}</div>
          <div className="text-muted" style={{ fontSize: '0.75rem' }}>
            {new Date(listing.created_at).toLocaleDateString()}
          </div>
        </div>

        {isMine && listing.status === 'active' && (
          <div className="d-flex gap-2 mt-2">
            <button className="btn btn-sm btn-outline-secondary flex-grow-1" onClick={() => onSold(listing.id)}>
              Mark Sold
            </button>
            <button className="btn btn-sm btn-outline-danger flex-grow-1" onClick={() => onDelete(listing.id)}>
              Delete
            </button>
          </div>
        )}
        {isMine && listing.status === 'sold' && (
          <button className="btn btn-sm btn-outline-danger mt-2 w-100" onClick={() => onDelete(listing.id)}>
            Delete
          </button>
        )}
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function Market() {
  const { user } = useAuth()
  const [tab, setTab]           = useState('browse')
  const [listings, setListings] = useState([])
  const [myListings, setMy]     = useState([])
  const [loading, setLoading]   = useState(false)
  const [form, setForm]         = useState(EMPTY_FORM)
  const [images, setImages]     = useState([])      // File objects
  const [previews, setPreviews] = useState([])      // Object URLs
  const [submitting, setSub]    = useState(false)
  const [toast, showToast]      = useToast()
  const fileRef = useRef()

  // Load listings on tab switch
  useEffect(() => {
    if (tab === 'browse')     loadBrowse()
    if (tab === 'mylistings') loadMine()
  }, [tab])

  async function loadBrowse() {
    setLoading(true)
    const d = await api.get('/api/market/listings')
    if (d.ok) setListings(d.listings)
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
    if (!form.contact.trim())     return showToast('Contact info is required.', 'danger')
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
              <div className="spinner-border text-primary" />
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
                      <label className="form-label fw-medium">Price (¥) <span className="text-danger">*</span></label>
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

                  <div className="mb-3">
                    <label className="form-label fw-medium">Contact Info <span className="text-danger">*</span></label>
                    <input
                      className="form-control"
                      placeholder="WeChat ID / Phone number"
                      value={form.contact}
                      onChange={e => setForm(f => ({ ...f, contact: e.target.value }))}
                    />
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
