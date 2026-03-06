import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
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

// ── Seller Avatar ─────────────────────────────────────────────────────────────
function SellerAvatar({ username, displayName, avatarUrl, size = 28, onClick }) {
  const style = {
    width: size, height: size, borderRadius: '50%', flexShrink: 0,
    cursor: onClick ? 'pointer' : 'default',
  }
  if (avatarUrl) return (
    <img src={avatarUrl} alt={displayName} style={{ ...style, objectFit: 'cover' }} onClick={onClick} />
  )
  return (
    <div style={{
      ...style,
      background: '#3a7bd5', color: '#fff',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontWeight: 700, fontSize: size * 0.4,
    }} onClick={onClick}>
      {(displayName || username)?.[0]?.toUpperCase() || '?'}
    </div>
  )
}

// ── Listing Card ──────────────────────────────────────────────────────────────
function ListingCard({ listing, currentUser, onSold, onDelete, onReachOut, onSellerClick, reachOutStatus }) {
  const isMine      = listing.seller_username === currentUser
  const firstImg    = listing.images?.[0]?.url
  const isSold      = listing.status === 'sold'
  const hasOriginal = listing.original_price && listing.original_price > listing.price

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
        <div
          className="market-card__seller"
          style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6 }}
          onClick={() => onSellerClick(listing.seller_username)}
          title={`View ${listing.seller_username}'s listings`}
        >
          <SellerAvatar
            username={listing.seller_username}
            displayName={listing.seller_display}
            avatarUrl={listing.seller_avatar}
            size={26}
          />
          <div>
            <div style={{ fontWeight: 600, fontSize: '.78rem' }}>{listing.seller_display || listing.seller_username}</div>
            <div style={{ fontSize: '.7rem' }}>{new Date(listing.created_at).toLocaleDateString()}</div>
          </div>
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

      {/* Reach Out for other users' active listings */}
      {!isMine && !isSold && (
        <div className="market-card__action">
          {reachOutStatus === 'friends' ? (
            <button
              className="market-card__btn"
              style={{ background: '#22c55e', border: 'none' }}
              onClick={() => onReachOut(listing)}
            >
              <i className="fas fa-comment-dots me-1" />Reach Out
            </button>
          ) : reachOutStatus === 'sent' ? (
            <span className="badge bg-warning text-dark px-3 py-2" style={{ fontSize: '.8rem' }}>
              <i className="fas fa-clock me-1" />Request Sent
            </span>
          ) : (
            <button
              className="market-card__btn"
              style={{ background: '#3a7bd5', color: '#fff', border: 'none' }}
              onClick={() => onReachOut(listing)}
            >
              <i className="fas fa-paper-plane me-1" />Reach Out
            </button>
          )}
        </div>
      )}
    </div>
  )
}

// ── Seller Modal ──────────────────────────────────────────────────────────────
function SellerModal({ seller, listings, onClose, onReachOut, reachOutStatus }) {
  return (
    <div className="modal show d-block" style={{ background: 'rgba(0,0,0,.45)' }} onClick={onClose}>
      <div className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable"
        onClick={e => e.stopPropagation()}>
        <div className="modal-content">
          <div className="modal-header">
            <div className="d-flex align-items-center gap-3">
              <SellerAvatar username={seller.username} displayName={seller.display_name} avatarUrl={seller.avatar_url} size={44} />
              <div>
                <div className="fw-bold">{seller.display_name || seller.username}</div>
                <div className="text-muted small">@{seller.username}</div>
              </div>
              {reachOutStatus === 'friends' ? (
                <button className="btn btn-sm btn-success ms-2" onClick={() => onReachOut(null)}>
                  <i className="fas fa-comment-dots me-1" />Chat
                </button>
              ) : reachOutStatus === 'sent' ? (
                <span className="badge bg-warning text-dark ms-2">Request Sent</span>
              ) : (
                <button className="btn btn-sm btn-primary ms-2" onClick={() => onReachOut(null)}>
                  <i className="fas fa-user-plus me-1" />Add Friend
                </button>
              )}
            </div>
            <button className="btn-close" onClick={onClose} />
          </div>
          <div className="modal-body">
            <p className="text-muted small mb-3">
              <i className="fas fa-store me-1" />{listings.length} active listing{listings.length !== 1 ? 's' : ''}
            </p>
            {listings.length === 0 ? (
              <div className="text-center py-4 text-muted">
                <i className="fas fa-box-open fa-2x mb-2 d-block opacity-25" />
                No active listings.
              </div>
            ) : (
              <div className="row row-cols-1 row-cols-sm-2 g-3">
                {listings.map(l => (
                  <div className="col" key={l.id}>
                    <div className="card h-100 p-2">
                      {l.images?.[0]?.url && (
                        <img src={l.images[0].url} alt={l.title}
                          style={{ width: '100%', height: 120, objectFit: 'cover', borderRadius: 6 }} />
                      )}
                      <div className="fw-semibold mt-2 small text-truncate">{l.title}</div>
                      <div className="text-primary fw-bold">¥{l.price}
                        {l.original_price > l.price && (
                          <span style={{ fontSize: '.7rem', color: '#999', textDecoration: 'line-through', marginLeft: 6 }}>
                            ¥{l.original_price}
                          </span>
                        )}
                      </div>
                      <div className="text-muted small text-truncate">{l.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function Market() {
  const { user }    = useAuth()
  const navigate    = useNavigate()
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
  // friendsMap: { [username]: { username, display_name, avatar_url } }
  const [friendsMap, setFriendsMap] = useState({})
  // sellerModal: { username, display_name, avatar_url } | null
  const [sellerModal, setSellerModal]   = useState(null)
  const [sellerListings, setSellerListings] = useState([])
  const [sellerLoading, setSellerLoading]  = useState(false)
  const fileRef = useRef()

  // Load listings on tab switch
  useEffect(() => {
    if (tab === 'browse')      loadBrowse()
    if (tab === 'mylistings')  loadMine()
  }, [tab])

  async function loadBrowse() {
    setLoading(true)
    const [listRes, friendsRes, sentRes] = await Promise.all([
      api.get('/api/market/listings'),
      api.get('/api/friends/list'),
      api.get('/api/friends/requests/sent'),
    ])

    // Build friendsMap
    const fMap = {}
    if (friendsRes.ok) friendsRes.friends.forEach(f => { fMap[f.username] = f })
    setFriendsMap(fMap)

    // Build interestedSet
    const map = {}
    if (friendsRes.ok)  friendsRes.friends.forEach(f => { map[f.username] = 'friends' })
    if (sentRes.ok)     sentRes.requests.filter(r => r.status === 'pending').forEach(r => { map[r.to_user] = 'sent' })
    setInterested(map)

    if (listRes.ok) setListings(listRes.listings)
    setLoading(false)
  }

  async function loadMine() {
    setLoading(true)
    const d = await api.get('/api/market/my')
    if (d.ok) setMy(d.listings)
    setLoading(false)
  }

  async function openSellerModal(username) {
    // Don't show modal for own profile
    if (username === user.username) return
    const friend = friendsMap[username]
    const sellerInfo = friend ?? { username, display_name: username, avatar_url: null }
    setSellerModal(sellerInfo)
    setSellerListings([])
    setSellerLoading(true)
    const d = await api.get(`/api/market/user/${username}`)
    if (d.ok) setSellerListings(d.listings)
    setSellerLoading(false)
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

  // Reach Out: friends → navigate to chat; not friends → send friend request
  async function handleReachOut(listing, sellerUsername) {
    const username = sellerUsername ?? listing?.seller_username
    const title    = listing?.title ?? null
    const status   = interestedSet[username]

    if (status === 'friends') {
      const friend = friendsMap[username]
      const initialMessage = title
        ? `嗨！我看到你发布的《${title}》，想聊一聊 😊`
        : ''
      // Close seller modal if open
      setSellerModal(null)
      navigate('/friends', { state: { openChat: friend, initialMessage } })
      return
    }

    // Not friends — send friend request
    const msg = listing
      ? `Hi! I saw your listing "${listing.title}" and would like to connect!`
      : `Hi! I'd like to connect with you!`
    const d = await api.post('/api/friends/requests', { to_user: username, message: msg })
    if (d.ok) {
      setInterested(prev => ({ ...prev, [username]: 'sent' }))
      showToast('Friend request sent!')
    } else if (d.error === 'Already friends') {
      setInterested(prev => ({ ...prev, [username]: 'friends' }))
    } else if (d.error === 'Request already pending') {
      setInterested(prev => ({ ...prev, [username]: 'sent' }))
    } else {
      showToast(d.error, 'danger')
    }
  }

  // Browse: filter out own listings
  const browseListing = listings.filter(l => l.seller_username !== user.username)
  const displayList   = tab === 'mylistings' ? myListings : browseListing

  return (
    <div className="container-fluid py-4">

      {/* Toast */}
      {toast && (
        <div className={`alert alert-${toast.type} alert-dismissible position-fixed top-0 end-0 m-3`}
             style={{ zIndex: 9999 }}>
          {toast.msg}
        </div>
      )}

      {/* Seller modal */}
      {sellerModal && (
        <SellerModal
          seller={sellerModal}
          listings={sellerLoading ? [] : sellerListings}
          onClose={() => setSellerModal(null)}
          onReachOut={(listing) => handleReachOut(listing, sellerModal.username)}
          reachOutStatus={interestedSet[sellerModal.username]}
        />
      )}

      <div className="d-flex align-items-center mb-4">
        <i className="fas fa-store fa-lg me-2 text-primary" />
        <h4 className="mb-0 fw-bold">Market</h4>
        <span className="text-muted ms-2 small">Second-hand trading</span>
      </div>

      {/* Tabs — Browse → My Listings → Post Item */}
      <ul className="nav nav-tabs mb-4">
        {[
          { key: 'browse',     label: 'Browse',      icon: 'fa-th-large' },
          { key: 'mylistings', label: 'My Listings',  icon: 'fa-list-ul' },
          { key: 'create',     label: 'Post Item',    icon: 'fa-plus-circle' },
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
                    onReachOut={handleReachOut}
                    onSellerClick={openSellerModal}
                    reachOutStatus={interestedSet[l.seller_username]}
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

                  <div className="mb-3">
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
