import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'
import HandLoader from '../components/HandLoader'

const FALLBACK_CATEGORIES = [
  { slug: 'clothing',    label: '衣服' },
  { slug: 'furniture',   label: '家具' },
  { slug: 'kitchen',     label: '厨具' },
  { slug: 'electronics', label: 'Electronics' },
  { slug: 'beauty',      label: '美妆' },
]

const CATEGORY_ICONS = {
  clothing: 'fa-tshirt', furniture: 'fa-couch', kitchen: 'fa-utensils',
  electronics: 'fa-laptop', beauty: 'fa-spa', books: 'fa-book', other: 'fa-box',
}

const EMPTY_FORM = { title: '', description: '', price: '', original_price: '', category: 'clothing', delivery_type: 'pickup', delivery_fee: '' }

// ── Toast ─────────────────────────────────────────────────────────────────────
function useToast() {
  const [toast, setToast] = useState(null)
  const show = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 2800)
  }
  return [toast, show]
}

// ── Price Display ────────────────────────────────────────────────────────────
function PriceDisplay({ listing, large = false }) {
  const { price, original_price, delivery_type, delivery_fee } = listing
  const hasOriginal = original_price && original_price > price
  const hasFee      = delivery_fee && delivery_fee > 0
  const showBoth    = delivery_type === 'both' && hasFee
  const deliveryOnly = delivery_type === 'delivery' && hasFee
  const freeDelivery = (delivery_type === 'both' || delivery_type === 'delivery') && !hasFee

  const bigSz  = large ? '1.5rem' : undefined
  const smSz   = large ? '.85rem' : '.75rem'
  const labelSz = large ? '.78rem' : '.65rem'

  if (showBoth) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
          <span style={{ fontSize: bigSz, fontWeight: 600 }}>${price}</span>
          {hasOriginal && <span style={{ fontSize: smSz, color: '#999', textDecoration: 'line-through' }}>${original_price}</span>}
          <span style={{ fontSize: labelSz, color: '#888', background: '#f0f0f0', borderRadius: 8, padding: '1px 5px' }}>自提</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
          <span style={{ fontSize: large ? '1.1rem' : '.85rem', fontWeight: 600, color: '#3b5bdb' }}>${price + delivery_fee}</span>
          <span style={{ fontSize: labelSz, color: '#3b5bdb', background: '#e8f0fe', borderRadius: 8, padding: '1px 5px' }}>含${delivery_fee}配送</span>
        </div>
      </div>
    )
  }

  if (deliveryOnly) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
          <span style={{ fontSize: bigSz, fontWeight: 600, color: '#3b5bdb' }}>${price + delivery_fee}</span>
          {hasOriginal && <span style={{ fontSize: smSz, color: '#999', textDecoration: 'line-through' }}>${original_price}</span>}
        </div>
        <span style={{ fontSize: labelSz, color: '#3b5bdb', background: '#e8f0fe', borderRadius: 8, padding: '1px 5px' }}>含${delivery_fee}配送费</span>
      </div>
    )
  }

  if (freeDelivery) {
    return (
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
        <span style={{ fontSize: bigSz, fontWeight: 600 }}>${price}</span>
        {hasOriginal && <span style={{ fontSize: smSz, color: '#999', textDecoration: 'line-through' }}>${original_price}</span>}
        <span style={{ fontSize: labelSz, color: '#27ae60', background: '#e8f8f0', borderRadius: 8, padding: '1px 5px' }}>包邮</span>
      </div>
    )
  }

  // pickup only or no fee
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
      <span style={{ fontSize: bigSz, fontWeight: 600 }}>${price}</span>
      {hasOriginal && <span style={{ fontSize: smSz, color: '#999', textDecoration: 'line-through' }}>${original_price}</span>}
    </div>
  )
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
      background: '#6b9cdb', color: '#fff',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontWeight: 700, fontSize: size * 0.4,
    }} onClick={onClick}>
      {(displayName || username)?.[0]?.toUpperCase() || '?'}
    </div>
  )
}

// ── Edit Modal ────────────────────────────────────────────────────────────────
function EditModal({ listing, onClose, onSave, categories }) {
  const [form, setForm] = useState({
    title:          listing.title,
    description:    listing.description,
    price:          String(listing.price),
    original_price: listing.original_price != null ? String(listing.original_price) : '',
    category:       listing.category,
    delivery_type:  listing.delivery_type || 'pickup',
    delivery_fee:   listing.delivery_fee != null ? String(listing.delivery_fee) : '',
  })
  const [saving, setSaving] = useState(false)
  const [err, setErr]       = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.title.trim())       return setErr('Title is required.')
    if (!form.description.trim()) return setErr('Description is required.')
    if (!form.price || isNaN(form.price)) return setErr('Enter a valid price.')
    setSaving(true)
    await onSave(listing.id, {
      title:          form.title.trim(),
      description:    form.description.trim(),
      price:          Number(form.price),
      original_price: form.original_price !== '' ? Number(form.original_price) : '',
      category:       form.category,
      delivery_type:  form.delivery_type,
      delivery_fee:   form.delivery_fee !== '' ? Number(form.delivery_fee) : null,
    })
    setSaving(false)
  }

  return (
    <div className="modal show d-block" style={{ background: 'rgba(0,0,0,.45)' }} onClick={onClose}>
      <div className="modal-dialog modal-dialog-centered" onClick={e => e.stopPropagation()}>
        <div className="modal-content">
          <form onSubmit={handleSubmit}>
            <div className="modal-header">
              <h5 className="modal-title fw-semibold">Edit Listing</h5>
              <button type="button" className="btn-close" onClick={onClose} />
            </div>
            <div className="modal-body">
              {err && <div className="alert alert-danger py-2 small">{err}</div>}
              <div className="mb-3">
                <label className="form-label fw-medium">Title</label>
                <input className="form-control" maxLength={100} value={form.title}
                  onChange={e => setForm(f => ({ ...f, title: e.target.value }))} required />
              </div>
              <div className="mb-3">
                <label className="form-label fw-medium">Description</label>
                <textarea className="form-control" rows={3} value={form.description}
                  onChange={e => setForm(f => ({ ...f, description: e.target.value }))} required />
              </div>
              <div className="row mb-3">
                <div className="col">
                  <label className="form-label fw-medium">Original Price ($)</label>
                  <input type="number" className="form-control" min={0} step="0.01"
                    placeholder="optional" value={form.original_price}
                    onChange={e => setForm(f => ({ ...f, original_price: e.target.value }))} />
                </div>
                <div className="col">
                  <label className="form-label fw-medium">Selling Price ($)</label>
                  <input type="number" className="form-control" min={0} step="0.01"
                    value={form.price}
                    onChange={e => setForm(f => ({ ...f, price: e.target.value }))} required />
                </div>
              </div>
              <div className="mb-1">
                <label className="form-label fw-medium">Category</label>
                <select className="form-select" value={form.category}
                  onChange={e => setForm(f => ({ ...f, category: e.target.value }))}>
                  {categories.map(c => (
                    <option key={c.slug} value={c.slug}>{c.label}</option>
                  ))}
                </select>
              </div>
              <div className="mt-3">
                <label className="form-label fw-medium">Delivery Options</label>
                <div className="d-flex gap-3 flex-wrap">
                  {[['pickup','Self-pickup'],['delivery','Delivery'],['both','Both']].map(([v,lbl]) => (
                    <div className="form-check" key={v}>
                      <input className="form-check-input" type="radio" name="edit-delivery"
                        id={`edit-dt-${v}`} value={v} checked={form.delivery_type === v}
                        onChange={() => setForm(f => ({ ...f, delivery_type: v }))} />
                      <label className="form-check-label small" htmlFor={`edit-dt-${v}`}>{lbl}</label>
                    </div>
                  ))}
                </div>
                {(form.delivery_type === 'delivery' || form.delivery_type === 'both') && (
                  <div className="mt-2">
                    <input type="number" className="form-control form-control-sm" min={0} step="0.01"
                      placeholder="Delivery fee ($, blank = free)" value={form.delivery_fee}
                      onChange={e => setForm(f => ({ ...f, delivery_fee: e.target.value }))} />
                  </div>
                )}
              </div>
            </div>
            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
              <button type="submit" className="btn btn-primary" disabled={saving}>
                {saving ? <span className="spinner-border spinner-border-sm me-1" /> : null}
                Save Changes
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

// ── Listing Card ──────────────────────────────────────────────────────────────
function ListingCard({ listing, currentUser, onSold, onRestore, onDelete, onEdit, onReachOut, onSellerClick, onDetail, reachOutStatus }) {
  const isMine      = listing.seller_username === currentUser
  const firstImg    = listing.images?.[0]?.url
  const isSold      = listing.status === 'sold'
  const hasOriginal = listing.original_price && listing.original_price > listing.price

  return (
    <div className="market-card">
      {/* Image — click opens detail */}
      <div className="market-card__img" onClick={onDetail} style={{ cursor: 'pointer' }}>
        {firstImg
          ? <img src={firstImg} alt={listing.title} />
          : <i className="fas fa-image placeholder-icon" />
        }
      </div>

      {/* Title — click opens detail */}
      <div className="market-card__title" title={listing.title}
           style={{ cursor: 'pointer' }} onClick={onDetail}>{listing.title}</div>

      {/* Category + delivery + sold badge + views */}
      <div className="market-card__meta">
        <span className="market-card__category">{listing.category}</span>
        {(listing.delivery_type === 'delivery' || listing.delivery_type === 'both') ? (
          <span style={{ fontSize: '.62rem', background: '#e8f0fe', color: '#3b5bdb', borderRadius: 10, padding: '1px 6px', whiteSpace: 'nowrap' }}>
            <i className="fas fa-truck me-1" />{listing.delivery_type === 'both' ? 'Delivery/Pickup' : 'Delivery'}
          </span>
        ) : (
          <span style={{ fontSize: '.62rem', background: '#f0f0f0', color: '#666', borderRadius: 10, padding: '1px 6px', whiteSpace: 'nowrap' }}>
            <i className="fas fa-walking me-1" />Pickup
          </span>
        )}
        {isSold && <span className="market-card__sold-badge">Sold</span>}
        {listing.view_count > 0 && (
          <span style={{ fontSize: '.62rem', color: '#aaa', whiteSpace: 'nowrap', marginLeft: 'auto' }}>
            <i className="fas fa-eye me-1" />{listing.view_count}
          </span>
        )}
      </div>

      {/* Description */}
      <p className="market-card__desc">{listing.description}</p>

      <hr className="market-card__divider" />

      {/* Price row */}
      {!isSold && (
        <div className="market-card__price mb-1">
          <PriceDisplay listing={listing} />
        </div>
      )}

      {/* Seller row */}
      <div
        className="market-card__seller"
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}
        onClick={() => onSellerClick(listing.seller_username)}
        title={`View ${listing.seller_username}'s listings`}
      >
        <SellerAvatar
          username={listing.seller_username}
          displayName={listing.seller_display}
          avatarUrl={listing.seller_avatar}
          size={24}
        />
        <div>
          <div style={{ fontWeight: 600, fontSize: '.75rem' }}>{listing.seller_display || listing.seller_username}</div>
          <div style={{ fontSize: '.68rem' }}>{new Date(listing.created_at).toLocaleDateString()}</div>
        </div>
      </div>

      {/* Owner actions */}
      {isMine && (
        <div className="market-card__action">
          {!isSold && (
            <>
              <button className="market-card__btn" onClick={() => onSold(listing.id)}>
                <i className="fas fa-check-circle" />Mark Sold
              </button>
              <button className="market-card__btn market-card__btn--edit" onClick={() => onEdit(listing)}>
                <i className="fas fa-pen" />Edit
              </button>
            </>
          )}
          {isSold && (
            <button className="market-card__btn market-card__btn--restore" onClick={() => onRestore(listing.id)}>
              <i className="fas fa-undo" />Restore
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
              className="market-card__btn market-card__btn--reach"
              onClick={() => onReachOut(listing)}
            >
              <i className="fas fa-comment-dots" />Reach Out
            </button>
          ) : reachOutStatus === 'sent' ? (
            <span className="badge bg-warning text-dark px-3 py-2" style={{ fontSize: '.8rem' }}>
              <i className="fas fa-clock me-1" />Request Sent
            </span>
          ) : (
            <button
              className="market-card__btn market-card__btn--reach"
              onClick={() => onReachOut(listing)}
            >
              <i className="fas fa-paper-plane" />Reach Out
            </button>
          )}
        </div>
      )}
    </div>
  )
}

// ── Listing Detail Modal ──────────────────────────────────────────────────────
function ListingDetailModal({ listing, currentUser, onClose, onSold, onRestore, onDelete, onEdit, onReachOut, onSellerClick, reachOutStatus }) {
  const [imgIndex, setImgIndex] = useState(0)
  const isMine      = listing.seller_username === currentUser
  const isSold      = listing.status === 'sold'
  const hasOriginal = listing.original_price && listing.original_price > listing.price

  return (
    <div className="modal show d-block" style={{ background: 'rgba(0,0,0,.55)' }} onClick={onClose}>
      <div className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable"
           onClick={e => e.stopPropagation()}>
        <div className="modal-content"
             style={{ border: '2px solid #323232', borderRadius: 6, boxShadow: '6px 6px #323232' }}>

          {/* Header */}
          <div className="modal-header" style={{ borderBottom: '1px solid #323232' }}>
            <h5 className="modal-title fw-semibold" style={{ fontSize: '1rem' }}>{listing.title}</h5>
            <button className="btn-close" onClick={onClose} />
          </div>

          <div className="modal-body">
            {/* Images */}
            {listing.images?.length > 0 && (
              <div className="mb-3">
                <img
                  src={listing.images[imgIndex].url}
                  alt={listing.title}
                  style={{
                    width: '100%', maxHeight: 380,
                    objectFit: 'contain',
                    background: '#f5f5f5',
                    borderRadius: 4,
                    border: '1px solid #e0e0e0',
                  }}
                />
                {listing.images.length > 1 && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                    {listing.images.map((img, i) => (
                      <img
                        key={img.id} src={img.url}
                        onClick={() => setImgIndex(i)}
                        style={{
                          width: 58, height: 58, objectFit: 'cover',
                          border: i === imgIndex ? '2px solid #323232' : '2px solid #e0e0e0',
                          borderRadius: 4, cursor: 'pointer',
                        }}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Price + badges */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span className="market-card__category">{listing.category}</span>
                {isSold && <span className="market-card__sold-badge">Sold</span>}
              </div>
              {!isSold && (
                <PriceDisplay listing={listing} large />
              )}
            </div>

            {/* Delivery info */}
            <div style={{ marginBottom: 12, fontSize: '.82rem' }}>
              {listing.delivery_type === 'pickup' && (
                <span style={{ background: '#f0f0f0', color: '#555', borderRadius: 12, padding: '2px 10px' }}>
                  <i className="fas fa-walking me-1" />Self-pickup only
                </span>
              )}
              {listing.delivery_type === 'delivery' && (
                <span style={{ background: '#e8f0fe', color: '#3b5bdb', borderRadius: 12, padding: '2px 10px' }}>
                  <i className="fas fa-truck me-1" />Delivery{listing.delivery_fee != null ? ` +$${listing.delivery_fee}` : ' (free)'}
                </span>
              )}
              {listing.delivery_type === 'both' && (
                <span style={{ background: '#e8f0fe', color: '#3b5bdb', borderRadius: 12, padding: '2px 10px' }}>
                  <i className="fas fa-truck me-1" />Delivery{listing.delivery_fee != null ? ` +$${listing.delivery_fee}` : ' (free)'}
                  <span className="mx-2 text-muted">·</span>
                  <i className="fas fa-walking me-1" />Self-pickup
                </span>
              )}
            </div>

            <hr className="market-card__divider" />

            {/* Description */}
            <p style={{ fontSize: '.9rem', color: '#555', lineHeight: 1.7, whiteSpace: 'pre-wrap', marginBottom: 16 }}>
              {listing.description}
            </p>

            <hr className="market-card__divider" />

            {/* Seller */}
            <div
              style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', marginTop: 12 }}
              onClick={() => { onSellerClick(listing.seller_username); onClose() }}
            >
              <SellerAvatar
                username={listing.seller_username}
                displayName={listing.seller_display}
                avatarUrl={listing.seller_avatar}
                size={34}
              />
              <div>
                <div style={{ fontWeight: 600, fontSize: '.85rem' }}>
                  {listing.seller_display || listing.seller_username}
                </div>
                <div style={{ fontSize: '.75rem', color: '#888' }}>
                  Posted {new Date(listing.created_at).toLocaleDateString()}
                  {listing.view_count > 0 && (
                    <span className="ms-2">
                      <i className="fas fa-eye me-1" />{listing.view_count} view{listing.view_count !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>
              </div>
              <i className="fas fa-chevron-right ms-auto text-muted" style={{ fontSize: '.75rem' }} />
            </div>
          </div>

          {/* Footer actions */}
          <div className="modal-footer" style={{ borderTop: '1px solid #323232' }}>
            {isMine ? (
              <div style={{ display: 'flex', gap: 8, width: '100%' }}>
                {!isSold && (
                  <>
                    <button className="market-card__btn" style={{ flex: 1 }}
                      onClick={() => { onSold(listing.id); onClose() }}>
                      <i className="fas fa-check-circle" />Mark Sold
                    </button>
                    <button className="market-card__btn market-card__btn--edit" style={{ flex: 1 }}
                      onClick={() => { onEdit(listing); onClose() }}>
                      <i className="fas fa-pen" />Edit
                    </button>
                  </>
                )}
                {isSold && (
                  <button className="market-card__btn market-card__btn--restore" style={{ flex: 1 }}
                    onClick={() => { onRestore(listing.id); onClose() }}>
                    <i className="fas fa-undo" />Restore
                  </button>
                )}
                <button className="market-card__btn market-card__btn--danger" style={{ flex: 1 }}
                  onClick={() => { onDelete(listing.id); onClose() }}>
                  <i className="fas fa-trash" />Delete
                </button>
              </div>
            ) : !isSold ? (
              <div style={{ display: 'flex', gap: 8, width: '100%' }}>
                {reachOutStatus === 'sent' ? (
                  <span className="badge bg-warning text-dark px-3 py-2" style={{ fontSize: '.85rem' }}>
                    <i className="fas fa-clock me-1" />Request Sent
                  </span>
                ) : (
                  <button className="market-card__btn market-card__btn--reach" style={{ flex: 1 }}
                    onClick={() => { onReachOut(listing); onClose() }}>
                    <i className={`fas ${reachOutStatus === 'friends' ? 'fa-comment-dots' : 'fa-paper-plane'}`} />
                    Reach Out
                  </button>
                )}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Seller Modal ──────────────────────────────────────────────────────────────
function SellerModal({ seller, listings, onClose, onReachOut, reachOutStatus }) {
  const navigate = useNavigate()
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
                <div className="text-muted small">
                  @{seller.username} ·{' '}
                  <span
                    style={{ color: '#6b9cdb', cursor: 'pointer' }}
                    onClick={() => { onClose(); navigate(`/u/${seller.username}`) }}
                  >
                    View Profile →
                  </span>
                </div>
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
              <div className="row row-cols-2 row-cols-sm-3 g-2">
                {listings.map(l => (
                  <div className="col" key={l.id}>
                    <ListingCard
                      listing={l} currentUser={null}
                      onSold={() => {}} onRestore={() => {}} onDelete={() => {}}
                      onEdit={() => {}} onReachOut={() => {}} onSellerClick={() => {}}
                      onDetail={() => {}} reachOutStatus={null}
                    />
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
  const [categories, setCategories] = useState(FALLBACK_CATEGORIES)
  const [searchQuery, setSearchQuery]   = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
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
  const [editListing, setEditListing]   = useState(null)
  const [detailListing, setDetailListing] = useState(null)
  // sellerModal: { username, display_name, avatar_url } | null
  const [sellerModal, setSellerModal]   = useState(null)
  const [sellerListings, setSellerListings] = useState([])
  const [sellerLoading, setSellerLoading]  = useState(false)
  const [showExport, setShowExport]     = useState(false)
  const [copied, setCopied]             = useState(false)
  const fileRef = useRef()
  const tabRef  = useRef(tab)
  useEffect(() => { tabRef.current = tab }, [tab])

  // Fetch active categories once on mount
  useEffect(() => {
    api.get('/api/market/categories').then(d => {
      if (d.ok && d.categories?.length)
        setCategories(d.categories.map(c => ({ slug: c.slug, label: c.label })))
    })
  }, [])

  // Load listings on tab switch; reset filters on browse
  useEffect(() => {
    if (tab === 'browse')     { loadBrowse(); setSearchQuery(''); setCategoryFilter('all') }
    if (tab === 'mylistings')   loadMine()
  }, [tab])

  // Refresh when user returns to this tab
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState !== 'visible') return
      if (tabRef.current === 'browse')      loadBrowse()
      else if (tabRef.current === 'mylistings') loadMine()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [])

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

  async function handleRestore(id) {
    const d = await api.post(`/api/market/listings/${id}/restore`)
    if (d.ok) {
      showToast('Listing restored to active.')
      if (tab === 'browse')     loadBrowse()
      if (tab === 'mylistings') loadMine()
    } else {
      showToast(d.error, 'danger')
    }
  }

  async function handleEditSave(id, fields) {
    const d = await api.put(`/api/market/listings/${id}`, fields)
    if (d.ok) {
      showToast('Listing updated.')
      setEditListing(null)
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
  async function openDetail(listing) {
    const key = `viewed_${listing.id}`
    if (sessionStorage.getItem(key)) {
      // Already viewed this session — show cached listing, skip increment
      setDetailListing(listing)
      return
    }
    // First view this session — fetch from API (server increments count)
    const d = await api.get(`/api/market/listings/${listing.id}`)
    if (d.ok) {
      sessionStorage.setItem(key, '1')
      setDetailListing(d.listing)
      // Update view_count in the local listings arrays too
      setListings(prev => prev.map(l => l.id === listing.id ? { ...l, view_count: d.listing.view_count } : l))
      setMy(prev => prev.map(l => l.id === listing.id ? { ...l, view_count: d.listing.view_count } : l))
    } else {
      setDetailListing(listing)
    }
  }

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

  function buildExportText() {
    return myListings
      .filter(l => l.status === 'active')
      .map((l, i) => {
        const orig = l.original_price && l.original_price > l.price ? `（原价 $${l.original_price}）` : ''
        return `${i + 1}. ${l.title}  $${l.price}${orig}`
      })
      .join('\n')
  }

  function handleCopyExport() {
    navigator.clipboard.writeText(buildExportText()).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  // Browse: filter out own listings, then apply search + category
  const browseListing = listings.filter(l => l.seller_username !== user.username)
  const filteredBrowse = browseListing.filter(l => {
    const q = searchQuery.trim().toLowerCase()
    const matchesSearch = !q ||
      l.title.toLowerCase().includes(q) ||
      l.description.toLowerCase().includes(q)
    const matchesCategory = categoryFilter === 'all' || l.category === categoryFilter
    return matchesSearch && matchesCategory
  })
  const displayList = tab === 'mylistings' ? myListings : filteredBrowse

  return (
    <div className="container-fluid py-4">

      {/* Listing Detail Modal */}
      {detailListing && (
        <ListingDetailModal
          listing={detailListing}
          currentUser={user.username}
          onClose={() => setDetailListing(null)}
          onSold={handleSold}
          onRestore={handleRestore}
          onDelete={handleDelete}
          onEdit={setEditListing}
          onReachOut={handleReachOut}
          onSellerClick={openSellerModal}
          reachOutStatus={interestedSet[detailListing.seller_username]}
        />
      )}

      {/* Toast */}
      {toast && (
        <div className={`alert alert-${toast.type} alert-dismissible position-fixed top-0 end-0 m-3`}
             style={{ zIndex: 9999 }}>
          {toast.msg}
        </div>
      )}

      {/* Export modal */}
      {showExport && (
        <div className="modal show d-block" style={{ background: 'rgba(0,0,0,.45)' }} onClick={() => setShowExport(false)}>
          <div className="modal-dialog modal-dialog-centered" onClick={e => e.stopPropagation()}>
            <div className="modal-content">
              <div className="modal-header">
                <h6 className="modal-title fw-semibold">
                  <i className="fas fa-file-export me-2 text-primary" />Export My Listings
                </h6>
                <button className="btn-close" onClick={() => setShowExport(false)} />
              </div>
              <div className="modal-body">
                <textarea
                  className="form-control font-monospace"
                  rows={Math.min(myListings.filter(l => l.status === 'active').length + 2, 14)}
                  readOnly
                  value={buildExportText()}
                  style={{ fontSize: '.85rem', resize: 'none' }}
                  onClick={e => e.target.select()}
                />
                <div className="text-muted small mt-2">Click the text area to select all, or use the button below.</div>
              </div>
              <div className="modal-footer">
                <button className="btn btn-outline-secondary btn-sm" onClick={() => setShowExport(false)}>Close</button>
                <button
                  className={`btn btn-sm ${copied ? 'btn-success' : 'btn-primary'}`}
                  onClick={handleCopyExport}
                >
                  <i className={`fas ${copied ? 'fa-check' : 'fa-copy'} me-1`} />
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Edit listing modal */}
      {editListing && (
        <EditModal
          listing={editListing}
          onClose={() => setEditListing(null)}
          onSave={handleEditSave}
          categories={categories}
        />
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
      <div className="radio-inputs mb-4">
        {[
          { key: 'browse',     label: 'Browse' },
          { key: 'mylistings', label: 'My Listings' },
          { key: 'create',     label: 'Post Item' },
        ].map(t => (
          <label className="radio" key={t.key}>
            <input
              type="radio"
              name="market-tab"
              checked={tab === t.key}
              onChange={() => setTab(t.key)}
            />
            <span className="name">{t.label}</span>
          </label>
        ))}
      </div>

      {/* ── Browse / My Listings ── */}
      {(tab === 'browse' || tab === 'mylistings') && (
        <>
          {/* Search + Category filter — browse only */}
          {tab === 'browse' && (
            <div className="mb-4">
              {/* Search bar */}
              <div className="search mb-3">
                <input
                  className="search__input"
                  placeholder="Search listings…"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
                <button className="search__button" onClick={() => {}}>
                  <i className="fas fa-search" />
                </button>
                {searchQuery && (
                  <button className="search__clear" onClick={() => setSearchQuery('')}>
                    <i className="fas fa-times" />
                  </button>
                )}
              </div>

              {/* Category pills — horizontally scrollable on mobile */}
              <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
                <div className="d-flex gap-2" style={{ width: 'max-content', paddingBottom: 4 }}>
                  {[{ slug: 'all', label: 'All' }, ...categories].map(cat => {
                    const icon   = CATEGORY_ICONS[cat.slug] || 'fa-tag'
                    const active = categoryFilter === cat.slug
                    return (
                      <button
                        key={cat.slug}
                        onClick={() => setCategoryFilter(cat.slug)}
                        className={`btn btn-sm ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
                        style={{ borderRadius: 20, whiteSpace: 'nowrap' }}
                      >
                        <i className={`fas ${icon} me-1`} />{cat.label}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Active filter summary */}
              {(searchQuery || categoryFilter !== 'all') && !loading && (
                <div className="d-flex align-items-center gap-2 mt-2">
                  <span className="text-muted small">
                    {filteredBrowse.length} result{filteredBrowse.length !== 1 ? 's' : ''}
                  </span>
                  <button className="btn btn-link btn-sm p-0 text-muted"
                    onClick={() => { setSearchQuery(''); setCategoryFilter('all') }}>
                    Clear filters
                  </button>
                </div>
              )}
            </div>
          )}

          {tab === 'mylistings' && myListings.length > 0 && (
            <div className="d-flex justify-content-end mb-3">
              <button className="btn btn-outline-secondary btn-sm" onClick={() => setShowExport(true)}>
                <i className="fas fa-file-export me-1" />Export List
              </button>
            </div>
          )}
          {loading ? (
            <div className="text-center py-5">
              <HandLoader />
            </div>
          ) : displayList.length === 0 ? (
            <div className="text-center py-5 text-muted">
              <i className="fas fa-box-open fa-3x mb-3" />
              {tab === 'browse' && (searchQuery || categoryFilter !== 'all') ? (
                <>
                  <p>No listings match your search.</p>
                  <button className="btn btn-outline-secondary btn-sm"
                    onClick={() => { setSearchQuery(''); setCategoryFilter('all') }}>
                    Clear filters
                  </button>
                </>
              ) : (
                <>
                  <p>{tab === 'mylistings' ? "You haven't posted anything yet." : 'No listings yet. Be the first to post!'}</p>
                  {tab === 'mylistings' && (
                    <button className="btn btn-primary" onClick={() => setTab('create')}>
                      Post a Listing
                    </button>
                  )}
                </>
              )}
            </div>
          ) : (
            <div className="row row-cols-2 row-cols-sm-3 row-cols-lg-4 row-cols-xl-5 g-2">
              {displayList.map(l => (
                <div className="col" key={l.id}>
                  <ListingCard
                    listing={l}
                    currentUser={user.username}
                    onSold={handleSold}
                    onRestore={handleRestore}
                    onDelete={handleDelete}
                    onEdit={setEditListing}
                    onReachOut={handleReachOut}
                    onSellerClick={openSellerModal}
                    onDetail={() => openDetail(l)}
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
                      <label className="form-label fw-medium">Original Price ($) <span className="text-muted small">(optional)</span></label>
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
                      <label className="form-label fw-medium">Selling Price ($) <span className="text-danger">*</span></label>
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
                      {categories.map(c => (
                        <option key={c.slug} value={c.slug}>{c.label}</option>
                      ))}
                    </select>
                  </div>

                  <div className="mb-3">
                    <label className="form-label fw-medium">Delivery Options</label>
                    <div className="d-flex gap-3 flex-wrap">
                      {[['pickup','Self-pickup only'],['delivery','Delivery only'],['both','Both']].map(([v,lbl]) => (
                        <div className="form-check" key={v}>
                          <input className="form-check-input" type="radio" name="delivery_type"
                            id={`dt-${v}`} value={v} checked={form.delivery_type === v}
                            onChange={() => setForm(f => ({ ...f, delivery_type: v, delivery_fee: v === 'pickup' ? '' : f.delivery_fee }))} />
                          <label className="form-check-label" htmlFor={`dt-${v}`}>{lbl}</label>
                        </div>
                      ))}
                    </div>
                    {(form.delivery_type === 'delivery' || form.delivery_type === 'both') && (
                      <div className="mt-2">
                        <input type="number" className="form-control" min={0} step="0.01"
                          placeholder="Delivery fee ($, leave blank if free)"
                          value={form.delivery_fee}
                          onChange={e => setForm(f => ({ ...f, delivery_fee: e.target.value }))} />
                      </div>
                    )}
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
   