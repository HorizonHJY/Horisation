import React, { useState, useEffect, useRef, useCallback, useMemo, useId } from 'react'
import { useNavigate, useParams, useSearchParams, useLocation } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'
import HandLoader from '../components/HandLoader'
import Modal, { ConfirmDialog } from '../components/Modal'

/* Interface language is English; Chinese rides along as an accent (PRODUCT.md).
   Categories carry the same pair from the database as `label` / `label_zh`. */
const FALLBACK_CATEGORIES = [
  { slug: 'clothing',    label: 'Clothing',    label_zh: '衣服',     icon: 'fa-tshirt' },
  { slug: 'furniture',   label: 'Furniture',   label_zh: '家具',     icon: 'fa-couch' },
  { slug: 'kitchen',     label: 'Kitchen',     label_zh: '厨具',     icon: 'fa-utensils' },
  { slug: 'electronics', label: 'Electronics', label_zh: '电子产品', icon: 'fa-laptop' },
  { slug: 'beauty',      label: 'Beauty',      label_zh: '美妆',     icon: 'fa-spa' },
]

const DELIVERY_OPTIONS = [
  { value: 'pickup',   label: 'Pickup only',        label_zh: '仅自提',      icon: 'fa-walking' },
  { value: 'delivery', label: 'Delivery only',      label_zh: '仅配送',      icon: 'fa-truck' },
  { value: 'both',     label: 'Pickup or delivery', label_zh: '可自提或配送', icon: 'fa-truck' },
]
const DELIVERY_BY_VALUE = Object.fromEntries(DELIVERY_OPTIONS.map(o => [o.value, o]))

const EMPTY_FORM = {
  title: '', description: '', price: '', original_price: '',
  category: 'clothing', delivery_type: 'pickup', delivery_fee: '',
}

/* Prices come back as SQLite REALs, so price + delivery_fee can land on
   25.000000000000004. Money is always formatted, never printed raw. */
const money = (n) => `$${(Number(n) || 0).toFixed(2)}`

/** English label with the Chinese accent beside it, when one exists. */
function Label({ en, zh }) {
  return <>{en}{zh ? <span className="label-zh">{zh}</span> : null}</>
}

// ── Trade intent status ───────────────────────────────────────────────────────
/* A buyer says "I want this"; the seller accepts (listing → reserved, every other
   pending intent auto-declined); the buyer confirms receipt (listing → sold). */
const INTENT_STATUS = {
  pending:   { en: 'Waiting for seller', zh: '等待卖家回复', tone: 'warn' },
  accepted:  { en: 'Deal agreed',        zh: '已谈成',       tone: 'warn' },
  completed: { en: 'Completed',          zh: '已完成',       tone: 'good' },
  declined:  { en: 'Seller declined',    zh: '卖家婉拒',     tone: '' },
  cancelled: { en: 'Cancelled',          zh: '已取消',       tone: '' },
}
const isLiveIntent = (it) => it && (it.status === 'pending' || it.status === 'accepted')

function IntentChip({ status, block = false }) {
  const meta = INTENT_STATUS[status]
  if (!meta) return <span className="badge-pill">{status}</span>
  return (
    <span className={`badge-pill${meta.tone ? ` badge-pill--${meta.tone}` : ''}${block ? ' badge-pill--block' : ''}`}>
      <Label en={meta.en} zh={meta.zh} />
    </span>
  )
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function useToast() {
  const [toast, setToast] = useState(null)
  const timer = useRef(null)

  const show = useCallback((msg, type = 'success') => {
    // Without clearing, a second toast inherits the first one's expiry.
    if (timer.current) clearTimeout(timer.current)
    setToast({ msg, type })
    timer.current = setTimeout(() => setToast(null), 4000)
  }, [])

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])
  return [toast, show]
}

function Toast({ toast }) {
  if (!toast) return null
  return (
    <div className={`alert alert-${toast.type} app-toast`} role="alert" aria-live="assertive">
      {toast.msg}
    </div>
  )
}

// ── Price Display ────────────────────────────────────────────────────────────
function PriceDisplay({ listing, large = false }) {
  const { price, original_price, delivery_type, delivery_fee } = listing
  const hasOriginal  = original_price && original_price > price
  const hasFee       = delivery_fee && delivery_fee > 0
  const showBoth     = delivery_type === 'both' && hasFee
  const deliveryOnly = delivery_type === 'delivery' && hasFee
  const freeDelivery = (delivery_type === 'both' || delivery_type === 'delivery') && !hasFee

  const bigSz = large ? '1.5rem' : undefined
  const smSz  = large ? '.85rem' : '.78rem'

  /* In a card everything else is left-aligned, so prices are too — otherwise
     two of the four branches sat flush right and the eye could not run down
     the column. In the detail modal the price is paired opposite the category,
     where flush right is correct. */
  const stackAlign = large ? 'flex-end' : 'flex-start'

  const Struck = () => hasOriginal ? (
    <span className="tnum" style={{ fontSize: smSz, color: 'var(--text-muted)', textDecoration: 'line-through' }}>
      {money(original_price)}
    </span>
  ) : null

  if (showBoth) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: stackAlign, gap: 2 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
          <span className="tnum" style={{ fontSize: bigSz, fontWeight: 600 }}>{money(price)}</span>
          <Struck />
          <span className="badge-pill">Pickup</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
          <span className="tnum" style={{ fontSize: large ? '1.1rem' : '.9rem', fontWeight: 600, color: 'var(--badge-info-fg)' }}>
            {money(Number(price) + Number(delivery_fee))}
          </span>
          <span className="badge-pill badge-pill--info">Delivered · {money(delivery_fee)} fee</span>
        </div>
      </div>
    )
  }

  if (deliveryOnly) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: stackAlign, gap: 2 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
          <span className="tnum" style={{ fontSize: bigSz, fontWeight: 600, color: 'var(--badge-info-fg)' }}>
            {money(Number(price) + Number(delivery_fee))}
          </span>
          <Struck />
        </div>
        <span className="badge-pill badge-pill--info">Includes {money(delivery_fee)} delivery</span>
      </div>
    )
  }

  if (freeDelivery) {
    return (
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
        <span className="tnum" style={{ fontSize: bigSz, fontWeight: 600 }}>{money(price)}</span>
        <Struck />
        <span className="badge-pill badge-pill--good">Free delivery</span>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
      <span className="tnum" style={{ fontSize: bigSz, fontWeight: 600 }}>{money(price)}</span>
      <Struck />
    </div>
  )
}

// ── Seller Avatar ─────────────────────────────────────────────────────────────
function SellerAvatar({ username, displayName, avatarUrl, size = 28 }) {
  const style = { width: size, height: size, borderRadius: '50%', flexShrink: 0 }
  const name = displayName || username || 'Unknown'

  if (avatarUrl) {
    return <img src={avatarUrl} alt="" style={{ ...style, objectFit: 'cover' }} />
  }
  return (
    <span
      aria-hidden="true"
      style={{
        ...style,
        /* --accent-text, not --accent: white on #6b9cdb is 2.84:1 and this
           initial stands in for a real person across every module. */
        background: 'var(--accent-text)', color: '#fff',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontWeight: 700, fontSize: size * 0.42,
      }}
    >
      {name[0]?.toUpperCase() || '?'}
    </span>
  )
}

// ── Delivery badge ────────────────────────────────────────────────────────────
function DeliveryBadge({ type }) {
  const opt = DELIVERY_BY_VALUE[type] || DELIVERY_BY_VALUE.pickup
  const tone = type === 'pickup' ? '' : ' badge-pill--info'
  return (
    <span className={`badge-pill${tone}`}>
      <i className={`fas ${opt.icon}`} aria-hidden="true" />
      <Label en={opt.label} zh={opt.label_zh} />
    </span>
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
    <Modal onClose={onClose} title="Edit Listing" scrollable={false}>
      {({ titleId }) => (
        <form onSubmit={handleSubmit}>
          <div className="modal-header">
            <h5 className="modal-title fw-semibold" id={titleId}>Edit Listing</h5>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose} />
          </div>
          <div className="modal-body">
            {err && <div className="alert alert-danger py-2 small" role="alert">{err}</div>}
            <div className="mb-3">
              <label className="form-label fw-medium" htmlFor="edit-title">Title</label>
              <input id="edit-title" className="form-control" maxLength={100} value={form.title}
                onChange={e => setForm(f => ({ ...f, title: e.target.value }))} required />
            </div>
            <div className="mb-3">
              <label className="form-label fw-medium" htmlFor="edit-desc">Description</label>
              <textarea id="edit-desc" className="form-control" rows={3} value={form.description}
                onChange={e => setForm(f => ({ ...f, description: e.target.value }))} required />
            </div>
            <div className="row g-2 mb-3">
              <div className="col-12 col-sm-6">
                <label className="form-label fw-medium" htmlFor="edit-orig">Original Price ($)</label>
                <input id="edit-orig" type="number" className="form-control" min={0} step="0.01"
                  placeholder="optional" value={form.original_price}
                  onChange={e => setForm(f => ({ ...f, original_price: e.target.value }))} />
              </div>
              <div className="col-12 col-sm-6">
                <label className="form-label fw-medium" htmlFor="edit-price">Selling Price ($)</label>
                <input id="edit-price" type="number" className="form-control" min={0} step="0.01"
                  value={form.price}
                  onChange={e => setForm(f => ({ ...f, price: e.target.value }))} required />
              </div>
            </div>
            <div className="mb-1">
              <label className="form-label fw-medium" htmlFor="edit-cat">Category</label>
              <select id="edit-cat" className="form-select" value={form.category}
                onChange={e => setForm(f => ({ ...f, category: e.target.value }))}>
                {categories.map(c => (
                  <option key={c.slug} value={c.slug}>
                    {c.label}{c.label_zh ? ` · ${c.label_zh}` : ''}
                  </option>
                ))}
              </select>
            </div>
            <fieldset className="mt-3">
              <legend className="form-label fw-medium">Delivery Options</legend>
              <div className="d-flex gap-3 flex-wrap">
                {DELIVERY_OPTIONS.map(opt => (
                  <div className="form-check" key={opt.value}>
                    <input className="form-check-input" type="radio" name="edit-delivery"
                      id={`edit-dt-${opt.value}`} value={opt.value}
                      checked={form.delivery_type === opt.value}
                      onChange={() => setForm(f => ({ ...f, delivery_type: opt.value }))} />
                    <label className="form-check-label small" htmlFor={`edit-dt-${opt.value}`}>
                      <Label en={opt.label} zh={opt.label_zh} />
                    </label>
                  </div>
                ))}
              </div>
              {(form.delivery_type === 'delivery' || form.delivery_type === 'both') && (
                <div className="mt-2">
                  <label className="form-label small" htmlFor="edit-fee">Delivery fee ($) — blank means free</label>
                  <input id="edit-fee" type="number" className="form-control form-control-sm" min={0} step="0.01"
                    placeholder="0.00" value={form.delivery_fee}
                    onChange={e => setForm(f => ({ ...f, delivery_fee: e.target.value }))} />
                </div>
              )}
            </fieldset>
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? <span className="spinner-border spinner-border-sm me-1" /> : null}
              Save Changes
            </button>
          </div>
        </form>
      )}
    </Modal>
  )
}

// ── Want Modal ────────────────────────────────────────────────────────────────
// Buyer expressing purchase interest in an ACTIVE (not reserved/sold) listing.
function WantModal({ listing, onClose, onConfirm }) {
  const [message, setMessage] = useState('')
  const [sending, setSending] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    if (sending) return
    setSending(true)
    await onConfirm(listing, message.trim())
    setSending(false)
  }

  return (
    <Modal onClose={sending ? () => {} : onClose} title="I want this" scrollable={false} dismissOnBackdrop={!sending}>
      {({ titleId }) => (
        <form onSubmit={submit}>
          <div className="modal-header">
            <h5 className="modal-title fw-semibold" id={titleId} style={{ fontSize: '1rem' }}>
              <i className="fas fa-hand-pointer me-2" aria-hidden="true" />
              <Label en="I want this" zh="我想要" />
            </h5>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose} disabled={sending} />
          </div>
          <div className="modal-body">
            <p className="small mb-1" style={{ color: 'var(--text-secondary)' }}>
              Tell the seller you would like to buy:
            </p>
            <p className="mb-1" style={{ fontFamily: 'var(--font-display)', fontSize: '1.05rem', fontWeight: 600 }}>
              {listing.title}
            </p>
            <div className="tnum mb-3" style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--accent-text)' }}>
              {money(listing.price)}
            </div>
            <label className="form-label fw-medium small mb-1" htmlFor="want-message">
              Message to the seller <span style={{ color: 'var(--text-muted)' }}>(optional)</span>
            </label>
            <textarea
              id="want-message"
              className="form-control"
              rows={2}
              maxLength={300}
              placeholder="e.g. Is it still available? When could I pick it up?"
              value={message}
              onChange={e => setMessage(e.target.value)}
            />
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose} disabled={sending}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={sending}>
              {sending
                ? <><span className="spinner-border spinner-border-sm me-1" />Sending…</>
                : <><i className="fas fa-hand-pointer me-1" aria-hidden="true" />Send interest</>
              }
            </button>
          </div>
        </form>
      )}
    </Modal>
  )
}

// ── Incoming Intents Modal (seller: who wants this listing) ───────────────────
function IncomingModal({ listing, intents, onClose, onAccept, onDecline, onContact, onCancelReservation }) {
  const [busyId, setBusyId] = useState(null)
  const run = async (fn, intent) => { if (busyId) return; setBusyId(intent.id); await fn(intent); setBusyId(null) }
  const accepted = intents.find(i => i.status === 'accepted')

  return (
    <Modal onClose={onClose} title="Who wants this">
      {({ titleId }) => (
        <>
          <div className="modal-header">
            <h5 className="modal-title fw-semibold" id={titleId} style={{ fontSize: '1rem' }}>
              <i className="fas fa-hand-holding-heart me-2" aria-hidden="true" />
              <Label en="Who wants this" zh="谁想要" />
            </h5>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose} />
          </div>

          <div className="modal-body">
            <p className="small mb-3" style={{ color: 'var(--text-muted)' }}>
              Buyer interest in <strong style={{ color: 'var(--text-primary)' }}>{listing.title}</strong>
            </p>

            {listing.status === 'reserved' && accepted && (
              <div className="alert alert-warning py-2 small" role="status">
                <i className="fas fa-hourglass-half me-1" aria-hidden="true" />
                Agreed with <strong>{accepted.buyer_display || accepted.buyer}</strong> — the sale completes
                once they confirm they have received it.
              </div>
            )}

            {intents.length === 0 ? (
              <div className="text-center py-4" style={{ color: 'var(--text-muted)' }}>
                <i className="fas fa-inbox fa-2x d-block opacity-25 mb-2" aria-hidden="true" />
                Nobody has expressed interest yet.
              </div>
            ) : (
              <div className="d-flex flex-column gap-2">
                {intents.map(intent => (
                  <div key={intent.id} className="border rounded p-2"
                       style={{ background: intent.status === 'accepted' ? 'var(--badge-warn-bg)'
                                          : intent.status === 'pending'  ? 'var(--accent-soft)'
                                          : 'var(--bg-subtle)' }}>
                    <div className="d-flex align-items-center gap-2 mb-1">
                      <SellerAvatar username={intent.buyer} displayName={intent.buyer_display} avatarUrl={intent.buyer_avatar} size={26} />
                      <strong style={{ fontSize: '.85rem' }} className="me-auto">
                        {intent.buyer_display || intent.buyer}
                      </strong>
                      <IntentChip status={intent.status} />
                    </div>

                    {intent.message && (
                      <div style={{
                        fontSize: '.8rem', color: 'var(--text-secondary)',
                        background: 'var(--bg-surface)', border: '1px solid var(--border-soft)',
                        borderRadius: 'var(--radius-sm)', padding: '4px 8px',
                        marginBottom: 6, whiteSpace: 'pre-wrap',
                      }}>
                        {intent.message}
                      </div>
                    )}

                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 6 }}>
                      {new Date(intent.created_at).toLocaleString()}
                    </div>

                    <div className="d-flex gap-2 flex-wrap">
                      {intent.status === 'pending' && (
                        <>
                          <button
                            className="market-card__btn market-card__btn--reach"
                            style={{ flex: 1 }}
                            disabled={busyId === intent.id}
                            onClick={() => run(onAccept, intent)}
                          >
                            <i className="fas fa-check" aria-hidden="true" />Accept
                          </button>
                          <button
                            className="market-card__btn market-card__btn--danger"
                            style={{ flex: 1 }}
                            disabled={busyId === intent.id}
                            onClick={() => run(onDecline, intent)}
                          >
                            <i className="fas fa-times" aria-hidden="true" />Decline
                          </button>
                        </>
                      )}
                      {intent.status === 'accepted' && onCancelReservation && (
                        <button
                          className="market-card__btn market-card__btn--danger"
                          style={{ flex: 1 }}
                          disabled={busyId === intent.id}
                          onClick={() => run(onCancelReservation, intent)}
                        >
                          <i className="fas fa-times-circle" aria-hidden="true" />Cancel trade
                        </button>
                      )}
                      {onContact && (
                        <button
                          className="market-card__btn"
                          style={{ flex: 1 }}
                          onClick={() => onContact(intent)}
                        >
                          <i className="fas fa-comment-dots" aria-hidden="true" />Message buyer
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="modal-footer">
            <button className="btn btn-secondary btn-sm" onClick={onClose}>Close</button>
          </div>
        </>
      )}
    </Modal>
  )
}

// ── Listing Card ──────────────────────────────────────────────────────────────
function ListingCard({
  listing, currentUser, categoryLabel,
  onSold, onRestore, onDelete, onEdit, onReachOut, onSellerClick, onDetail,
  onWant, myIntent, onComplete, onCancelIntent, onShowIntents, incomingCount = 0, onCancelReservation,
}) {
  const isMine     = listing.seller_username === currentUser
  const firstImg   = listing.images?.[0]?.url
  const isSold     = listing.status === 'sold'
  const isReserved = listing.status === 'reserved'
  const cat        = categoryLabel(listing.category)

  return (
    <div className="market-card">
      {/* Image and title are ONE focusable control. They used to be divs with
          onClick, which left the grid with no keyboard path into any listing. */}
      <button
        type="button"
        className="market-card__open"
        onClick={onDetail}
        aria-label={`Open listing: ${listing.title}`}
      >
        <span className="market-card__img">
          {firstImg
            ? <img src={firstImg} alt="" />
            : <i className="fas fa-image placeholder-icon" aria-hidden="true" />
          }
        </span>
        <span className="market-card__title" title={listing.title}>{listing.title}</span>
      </button>

      <div className="market-card__meta">
        <span className="market-card__category">
          <i className={`fas ${listing.category_icon || 'fa-tag'} me-1`} aria-hidden="true" />
          {cat.label}
        </span>
        {isSold && <span className="market-card__sold-badge">Sold</span>}
        {isReserved && !isSold && (
          <span className="badge-pill badge-pill--warn">
            <Label en="Reserved" zh="已谈成" />
          </span>
        )}
        {listing.view_count > 0 && (
          <span className="tnum" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', whiteSpace: 'nowrap', marginLeft: 'auto' }}>
            <i className="fas fa-eye me-1" aria-hidden="true" />
            {listing.view_count}
            <span className="visually-hidden"> views</span>
          </span>
        )}
      </div>

      <div style={{ marginTop: 3, marginBottom: 2 }}>
        <DeliveryBadge type={listing.delivery_type} />
      </div>

      <p className="market-card__desc">{listing.description}</p>

      <hr className="market-card__divider" />

      {!isSold && (
        <div className="market-card__price mb-1">
          <PriceDisplay listing={listing} />
        </div>
      )}

      <button
        type="button"
        className="market-card__seller"
        onClick={() => onSellerClick(listing.seller_username)}
        aria-label={`View listings from ${listing.seller_display || listing.seller_username}`}
      >
        <SellerAvatar
          username={listing.seller_username}
          displayName={listing.seller_display}
          avatarUrl={listing.seller_avatar}
          size={24}
        />
        <span>
          <span style={{ display: 'block', fontWeight: 600, fontSize: '.75rem', color: 'var(--mc-font)' }}>
            {listing.seller_display || listing.seller_username}
          </span>
          <span style={{ display: 'block', fontSize: '.68rem' }}>
            {new Date(listing.created_at).toLocaleDateString()}
          </span>
        </span>
      </button>

      {/* Owner actions */}
      {isMine && (
        <div className="market-card__action">
          {isReserved ? (
            <>
              <span className="badge-pill badge-pill--warn badge-pill--block" style={{ padding: '6px 10px' }}>
                <i className="fas fa-hourglass-half" aria-hidden="true" />
                <Label en="Agreed — awaiting buyer" zh="待买家确认" />
              </span>
              {onCancelReservation && (
                <button
                  className="market-card__btn market-card__btn--danger"
                  onClick={() => onCancelReservation(listing)}
                >
                  <i className="fas fa-times-circle" aria-hidden="true" />Cancel trade
                </button>
              )}
            </>
          ) : (
            !isSold && (
              <>
                {incomingCount > 0 && onShowIntents && (
                  <button
                    className="market-card__btn"
                    style={{ color: 'var(--badge-interest-fg)', borderColor: 'var(--badge-interest-fg)', background: 'var(--badge-interest-bg)' }}
                    onClick={() => onShowIntents(listing)}
                  >
                    <i className="fas fa-hand-holding-heart" aria-hidden="true" />
                    Who wants this ({incomingCount})
                  </button>
                )}
                <button className="market-card__btn" onClick={() => onSold(listing.id)}>
                  <i className="fas fa-check-circle" aria-hidden="true" />Mark Sold
                </button>
                <button className="market-card__btn market-card__btn--edit" onClick={() => onEdit(listing)}>
                  <i className="fas fa-pen" aria-hidden="true" />Edit
                </button>
              </>
            )
          )}
          {isSold && (
            <button className="market-card__btn market-card__btn--restore" onClick={() => onRestore(listing.id)}>
              <i className="fas fa-undo" aria-hidden="true" />Restore
            </button>
          )}
          <button
            className="market-card__btn market-card__btn--danger"
            onClick={() => onDelete(listing)}
            aria-label={`Delete listing: ${listing.title}`}
          >
            <i className="fas fa-trash" aria-hidden="true" />Delete
          </button>
        </div>
      )}

      {/* Buyer actions */}
      {!isMine && !isSold && (
        <div className="market-card__action">
          {myIntent && <IntentChip status={myIntent.status} block />}

          {myIntent?.status === 'accepted' && onComplete && (
            <button
              className="market-card__btn market-card__btn--reach"
              style={{ flexBasis: '100%' }}
              onClick={() => onComplete(myIntent.id)}
            >
              <i className="fas fa-check-circle" aria-hidden="true" />
              <Label en="Confirm received" zh="确认收到" />
            </button>
          )}

          {!isReserved && (!myIntent || ['declined', 'cancelled'].includes(myIntent.status)) && onWant && (
            <button
              className="market-card__btn"
              style={{ flexBasis: '100%', color: 'var(--badge-info-fg)', borderColor: 'var(--badge-info-fg)', background: 'var(--badge-info-bg)' }}
              onClick={() => onWant(listing)}
            >
              <i className="fas fa-hand-pointer" aria-hidden="true" />
              <Label en="I want this" zh="我想要" />
            </button>
          )}

          {!isReserved && myIntent?.status === 'pending' && onCancelIntent && (
            <button
              className="market-card__btn market-card__btn--danger"
              style={{ flexBasis: '100%' }}
              onClick={() => onCancelIntent(myIntent.id)}
            >
              <i className="fas fa-times-circle" aria-hidden="true" />Withdraw interest
            </button>
          )}

          {isReserved && myIntent?.status !== 'accepted' && (
            <span className="badge-pill badge-pill--warn badge-pill--block">
              <Label en="Reserved by another buyer" zh="已被其他买家预定" />
            </span>
          )}

          <button
            className="market-card__btn market-card__btn--reach"
            onClick={() => onReachOut(listing)}
            aria-label={`Message the seller about ${listing.title}`}
          >
            <i className="fas fa-comment-dots" aria-hidden="true" />Message
          </button>
        </div>
      )}
    </div>
  )
}

// ── Listing Detail Modal ──────────────────────────────────────────────────────
function ListingDetailModal({
  listing, currentUser, categoryLabel, shareUrl,
  onClose, onSold, onRestore, onDelete, onEdit, onReachOut, onSellerClick, onCopyLink,
  onWant, myIntent, onComplete, onCancelIntent, onShowIntents, incomingCount = 0, onCancelReservation,
}) {
  const [imgIndex, setImgIndex] = useState(0)
  const isMine     = listing.seller_username === currentUser
  const isSold     = listing.status === 'sold'
  const isReserved = listing.status === 'reserved'
  const cat        = categoryLabel(listing.category)
  const opt        = DELIVERY_BY_VALUE[listing.delivery_type] || DELIVERY_BY_VALUE.pickup

  return (
    <Modal onClose={onClose} title={listing.title} size="modal-lg">
      {({ titleId }) => (
        <>
          <div className="modal-header">
            <h5 className="modal-title fw-semibold" id={titleId}
                style={{ fontSize: '1.15rem', fontFamily: 'var(--font-display)' }}>
              {listing.title}
            </h5>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose} />
          </div>

          <div className="modal-body">
            {listing.images?.length > 0 && (
              <div className="mb-3">
                <img
                  src={listing.images[imgIndex].url}
                  alt={`${listing.title} — photo ${imgIndex + 1} of ${listing.images.length}`}
                  style={{
                    width: '100%', maxHeight: 380, objectFit: 'contain',
                    background: 'var(--bg-page)',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-soft)',
                  }}
                />
                {listing.images.length > 1 && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 8 }} role="group" aria-label="Photos">
                    {listing.images.map((img, i) => (
                      <button
                        key={img.id}
                        type="button"
                        onClick={() => setImgIndex(i)}
                        aria-label={`Show photo ${i + 1}`}
                        aria-pressed={i === imgIndex}
                        style={{
                          padding: 0, lineHeight: 0, cursor: 'pointer',
                          background: 'none',
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

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 10, flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span className="market-card__category">{cat.label}</span>
                {isSold && <span className="market-card__sold-badge">Sold</span>}
                {isReserved && !isSold && (
                  <span className="badge-pill badge-pill--warn"><Label en="Reserved" zh="已谈成" /></span>
                )}
              </div>
              {!isSold && <PriceDisplay listing={listing} large />}
            </div>

            <div style={{ marginBottom: 12 }}>
              <span className={`badge-pill${listing.delivery_type === 'pickup' ? '' : ' badge-pill--info'}`}>
                <i className={`fas ${opt.icon}`} aria-hidden="true" />
                <Label en={opt.label} zh={opt.label_zh} />
                {listing.delivery_type !== 'pickup' && (
                  <span className="tnum">
                    {listing.delivery_fee != null && listing.delivery_fee > 0
                      ? ` · ${money(listing.delivery_fee)}`
                      : ' · free'}
                  </span>
                )}
              </span>
            </div>

            <hr className="market-card__divider" />

            <p style={{ fontSize: '.9rem', color: 'var(--text-secondary)', lineHeight: 1.7, whiteSpace: 'pre-wrap', margin: '16px 0' }}>
              {listing.description}
            </p>

            <hr className="market-card__divider" />

            <button
              type="button"
              onClick={() => onSellerClick(listing.seller_username)}
              aria-label={`View listings from ${listing.seller_display || listing.seller_username}`}
              style={{
                display: 'flex', alignItems: 'center', gap: 10, width: '100%',
                marginTop: 12, padding: '6px 0',
                background: 'none', border: 'none', font: 'inherit', textAlign: 'left', cursor: 'pointer',
              }}
            >
              <SellerAvatar
                username={listing.seller_username}
                displayName={listing.seller_display}
                avatarUrl={listing.seller_avatar}
                size={34}
              />
              <span>
                <span style={{ display: 'block', fontWeight: 600, fontSize: '.9rem', color: 'var(--text-primary)' }}>
                  {listing.seller_display || listing.seller_username}
                </span>
                <span style={{ display: 'block', fontSize: '.75rem', color: 'var(--text-muted)' }}>
                  Posted {new Date(listing.created_at).toLocaleDateString()}
                  {listing.view_count > 0 && (
                    <span className="ms-2 tnum">
                      <i className="fas fa-eye me-1" aria-hidden="true" />
                      {listing.view_count} view{listing.view_count !== 1 ? 's' : ''}
                    </span>
                  )}
                </span>
              </span>
              <i className="fas fa-chevron-right ms-auto" aria-hidden="true" style={{ fontSize: '.75rem', color: 'var(--text-muted)' }} />
            </button>
          </div>

          <div className="modal-footer" style={{ gap: 8 }}>
            {/* The whole point of a shared friend graph: hand someone the item. */}
            <button className="market-card__btn" style={{ flex: '0 0 auto' }}
              onClick={() => onCopyLink(shareUrl)}>
              <i className="fas fa-link" aria-hidden="true" />Copy link
            </button>

            {isMine ? (
              <div style={{ display: 'flex', gap: 8, flex: 1, flexWrap: 'wrap' }}>
                {isReserved ? (
                  <>
                    <span className="badge-pill badge-pill--warn" style={{ flex: 1, justifyContent: 'center', padding: '6px 10px' }}>
                      <i className="fas fa-hourglass-half" aria-hidden="true" />
                      <Label en="Agreed — awaiting buyer" zh="待买家确认" />
                    </span>
                    {onCancelReservation && (
                      <button className="market-card__btn market-card__btn--danger" style={{ flex: 1 }}
                        onClick={() => { onCancelReservation(listing); onClose() }}>
                        <i className="fas fa-times-circle" aria-hidden="true" />Cancel trade
                      </button>
                    )}
                  </>
                ) : (
                  !isSold && (
                    <>
                      {incomingCount > 0 && onShowIntents && (
                        <button
                          className="market-card__btn"
                          style={{ flex: 1, color: 'var(--badge-interest-fg)', borderColor: 'var(--badge-interest-fg)', background: 'var(--badge-interest-bg)' }}
                          onClick={() => { onShowIntents(listing); onClose() }}
                        >
                          <i className="fas fa-hand-holding-heart" aria-hidden="true" />
                          Who wants this ({incomingCount})
                        </button>
                      )}
                      <button className="market-card__btn" style={{ flex: 1 }}
                        onClick={() => { onSold(listing.id); onClose() }}>
                        <i className="fas fa-check-circle" aria-hidden="true" />Mark Sold
                      </button>
                      <button className="market-card__btn market-card__btn--edit" style={{ flex: 1 }}
                        onClick={() => { onEdit(listing); onClose() }}>
                        <i className="fas fa-pen" aria-hidden="true" />Edit
                      </button>
                    </>
                  )
                )}
                {isSold && (
                  <button className="market-card__btn market-card__btn--restore" style={{ flex: 1 }}
                    onClick={() => { onRestore(listing.id); onClose() }}>
                    <i className="fas fa-undo" aria-hidden="true" />Restore
                  </button>
                )}
                <button className="market-card__btn market-card__btn--danger" style={{ flex: 1 }}
                  onClick={() => onDelete(listing)}>
                  <i className="fas fa-trash" aria-hidden="true" />Delete
                </button>
              </div>
            ) : !isSold ? (
              <div style={{ display: 'flex', gap: 8, flex: 1, flexWrap: 'wrap' }}>
                {myIntent && <IntentChip status={myIntent.status} block />}

                {myIntent?.status === 'accepted' && onComplete && (
                  <button className="market-card__btn market-card__btn--reach" style={{ flex: 1 }}
                    onClick={() => { onComplete(myIntent.id); onClose() }}>
                    <i className="fas fa-check-circle" aria-hidden="true" />
                    <Label en="Confirm received" zh="确认收到" />
                  </button>
                )}

                {!isReserved && (!myIntent || ['declined', 'cancelled'].includes(myIntent.status)) && onWant && (
                  <button
                    className="market-card__btn"
                    style={{ flex: 1, color: 'var(--badge-info-fg)', borderColor: 'var(--badge-info-fg)', background: 'var(--badge-info-bg)' }}
                    onClick={() => { onWant(listing); onClose() }}
                  >
                    <i className="fas fa-hand-pointer" aria-hidden="true" />
                    <Label en="I want this" zh="我想要" />
                  </button>
                )}

                {!isReserved && myIntent?.status === 'pending' && onCancelIntent && (
                  <button className="market-card__btn market-card__btn--danger" style={{ flex: 1 }}
                    onClick={() => onCancelIntent(myIntent.id)}>
                    <i className="fas fa-times-circle" aria-hidden="true" />Withdraw interest
                  </button>
                )}

                {isReserved && myIntent?.status !== 'accepted' && (
                  <span className="badge-pill badge-pill--warn" style={{ flex: 1, justifyContent: 'center' }}>
                    <Label en="Reserved by another buyer" zh="已被其他买家预定" />
                  </span>
                )}

                <button className="market-card__btn market-card__btn--reach" style={{ flex: 1 }}
                  onClick={() => onReachOut(listing)}>
                  <i className="fas fa-comment-dots" aria-hidden="true" />Message Seller
                </button>
              </div>
            ) : null}
          </div>
        </>
      )}
    </Modal>
  )
}

// ── Seller Modal ──────────────────────────────────────────────────────────────
/* Read-only summary card. The old version reused ListingCard with every
   callback stubbed to () => {}, so it rendered a grid of buttons that looked
   live and did nothing. */
function SellerListingCard({ listing, categoryLabel, onOpen }) {
  const firstImg = listing.images?.[0]?.url
  const cat = categoryLabel(listing.category)
  return (
    <div className="market-card">
      <button
        type="button"
        className="market-card__open"
        onClick={onOpen}
        aria-label={`Open listing: ${listing.title}`}
      >
        <span className="market-card__img">
          {firstImg
            ? <img src={firstImg} alt="" />
            : <i className="fas fa-image placeholder-icon" aria-hidden="true" />
          }
        </span>
        <span className="market-card__title" title={listing.title}>{listing.title}</span>
      </button>
      <div className="market-card__meta">
        <span className="market-card__category">{cat.label}</span>
        {listing.status === 'reserved' && (
          <span className="badge-pill badge-pill--warn"><Label en="Reserved" zh="已谈成" /></span>
        )}
      </div>
      <div className="market-card__price mb-1">
        <PriceDisplay listing={listing} />
      </div>
    </div>
  )
}

function SellerModal({ seller, listings, loading, categoryLabel, onClose, onReachOut, onOpenListing, onViewProfile }) {
  return (
    <Modal onClose={onClose} title={seller.display_name || seller.username} size="modal-lg">
      {({ titleId }) => (
        <>
          <div className="modal-header">
            <div className="d-flex align-items-center gap-3 flex-wrap">
              <SellerAvatar username={seller.username} displayName={seller.display_name} avatarUrl={seller.avatar_url} size={44} />
              <div>
                <div className="fw-bold" id={titleId}>{seller.display_name || seller.username}</div>
                <div className="small" style={{ color: 'var(--text-muted)' }}>
                  @{seller.username} ·{' '}
                  <button
                    type="button"
                    onClick={() => onViewProfile(seller.username)}
                    style={{ background: 'none', border: 'none', padding: 0, font: 'inherit', color: 'var(--accent-text)', cursor: 'pointer' }}
                  >
                    View Profile →
                  </button>
                </div>
              </div>
              <button className="btn btn-sm btn-primary ms-2" onClick={() => onReachOut(null)}>
                <i className="fas fa-comment-dots me-1" aria-hidden="true" />Message
              </button>
            </div>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose} />
          </div>

          <div className="modal-body">
            {loading ? (
              <div className="text-center py-4"><HandLoader /></div>
            ) : (
              <>
                <p className="small mb-3" style={{ color: 'var(--text-muted)' }}>
                  <i className="fas fa-store me-1" aria-hidden="true" />
                  {listings.length} active listing{listings.length !== 1 ? 's' : ''}
                </p>
                {listings.length === 0 ? (
                  <div className="text-center py-4" style={{ color: 'var(--text-muted)' }}>
                    <i className="fas fa-box-open fa-2x mb-2 d-block opacity-25" aria-hidden="true" />
                    No active listings.
                  </div>
                ) : (
                  <div className="row row-cols-2 row-cols-sm-3 g-2">
                    {listings.map(l => (
                      <div className="col" key={l.id}>
                        <SellerListingCard
                          listing={l}
                          categoryLabel={categoryLabel}
                          onOpen={() => onOpenListing(l.id)}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </>
      )}
    </Modal>
  )
}

// ── Export Modal ──────────────────────────────────────────────────────────────
/* Exporting your own listings for sharing outside the circle is a confirmed
   product need (PRODUCT.md, 2026-09-07). It has to work on iOS Safari, where
   a scripted click on a data: URL silently does nothing — so the rendered
   image is always shown in-page and can be saved by long-press. */
function ExportModal({ listings, categoryLabel, onClose, showToast }) {
  const exportRef = useRef(null)
  const [rendering, setRendering] = useState(false)
  const [rendered, setRendered]   = useState(null)
  const active = listings.filter(l => l.status === 'active')
  const stamp  = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })

  async function handleRender() {
    if (!exportRef.current) return
    setRendering(true)
    try {
      // Only loaded when someone actually exports — it used to be a top-level
      // import paid for on every Market mount.
      const { default: html2canvas } = await import('html2canvas')
      await new Promise(r => setTimeout(r, 100))
      const canvas = await html2canvas(exportRef.current, {
        scale: 2, useCORS: true, logging: false, backgroundColor: '#ffffff',
      })
      setRendered(canvas.toDataURL('image/png'))
    } catch (err) {
      showToast('Could not render the image. Some photos may be blocked from export.', 'danger')
    } finally {
      setRendering(false)
    }
  }

  return (
    <Modal onClose={onClose} title="Export as image">
      {({ titleId }) => (
        <>
          <div className="modal-header">
            <h6 className="modal-title fw-semibold" id={titleId}>
              <i className="fas fa-image me-2" aria-hidden="true" />Export as image
            </h6>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose} />
          </div>

          <div className="modal-body">
            <p className="small mb-1" style={{ color: 'var(--text-secondary)' }}>
              Turns your {active.length} active listing{active.length !== 1 ? 's' : ''} into one
              tall image you can share anywhere.
            </p>
            <p className="small" style={{ color: 'var(--text-muted)' }}>
              一张长图，随手分享
            </p>

            {rendered ? (
              <div className="text-center">
                <img
                  src={rendered}
                  alt="Your listings, rendered for sharing"
                  style={{ maxWidth: '100%', border: '1px solid var(--border-soft)', borderRadius: 'var(--radius-md)' }}
                />
                <p className="small mt-2 mb-0" style={{ color: 'var(--text-muted)' }}>
                  On a phone, press and hold the image to save it.
                </p>
              </div>
            ) : (
              /* Preview scales down instead of overflowing a ~343px dialog. */
              <div style={{ overflowX: 'auto' }}>
                <div ref={exportRef} style={{
                  width: 420, padding: '20px 24px', background: '#fff',
                  borderRadius: 8, margin: '0 auto', color: '#1a1a1a',
                  fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                }}>
                  <div style={{ textAlign: 'center', marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid #d8d3c8' }}>
                    <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'Playfair Display, Georgia, serif', letterSpacing: '-0.01em' }}>
                      Arch Bay
                    </div>
                    <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>{stamp}</div>
                  </div>

                  {active.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '40px 0', color: '#666', fontSize: 14 }}>
                      No active listings yet.
                    </div>
                  ) : active.map((l, i) => (
                    <div key={l.id} style={{
                      display: 'flex', gap: 12, padding: '12px 0',
                      borderBottom: i < active.length - 1 ? '1px solid #ece8e0' : 'none',
                    }}>
                      <div style={{
                        width: 88, height: 88, borderRadius: 6, overflow: 'hidden', flexShrink: 0,
                        background: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center',
                      }}>
                        {l.images?.[0]?.url
                          ? <img src={l.images[0].url} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                          : <i className="fas fa-image" style={{ fontSize: 24, opacity: 0.3 }} aria-hidden="true" />
                        }
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {l.title}
                        </div>
                        <div style={{ color: '#a5321f', fontWeight: 700, fontSize: 16 }}>
                          {money(l.price)}
                          {l.original_price && l.original_price > l.price && (
                            <span style={{ color: '#666', textDecoration: 'line-through', fontWeight: 400, fontSize: 12, marginLeft: 6 }}>
                              {money(l.original_price)}
                            </span>
                          )}
                        </div>
                        <div style={{ fontSize: 11, color: '#5a6270', marginTop: 3 }}>
                          <span style={{ background: '#eef1f5', borderRadius: 8, padding: '1px 6px', marginRight: 4 }}>
                            {categoryLabel(l.category).label}
                          </span>
                          <span style={{ background: '#eef1f5', borderRadius: 8, padding: '1px 6px' }}>
                            {l.delivery_type === 'pickup'
                              ? 'Pickup'
                              : (l.delivery_fee != null && l.delivery_fee > 0
                                  ? `Delivery ${money(l.delivery_fee)}`
                                  : 'Free delivery')}
                          </span>
                        </div>
                        <div style={{ fontSize: 11, color: '#666', marginTop: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {l.description}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="modal-footer">
            <button className="btn btn-outline-secondary btn-sm" onClick={onClose}>Close</button>
            {rendered ? (
              <a className="btn btn-primary btn-sm" href={rendered}
                 download={`arch-bay-listings-${new Date().toISOString().slice(0, 10)}.png`}>
                <i className="fas fa-download me-1" aria-hidden="true" />Download
              </a>
            ) : (
              <button className="btn btn-primary btn-sm" onClick={handleRender} disabled={rendering || active.length === 0}>
                {rendering
                  ? <><span className="spinner-border spinner-border-sm me-1" />Rendering…</>
                  : <><i className="fas fa-wand-magic-sparkles me-1" aria-hidden="true" />Create image</>
                }
              </button>
            )}
          </div>
        </>
      )}
    </Modal>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function Market() {
  const { user }   = useAuth()
  const navigate   = useNavigate()
  const location   = useLocation()
  const { listingId, sellerUsername } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const formErrId = useId()

  const [categories, setCategories] = useState(FALLBACK_CATEGORIES)
  const [listings, setListings]     = useState([])
  const [myListings, setMy]         = useState([])
  const [loading, setLoading]       = useState(false)
  const [form, setForm]             = useState(EMPTY_FORM)
  const [fieldErrors, setFieldErrors] = useState({})
  const [images, setImages]         = useState([])
  const [previews, setPreviews]     = useState([])
  const [submitting, setSub]        = useState(false)
  const [toast, showToast]          = useToast()
  const [friendsMap, setFriendsMap] = useState({})
  const [editListing, setEditListing]       = useState(null)
  const [detailListing, setDetailListing]   = useState(null)
  const [sellerInfo, setSellerInfo]         = useState(null)
  const [sellerListings, setSellerListings] = useState([])
  const [sellerLoading, setSellerLoading]   = useState(false)
  const [showExport, setShowExport]   = useState(false)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting]       = useState(false)
  const [highlightId, setHighlightId] = useState(null)

  // ── Trade-intent state ──────────────────────────────────────────────────────
  const [outgoingIntents, setOutgoingIntents] = useState([])   // intents I placed as buyer
  const [incomingIntents, setIncomingIntents] = useState([])   // intents on MY listings
  const [wantListing, setWantListing]         = useState(null)
  const [incomingModalId, setIncomingModalId] = useState(null)
  const [pendingCancelIntent, setPendingCancelIntent] = useState(null)
  const [cancellingIntent, setCancellingIntent]       = useState(false)
  // Bumped after any mutation so an open detail overlay re-fetches its listing.
  const [refreshTick, setRefreshTick] = useState(0)

  const fileRef  = useRef()
  const titleRef = useRef()
  const descRef  = useRef()
  const priceRef = useRef()

  // ── URL is the source of truth for tab and filters ──────────────────────────
  const tab            = searchParams.get('tab') || 'browse'
  const searchQuery    = searchParams.get('q') || ''
  const categoryFilter = useMemo(
    () => new Set((searchParams.get('cat') || '').split(',').filter(Boolean)), [searchParams])
  const deliveryFilter = useMemo(
    () => new Set((searchParams.get('del') || '').split(',').filter(Boolean)), [searchParams])

  const patchParams = useCallback((patch, opts = {}) => {
    setSearchParams(prev => {
      const next = new URLSearchParams(prev)
      Object.entries(patch).forEach(([k, v]) => {
        if (v === '' || v == null) next.delete(k)
        else next.set(k, v)
      })
      return next
    }, { replace: opts.replace ?? true })
  }, [setSearchParams])

  const setTab = useCallback((t) => patchParams({ tab: t === 'browse' ? '' : t }, { replace: false }), [patchParams])
  const setSearchQuery = useCallback((q) => patchParams({ q }), [patchParams])

  const tabRef = useRef(tab)
  useEffect(() => { tabRef.current = tab }, [tab])

  const categoryLabel = useCallback((slug) => {
    const c = categories.find(x => x.slug === slug)
    return c || { slug, label: slug, label_zh: '', icon: 'fa-tag' }
  }, [categories])

  // ── Categories ──────────────────────────────────────────────────────────────
  useEffect(() => {
    api.get('/api/market/categories').then(d => {
      if (d.ok && d.categories?.length) {
        const cats = d.categories.map(c => ({
          slug: c.slug, label: c.label, label_zh: c.label_zh || '', icon: c.icon || 'fa-tag',
        }))
        setCategories(cats)
        setListings(prev => enrichWithIcon(prev, cats))
        setMy(prev => enrichWithIcon(prev, cats))
      }
    })
  }, [])

  // Filters are no longer wiped on every return to Browse — they live in the URL.
  useEffect(() => {
    if (tab === 'browse')     loadBrowse()
    if (tab === 'mylistings') loadMine()
    refreshIntents()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab])

  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState !== 'visible') return
      if (tabRef.current === 'browse')          loadBrowse()
      else if (tabRef.current === 'mylistings') loadMine()
      refreshIntents()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function enrichWithIcon(list, cats) {
    const iconMap = Object.fromEntries((cats || []).map(c => [c.slug, c.icon || 'fa-tag']))
    return list.map(l => ({ ...l, category_icon: iconMap[l.category] || 'fa-tag' }))
  }

  async function loadBrowse() {
    setLoading(true)
    const [listRes, friendsRes] = await Promise.all([
      api.get('/api/market/listings'),
      api.get('/api/friends/list'),
    ])

    const fMap = {}
    if (friendsRes.ok) friendsRes.friends.forEach(f => { fMap[f.username] = f })
    setFriendsMap(fMap)

    if (listRes.ok) setListings(enrichWithIcon(listRes.listings, categories))
    // A failed fetch used to fall through to "No listings yet. Be the first to
    // post!", making an outage look like an empty market.
    else showToast(listRes.error || 'Could not load listings.', 'danger')

    setLoading(false)
  }

  async function loadMine() {
    setLoading(true)
    const d = await api.get('/api/market/my')
    if (d.ok) setMy(enrichWithIcon(d.listings, categories))
    else showToast(d.error || 'Could not load your listings.', 'danger')
    setLoading(false)
  }

  async function refreshIntents() {
    const [outRes, inRes] = await Promise.all([
      api.get('/api/market/intents/outgoing'),
      api.get('/api/market/intents/incoming'),
    ])
    if (outRes.ok) setOutgoingIntents(outRes.intents || [])
    if (inRes.ok)  setIncomingIntents(inRes.intents || [])
  }

  // After any intent/listing mutation, re-fetch the current tab, the intents,
  // and whatever overlay is open.
  async function reloadAfterMutation() {
    if (tabRef.current === 'browse')          loadBrowse()
    else if (tabRef.current === 'mylistings') loadMine()
    refreshIntents()
    setRefreshTick(t => t + 1)
  }

  // ── Route-driven overlays ───────────────────────────────────────────────────
  useEffect(() => {
    if (!listingId) { setDetailListing(null); return }
    let cancelled = false
    ;(async () => {
      const key = `viewed_${listingId}`
      const alreadyCounted = sessionStorage.getItem(key)
      // Always fetch so price and status are current; only count the first view.
      const d = await api.get(`/api/market/listings/${listingId}${alreadyCounted ? '?track=0' : ''}`)
      if (cancelled) return
      if (d.ok) {
        if (!alreadyCounted) sessionStorage.setItem(key, '1')
        setDetailListing(d.listing)
        setListings(prev => prev.map(l => l.id === listingId ? { ...l, view_count: d.listing.view_count } : l))
        setMy(prev => prev.map(l => l.id === listingId ? { ...l, view_count: d.listing.view_count } : l))
      } else {
        showToast(d.error || 'That listing is no longer available.', 'danger')
        navigate(`/market${location.search}`, { replace: true })
      }
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [listingId, refreshTick])

  useEffect(() => {
    if (!sellerUsername) { setSellerInfo(null); setSellerListings([]); return }
    if (sellerUsername === user.username) {
      navigate(`/market?tab=mylistings`, { replace: true })
      return
    }
    let cancelled = false
    ;(async () => {
      // Seed from whatever real identity we already hold, so the avatar and
      // display name never degrade to a grey initial on the way in.
      const known = friendsMap[sellerUsername]
      const fromListing = [...listings, ...myListings].find(l => l.seller_username === sellerUsername)
      setSellerInfo(known ?? {
        username: sellerUsername,
        display_name: fromListing?.seller_display || sellerUsername,
        avatar_url: fromListing?.seller_avatar || null,
      })
      setSellerLoading(true)
      const [profileRes, listRes] = await Promise.all([
        api.get(`/api/auth/users/${sellerUsername}/public`),
        api.get(`/api/market/user/${sellerUsername}`),
      ])
      if (cancelled) return
      // Canonical profile wins, so Market, Friends and /u/:username agree.
      if (profileRes.ok && profileRes.user) {
        setSellerInfo(prev => ({ ...prev, ...profileRes.user, username: sellerUsername }))
      }
      if (listRes.ok) setSellerListings(enrichWithIcon(listRes.listings, categories))
      setSellerLoading(false)
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sellerUsername])

  // ── Navigation helpers ──────────────────────────────────────────────────────
  const openDetail   = (id) => navigate(`/market/l/${id}${location.search}`)
  const openSeller   = (username) => navigate(`/market/u/${username}${location.search}`)
  const closeOverlay = () => navigate(`/market${location.search}`)

  const shareUrlFor = (id) => `${window.location.origin}/market/l/${id}`

  async function handleCopyLink(url) {
    try {
      await navigator.clipboard.writeText(url)
      showToast('Link copied — paste it anywhere.')
    } catch {
      showToast(url, 'info')
    }
  }

  // ── Images ──────────────────────────────────────────────────────────────────
  function handleFileChange(e) {
    const incoming = Array.from(e.target.files)
    const incomingPreviews = incoming.map(f => URL.createObjectURL(f))

    setImages(prev => [...prev, ...incoming].slice(0, 3))
    setPreviews(prev => {
      const combined = [...prev, ...incomingPreviews]
      if (combined.length > 3) combined.slice(3).forEach(url => URL.revokeObjectURL(url))
      return combined.slice(0, 3)
    })
    if (fileRef.current) fileRef.current.value = ''
  }

  function removeImage(idx) {
    URL.revokeObjectURL(previews[idx])
    setImages(prev => prev.filter((_, i) => i !== idx))
    setPreviews(prev => prev.filter((_, i) => i !== idx))
  }

  // ── Create ──────────────────────────────────────────────────────────────────
  function validateForm() {
    const errs = {}
    if (!form.title.trim())       errs.title = 'Give your item a title.'
    if (!form.description.trim()) errs.description = 'Describe the condition so people know what to expect.'
    if (!form.price || isNaN(form.price) || Number(form.price) < 0) {
      errs.price = 'Enter a price, using numbers only.'
    } else if (form.original_price && Number(form.original_price) > 0 &&
               Number(form.original_price) <= Number(form.price)) {
      // Otherwise the strikethrough silently vanishes with no explanation.
      errs.original_price = 'The original price should be higher than what you are asking.'
    }
    return errs
  }

  async function handleCreate(e) {
    e.preventDefault()
    const errs = validateForm()
    setFieldErrors(errs)

    if (Object.keys(errs).length > 0) {
      // A toast alone fires off-screen when the submit button is at the bottom
      // of a scrolled phone form — take the user to the field instead.
      const first = errs.title ? titleRef : errs.description ? descRef : priceRef
      first.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      first.current?.focus({ preventScroll: true })
      showToast('Check the highlighted field.', 'danger')
      return
    }

    setSub(true)
    const fd = new FormData()
    Object.entries(form).forEach(([k, v]) => fd.append(k, v))
    images.forEach(img => fd.append('images', img))

    const d = await api.upload('/api/market/listings', fd)
    setSub(false)

    if (d.ok) {
      showToast('Listing posted.')
      setForm(EMPTY_FORM)
      setFieldErrors({})
      previews.forEach(url => URL.revokeObjectURL(url))
      setImages([])
      setPreviews([])
      if (fileRef.current) fileRef.current.value = ''
      // Browse hides your own listings, so landing there after posting showed
      // the seller a market without the thing they had just posted.
      setHighlightId(d.listing?.id || null)
      setTab('mylistings')
    } else {
      showToast(d.error || 'Could not post the listing.', 'danger')
    }
  }

  useEffect(() => {
    if (!highlightId) return
    const t = setTimeout(() => setHighlightId(null), 2600)
    return () => clearTimeout(t)
  }, [highlightId])

  // ── Listing mutations ───────────────────────────────────────────────────────
  async function handleSold(id) {
    const d = await api.post(`/api/market/listings/${id}/sold`)
    if (d.ok) { showToast('Marked as sold.'); reloadAfterMutation() }
    else showToast(d.error || 'Could not update the listing.', 'danger')
  }

  async function handleRestore(id) {
    const d = await api.post(`/api/market/listings/${id}/restore`)
    if (d.ok) { showToast('Listing is active again.'); reloadAfterMutation() }
    else showToast(d.error || 'Could not restore the listing.', 'danger')
  }

  async function handleEditSave(id, fields) {
    const d = await api.put(`/api/market/listings/${id}`, fields)
    if (d.ok) { showToast('Listing updated.'); setEditListing(null); reloadAfterMutation() }
    else showToast(d.error || 'Could not save your changes.', 'danger')
  }

  async function confirmDelete() {
    if (!pendingDelete) return
    setDeleting(true)
    const d = await api.delete(`/api/market/listings/${pendingDelete.id}`)
    setDeleting(false)
    setPendingDelete(null)
    if (d.ok) {
      showToast('Listing deleted.')
      if (listingId === pendingDelete.id) navigate(`/market${location.search}`, { replace: true })
      reloadAfterMutation()
    } else {
      showToast(d.error || 'Could not delete the listing.', 'danger')
    }
  }

  // ── Trade-intent handlers ───────────────────────────────────────────────────
  // Buyer expresses purchase intent on an active listing.
  async function expressIntent(listing, message) {
    const d = await api.post(`/api/market/listings/${listing.id}/intent`, { message: message || undefined })
    if (d.ok) {
      showToast('Interest sent — waiting for the seller.')
      setWantListing(null)
      reloadAfterMutation()
    } else {
      showToast(d.error || 'Could not send your interest.', 'danger')
    }
  }

  // Seller accepts a pending intent (listing → reserved; others auto-declined).
  async function acceptIntent(intent) {
    const d = await api.put(`/api/market/intents/${intent.id}/accept`)
    if (d.ok) {
      showToast(`Accepted ${intent.buyer_display || intent.buyer}.`)
      setIncomingModalId(null)
      reloadAfterMutation()
    } else showToast(d.error || 'Could not accept.', 'danger')
  }

  async function declineIntent(intent) {
    const d = await api.put(`/api/market/intents/${intent.id}/decline`)
    if (d.ok) { showToast('Declined.'); reloadAfterMutation() }
    else showToast(d.error || 'Could not decline.', 'danger')
  }

  // Either party aborts an intent (frees a reserved listing back to active).
  // Confirmation goes through ConfirmDialog, not window.confirm.
  async function reallyCancelIntent() {
    const id = pendingCancelIntent
    if (!id) return
    setCancellingIntent(true)
    const d = await api.put(`/api/market/intents/${id}/cancel`)
    setCancellingIntent(false)
    setPendingCancelIntent(null)
    if (d.ok) {
      showToast('Cancelled.')
      setIncomingModalId(null)
      reloadAfterMutation()
    } else showToast(d.error || 'Could not cancel.', 'danger')
  }

  // Buyer confirms receipt → intent completed, listing sold.
  async function completeIntent(id) {
    const d = await api.put(`/api/market/intents/${id}/complete`)
    if (d.ok) {
      showToast('Trade complete — thank you.')
      if (listingId) navigate(`/market${location.search}`, { replace: true })
      reloadAfterMutation()
    } else showToast(d.error || 'Could not confirm.', 'danger')
  }

  // Seller releases their own reserved listing by cancelling the accepted intent.
  function cancelReservationByListing(listing) {
    const accepted = incomingIntents.find(i => i.listing_id === listing.id && i.status === 'accepted')
    if (accepted) return setPendingCancelIntent(accepted.id)
    const anyIntent = incomingIntents.find(
      i => i.listing_id === listing.id && ['pending', 'accepted'].includes(i.status))
    if (anyIntent) return setPendingCancelIntent(anyIntent.id)
    showToast('There is no reservation to cancel.', 'danger')
  }

  function contactBuyer(intent) {
    setIncomingModalId(null)
    handleReachOut({ seller_username: intent.buyer, title: null })
  }

  function handleReachOut(listing, sellerName) {
    const username = sellerName ?? listing?.seller_username
    let chatPartner = friendsMap[username]
    if (!chatPartner) {
      if (listing && listing.seller_display) {
        chatPartner = {
          username,
          display_name: listing.seller_display || username,
          avatar_url:   listing.seller_avatar  || null,
        }
      } else if (sellerInfo && sellerInfo.username === username) {
        chatPartner = { ...sellerInfo }
      } else {
        chatPartner = { username, display_name: username, avatar_url: null }
      }
    }

    // English, because it is put in the user's own mouth and they must be able
    // to read it. It carries a real link now, instead of naming the item.
    const initialMessage = listing?.title
      ? `Hi! Is "${listing.title}" still available? ${shareUrlFor(listing.id)}`
      : ''
    navigate('/friends', { state: { openChat: chatPartner, initialMessage } })
  }

  // ── Derived intent data ─────────────────────────────────────────────────────
  const outgoingByListing = {}
  outgoingIntents.forEach(it => {
    (outgoingByListing[it.listing_id] = outgoingByListing[it.listing_id] || []).push(it)
  })
  const incomingByListing = {}
  incomingIntents.forEach(it => {
    (incomingByListing[it.listing_id] = incomingByListing[it.listing_id] || []).push(it)
  })

  function pickBuyerIntent(list) {
    const all = outgoingByListing[list.id] || []
    if (!all.length) return null
    // Prefer whichever intent is currently driving the negotiation.
    const active = all.find(i => i.status === 'pending')
    if (active) return active
    const accepted = all.find(i => i.status === 'accepted')
    if (accepted) return accepted
    return all.slice().sort((a, b) =>
      new Date(b.updated_at || b.created_at) - new Date(a.updated_at || a.created_at))[0]
  }
  const buyerIntentFor   = (list) => (list ? pickBuyerIntent(list) : null)
  const listingIncoming  = (list) => (incomingByListing[list.id] || []).slice()
  // The badge counts buyers currently wanting or negotiating, not history.
  const incomingCountFor = (list) =>
    (list ? (incomingByListing[list.id] || []).filter(isLiveIntent).length : 0)

  // ── Filters ─────────────────────────────────────────────────────────────────
  const browseListing = listings.filter(l => l.seller_username !== user.username)
  const filteredBrowse = browseListing.filter(l => {
    const q = searchQuery.trim().toLowerCase()
    const matchesSearch = !q ||
      l.title.toLowerCase().includes(q) ||
      l.description.toLowerCase().includes(q)
    const matchesCategory = categoryFilter.size === 0 || categoryFilter.has(l.category)
    const matchesDelivery = deliveryFilter.size === 0 || deliveryFilter.has(l.delivery_type)
    return matchesSearch && matchesCategory && matchesDelivery
  })
  const hasActiveFilters = Boolean(searchQuery) || categoryFilter.size > 0 || deliveryFilter.size > 0

  const toggleIn = (set, key, param) => {
    const next = new Set(set)
    next.has(key) ? next.delete(key) : next.add(key)
    patchParams({ [param]: [...next].join(',') })
  }
  const clearAllFilters = () => patchParams({ q: '', cat: '', del: '' })

  const displayList = tab === 'mylistings' ? myListings : filteredBrowse

  return (
    <div className="container-fluid py-4">
      <Toast toast={toast} />

      {detailListing && (
        <ListingDetailModal
          listing={detailListing}
          currentUser={user.username}
          categoryLabel={categoryLabel}
          shareUrl={shareUrlFor(detailListing.id)}
          onClose={closeOverlay}
          onSold={handleSold}
          onRestore={handleRestore}
          onDelete={setPendingDelete}
          onEdit={setEditListing}
          onReachOut={handleReachOut}
          onSellerClick={openSeller}
          onCopyLink={handleCopyLink}
          onWant={setWantListing}
          myIntent={buyerIntentFor(detailListing)}
          onComplete={completeIntent}
          onCancelIntent={setPendingCancelIntent}
          incomingCount={incomingCountFor(detailListing)}
          onShowIntents={l => setIncomingModalId(l.id)}
          onCancelReservation={cancelReservationByListing}
        />
      )}

      {sellerInfo && (
        <SellerModal
          seller={sellerInfo}
          listings={sellerListings}
          loading={sellerLoading}
          categoryLabel={categoryLabel}
          onClose={closeOverlay}
          onReachOut={(listing) => handleReachOut(listing, sellerInfo.username)}
          onOpenListing={openDetail}
          onViewProfile={(username) => navigate(`/u/${username}`)}
        />
      )}

      {editListing && (
        <EditModal
          listing={editListing}
          onClose={() => setEditListing(null)}
          onSave={handleEditSave}
          categories={categories}
        />
      )}

      {wantListing && (
        <WantModal
          listing={wantListing}
          onClose={() => setWantListing(null)}
          onConfirm={expressIntent}
        />
      )}

      {incomingModalId != null && (() => {
        const target = [...myListings, ...listings].find(l => l.id === incomingModalId)
        if (!target) return null
        return (
          <IncomingModal
            listing={target}
            intents={listingIncoming(target)}
            onClose={() => setIncomingModalId(null)}
            onAccept={acceptIntent}
            onDecline={declineIntent}
            onContact={contactBuyer}
            onCancelReservation={(intent) => setPendingCancelIntent(intent.id)}
          />
        )
      })()}

      {pendingDelete && (
        <ConfirmDialog
          title="Delete this listing?"
          itemName={pendingDelete.title}
          message="The listing and its photos are removed for good. This cannot be undone."
          confirmLabel="Delete listing"
          busy={deleting}
          onConfirm={confirmDelete}
          onClose={() => setPendingDelete(null)}
        />
      )}

      {pendingCancelIntent && (
        <ConfirmDialog
          title="Cancel this trade?"
          message="The interest is withdrawn and the listing goes back on sale for everyone."
          confirmLabel="Cancel trade"
          cancelLabel="Keep it"
          busy={cancellingIntent}
          onConfirm={reallyCancelIntent}
          onClose={() => setPendingCancelIntent(null)}
        />
      )}

      {showExport && (
        <ExportModal
          listings={myListings}
          categoryLabel={categoryLabel}
          onClose={() => setShowExport(false)}
          showToast={showToast}
        />
      )}

      <div className="mb-4">
        <h1 className="page-title mb-0">Market</h1>
        <p className="page-subtitle mb-0">二手好物 — 朋友之间</p>
      </div>

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

      {(tab === 'browse' || tab === 'mylistings') && (
        <>
          {tab === 'browse' && (
            <div className="mb-4">
              <div className="search mb-3">
                <label className="visually-hidden" htmlFor="market-search">Search listings</label>
                <input
                  id="market-search"
                  className="search__input"
                  type="search"
                  placeholder="Search listings…"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
                <span className="search__button" aria-hidden="true">
                  <i className="fas fa-search" />
                </span>
                {searchQuery && (
                  <button className="search__clear" onClick={() => setSearchQuery('')} aria-label="Clear search">
                    <i className="fas fa-times" aria-hidden="true" />
                  </button>
                )}
              </div>

              <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }} className="mb-2">
                <div className="d-flex gap-2 align-items-center" style={{ width: 'max-content', paddingBottom: 2 }}
                     role="group" aria-label="Filter by category">
                  <span className="small me-1" style={{ whiteSpace: 'nowrap', color: 'var(--text-muted)' }}>Category</span>
                  {categories.map(cat => {
                    const active = categoryFilter.has(cat.slug)
                    return (
                      <button
                        key={cat.slug}
                        onClick={() => toggleIn(categoryFilter, cat.slug, 'cat')}
                        aria-pressed={active}
                        className={`btn btn-sm ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
                        style={{ borderRadius: 20, whiteSpace: 'nowrap' }}
                      >
                        <i className={`fas ${cat.icon || 'fa-tag'} me-1`} aria-hidden="true" />
                        <Label en={cat.label} zh={cat.label_zh} />
                      </button>
                    )
                  })}
                </div>
              </div>

              <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
                <div className="d-flex gap-2 align-items-center" style={{ width: 'max-content', paddingBottom: 4 }}
                     role="group" aria-label="Filter by delivery">
                  <span className="small me-1" style={{ whiteSpace: 'nowrap', color: 'var(--text-muted)' }}>Delivery</span>
                  {DELIVERY_OPTIONS.map(opt => {
                    const active = deliveryFilter.has(opt.value)
                    return (
                      <button
                        key={opt.value}
                        onClick={() => toggleIn(deliveryFilter, opt.value, 'del')}
                        aria-pressed={active}
                        className={`btn btn-sm ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
                        style={{ borderRadius: 20, whiteSpace: 'nowrap' }}
                      >
                        <i className={`fas ${opt.icon} me-1`} aria-hidden="true" />
                        <Label en={opt.label} zh={opt.label_zh} />
                      </button>
                    )
                  })}
                </div>
              </div>

              {hasActiveFilters && !loading && (
                <div className="d-flex align-items-center gap-2 mt-2 flex-wrap">
                  <span className="small" style={{ color: 'var(--text-muted)' }}>
                    {filteredBrowse.length} result{filteredBrowse.length !== 1 ? 's' : ''}
                  </span>
                  {[...categoryFilter].map(slug => (
                    <button key={slug} className="badge-pill" style={{ border: 'none', cursor: 'pointer' }}
                            onClick={() => toggleIn(categoryFilter, slug, 'cat')}>
                      {categoryLabel(slug).label} <i className="fas fa-times" aria-hidden="true" />
                      <span className="visually-hidden">Remove filter</span>
                    </button>
                  ))}
                  {[...deliveryFilter].map(v => (
                    <button key={v} className="badge-pill badge-pill--info" style={{ border: 'none', cursor: 'pointer' }}
                            onClick={() => toggleIn(deliveryFilter, v, 'del')}>
                      {(DELIVERY_BY_VALUE[v] || {}).label} <i className="fas fa-times" aria-hidden="true" />
                      <span className="visually-hidden">Remove filter</span>
                    </button>
                  ))}
                  <button className="btn btn-link btn-sm p-0" onClick={clearAllFilters}
                          style={{ color: 'var(--text-muted)' }}>
                    Clear all
                  </button>
                </div>
              )}
            </div>
          )}

          {tab === 'mylistings' && myListings.length > 0 && (
            <div className="d-flex justify-content-end mb-3">
              <button className="btn btn-outline-secondary btn-sm" onClick={() => setShowExport(true)}>
                <i className="fas fa-image me-1" aria-hidden="true" />Export as image
              </button>
            </div>
          )}

          {loading ? (
            <div className="text-center py-5"><HandLoader /></div>
          ) : displayList.length === 0 ? (
            <div className="text-center py-5" style={{ color: 'var(--text-muted)' }}>
              <i className="fas fa-box-open fa-3x mb-3" aria-hidden="true" />
              {tab === 'browse' && hasActiveFilters ? (
                <>
                  <p>No listings match your search.</p>
                  <button className="btn btn-outline-secondary btn-sm" onClick={clearAllFilters}>
                    Clear filters
                  </button>
                </>
              ) : (
                <>
                  <p>{tab === 'mylistings'
                    ? "You haven't posted anything yet."
                    : 'Nothing up for grabs right now.'}</p>
                  <button className="btn btn-primary" onClick={() => setTab('create')}>
                    Post a Listing
                  </button>
                </>
              )}
            </div>
          ) : (
            <div className="row row-cols-2 row-cols-sm-3 row-cols-lg-4 row-cols-xl-5 g-2">
              {displayList.map(l => {
                const mineHere = l.seller_username === user.username
                return (
                  <div className="col" key={l.id}>
                    <div style={highlightId === l.id
                      ? { outline: '2px solid var(--accent)', outlineOffset: 3, borderRadius: 'var(--radius-lg)' }
                      : undefined}>
                      <ListingCard
                        listing={l}
                        currentUser={user.username}
                        categoryLabel={categoryLabel}
                        onSold={handleSold}
                        onRestore={handleRestore}
                        onDelete={setPendingDelete}
                        onEdit={setEditListing}
                        onReachOut={handleReachOut}
                        onSellerClick={openSeller}
                        onDetail={() => openDetail(l.id)}
                        onWant={setWantListing}
                        myIntent={mineHere ? null : buyerIntentFor(l)}
                        onComplete={completeIntent}
                        onCancelIntent={setPendingCancelIntent}
                        incomingCount={mineHere ? incomingCountFor(l) : 0}
                        onShowIntents={list => setIncomingModalId(list.id)}
                        onCancelReservation={mineHere ? cancelReservationByListing : null}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </>
      )}

      {tab === 'create' && (
        <div className="row justify-content-center">
          <div className="col-lg-7">
            <div className="card">
              <div className="card-body p-4">
                <h5 className="card-title mb-4 fw-semibold">
                  <i className="fas fa-tag me-2 text-primary" aria-hidden="true" />Post a Listing
                </h5>

                <form onSubmit={handleCreate} noValidate>
                  <div className="mb-3">
                    <label className="form-label fw-medium" htmlFor="new-title">
                      Title <span className="text-danger" aria-hidden="true">*</span>
                    </label>
                    <input
                      id="new-title"
                      ref={titleRef}
                      className="form-control"
                      maxLength={100}
                      required
                      aria-required="true"
                      aria-invalid={fieldErrors.title ? 'true' : undefined}
                      aria-describedby={fieldErrors.title ? `${formErrId}-title` : undefined}
                      placeholder="e.g. iPhone 13 128GB"
                      value={form.title}
                      onChange={e => setForm(f => ({ ...f, title: e.target.value }))}
                    />
                    {fieldErrors.title && (
                      <p className="field-error" id={`${formErrId}-title`}>
                        <i className="fas fa-circle-exclamation" aria-hidden="true" />{fieldErrors.title}
                      </p>
                    )}
                  </div>

                  <div className="mb-3">
                    <label className="form-label fw-medium" htmlFor="new-desc">
                      Description <span className="text-danger" aria-hidden="true">*</span>
                    </label>
                    <textarea
                      id="new-desc"
                      ref={descRef}
                      className="form-control"
                      rows={4}
                      required
                      aria-required="true"
                      aria-invalid={fieldErrors.description ? 'true' : undefined}
                      aria-describedby={fieldErrors.description ? `${formErrId}-desc` : undefined}
                      placeholder="Condition, reason for selling, included accessories…"
                      value={form.description}
                      onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                    />
                    {fieldErrors.description && (
                      <p className="field-error" id={`${formErrId}-desc`}>
                        <i className="fas fa-circle-exclamation" aria-hidden="true" />{fieldErrors.description}
                      </p>
                    )}
                  </div>

                  {/* col-12 col-sm-6: bare `col` never stacked, so on a 375px
                      phone these were two ~150px inputs with 3-line labels. */}
                  <div className="row g-2 mb-3">
                    <div className="col-12 col-sm-6">
                      <label className="form-label fw-medium" htmlFor="new-orig">
                        Original Price ($) <span style={{ color: 'var(--text-muted)' }} className="small">(optional)</span>
                      </label>
                      <input
                        id="new-orig"
                        type="number" className="form-control" min={0} step="0.01"
                        aria-invalid={fieldErrors.original_price ? 'true' : undefined}
                        aria-describedby={fieldErrors.original_price ? `${formErrId}-orig` : undefined}
                        placeholder="e.g. 500.00"
                        value={form.original_price}
                        onChange={e => setForm(f => ({ ...f, original_price: e.target.value }))}
                      />
                      {fieldErrors.original_price && (
                        <p className="field-error" id={`${formErrId}-orig`}>
                          <i className="fas fa-circle-exclamation" aria-hidden="true" />{fieldErrors.original_price}
                        </p>
                      )}
                    </div>
                    <div className="col-12 col-sm-6">
                      <label className="form-label fw-medium" htmlFor="new-price">
                        Selling Price ($) <span className="text-danger" aria-hidden="true">*</span>
                      </label>
                      <input
                        id="new-price"
                        ref={priceRef}
                        type="number" className="form-control" min={0} step="0.01"
                        required
                        aria-required="true"
                        aria-invalid={fieldErrors.price ? 'true' : undefined}
                        aria-describedby={fieldErrors.price ? `${formErrId}-price` : undefined}
                        placeholder="0.00"
                        value={form.price}
                        onChange={e => setForm(f => ({ ...f, price: e.target.value }))}
                      />
                      {fieldErrors.price && (
                        <p className="field-error" id={`${formErrId}-price`}>
                          <i className="fas fa-circle-exclamation" aria-hidden="true" />{fieldErrors.price}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="mb-3">
                    <label className="form-label fw-medium" htmlFor="new-cat">Category</label>
                    <select
                      id="new-cat"
                      className="form-select"
                      value={form.category}
                      onChange={e => setForm(f => ({ ...f, category: e.target.value }))}
                    >
                      {categories.map(c => (
                        <option key={c.slug} value={c.slug}>
                          {c.label}{c.label_zh ? ` · ${c.label_zh}` : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  <fieldset className="mb-3">
                    <legend className="form-label fw-medium">Delivery Options</legend>
                    <div className="d-flex gap-3 flex-wrap">
                      {DELIVERY_OPTIONS.map(opt => (
                        <div className="form-check" key={opt.value}>
                          <input className="form-check-input" type="radio" name="delivery_type"
                            id={`dt-${opt.value}`} value={opt.value}
                            checked={form.delivery_type === opt.value}
                            onChange={() => setForm(f => ({
                              ...f, delivery_type: opt.value,
                              delivery_fee: opt.value === 'pickup' ? '' : f.delivery_fee,
                            }))} />
                          <label className="form-check-label" htmlFor={`dt-${opt.value}`}>
                            <Label en={opt.label} zh={opt.label_zh} />
                          </label>
                        </div>
                      ))}
                    </div>
                    {(form.delivery_type === 'delivery' || form.delivery_type === 'both') && (
                      <div className="mt-2">
                        <label className="form-label small" htmlFor="new-fee">
                          Delivery fee ($) — leave blank if free
                        </label>
                        <input id="new-fee" type="number" className="form-control" min={0} step="0.01"
                          placeholder="0.00"
                          value={form.delivery_fee}
                          onChange={e => setForm(f => ({ ...f, delivery_fee: e.target.value }))} />
                      </div>
                    )}
                  </fieldset>

                  <div className="mb-4">
                    <label className="form-label fw-medium" htmlFor="new-photos">Photos</label>
                    <p className="small mb-2" style={{ color: 'var(--text-muted)' }}>
                      Up to 3 photos · JPEG or PNG · 5 MB each
                      {images.length > 0 && images.length < 3 && ` — ${images.length} added, ${3 - images.length} to go`}
                      {images.length >= 3 && ' — limit reached'}
                    </p>
                    <input
                      id="new-photos"
                      ref={fileRef}
                      type="file"
                      className="form-control"
                      accept=".jpg,.jpeg,.png"
                      multiple
                      disabled={images.length >= 3}
                      onChange={handleFileChange}
                    />
                    {previews.length > 0 && (
                      <div className="d-flex gap-3 mt-3 flex-wrap">
                        {previews.map((src, i) => (
                          <div key={src} className="position-relative">
                            <img
                              src={src}
                              alt={`Selected photo ${i + 1}`}
                              style={{ width: 90, height: 90, objectFit: 'cover', borderRadius: 'var(--radius-md)' }}
                            />
                            {/* Was ~18x16px sitting on the thumbnail corner. */}
                            <button
                              type="button"
                              className="btn btn-danger position-absolute d-flex align-items-center justify-content-center"
                              style={{ width: 30, height: 30, padding: 0, borderRadius: '50%', top: -10, right: -10 }}
                              onClick={() => removeImage(i)}
                              aria-label={`Remove photo ${i + 1}`}
                            >
                              <i className="fas fa-times" aria-hidden="true" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <button type="submit" className="btn btn-primary w-100" disabled={submitting}>
                    {submitting
                      ? <><span className="spinner-border spinner-border-sm me-2" />Posting…</>
                      : <><i className="fas fa-paper-plane me-2" aria-hidden="true" />Post Listing</>
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
