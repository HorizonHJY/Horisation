import React, { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'
import HandLoader from '../components/HandLoader'
import { fmtDateJoin } from '../utils'
import ReviewComposer from '../components/ReviewComposer'
import Cat from '../components/CatArt'

const REVIEW_TAGS = ['准时', '描述相符', '态度好', '包装仔细']

export default function UserProfile() {
  const { username }      = useParams()
  const { user: me }      = useAuth()
  const navigate          = useNavigate()
  const [profile, setProfile]   = useState(null)
  const [listings, setListings] = useState([])
  const [loading, setLoading]   = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [copied, setCopied]     = useState(false)
  const [mbti, setMbti]         = useState(null)
  const [pet, setPet]           = useState(null)      // {breed,name,stage,bond,touched_today} or null
  const [petting, setPetting]   = useState(false)
  const [petNote, setPetNote]   = useState('')

  const [tab, setTab] = useState('reviews')

  // Reviews
  const [reviewsData, setReviewsData] = useState(null)     // {avg,count,reviews}
  const [pendingDeals, setPendingDeals] = useState([])     // own pending reviews about THIS profile
  const [rateFor, setRateFor]           = useState(null)   // pending deal being rated

  // Wall
  const [wall, setWall]       = useState([])
  const [wallDraft, setDraft] = useState('')
  const [wallSaving, setWallSaving] = useState(false)
  const [wallErr, setWallErr] = useState('')

  const loadReviews = useCallback(async () => {
    const r = await api.get(`/api/auth/users/${username}/reviews`)
    if (r.ok) setReviewsData({ avg: r.avg, count: r.count, reviews: r.reviews || [] })
  }, [username])

  const loadWall = useCallback(async () => {
    const r = await api.get(`/api/auth/users/${username}/wall`)
    if (r.ok) setWall(r.posts || [])
  }, [username])

  const loadPending = useCallback(async () => {
    if (!me) { setPendingDeals([]); return }
    const r = await api.get('/api/market/deals/pending')
    if (r.ok) {
      setPendingDeals((r.deals || []).filter(d => d.counterpart?.username === username))
    }
  }, [me, username])

  useEffect(() => {
    setLoading(true)
    setNotFound(false)
    setMbti(null)
    const safe = p => p.catch(() => ({ ok: false }))
    Promise.all([
      safe(api.get(`/api/auth/users/${username}/public`)),
      safe(api.get(`/api/market/user/${username}`)),
      safe(api.get(`/api/mbti/user/${username}`)),
    ]).then(([u, l, m]) => {
      if (!u || !u.ok || !u.user) { setNotFound(true); setLoading(false); return }
      setProfile(u.user)
      setPet(u.user.pet || null)
      if (l && l.ok) setListings(l.listings)
      if (m && m.ok && m.result) setMbti({ ...m.result, type_card: m.type_card })
      setLoading(false)
    })
    loadReviews()
    loadWall()
    loadPending()
  }, [username, loadReviews, loadWall, loadPending])

  const copyLink = () => {
    navigator.clipboard.writeText(`${window.location.origin}/u/${username}`)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  async function submitReview(payload) {
    if (!rateFor) return
    const r = await api.post(`/api/market/deals/${rateFor.id}/review`, payload)
    if (r.ok) {
      setRateFor(null)
      loadReviews()
      loadPending()
      return true
    }
    return r.error || 'Could not submit the review.'
  }

  async function submitWall(e) {
    e.preventDefault()
    setWallErr('')
    const content = wallDraft.trim()
    if (!content) { setWallErr('内容不能为空。'); return }
    setWallSaving(true)
    const r = await api.post(`/api/auth/users/${username}/wall`, { content })
    setWallSaving(false)
    if (r.ok) {
      setDraft('')
      loadWall()
    } else {
      setWallErr(r.error || 'Could not post.')
    }
  }

  async function removeWall(id) {
    const r = await api.delete(`/api/auth/users/${username}/wall/${id}`)
    if (r.ok) loadWall()
  }

  async function touchPet() {
    if (petting || !pet || pet.touched_today) return
    setPetting(true)
    setPetNote('')
    const r = await api.post('/api/pet/pet', { username })
    setPetting(false)
    if (!r.ok) { setPetNote(r.error || 'Could not pet it.'); return }
    setPet(r.pet)
    if (r.applied) setPetNote(r.xp_added ? `+${r.xp_delta} 经验` : '')
  }

  if (loading) return <HandLoader fullPage />

  if (notFound) return (
    <div className="container-fluid py-5 text-center text-muted">
      <i className="fas fa-user-slash fa-3x mb-3 d-block" style={{ opacity: 0.25 }} />
      <p>User not found.</p>
      <button className="btn btn-outline-secondary btn-sm" onClick={() => navigate(-1)}>
        <i className="fas fa-arrow-left me-1" />Go Back
      </button>
    </div>
  )

  const isMe   = me?.username === username
  const rating = reviewsData && reviewsData.count > 0
    ? { avg: reviewsData.avg.toFixed(1), count: reviewsData.count }
    : null

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 960 }}>

      {/* ── Profile header ─────────────────────────────────────────── */}
      <div className="card p-4 mb-4">
        <div className="d-flex align-items-center gap-3 flex-wrap">

          {profile.avatar_url ? (
            <img
              src={profile.avatar_url}
              alt={profile.display_name}
              className="rounded-circle flex-shrink-0"
              style={{ width: 72, height: 72, objectFit: 'cover' }}
            />
          ) : (
            <div
              className="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
              style={{
                width: 72, height: 72,
                background: '#6b9cdb1a', color: '#6b9cdb',
                fontSize: '1.8rem', fontWeight: 700,
              }}
            >
              {profile.display_name?.[0]?.toUpperCase() || '?'}
            </div>
          )}

          <div className="flex-grow-1" style={{ minWidth: 0 }}>
            <div className="d-flex align-items-center gap-2 flex-wrap">
              <h5 className="mb-0 fw-bold">{profile.display_name}</h5>
              {profile.level_name && (
                <span className="xp-level xp-level--sm" title={`等级 ${profile.level}`}>
                  <span className="xp-level__lv">Lv.{profile.level}</span>{profile.level_name}
                </span>
              )}
              {profile.hidden_tag && (
                <span className="xp-native" title="注册满一年">{profile.hidden_tag}</span>
              )}
              {mbti && (
                <Link to="/mbti" className="mbti-badge mbti-badge--sm text-decoration-none"
                      title={mbti.type_card?.nick || mbti.type}>
                  {mbti.type}
                </Link>
              )}
              {rating && (
                <span
                  className="badge rounded-pill"
                  style={{
                    background: '#f5efe3', color: '#6b5a2a',
                    border: '1px solid #e3d8be', fontWeight: 600,
                  }}
                  title="Marketplace reviews"
                >
                  <i className="fas fa-star me-1" style={{ color: '#d6a73a' }} aria-hidden="true" />
                  {rating.avg} · {rating.count}条
                </span>
              )}
            </div>
            {profile.bio && (
              <div
                className="text-muted mt-1"
                style={{
                  fontSize: '.85rem',
                  display: '-webkit-box',
                  WebkitBoxOrient: 'vertical',
                  WebkitLineClamp: 2,
                  overflow: 'hidden',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {profile.bio}
              </div>
            )}
          </div>

          <div className="d-flex gap-2 flex-shrink-0">
            {isMe && (
              <button className="btn btn-outline-secondary btn-sm" onClick={() => navigate('/profile')}>
                <i className="fas fa-pen me-1" />Edit
              </button>
            )}
            <button className="btn btn-outline-secondary btn-sm" onClick={copyLink}>
              <i className={`fas ${copied ? 'fa-check' : 'fa-link'} me-1`} />
              {copied ? 'Copied!' : 'Copy Link'}
            </button>
          </div>
        </div>
      </div>

      {/* ── The cat ───────────────────────────────────────────────── */}
      {pet && (
        <div className="pet-visit card mb-4">
          <span className="pet-visit__art"><Cat breed={pet.breed} stage={pet.stage} size={96} title={pet.name} /></span>
          <div className="pet-visit__body">
            <div className="pet-visit__name">
              {pet.name}
              <span className="pet-pill pet-pill--sm">{pet.stage_en} · {pet.stage_zh}</span>
            </div>
            <div className="pet-visit__breed">{pet.breed_en} · {pet.breed_zh}</div>
            {petNote && <div className="pet-visit__note" role="status">{petNote}</div>}
          </div>
          {!isMe && (
            <button
              type="button"
              className={'pet-touch' + (pet.touched_today ? ' is-done' : '')}
              disabled={petting || pet.touched_today}
              onClick={touchPet}
            >
              <i className="fas fa-hand" aria-hidden="true" />
              <span>{pet.touched_today ? 'Petted today' : 'Pet it'}</span>
              <span className="pet-touch__zh">{pet.touched_today ? '今天摸过了' : '摸一下'}</span>
            </button>
          )}
        </div>
      )}

      {/* ── Listings ───────────────────────────────────────────────── */}
      <div className="d-flex align-items-center gap-2 mb-3">
        <h6
          className="fw-semibold text-muted text-uppercase mb-0"
          style={{ fontSize: '.75rem', letterSpacing: '.08em' }}
        >
          Active Listings
        </h6>
        <span className="badge text-bg-secondary" style={{ fontSize: '.7rem' }}>
          {listings.length}
        </span>
      </div>

      {listings.length === 0 ? (
        <div className="text-center text-muted py-4">
          <i className="fas fa-store-slash fa-2x mb-2 d-block" style={{ opacity: 0.2 }} />
          <p className="mb-0 small">
            {isMe ? 'You have no active listings — head to Market to post one!' : 'No active listings.'}
          </p>
        </div>
      ) : (
        <div className="row row-cols-2 row-cols-sm-3 row-cols-lg-4 row-cols-xl-5 g-2">
          {listings.map(l => {
            const firstImg    = l.images?.[0]?.url
            const hasOriginal = l.original_price && l.original_price > l.price
            return (
              <div className="col" key={l.id}>
                <div className="market-card h-100">
                  <div className="market-card__img">
                    {firstImg
                      ? <img src={firstImg} alt={l.title} />
                      : <i className="fas fa-image placeholder-icon" />
                    }
                  </div>
                  <div className="market-card__title" title={l.title}>{l.title}</div>
                  <div className="market-card__meta">
                    <span className="market-card__category">{l.category}</span>
                  </div>
                  <p className="market-card__desc">{l.description}</p>
                  <hr className="market-card__divider" />
                  <div className="market-card__footer">
                    <div />
                    <div className="market-card__price">
                      <span>¥{l.price}</span>
                      {hasOriginal && (
                        <span style={{
                          fontSize: '.75rem', color: '#999',
                          textDecoration: 'line-through', marginLeft: 6,
                        }}>
                          ¥{l.original_price}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ── Reviews + Wall ─────────────────────────────────────────── */}
      <div className="card p-3 mt-4">
        <div className="btn-group btn-group-sm mb-3" role="tablist" style={{ alignSelf: 'flex-start' }}>
          <button
            className={`btn ${tab === 'reviews' ? 'btn-primary' : 'btn-outline-secondary'}`}
            onClick={() => setTab('reviews')}
          >
            <i className="fas fa-star me-1" aria-hidden="true" />
            Reviews{rating ? ` · ${rating.count}` : ''}
          </button>
          <button
            className={`btn ${tab === 'wall' ? 'btn-primary' : 'btn-outline-secondary'}`}
            onClick={() => setTab('wall')}
          >
            <i className="fas fa-comment-dots me-1" aria-hidden="true" />
            留言墙{wall.length ? ` · ${wall.length}` : ''}
          </button>
        </div>

        {tab === 'reviews' && (
          <div>
            {pendingDeals.length > 0 && (
              <div className="mb-3 p-3" style={{
                background: '#fbf6ec', border: '1px solid #e3d8be', borderRadius: 10,
              }}>
                <div className="fw-semibold small mb-2">
                  <i className="fas fa-pen-to-square me-1" aria-hidden="true" />
                  有待评价的交易
                </div>
                {pendingDeals.map(d => (
                  <div key={d.id} className="d-flex align-items-center justify-content-between gap-2 py-1">
                    <span className="small" style={{ minWidth: 0, flex: 1 }}>
                      <i className="fas fa-box me-1 text-muted" aria-hidden="true" />
                      {d.listing_title || d.listing_id}
                    </span>
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => setRateFor(d)}
                    >
                      去评价
                    </button>
                  </div>
                ))}
              </div>
            )}

            {rateFor && (
              <ReviewComposer
                deal={rateFor}
                tags={REVIEW_TAGS}
                onCancel={() => setRateFor(null)}
                onSubmit={submitReview}
              />
            )}

            {!reviewsData || reviewsData.count === 0 ? (
              <div className="text-center text-muted py-3 small">
                暂无评价。
              </div>
            ) : (
              <div className="d-flex flex-column gap-2">
                <div className="small text-muted">
                  平均 <strong style={{ color: 'var(--text-primary)' }}>{rating.avg}</strong>
                  &nbsp;·&nbsp;共 {rating.count} 条
                </div>
                {reviewsData.reviews.map(r => (
                  <ReviewItem key={r.id} review={r} />
                ))}
              </div>
            )}
          </div>
        )}

        {tab === 'wall' && (
          <div>
            {me && !isMe && (
              <form className="mb-3" onSubmit={submitWall}>
                <textarea
                  className="form-control"
                  rows={2}
                  maxLength={200}
                  placeholder={`给 ${profile.display_name} 留个言…`}
                  value={wallDraft}
                  onChange={e => setDraft(e.target.value)}
                />
                <div className="d-flex align-items-center justify-content-between mt-2 gap-2">
                  <span className="small text-muted">{wallDraft.length}/200</span>
                  {wallErr && <span className="small text-danger flex-grow-1 text-center">{wallErr}</span>}
                  <button className="btn btn-primary btn-sm" disabled={wallSaving || !wallDraft.trim()}>
                    {wallSaving ? '发送中…' : '留言'}
                  </button>
                </div>
              </form>
            )}
            {isMe && (
              <div className="small text-muted mb-2">
                这是你自己的留言墙,别人可以在这里给你留言。
              </div>
            )}

            {wall.length === 0 ? (
              <div className="text-center text-muted py-3 small">
                还没人留言,成为第一个吧。
              </div>
            ) : (
              <div className="d-flex flex-column gap-2">
                {wall.map(p => (
                  <WallItem
                    key={p.id}
                    post={p}
                    canDelete={!!me && (me.username === p.author || me.role === 'admin')}
                    onDelete={() => removeWall(p.id)}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ReviewItem({ review }) {
  const tags = Array.isArray(review.tags) ? review.tags : []
  if (review.hidden) {
    return (
      <div className="p-2 rounded" style={{ background: '#faf7f0', border: '1px solid #ece3cf' }}>
        <div className="d-flex align-items-center justify-content-between">
          <span className="small text-muted">
            <i className="fas fa-eye-slash me-1" aria-hidden="true" />
            已评价 · 双方评完或 14 天后公开
          </span>
          <span className="small text-muted">{fmtDateJoin(review.created_at)}</span>
        </div>
      </div>
    )
  }
  return (
    <div className="p-2 rounded" style={{ background: '#fff', border: '1px solid var(--border-soft, #e8e2d4)' }}>
      <div className="d-flex align-items-center justify-content-between">
        <div className="d-flex align-items-center gap-2">
          <span className="fw-semibold small">
            {review.from_display || review.from_user}
          </span>
          <span className="small" style={{ color: '#d6a73a' }}>
            {'★'.repeat(review.rating)}
            <span style={{ color: '#ddd' }}>{'★'.repeat(5 - review.rating)}</span>
          </span>
        </div>
        <span className="small text-muted">{fmtDateJoin(review.created_at)}</span>
      </div>
      {tags.length > 0 && (
        <div className="d-flex gap-1 flex-wrap mt-1">
          {tags.map(t => (
            <span key={t} className="badge text-bg-light border" style={{ fontWeight: 500 }}>{t}</span>
          ))}
        </div>
      )}
      {review.note && <div className="small mt-1" style={{ whiteSpace: 'pre-wrap' }}>{review.note}</div>}
    </div>
  )
}

function WallItem({ post, canDelete, onDelete }) {
  return (
    <div className="p-2 rounded d-flex gap-2" style={{ background: '#fff', border: '1px solid var(--border-soft, #e8e2d4)' }}>
      {post.author_avatar
        ? <img src={post.author_avatar} alt="" width={32} height={32} style={{ borderRadius: '50%', objectFit: 'cover', flexShrink: 0 }} />
        : <span style={{
            width: 32, height: 32, borderRadius: '50%', background: '#f0ead8',
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            color: '#9a8b60', flexShrink: 0,
          }}><i className="fas fa-user" aria-hidden="true" /></span>
      }
      <div className="flex-grow-1" style={{ minWidth: 0 }}>
        <div className="d-flex align-items-center justify-content-between gap-2">
          <Link to={`/u/${post.author}`} className="fw-semibold small text-decoration-none">
            {post.author_display || post.author}
          </Link>
          <div className="d-flex align-items-center gap-2">
            <span className="small text-muted">{fmtDateJoin(post.created_at)}</span>
            {canDelete && (
              <button
                className="btn btn-link btn-sm p-0 text-danger"
                onClick={onDelete}
                aria-label="Delete"
                title="Delete"
              >
                <i className="fas fa-trash" aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
        <div className="small mt-1" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {post.content}
        </div>
      </div>
    </div>
  )
}
