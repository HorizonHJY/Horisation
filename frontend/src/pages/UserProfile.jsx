import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'
import HandLoader from '../components/HandLoader'

export default function UserProfile() {
  const { username }      = useParams()
  const { user: me }      = useAuth()
  const navigate          = useNavigate()
  const [profile, setProfile]   = useState(null)
  const [listings, setListings] = useState([])
  const [loading, setLoading]   = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [copied, setCopied]     = useState(false)

  useEffect(() => {
    setLoading(true)
    setNotFound(false)
    Promise.all([
      api.get(`/api/auth/users/${username}/public`),
      api.get(`/api/market/user/${username}`),
    ]).then(([u, l]) => {
      if (!u.ok) { setNotFound(true); setLoading(false); return }
      setProfile(u.user)
      if (l.ok) setListings(l.listings)
      setLoading(false)
    })
  }, [username])

  const copyLink = () => {
    navigator.clipboard.writeText(`${window.location.origin}/u/${username}`)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
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

  const isMe = me?.username === username

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 960 }}>

      {/* ── Profile header ─────────────────────────────────────────── */}
      <div className="card p-4 mb-4">
        <div className="d-flex align-items-center gap-3">

          {/* Avatar */}
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

          {/* Info */}
          <div className="flex-grow-1">
            <h5 className="mb-0 fw-bold">{profile.display_name}</h5>
            <div className="text-muted small">@{profile.username}</div>
            {profile.created_at && (
              <div className="text-muted mt-1" style={{ fontSize: '.75rem' }}>
                Joined {new Date(profile.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
              </div>
            )}
          </div>

          {/* Actions */}
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
        <div className="text-center text-muted py-5">
          <i className="fas fa-store-slash fa-3x mb-3 d-block" style={{ opacity: 0.2 }} />
          <p className="mb-0">
            {isMe
              ? 'You have no active listings — head to Market to post one!'
              : 'No active listings.'}
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
    </div>
  )
}
