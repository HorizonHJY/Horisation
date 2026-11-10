import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'

/**
 * Today's one card — a single card the server picked for this user today,
 * with two short sentences, linking into the full three-card spread.
 *
 * Deliberately quiet on failure: if the request errors (not signed in, API
 * down, AI off), the component renders nothing. The home page should still
 * work without it.
 */
export default function DailyCard() {
  const [state, setState] = useState({ loading: true, daily: null, error: false })

  useEffect(() => {
    let cancelled = false
    api.get('/api/tarot/daily').then((res) => {
      if (cancelled) return
      if (res && res.ok && res.daily) {
        setState({ loading: false, daily: res.daily, error: false })
      } else {
        setState({ loading: false, daily: null, error: true })
      }
    })
    return () => { cancelled = true }
  }, [])

  if (state.error) return null                    // never block the home page
  if (state.loading) return <DailyCardSkeleton />
  if (!state.daily) return null

  const { card, upright, text } = state.daily
  const nameZh = card.name_zh || ''
  const orient = upright ? '正位' : '逆位'

  return (
    <div
      className="card mb-4"
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border-soft)',
        borderRadius: 12,
        overflow: 'hidden',
      }}
    >
      <div className="card-body p-3 p-sm-4">
        <div
          className="d-flex align-items-center gap-2 mb-3"
          style={{ color: 'var(--text-muted)' }}
        >
          <i className="fas fa-moon" style={{ color: '#8b5fbf', fontSize: '.95rem' }} />
          <span style={{
            fontFamily: "'Playfair Display', serif",
            fontWeight: 700,
            fontSize: '.95rem',
            letterSpacing: '.02em',
            color: 'var(--text-primary)',
          }}>
            Today's Card
          </span>
          <span style={{
            fontFamily: "'Noto Serif SC', serif",
            fontStyle: 'italic',
            fontSize: '.82rem',
          }}>
            今日一签
          </span>
        </div>

        <div className="d-flex gap-3 gap-sm-4 align-items-stretch">
          <div
            style={{
              flex: '0 0 auto',
              width: 'clamp(76px, 22vw, 108px)',
              aspectRatio: '350 / 600',
              borderRadius: 8,
              overflow: 'hidden',
              background: '#2b2434',
              boxShadow: '0 4px 14px rgba(26,26,26,0.14)',
            }}
          >
            {card.img && (
              <img
                src={`/tarot/${card.img}`}
                alt={card.name}
                draggable="false"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: 'block',
                  transform: upright ? 'none' : 'rotate(180deg)',
                }}
              />
            )}
          </div>

          <div className="d-flex flex-column" style={{ flex: '1 1 auto', minWidth: 0 }}>
            <div style={{
              fontFamily: "'Playfair Display', serif",
              fontWeight: 700,
              fontSize: 'clamp(1rem, 3.6vw, 1.15rem)',
              color: 'var(--text-primary)',
              lineHeight: 1.25,
              marginBottom: 2,
            }}>
              {card.name}
            </div>
            <div style={{
              fontFamily: "'Noto Serif SC', serif",
              fontSize: '.85rem',
              color: 'var(--text-muted)',
              marginBottom: 10,
            }}>
              {nameZh ? `${nameZh} · ` : ''}{orient}
            </div>

            <p style={{
              fontFamily: "'Noto Serif SC', serif",
              fontSize: 'clamp(.88rem, 3.4vw, .95rem)',
              lineHeight: 1.7,
              color: 'var(--text-secondary, var(--text-primary))',
              margin: 0,
              flex: '1 1 auto',
            }}>
              {text}
            </p>

            <div style={{ marginTop: 12 }}>
              <Link
                to="/tarot"
                style={{
                  fontFamily: "'Noto Serif SC', serif",
                  fontSize: '.86rem',
                  color: '#8b5fbf',
                  textDecoration: 'none',
                  fontWeight: 500,
                }}
              >
                体验完整功能 →
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}


function DailyCardSkeleton() {
  // Same layout as the real card so the home page does not reflow once the
  // real data lands.
  const shimmer = {
    background: 'linear-gradient(90deg, var(--bg-subtle) 0%, var(--bg-surface) 50%, var(--bg-subtle) 100%)',
    backgroundSize: '200% 100%',
    animation: 'daily-shimmer 1.4s ease-in-out infinite',
    borderRadius: 6,
  }
  return (
    <div
      className="card mb-4"
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border-soft)',
        borderRadius: 12,
        overflow: 'hidden',
      }}
    >
      <style>{`
        @keyframes daily-shimmer { 0% { background-position: 200% 0 } 100% { background-position: -200% 0 } }
      `}</style>
      <div className="card-body p-3 p-sm-4">
        <div style={{ height: 18, width: 160, marginBottom: 14, ...shimmer }} />
        <div className="d-flex gap-3 gap-sm-4">
          <div style={{
            flex: '0 0 auto',
            width: 'clamp(76px, 22vw, 108px)',
            aspectRatio: '350 / 600',
            ...shimmer,
          }} />
          <div style={{ flex: '1 1 auto', minWidth: 0 }}>
            <div style={{ height: 16, width: '55%', marginBottom: 8, ...shimmer }} />
            <div style={{ height: 12, width: '35%', marginBottom: 16, ...shimmer }} />
            <div style={{ height: 12, width: '95%', marginBottom: 8, ...shimmer }} />
            <div style={{ height: 12, width: '80%', ...shimmer }} />
          </div>
        </div>
      </div>
    </div>
  )
}
