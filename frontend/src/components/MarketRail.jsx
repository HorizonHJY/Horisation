import React, { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'

const money = (n) => `$${(Number(n) || 0).toFixed(2)}`

const SLOT_LABEL = {
  hot:    '热门',
  fresh:  '上新',
  rotate: '轮换',
}

export default function MarketRail() {
  const [items, setItems]   = useState(null)   // null = loading, [] = empty, [...] = loaded
  const [seed,  setSeed]    = useState(0)

  const load = useCallback((s) => {
    let cancelled = false
    api.get(`/api/market/rail?seed=${s}`).then(d => {
      if (cancelled) return
      if (d && d.ok && Array.isArray(d.items)) {
        setItems(d.items)
      } else {
        setItems([])
      }
    }).catch(() => { if (!cancelled) setItems([]) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => load(seed), [seed, load])

  if (!items || items.length === 0) return null

  return (
    <section className="market-rail" aria-label="Today's shelf">
      <div className="market-rail__head">
        <div>
          <h5 className="market-rail__title">Today's shelf</h5>
          <p className="market-rail__sub">今日摊位</p>
        </div>
        <div className="market-rail__actions">
          <button
            type="button"
            className="market-rail__shuffle"
            onClick={() => setSeed(s => s + 1)}
            aria-label="Shuffle rotate slot"
          >
            换一批
          </button>
          <Link to="/market" className="market-rail__all">
            全部 <i className="fas fa-arrow-right" aria-hidden="true" />
          </Link>
        </div>
      </div>

      <div className="market-rail__track">
        {items.map(l => (
          <Link
            key={l.id}
            to={`/market/l/${l.id}`}
            className="market-rail__card"
            aria-label={`${l.title} — ${money(l.price)}`}
          >
            <div className="market-rail__thumb">
              <img
                src={l.images[0].url}
                alt=""
                loading="lazy"
              />
              {SLOT_LABEL[l.slot] && (
                <span className={`market-rail__slot market-rail__slot--${l.slot}`}>
                  {SLOT_LABEL[l.slot]}
                </span>
              )}
            </div>
            <div className="market-rail__body">
              <div className="market-rail__price tnum">{money(l.price)}</div>
              <div className="market-rail__name" title={l.title}>{l.title}</div>
              <div className="market-rail__seller">
                {l.seller_display || l.seller_username}
              </div>
            </div>
          </Link>
        ))}
      </div>
    </section>
  )
}
