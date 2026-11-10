import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'

const money = (n) => `$${(Number(n) || 0).toFixed(2)}`

export default function MarketRail() {
  const [items, setItems] = useState(null)   // null = loading, [] = empty, [...] = loaded

  useEffect(() => {
    let cancelled = false
    api.get('/api/market/listings?limit=3&offset=0').then(d => {
      if (cancelled) return
      if (d && d.ok && Array.isArray(d.listings)) {
        setItems(d.listings.filter(l => l.images && l.images[0] && l.images[0].url).slice(0, 3))
      } else {
        setItems([])
      }
    }).catch(() => { if (!cancelled) setItems([]) })
    return () => { cancelled = true }
  }, [])

  if (!items || items.length === 0) return null

  return (
    <section className="market-rail" aria-label="Just listed">
      <div className="market-rail__head">
        <div>
          <h5 className="market-rail__title">Just listed</h5>
          <p className="market-rail__sub">最新上架</p>
        </div>
        <Link to="/market" className="market-rail__all">
          全部 <i className="fas fa-arrow-right" aria-hidden="true" />
        </Link>
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
