import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import ReadingFeedback from '../components/ReadingFeedback'
import HandLoader from '../components/HandLoader'

/**
 * Past spreads — every reading of yours that was interpreted, newest first.
 *
 * Draws that were never read are not here (GET /api/tarot/readings skips
 * them): three cards with no words is nothing to look back at. Each row is a
 * <details>, so the list stays scannable and a reading opens in place; the
 * point of the page is "what did I draw last month, and did it fit".
 */

const PAGE = 20
const KEYS = ['past', 'present', 'future']

/** read_at is naive UTC from the server; mark it so before formatting. */
function when(iso) {
  if (!iso) return ''
  const d = new Date(iso.endsWith('Z') ? iso : `${iso}Z`)
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  })
}

function Stars({ value }) {
  if (!value) return <span className="tarot-hist__unrated">Not rated · 未评分</span>
  return (
    <span className="tarot-hist__stars" aria-label={`Rated ${value} of 5`}>
      {[1, 2, 3, 4, 5].map(n => (
        <i key={n} className={`fas fa-star${n <= value ? ' is-on' : ''}`} aria-hidden="true" />
      ))}
    </span>
  )
}

function Row({ item }) {
  const r = item.reading
  const spread = item.spread || []
  const label = (key) => spread.find(e => e.position.key === key)?.position || { label: key, label_zh: '' }
  const teaser = item.question || r?.summary || item.reading_text || ''

  return (
    <details className="tarot-hist__item">
      <summary className="tarot-hist__summary">
        <div className="tarot-hist__cards" aria-hidden="true">
          {spread.map(e => (
            <img key={e.position.key} src={`/tarot/${e.card.img}`} alt="" draggable="false" />
          ))}
        </div>
        <div className="tarot-hist__meta">
          <div className="tarot-hist__top">
            <time dateTime={item.read_at}>{when(item.read_at)}</time>
            <Stars value={item.rating} />
          </div>
          <div className="tarot-hist__names">
            {spread.map(e => e.card.name_zh || e.card.name).join(' · ')}
          </div>
          {teaser && (
            <p className="tarot-hist__teaser">
              {item.question && <span className="tarot-read__q-label">You asked</span>}
              {teaser}
            </p>
          )}
        </div>
        <i className="fas fa-chevron-down tarot-hist__chev" aria-hidden="true" />
      </summary>

      <div className="tarot-hist__body">
        {r ? (
          <>
            <p className="tarot-read__summary">{r.summary}</p>
            <div className="tarot-read__positions">
              {KEYS.map(key => {
                const p = label(key)
                const card = spread.find(e => e.position.key === key)?.card
                return (
                  <div className="tarot-read__pos" key={key}>
                    <div className="tarot-read__pos-label">
                      {p.label}<span className="label-zh">{p.label_zh}</span>
                      {card && <span className="tarot-hist__pos-card"> · {card.name_zh || card.name}</span>}
                    </div>
                    <p className="mb-0">{r[key]}</p>
                  </div>
                )
              })}
            </div>
            {r.next_step && (
              <div className="tarot-read__next">
                <div className="tarot-read__pos-label">One small thing<span className="label-zh">当时建议的一件小事</span></div>
                <p className="mb-0">{r.next_step}</p>
              </div>
            )}
          </>
        ) : (
          <p className="tarot-read__summary" style={{ whiteSpace: 'pre-wrap' }}>{item.reading_text}</p>
        )}
        {item.rating_note && <p className="tarot-hist__note">“{item.rating_note}”</p>}

        <ReadingFeedback feature="tarot" readingId={item.id} />
      </div>
    </details>
  )
}

export default function TarotHistory() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [more, setMore] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async (before) => {
    setLoading(true)
    const q = new URLSearchParams({ limit: String(PAGE) })
    if (before) q.set('before', before)
    const d = await api.get(`/api/tarot/readings?${q}`)
    setLoading(false)
    if (!d.ok) { setError(d.error || 'Could not load your readings.'); return }
    setItems(prev => (before ? [...prev, ...d.readings] : d.readings))
    // A full page might have more behind it; a short one is the end.
    setMore(d.readings.length === PAGE)
  }, [])

  useEffect(() => { load() }, [load])

  return (
    <div className="tarot">
      <div className="tarot__inner tarot-hist">
        <div className="tarot__head">
          <div>
            <h1 className="tarot__title">History</h1>
            <p className="tarot__subtitle">历史牌阵 — 你解读过的每一次</p>
          </div>
          <Link to="/tarot" className="tarot__head-link">
            <i className="fas fa-arrow-left" aria-hidden="true" /> New reading 抽新牌
          </Link>
        </div>

        {error && <div className="alert alert-danger" role="alert">{error}</div>}

        {!loading && !error && items.length === 0 && (
          <div className="tarot-hist__empty">
            <p>No readings yet. A spread shows up here once you ask for its reading.</p>
            <p className="label-zh">还没有记录。抽完三张牌、点了「解读」之后，这里就会出现。</p>
            <Link to="/tarot" className="tarot__btn">Start 开始</Link>
          </div>
        )}

        <div className="tarot-hist__list">
          {items.map(item => <Row key={item.id} item={item} />)}
        </div>

        {loading && <div className="text-center py-4"><HandLoader /></div>}

        {!loading && more && (
          <div className="tarot__actions">
            <button
              type="button"
              className="tarot__btn tarot__btn--quiet"
              onClick={() => load(items[items.length - 1]?.read_at)}
            >
              Older 更早的
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
