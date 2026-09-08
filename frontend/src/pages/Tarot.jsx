import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import HandLoader from '../components/HandLoader'

/**
 * Three-card spread — past, present, future.
 *
 * The draw happens on the server (see tarot_controller): a reading you can
 * re-roll from devtools is not a reading. This file owns the theatre only.
 *
 * Cards: Rider-Waite-Smith scans from metabismuth/tarot-json (MIT); the deck
 * is public domain in the US. Card text is Waite's Pictorial Key (1911).
 */

const POSITION_COUNT = 3

/* Laying the fan out by rotating every card about one distant origin turns 78
   cards into a wheel. A real spread is a wide shallow arc, so position is
   driven horizontally and the curve is a parabola on top of it:
     x     spreads the deck evenly across the table
     y     dips at the centre and lifts at the ends
     tilt  small, proportional to distance from the middle
   Cards overlap heavily by design — at this count only a sliver of each shows,
   which is exactly what a fanned deck looks like. */
const FAN_SPAN_PCT = 88     // share of the container the fan occupies
const FAN_ARC_PX = 54       // how far the ends lift above the centre
const FAN_TILT_DEG = 26     // tilt at the outermost card

function fanStyle(i, total) {
  const t = total > 1 ? i / (total - 1) - 0.5 : 0    // -0.5 … +0.5
  const lift = (t * t) * 4 * FAN_ARC_PX              // 0 at centre, FAN_ARC_PX at ends
  const tilt = t * 2 * FAN_TILT_DEG
  return {
    left: `${50 + t * FAN_SPAN_PCT}%`,
    transform: `translateX(-50%) translateY(${lift.toFixed(1)}px) rotate(${tilt.toFixed(2)}deg)`,
  }
}

function Fan({ cards, dealing }) {
  return (
    <div className="tarot__fan" aria-hidden="true">
      {cards.map((c, i) => {
        const s = fanStyle(i, cards.length)
        return (
          <div
            key={c.id}
            className="tarot__fan-card"
            style={{
              ...s,
              zIndex: i,
              // Stagger so the fan gathers itself rather than snapping as a block.
              transitionDelay: dealing ? `${i * 5}ms` : '0ms',
            }}
          />
        )
      })}
    </div>
  )
}

function Slot({ position, card, revealed }) {
  return (
    <div className={[
      'tarot__slot',
      card ? 'tarot__slot--dealt' : 'tarot__slot--empty',
      revealed ? 'tarot__slot--revealed' : '',
    ].join(' ')}>
      <div className="tarot__slot-frame">
        {/* Before the draw the slot is an empty place on the cloth, not a
            face-down card — the back only arrives once a card is dealt into it. */}
        {card && (
          <div className="tarot__flipper">
            <div className="tarot__face tarot__face--back" />
            <div className="tarot__face tarot__face--front">
              <img src={`/tarot/${card.img}`} alt={revealed ? card.name : ''} draggable="false" />
            </div>
          </div>
        )}
      </div>
      <div className="tarot__slot-label">
        {position.label}
        <span className="label-zh">{position.label_zh}</span>
      </div>
      <div className="tarot__card-name">{revealed && card ? card.name : ''}</div>
    </div>
  )
}

export default function Tarot() {
  const [deck, setDeck] = useState([])
  const [positions, setPositions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [spread, setSpread] = useState(null)     // [{position, card}, ...]
  const [phase, setPhase] = useState('idle')     // idle | dealing | revealing | done
  const [revealedCount, setRevealedCount] = useState(0)
  const timers = useRef([])

  const clearTimers = useCallback(() => {
    timers.current.forEach(clearTimeout)
    timers.current = []
  }, [])

  useEffect(() => () => clearTimers(), [clearTimers])

  useEffect(() => {
    api.get('/api/tarot/deck').then(d => {
      if (d.ok) {
        setDeck(d.cards || [])
        setPositions(d.positions || [])
      } else {
        setError(d.error || 'Could not load the deck.')
      }
      setLoading(false)
    })
  }, [])

  const prefersReducedMotion = useMemo(
    () => typeof window !== 'undefined'
      && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
    [])

  async function drawSpread() {
    if (phase === 'dealing' || phase === 'revealing') return
    clearTimers()
    setError('')
    setSpread(null)
    setRevealedCount(0)
    setPhase('dealing')

    const d = await api.post('/api/tarot/draw')
    if (!d.ok) {
      setError(d.error || 'The cards would not come.')
      setPhase('idle')
      return
    }

    // Let the fan finish gathering before the slots start turning, unless the
    // reader asked for less motion — then land everything at once.
    const gather = prefersReducedMotion ? 0 : 620
    const step = prefersReducedMotion ? 0 : 700

    timers.current.push(setTimeout(() => {
      setSpread(d.spread)
      setPhase('revealing')
      for (let i = 1; i <= POSITION_COUNT; i++) {
        timers.current.push(setTimeout(() => {
          setRevealedCount(i)
          if (i === POSITION_COUNT) setPhase('done')
        }, step * i))
      }
    }, gather))
  }

  if (loading) {
    return (
      <div className="tarot">
        <div className="tarot__inner text-center py-5"><HandLoader /></div>
      </div>
    )
  }

  const slots = positions.length ? positions : []
  const dealing = phase === 'dealing'
  const busy = dealing || phase === 'revealing'

  return (
    <div className={`tarot ${dealing && !prefersReducedMotion ? 'tarot--dealing' : ''}`}>
      <div className="tarot__inner">
        <h1 className="tarot__title">Tarot</h1>
        <p className="tarot__subtitle">三张牌 — 过去 · 现在 · 未来</p>

        {error && (
          <div className="alert alert-danger" role="alert">{error}</div>
        )}

        {/* The full deck, face down. Hidden from assistive tech: it is scenery,
            and the reading itself is announced below. It is drawn for everyone
            — reduced motion removes the gathering animation, not the deck. */}
        {deck.length > 0 && (
          <Fan cards={deck} dealing={dealing && !prefersReducedMotion} />
        )}

        <div className="tarot__spread">
          {slots.map((pos, i) => (
            <Slot
              key={pos.key}
              position={pos}
              card={spread?.[i]?.card}
              revealed={Boolean(spread) && revealedCount > i}
            />
          ))}
        </div>

        <div className="tarot__actions">
          <button
            type="button"
            className="tarot__btn"
            onClick={drawSpread}
            disabled={busy}
          >
            {busy ? 'Drawing…' : spread ? 'Draw again' : 'Draw three cards'}
          </button>
        </div>

        {/* Announced as one block once the turn is over, so a screen reader
            hears the finished reading rather than three interruptions. */}
        <div className="tarot__reading" role="status" aria-live="polite">
          {phase === 'done' && spread?.map(({ position, card }) => (
            <div className="tarot__entry" key={position.key}>
              <div className="tarot__entry-head">
                <span className="tarot__entry-pos">
                  {position.label}<span className="label-zh">{position.label_zh}</span>
                </span>
                <span className="tarot__entry-name">{card.name}</span>
              </div>
              {card.keywords && <p className="tarot__entry-keywords mb-0">{card.keywords}</p>}
              {card.description && <p className="tarot__entry-desc mb-0">{card.description}</p>}
            </div>
          ))}
        </div>

        <p className="tarot__footnote">
          Rider–Waite–Smith deck, public domain in the US · card text from
          A. E. Waite, <em>The Pictorial Key to the Tarot</em> (1911)
        </p>
      </div>
    </div>
  )
}
