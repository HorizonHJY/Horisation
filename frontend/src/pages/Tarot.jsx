import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import HandLoader from '../components/HandLoader'
import Modal from '../components/Modal'

/**
 * Three-card spread — past, present, future.
 *
 * The draw happens on the server (see tarot_controller): a reading you can
 * re-roll from devtools is not a reading. This file owns the theatre.
 *
 * The theatre is the point. A reading is a small ritual with a fixed order:
 * you hold a question, the reader shuffles, you touch three cards, they turn
 * over. Each of those beats is a phase here, and the deck is the one thing on
 * screen that moves.
 *
 * On which card is "yours": the server picks the three the moment the shuffle
 * starts, and the card you touch takes the next one in that order. That is
 * exactly as true as a physical reading — the deck was shuffled before your
 * hand went near it, and which slice of a shuffled deck you touch tells you
 * nothing. What the interaction has to be honest about is that you couldn't
 * have known, and you couldn't.
 *
 * Cards: Rider-Waite-Smith scans from metabismuth/tarot-json (MIT); the deck
 * is public domain in the US. Card text is Waite's Pictorial Key (1911).
 */

const POSITION_COUNT = 3

/* Layout. A single arc of 78 cards leaves each one a sliver too thin to aim
   at, so the deck breaks into rows — two on a wide screen, three otherwise. */
const WIDE_QUERY = '(min-width: 900px)'
const ROWS_WIDE = 2
const ROWS_NARROW = 3

/* Arc shape, per row. Rotating every card about one distant origin makes a
   wheel; a real spread is a wide shallow arc, so the three quantities are
   computed separately: position spreads evenly, the curve is a parabola on
   top of it, and the tilt is proportional to distance from the middle. */
const ARC_LIFT_PX = 30      // how far a row's ends rise above its centre
const ARC_TILT_DEG = 13     // tilt of the outermost card in a row
const CARD_RATIO = 600 / 350

/* Timings. The shuffle is deliberately long — it is the one authored moment
   on this page, and a reader who shuffles in 200ms is not shuffling. */
const RIFFLE_MS = 1500
const RIFFLE_STAGGER_MS = 5
const FLIGHT_MS = 620
const REVEAL_GAP_MS = 700
const SETTLE_MS = 420

/** Stable per-card jitter, so the idle pile looks hand-squared and does not
 *  reshuffle itself on every render the way Math.random() would. */
function jitter(i, spread) {
  const n = Math.sin((i + 1) * 12.9898) * 43758.5453
  return ((n - Math.floor(n)) - 0.5) * 2 * spread
}

/** Split 78 cards into `rows` near-equal rows, order preserved. */
function splitRows(items, rows) {
  const out = []
  let start = 0
  for (let r = 0; r < rows; r++) {
    const size = Math.ceil((items.length - start) / (rows - r))
    out.push(items.slice(start, start + size))
    start += size
  }
  return out
}

/** Resting transform of one card, as CSS custom properties. */
function seatStyle(indexInRow, rowLength, rowWidth, cardWidth) {
  const t = rowLength > 1 ? indexInRow / (rowLength - 1) - 0.5 : 0   // -0.5 … +0.5
  const span = Math.max(0, rowWidth - cardWidth)
  return {
    '--seat-x': `${(t * span).toFixed(1)}px`,
    '--seat-y': `${(t * t * 4 * ARC_LIFT_PX).toFixed(1)}px`,
    '--seat-rot': `${(t * 2 * ARC_TILT_DEG).toFixed(2)}deg`,
  }
}

/* Tilt range when you hold a card up to the light. Past about 15° the
   foreshortening starts to fight the artwork instead of describing a surface. */
const TILT_MAX_DEG = 14
const OPEN_MS = 460

/**
 * One card, held up and turned over.
 *
 * The scans are 350×600, so the slot on the table shows perhaps a fifth of what
 * is actually there — the small figures at the edges of a Waite card are half
 * of what the card says. This is the place to look at them. It deliberately
 * does not zoom past the source resolution: enlarging a 600px scan to 1200
 * shows JPEG artefacts, not detail, so the card is capped at roughly its own
 * size and the honest answer to "bigger?" is that there is no more to see.
 */
function Inspector({ entry, index, count, fromRect, onClose, onStep, reducedMotion }) {
  const { position, card } = entry
  const [flipped, setFlipped] = useState(false)
  const cardRef = useRef(null)
  const surfaceRef = useRef(null)
  const draggingRef = useRef(false)
  const pendingRef = useRef(null)
  const frameRef = useRef(0)

  // Reset the turn when stepping to another card — you asked to see this one.
  useEffect(() => { setFlipped(false) }, [index])

  /* Opens out of the slot it came from rather than fading in from nowhere, so
     there is never a question about which of the three you are looking at. */
  useLayoutEffect(() => {
    const el = cardRef.current
    if (!el || !fromRect || reducedMotion) return
    const to = el.getBoundingClientRect()
    if (!to.width) return
    const dx = (fromRect.left + fromRect.width / 2) - (to.left + to.width / 2)
    const dy = (fromRect.top + fromRect.height / 2) - (to.top + to.height / 2)
    const anim = el.animate([
      { transform: `translate(${dx}px, ${dy}px) scale(${fromRect.width / to.width})`, opacity: .6 },
      { transform: 'none', opacity: 1 },
    ], { duration: OPEN_MS, easing: 'cubic-bezier(0.16, 1, 0.3, 1)' })
    return () => anim.cancel()
  }, [fromRect, reducedMotion])

  /* Tilt follows the pointer: on a mouse just by hovering, on a touch screen
     only while a finger is down, so scrolling the page still works. Written
     as custom properties — no re-render per frame. */
  /* Tilt follows the pointer: on a mouse just by hovering, on a touch screen
     only while a finger is down, so scrolling the page still works.
     Written as custom properties, and only once per frame — a pointermove can
     fire several times between paints, and writing on every one of them is
     work the screen never shows. */
  const tilt = useCallback((e) => {
    if (reducedMotion) return
    const el = surfaceRef.current
    if (!el) return
    if (e.pointerType !== 'mouse' && !draggingRef.current) return

    const r = el.getBoundingClientRect()
    pendingRef.current = {
      px: (e.clientX - r.left) / r.width - 0.5,      // -0.5 … 0.5
      py: (e.clientY - r.top) / r.height - 0.5,
      w: r.width, h: r.height,
    }
    if (frameRef.current) return
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0
      const p = pendingRef.current
      const node = surfaceRef.current
      if (!p || !node) return
      // Chasing the pointer must be immediate. A transition here would restart
      // itself on every move and the card would lag behind the hand instead of
      // sitting under it.
      node.classList.remove('is-settling')
      node.style.setProperty('--ry', `${(p.px * 2 * TILT_MAX_DEG).toFixed(2)}deg`)
      node.style.setProperty('--rx', `${(-p.py * 2 * TILT_MAX_DEG).toFixed(2)}deg`)
      // The sheen is a fixed gradient that gets moved, never a gradient that
      // gets redrawn at a new centre — one is a composite, the other repaints
      // the whole card every frame.
      node.style.setProperty('--sx', `${(p.px * p.w).toFixed(1)}px`)
      node.style.setProperty('--sy', `${(p.py * p.h).toFixed(1)}px`)
    })
  }, [reducedMotion])

  const rest = useCallback(() => {
    draggingRef.current = false
    if (frameRef.current) { cancelAnimationFrame(frameRef.current); frameRef.current = 0 }
    pendingRef.current = null
    const el = surfaceRef.current
    if (!el) return
    // Letting go is the one moment that should ease rather than snap.
    el.classList.add('is-settling')
    el.style.setProperty('--ry', '0deg')
    el.style.setProperty('--rx', '0deg')
    el.style.setProperty('--sx', '0px')
    el.style.setProperty('--sy', '0px')
  }, [])

  useEffect(() => () => {
    if (frameRef.current) cancelAnimationFrame(frameRef.current)
  }, [])

  const onKeyDown = useCallback((e) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); onStep(-1) }
    else if (e.key === 'ArrowRight') { e.preventDefault(); onStep(1) }
  }, [onStep])

  return (
    <Modal
      onClose={onClose}
      title={card.name}
      scrollable={false}
      backdropClassName="tarot-inspect__backdrop"
      className="tarot-inspect__dialog"
      contentStyle={{ background: 'transparent', border: 0, boxShadow: 'none' }}
    >
      {({ titleId }) => (
        <div className="tarot-inspect" onKeyDown={onKeyDown}>
          <div className="tarot-inspect__stage">
            {/* Three nested transforms, deliberately: the open animation, the
                turn, and the tilt each change on their own schedule, and one
                element carrying all three means every pointer move restarts
                the turn's transition. */}
            <div
              ref={cardRef}
              className={`tarot-inspect__card${flipped ? ' is-flipped' : ''}`}
            >
              <div className="tarot-inspect__turn">
                <div
                  ref={surfaceRef}
                  className="tarot-inspect__surface is-settling"
                  onPointerMove={tilt}
                  onPointerDown={(e) => { draggingRef.current = true; tilt(e) }}
                  onPointerUp={rest}
                  onPointerLeave={rest}
                  onPointerCancel={rest}
                >
                  <div className="tarot-inspect__face tarot-inspect__face--front">
                    <img src={`/tarot/${card.img}`} alt={card.name} draggable="false" />
                    <span className="tarot-inspect__sheen" aria-hidden="true" />
                  </div>
                  <div className="tarot-inspect__face tarot-inspect__face--back" />
                </div>
              </div>
            </div>
          </div>

          <div className="tarot-inspect__panel">
            <p className="tarot-inspect__pos">
              {position.label}<span className="label-zh">{position.label_zh}</span>
            </p>
            <h2 className="tarot-inspect__name" id={titleId}>{card.name}</h2>
            <p className="tarot-inspect__meta">
              {card.arcana === 'major' ? 'Major Arcana 大阿卡纳' : 'Minor Arcana 小阿卡纳'}
              {' · '}Upright 正位
            </p>

            <div className="tarot-inspect__text">
              {card.keywords && <p className="tarot-inspect__keywords">{card.keywords}</p>}
              {card.description && <p className="tarot-inspect__desc">{card.description}</p>}
            </div>

            <div className="tarot-inspect__controls">
              <button
                type="button"
                className="tarot-inspect__btn"
                onClick={() => onStep(-1)}
                disabled={count < 2}
              >
                <i className="fas fa-chevron-left" aria-hidden="true" />
                <span className="visually-hidden">Previous card</span>
              </button>
              <span className="tarot-inspect__count">{index + 1} / {count}</span>
              <button
                type="button"
                className="tarot-inspect__btn"
                onClick={() => onStep(1)}
                disabled={count < 2}
              >
                <i className="fas fa-chevron-right" aria-hidden="true" />
                <span className="visually-hidden">Next card</span>
              </button>

              <button
                type="button"
                className="tarot-inspect__btn tarot-inspect__btn--wide"
                onClick={() => setFlipped(f => !f)}
                aria-pressed={flipped}
              >
                <i className="fas fa-sync-alt" aria-hidden="true" />
                {flipped ? 'Show the face 看正面' : 'Turn it over 翻面'}
              </button>
              <button type="button" className="tarot-inspect__btn tarot-inspect__btn--wide" onClick={onClose}>
                <i className="fas fa-xmark" aria-hidden="true" />
                Close 关闭
              </button>
            </div>
          </div>
        </div>
      )}
    </Modal>
  )
}

export default function Tarot() {
  const [deck, setDeck] = useState([])
  const [positions, setPositions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // ready → shuffling → choosing → revealing → done
  const [phase, setPhase] = useState('ready')
  const [remaining, setRemaining] = useState([])   // cards still face down in the fan
  const [slots, setSlots] = useState([])           // cards that have landed, in pick order
  const [revealedCount, setRevealedCount] = useState(0)
  const [flight, setFlight] = useState(null)       // the card currently in the air
  const [rowCount, setRowCount] = useState(ROWS_WIDE)
  const [deckWidth, setDeckWidth] = useState(0)

  const [focusSeatIndex, setFocusSeatIndex] = useState(0)   // roving tab stop
  const [inspecting, setInspecting] = useState(null)        // { index, fromRect }

  const drawnRef = useRef([])        // the server's three, in order
  const deckElRef = useRef(null)
  const slotRefs = useRef([])
  const flyerRef = useRef(null)
  const timers = useRef([])
  const restoreFocusRef = useRef(false)

  const clearTimers = useCallback(() => {
    timers.current.forEach(clearTimeout)
    timers.current = []
  }, [])
  useEffect(() => () => clearTimers(), [clearTimers])

  const reducedMotion = useMemo(
    () => typeof window !== 'undefined'
      && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
    [])

  // ── Load the deck ───────────────────────────────────────────
  useEffect(() => {
    api.get('/api/tarot/deck').then(d => {
      if (d.ok) {
        setDeck(d.cards || [])
        setRemaining(d.cards || [])
        setPositions(d.positions || [])
      } else {
        setError(d.error || 'Could not load the deck.')
      }
      setLoading(false)
    })
  }, [])

  // ── Row count follows the viewport ──────────────────────────
  useEffect(() => {
    const mq = window.matchMedia(WIDE_QUERY)
    const apply = () => setRowCount(mq.matches ? ROWS_WIDE : ROWS_NARROW)
    apply()
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [])

  /* Seats are computed in pixels, so every position lives in `transform` and
     nothing in the fan touches layout. That needs the measured width. */
  useEffect(() => {
    const el = deckElRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setDeckWidth(entry.contentRect.width))
    ro.observe(el)
    setDeckWidth(el.getBoundingClientRect().width)
    return () => ro.disconnect()
  }, [loading])

  const cardWidth = Math.max(52, Math.min(96, deckWidth / 13))

  /* Seats are derived from each card's index among the cards *still there*, so
     removing one closes the gap: every card behind it slides up a seat, and
     the base transition carries them. */
  const rows = useMemo(
    () => splitRows(remaining, rowCount),
    [remaining, rowCount])

  const seats = useMemo(() => {
    const map = new Map()
    let flat = 0
    rows.forEach(row => {
      row.forEach((card, i) => {
        map.set(card.id, { ...seatStyle(i, row.length, deckWidth, cardWidth), flat: flat++ })
      })
    })
    return map
  }, [rows, deckWidth, cardWidth])

  // ── Shuffle ─────────────────────────────────────────────────
  const shuffle = useCallback(async () => {
    if (phase === 'shuffling' || phase === 'revealing') return
    clearTimers()
    setError('')
    setSlots([])
    setRevealedCount(0)
    setRemaining(deck)
    setFocusSeatIndex(0)
    setPhase('shuffling')

    // Fetched while the riffle plays, so the request has nowhere to show.
    const d = await api.post('/api/tarot/draw')
    if (!d.ok) {
      setError(d.error || 'The cards would not come.')
      setPhase('ready')
      return
    }
    drawnRef.current = d.spread

    const wait = reducedMotion ? 0 : RIFFLE_MS + deck.length * RIFFLE_STAGGER_MS
    timers.current.push(setTimeout(() => setPhase('choosing'), wait))
  }, [phase, deck, reducedMotion, clearTimers])

  // ── Reveal, once all three have landed ──────────────────────
  const startReveal = useCallback(() => {
    setPhase('revealing')
    const gap = reducedMotion ? 0 : REVEAL_GAP_MS
    const lead = reducedMotion ? 0 : SETTLE_MS
    for (let i = 1; i <= POSITION_COUNT; i++) {
      timers.current.push(setTimeout(() => {
        setRevealedCount(i)
        if (i === POSITION_COUNT) setPhase('done')
      }, lead + gap * (i - 1)))
    }
  }, [reducedMotion])

  // ── Picking a card ──────────────────────────────────────────
  const land = useCallback((slotIndex) => {
    setSlots(prev => {
      const next = [...prev]
      next[slotIndex] = drawnRef.current[slotIndex]
      return next
    })
    if (slotIndex === POSITION_COUNT - 1) startReveal()
  }, [startReveal])

  const pick = useCallback((card, el) => {
    // One card in the air at a time, or two fast clicks would both aim at the
    // same slot and the second would overwrite the first.
    if (phase !== 'choosing' || flight) return
    const slotIndex = slots.filter(Boolean).length
    if (slotIndex >= POSITION_COUNT) return

    const target = slotRefs.current[slotIndex]

    /* The card that was taken is about to unmount, and with it the deck's only
       tab stop — leaving a keyboard user on <body> with no way back in. The
       card behind it slides into the vacated seat, so the tab stop stays at
       that seat, and if the pick came from the keyboard the focus follows it. */
    restoreFocusRef.current = el != null && el === document.activeElement
    setFocusSeatIndex(i => Math.max(0, Math.min(i, remaining.length - 2)))
    setRemaining(prev => prev.filter(c => c.id !== card.id))

    if (reducedMotion || !el || !target) {
      land(slotIndex)
      return
    }

    /* FLIP: the card is measured where the hand left it and animated to the
       slot, so the card that lands is visibly the card that was touched.
       Measured by centre, not by corner — getBoundingClientRect on a tilted
       card returns its bounding box, which is wider than the card itself. */
    const from = el.getBoundingClientRect()
    const to = target.getBoundingClientRect()
    const w = el.offsetWidth
    const h = el.offsetHeight
    setFlight({
      slotIndex, w, h,
      rot: parseFloat(getComputedStyle(el).getPropertyValue('--seat-rot')) || 0,
      from: { cx: from.left + from.width / 2, cy: from.top + from.height / 2 },
      to: { cx: to.left + to.width / 2, cy: to.top + to.height / 2, scale: to.width / w },
    })
  }, [phase, flight, slots, reducedMotion, land])

  useLayoutEffect(() => {
    if (!flight || !flyerRef.current) return
    const { from, to, rot, w, h, slotIndex } = flight
    const place = (cx, cy) => `translate(${cx - w / 2}px, ${cy - h / 2}px)`
    const anim = flyerRef.current.animate([
      { transform: `${place(from.cx, from.cy)} rotate(${rot}deg) scale(1)` },
      // A shallow rise between the two, so the card travels over the cloth
      // rather than sliding along it.
      {
        offset: 0.5,
        transform: `${place((from.cx + to.cx) / 2, (from.cy + to.cy) / 2 - 46)}`
          + ` rotate(${rot * 0.35}deg) scale(${(1 + to.scale) / 2 * 1.06})`,
      },
      { transform: `${place(to.cx, to.cy)} rotate(0deg) scale(${to.scale})` },
    ], { duration: FLIGHT_MS, easing: 'cubic-bezier(0.16, 1, 0.3, 1)', fill: 'forwards' })

    let done = false
    const finish = () => {
      if (done) return
      done = true
      land(slotIndex)
      setFlight(null)
    }
    anim.addEventListener('finish', finish)
    // A tab backgrounded mid-flight never fires `finish`; the card must still land.
    const guard = setTimeout(finish, FLIGHT_MS + 400)
    return () => { clearTimeout(guard); anim.cancel() }
  }, [flight, land])

  useEffect(() => {
    if (!restoreFocusRef.current) return
    restoreFocusRef.current = false
    deckElRef.current?.querySelector(`[data-seat="${focusSeatIndex}"]`)?.focus()
  }, [remaining, focusSeatIndex])

  /** Open one of the three face-up cards, out of the slot it is lying in. */
  const inspect = useCallback((i, el) => {
    setInspecting({ index: i, fromRect: el?.getBoundingClientRect() ?? null })
  }, [])

  /** Step between the three without going back to the table. */
  const stepInspect = useCallback((delta) => {
    setInspecting(prev => {
      if (!prev) return prev
      const next = (prev.index + delta + POSITION_COUNT) % POSITION_COUNT
      const el = slotRefs.current[next]
      return { index: next, fromRect: el?.getBoundingClientRect() ?? null }
    })
  }, [])

  // ── Keyboard: one roving stop for the whole deck ────────────
  const focusSeat = useCallback((flatIndex) => {
    const el = deckElRef.current?.querySelector(`[data-seat="${flatIndex}"]`)
    if (el) { setFocusSeatIndex(flatIndex); el.focus() }
  }, [])

  const onDeckKeyDown = useCallback((e) => {
    if (phase !== 'choosing') return
    const total = remaining.length
    if (!total) return
    const perRow = Math.ceil(total / rowCount)
    const cur = focusSeatIndex
    const go = (n) => { e.preventDefault(); focusSeat(Math.max(0, Math.min(total - 1, n))) }

    if (e.key === 'ArrowRight') go(cur + 1)
    else if (e.key === 'ArrowLeft') go(cur - 1)
    else if (e.key === 'ArrowDown') go(cur + perRow)
    else if (e.key === 'ArrowUp') go(cur - perRow)
    else if (e.key === 'Home') go(0)
    else if (e.key === 'End') go(total - 1)
  }, [phase, remaining.length, rowCount, focusSeatIndex, focusSeat])

  /* Aiming at a 20px sliver is a poor ask, and on a touch screen an unfair
     one. A press that misses every card resolves to the nearest one, so the
     gesture is "somewhere along here" rather than "hit this exact strip". */
  const onRowPointerDown = useCallback((e, row) => {
    if (phase !== 'choosing' || e.button > 0) return
    if (e.target.closest('[data-seat]')) return          // a real hit; let it through
    const cards = [...e.currentTarget.querySelectorAll('[data-seat]')]
    if (!cards.length) return
    let best = cards[0], bestD = Infinity
    for (const el of cards) {
      const r = el.getBoundingClientRect()
      const d = Math.abs(r.left + r.width / 2 - e.clientX)
      if (d < bestD) { bestD = d; best = el }
    }
    const card = row.find(c => String(c.id) === best.dataset.cardId)
    if (card) pick(card, best)
  }, [phase, pick])

  /* Neighbours part around the card under the cursor. Six writes of one custom
     property, and the base transform reads it — no layout, no re-render. */
  const partNeighbours = useCallback((rowEl, seat, on) => {
    if (reducedMotion || !rowEl) return
    for (let d = 1; d <= 3; d++) {
      const push = on ? (4 - d) * 7 : 0
      const before = rowEl.querySelector(`[data-seat="${seat - d}"]`)
      const after = rowEl.querySelector(`[data-seat="${seat + d}"]`)
      if (before) before.style.setProperty('--push-x', `${-push}px`)
      if (after) after.style.setProperty('--push-x', `${push}px`)
    }
  }, [reducedMotion])

  if (loading) {
    return (
      <div className="tarot">
        <div className="tarot__inner text-center py-5"><HandLoader /></div>
      </div>
    )
  }

  const chosen = slots.filter(Boolean).length
  const nextPosition = positions[chosen]
  /* The deck leads until the cards are chosen, then it has to get out of the
     way: three cards face up are the page now, and 75 face-down ones are the
     brightest thing on it. So it gathers itself back into a pile and dims —
     which also leaves it where the next shuffle starts. */
  const deckState =
    phase === 'ready' ? 'stacked'
      : phase === 'shuffling' ? 'shuffling'
        : phase === 'choosing' ? 'open'
          : 'gathered'

  const instruction = {
    ready: {
      lead: 'Hold one question in your mind.',
      sub: '心里想着一个问题 — 想清楚，别说出口。',
    },
    shuffling: { lead: 'Shuffling.', sub: '洗牌中 — 让牌自己排好。' },
    choosing: {
      lead: chosen === 0 ? 'Now take three.' : `${POSITION_COUNT - chosen} to go.`,
      sub: nextPosition
        ? `下一张是「${nextPosition.label_zh}」 · ${nextPosition.label}`
        : '',
    },
    revealing: { lead: 'Turning them over.', sub: '翻牌中。' },
    // The cards are worth looking at properly, and nothing else on the page
    // says so — the slot is small enough to read as finished.
    done: { lead: 'Your three cards.', sub: 'Tap one to look closer · 点开任意一张细看' },
  }[phase]

  return (
    <div className="tarot">
      <div className="tarot__inner">
        <h1 className="tarot__title">Tarot</h1>
        <p className="tarot__subtitle">三张牌 — 过去 · 现在 · 未来</p>

        {error && <div className="alert alert-danger" role="alert">{error}</div>}

        {/* Live, because during the choosing phase this line is the only place
            the count of cards left to take is reported. */}
        <div className="tarot__instruction" aria-live="polite">
          <p className="tarot__instruction-lead">{instruction.lead}</p>
          <p className="tarot__instruction-sub">{instruction.sub}</p>
        </div>

        {/* The deck. Face-down cards carry no information, so the whole thing
            is one labelled group rather than 78 announcements. */}
        <div
          ref={deckElRef}
          className={`tarot__deck tarot__deck--${deckState}`}
          style={{
            '--card-w': `${cardWidth.toFixed(1)}px`,
            '--card-h': `${(cardWidth * CARD_RATIO).toFixed(1)}px`,
            '--rows': rowCount,
            // The rows must not separate before the last staggered card has
            // finished arriving, so they wait out the whole stagger tail.
            '--tail': `${deck.length * RIFFLE_STAGGER_MS}ms`,
          }}
          role={phase === 'choosing' ? 'group' : undefined}
          aria-label={phase === 'choosing'
            ? `The shuffled deck, ${remaining.length} cards face down. Arrow keys to move along it, Enter to take a card.`
            : undefined}
          onKeyDown={onDeckKeyDown}
        >
          {rows.map((row, r) => (
            <div
              className="tarot__row"
              key={r}
              style={{ '--r': r }}
              onPointerDown={(e) => onRowPointerDown(e, row)}
            >
              {row.map((card, i) => {
                const seat = seats.get(card.id)
                if (!seat) return null
                return (
                  <button
                    type="button"
                    key={card.id}
                    data-seat={seat.flat}
                    data-card-id={card.id}
                    className="tarot__card"
                    style={{
                      ...seat,
                      '--i': seat.flat,
                      '--jitter': `${jitter(seat.flat, 2.6).toFixed(2)}deg`,
                      '--stack-y': `${(-seat.flat * 0.13).toFixed(2)}px`,
                      '--split-x': `${seat.flat < deck.length / 2 ? -78 : 78}px`,
                    }}
                    tabIndex={phase === 'choosing' && seat.flat === focusSeatIndex ? 0 : -1}
                    disabled={phase !== 'choosing'}
                    aria-label={`Take card ${i + 1} of row ${r + 1}`}
                    onMouseEnter={(e) => partNeighbours(e.currentTarget.parentElement, seat.flat, true)}
                    onMouseLeave={(e) => partNeighbours(e.currentTarget.parentElement, seat.flat, false)}
                    onFocus={() => setFocusSeatIndex(seat.flat)}
                    onClick={(e) => pick(card, e.currentTarget)}
                  />
                )
              })}
            </div>
          ))}
        </div>

        <div className="tarot__spread">
          {positions.map((pos, i) => {
            const card = slots[i]
            const revealed = revealedCount > i
            // Face up, it becomes a control: this is where you go to actually
            // look at the card rather than at a 132px thumbnail of it.
            const Frame = revealed ? 'button' : 'div'
            return (
              <div
                key={pos.key}
                className={[
                  'tarot__slot',
                  card ? 'tarot__slot--dealt' : 'tarot__slot--empty',
                  revealed ? 'tarot__slot--revealed' : '',
                  phase === 'choosing' && i === chosen ? 'tarot__slot--next' : '',
                ].join(' ')}
              >
                <Frame
                  className="tarot__slot-frame"
                  ref={el => { slotRefs.current[i] = el }}
                  {...(revealed ? {
                    type: 'button',
                    'aria-label': `Look closer at ${card.card.name} — ${pos.label}`,
                    onClick: (e) => inspect(i, e.currentTarget),
                  } : {})}
                >
                  {/* Before a card lands the slot is an empty place on the
                      cloth, not a card lying face down. */}
                  {card && (
                    <div className="tarot__flipper">
                      <div className="tarot__face tarot__face--back" />
                      <div className="tarot__face tarot__face--front">
                        <img src={`/tarot/${card.card.img}`} alt={revealed ? card.card.name : ''} draggable="false" />
                      </div>
                    </div>
                  )}
                </Frame>
                <div className="tarot__slot-label">
                  {pos.label}
                  <span className="label-zh">{pos.label_zh}</span>
                </div>
                <div className="tarot__card-name">
                  {revealed && card ? card.card.name : ''}
                </div>
              </div>
            )
          })}
        </div>

        <div className="tarot__actions">
          {phase === 'choosing' ? (
            /* Quiet, because the instruction above is what you should be
               reading — but someone who changed their mind after one card
               should not have to reload the page to start over. */
            <button type="button" className="tarot__btn tarot__btn--quiet" onClick={shuffle}>
              Start over 重来
            </button>
          ) : (
            <button
              type="button"
              className="tarot__btn"
              onClick={shuffle}
              disabled={phase === 'shuffling' || phase === 'revealing'}
            >
              {phase === 'shuffling' ? 'Shuffling… 洗牌中'
                : phase === 'ready' ? 'Shuffle the deck 洗牌'
                  : phase === 'revealing' ? 'Turning… 翻牌中'
                    : 'Ask again 再来一次'}
            </button>
          )}
        </div>

        {/* Announced as one block once the turn is over, so a screen reader
            hears the finished reading rather than three interruptions. */}
        <div className="tarot__reading" role="status" aria-live="polite">
          {phase === 'done' && slots.map(({ position, card }, i) => (
            <div
              className="tarot__entry"
              key={position.key}
              style={{ '--entry': i }}
            >
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

      {inspecting && slots[inspecting.index] && (
        <Inspector
          entry={slots[inspecting.index]}
          index={inspecting.index}
          count={POSITION_COUNT}
          fromRect={inspecting.fromRect}
          reducedMotion={reducedMotion}
          onStep={stepInspect}
          onClose={() => setInspecting(null)}
        />
      )}

      {/* The card in the air. Fixed to the viewport so neither the deck's
          clipping nor the slots' stacking order can cut it off. */}
      {flight && (
        <div
          ref={flyerRef}
          className="tarot__flyer"
          aria-hidden="true"
          style={{
            width: flight.w,
            height: flight.h,
            transform: `translate(${flight.from.cx - flight.w / 2}px, ${flight.from.cy - flight.h / 2}px)`
              + ` rotate(${flight.rot}deg)`,
          }}
        />
      )}
    </div>
  )
}
