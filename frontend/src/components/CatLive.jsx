import React, { useEffect, useRef, useState } from 'react'
import { PALETTES } from './CatArt'

/**
 * The cat that lives inside the pet page.
 *
 *   <CatLive breed="calico" stageRef={stageRef} size={140} />
 *
 * Side-view cat, one body with movable groups (head, ears, tail, four legs).
 * The ten breeds reuse CatArt's palette table — same coat / belly / ear / eye
 * colours, same pattern names, so adding a new breed in one place changes
 * both the still cards and the live cat.
 *
 * A single rAF loop drives a four-state machine (idle, walk, sit, sleep) and
 * writes `transform: translate3d` on the wrapper — no layout work per frame.
 * Legs, tail and breathing are CSS keyframes keyed on the state class. The
 * loop skips while `document.hidden`, stops at idle under
 * `prefers-reduced-motion: reduce`, and clamps the cat inside its stage.
 */

const CAT_W = 160
const CAT_H = 120

// ── Pattern overlays ──────────────────────────────────────────────────────
// Short, side-view versions of the ones CatArt paints over its seated cat.
// Nothing here moves — the body pattern rides with the body group.
function BodyPattern({ p }) {
  const ink = p.ink || {}
  switch (p.pattern) {
    case 'tabby':
      return (
        <g stroke={ink.a} strokeWidth="2.6" fill="none" strokeLinecap="round" opacity="0.9">
          <path d="M60 60 C62 68 62 80 60 88" />
          <path d="M78 58 C80 66 80 82 78 90" />
          <path d="M96 60 C98 68 98 80 96 88" />
        </g>
      )
    case 'shaded':
      return (
        <ellipse cx="80" cy="82" rx="38" ry="9" fill={ink.a} opacity="0.35" />
      )
    case 'calico':
      return (
        <g>
          <path d="M50 70 C56 62 70 62 74 72 C70 84 58 86 50 80 Z" fill={ink.a} />
          <path d="M92 64 C104 60 114 68 110 82 C102 86 92 82 90 74 Z" fill={ink.b} />
        </g>
      )
    case 'tuxedo':
      return (
        <ellipse cx="80" cy="84" rx="30" ry="9" fill={ink.a} />
      )
    case 'ragdoll':
      return (
        <ellipse cx="80" cy="82" rx="36" ry="9" fill={p.belly} />
      )
    default:
      return null
  }
}

function HeadPattern({ p }) {
  const ink = p.ink || {}
  switch (p.pattern) {
    case 'siamese':
      // Dark mask across muzzle.
      return <ellipse cx="135" cy="60" rx="14" ry="10" fill={ink.a} opacity="0.95" />
    case 'ragdoll':
      // Darker cap over the top of the head.
      return <path d="M110 46 C116 36 146 36 150 50 C140 48 120 48 110 54 Z" fill={ink.a} opacity="0.9" />
    case 'tabby':
      return (
        <g stroke={ink.a} strokeWidth="2.4" fill="none" strokeLinecap="round">
          <path d="M122 40 L124 48" />
          <path d="M130 38 L132 46" />
          <path d="M138 40 L140 48" />
        </g>
      )
    case 'calico':
      return <path d="M110 52 C118 44 126 44 132 48 C130 58 118 60 110 56 Z" fill={ink.b} opacity="0.9" />
    default:
      return null
  }
}

// ── The still cat drawing, side view, facing right ────────────────────────
function CatSvg({ p, mode }) {
  const closedEye = mode === 'sleep'
  const squint = mode === 'pounce'
  const dark = p.coat === '#2b2b2b'
  const pawFill = p.pattern === 'tuxedo' ? p.belly : p.pattern === 'siamese' ? (p.ink?.a || p.coat) : p.coat
  const tailFill = p.pattern === 'siamese' || p.pattern === 'ragdoll' ? (p.ink?.a || p.coat) : p.coat
  const noseFill = p.pattern === 'siamese' ? '#2b2220' : '#d98a8a'
  const whiskerInk = dark ? '#cfcfcf' : p.line

  return (
    <svg viewBox="0 0 160 120" width="100%" height="100%" aria-hidden="true" focusable="false"
         style={{ display: 'block', overflow: 'visible' }}>
      {/* tail behind body */}
      <g className="catlive__tail-g">
        <path className="catlive__tail" d="M34 72 Q10 48 20 22"
              stroke={tailFill} strokeWidth="10" fill="none" strokeLinecap="round" />
      </g>
      {/* back legs (behind body) */}
      <g className="catlive__leg catlive__leg--bl">
        <rect x="38" y="88" width="9" height="22" rx="4" fill={pawFill} />
      </g>
      <g className="catlive__leg catlive__leg--br">
        <rect x="52" y="88" width="9" height="22" rx="4" fill={pawFill} />
      </g>
      {/* body group — one group so it can scale as a whole */}
      <g className="catlive__body-g">
        <ellipse className="catlive__body" cx="80" cy="74" rx="52" ry="22" fill={p.coat} />
        {p.belly !== p.coat && (
          <ellipse cx="80" cy="82" rx="42" ry="12" fill={p.belly} />
        )}
        <BodyPattern p={p} />
      </g>
      {/* front legs */}
      <g className="catlive__leg catlive__leg--fl">
        <rect x="102" y="88" width="9" height="22" rx="4" fill={pawFill} />
      </g>
      <g className="catlive__leg catlive__leg--fr">
        <rect x="116" y="88" width="9" height="22" rx="4" fill={pawFill} />
      </g>
      {/* head group — pivots at the neck for sleep tilt */}
      <g className="catlive__head" style={{ transformOrigin: '118px 72px' }}>
        {/* back ear */}
        <path d="M112 36 L108 18 L122 32 Z" fill={p.coat} />
        <path d="M113 34 L111 24 L120 32 Z" fill={p.ear} />
        {/* front ear */}
        <path d="M144 36 L148 18 L134 32 Z" fill={p.coat} />
        <path d="M143 34 L145 24 L136 32 Z" fill={p.ear} />
        <circle cx="130" cy="54" r="22" fill={p.coat} />
        <HeadPattern p={p} />
        {/* eye */}
        {closedEye ? (
          <path d="M138 52 Q142 56 146 52"
                stroke={whiskerInk} strokeWidth="2" fill="none" strokeLinecap="round" />
        ) : squint ? (
          <path d="M138 55 Q142 51 146 55"
                stroke={whiskerInk} strokeWidth="2" fill="none" strokeLinecap="round" />
        ) : (
          <g>
            <ellipse cx="142" cy="53" rx="3.4" ry="4.2" fill={p.eye} />
            <ellipse cx="142" cy="53" rx="1.3" ry="3.4" fill="#1a1a1a" />
            <circle cx="143" cy="51" r="0.9" fill="#ffffff" />
          </g>
        )}
        {/* nose */}
        <path d="M149 60 L153.5 62 L149 64 Z" fill={noseFill} />
        {/* mouth */}
        <path d="M149 65 Q146 68 143 66"
              stroke={whiskerInk} strokeWidth="1.5" fill="none" strokeLinecap="round" />
        {/* whiskers */}
        <g stroke={whiskerInk} strokeWidth="1" opacity="0.75" strokeLinecap="round">
          <line x1="150" y1="62" x2="164" y2="60" />
          <line x1="150" y1="64" x2="164" y2="64" />
          <line x1="150" y1="66" x2="164" y2="68" />
        </g>
      </g>
    </svg>
  )
}

// ── The live wrapper ──────────────────────────────────────────────────────
export default function CatLive({ breed = 'blue', stageRef, size = 140 }) {
  const wrapRef = useRef(null)
  const flipRef = useRef(null)
  const [mode, setMode] = useState('idle')
  const modeRef = useRef('idle')
  // Position in stage pixels (top-left of the cat box).
  const posRef = useRef({ x: 0, y: 0, tx: 0, ty: 0, dir: 1 })
  const timeRef = useRef({ last: 0, lastInteract: 0, nextDecision: 0 })
  const reducedRef = useRef(false)
  const catW = size
  const catH = Math.round(size * (CAT_H / CAT_W))

  const setModeBoth = (m) => { modeRef.current = m; setMode(m) }

  const applyTransform = () => {
    const p = posRef.current
    if (wrapRef.current) {
      wrapRef.current.style.transform = `translate3d(${p.x}px, ${p.y}px, 0)`
    }
    if (flipRef.current) {
      flipRef.current.style.transform = `scaleX(${p.dir})`
    }
  }

  const stageBox = () => {
    const el = stageRef.current
    if (!el) return { w: 400, h: 320 }
    const r = el.getBoundingClientRect()
    return { w: r.width, h: r.height }
  }

  const clampTarget = (x, y) => {
    const { w, h } = stageBox()
    const m = 8
    return {
      x: Math.max(m, Math.min(w - catW - m, x)),
      y: Math.max(m, Math.min(h - catH - m, y)),
    }
  }

  const pickRoam = () => {
    const { w, h } = stageBox()
    const m = 12
    const nx = m + Math.random() * Math.max(10, w - catW - m * 2)
    const ny = m + Math.random() * Math.max(10, h - catH - m * 2)
    const c = clampTarget(nx, ny)
    const p = posRef.current
    p.tx = c.x
    p.ty = c.y
    p.dir = p.tx > p.x ? 1 : -1
  }

  // Init position and wire up loops.
  useEffect(() => {
    reducedRef.current = typeof window !== 'undefined'
      && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

    // Start near the centre.
    const { w, h } = stageBox()
    const p = posRef.current
    p.x = Math.max(8, w / 2 - catW / 2)
    p.y = Math.max(8, h / 2 - catH / 2)
    p.tx = p.x
    p.ty = p.y
    applyTransform()

    const now = performance.now()
    timeRef.current.last = now
    timeRef.current.lastInteract = now
    timeRef.current.nextDecision = now + 1500

    let rafId = 0
    const tick = (ts) => {
      const t = timeRef.current
      const dt = Math.min(60, ts - t.last)
      t.last = ts
      if (!document.hidden) {
        const curMode = modeRef.current

        if (curMode === 'walk' && !reducedRef.current) {
          const pp = posRef.current
          const dx = pp.tx - pp.x
          const dy = pp.ty - pp.y
          const dist = Math.hypot(dx, dy)
          const speed = 55 // px/s
          const step = speed * dt / 1000
          if (dist < step + 0.5) {
            pp.x = pp.tx; pp.y = pp.ty
            setModeBoth(Math.random() < 0.4 ? 'sit' : 'idle')
            t.nextDecision = ts + 2500 + Math.random() * 4500
          } else {
            pp.x += dx / dist * step
            pp.y += dy / dist * step
            pp.dir = dx >= 0 ? 1 : -1
          }
          applyTransform()
        }

        if (ts >= t.nextDecision) {
          const sinceTouch = ts - t.lastInteract
          if (reducedRef.current) {
            setModeBoth('idle')
            t.nextDecision = ts + 60000
          } else if (sinceTouch > 60000 && curMode !== 'sleep') {
            setModeBoth('sleep')
            t.nextDecision = ts + 8000 + Math.random() * 12000
          } else if (curMode === 'sleep') {
            // wake up eventually
            setModeBoth('idle')
            t.lastInteract = ts // reset timer so sleep is not immediate
            t.nextDecision = ts + 2500 + Math.random() * 3500
          } else {
            const r = Math.random()
            if (r < 0.55) {
              pickRoam()
              setModeBoth('walk')
              t.nextDecision = ts + 18000 // safety: give up and re-decide
            } else if (r < 0.8) {
              setModeBoth('sit')
              t.nextDecision = ts + 3000 + Math.random() * 4000
            } else {
              setModeBoth('idle')
              t.nextDecision = ts + 3000 + Math.random() * 3500
            }
            applyTransform()
          }
        }
      }
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)

    const onVis = () => {
      // When the tab comes back, reset dt so we don't jump.
      if (!document.hidden) timeRef.current.last = performance.now()
    }
    document.addEventListener('visibilitychange', onVis)

    // ResizeObserver: if the stage resizes, keep the cat inside.
    let ro = null
    if (stageRef.current && 'ResizeObserver' in window) {
      ro = new ResizeObserver(() => {
        const c = clampTarget(posRef.current.x, posRef.current.y)
        posRef.current.x = c.x; posRef.current.y = c.y
        posRef.current.tx = c.x; posRef.current.ty = c.y
        applyTransform()
      })
      ro.observe(stageRef.current)
    }

    return () => {
      cancelAnimationFrame(rafId)
      document.removeEventListener('visibilitychange', onVis)
      ro?.disconnect()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Stage click: walk toward the click point (cat clicks are stopped below).
  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return
    const onStageClick = (e) => {
      if (reducedRef.current) return
      if (e.target.closest('[data-cat-live]')) return
      const r = stage.getBoundingClientRect()
      const x = e.clientX - r.left - catW / 2
      const y = e.clientY - r.top - catH / 2
      const c = clampTarget(x, y)
      const p = posRef.current
      p.tx = c.x; p.ty = c.y
      p.dir = p.tx > p.x ? 1 : -1
      timeRef.current.lastInteract = performance.now()
      timeRef.current.nextDecision = performance.now() + 18000
      setModeBoth('walk')
      applyTransform()
    }
    stage.addEventListener('click', onStageClick)
    return () => stage.removeEventListener('click', onStageClick)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stageRef])

  const onCatPointer = (e) => {
    e.stopPropagation()
    if (reducedRef.current) return
    timeRef.current.lastInteract = performance.now()
    setModeBoth('pounce')
    // Clear to idle after a beat; the decision timer handles what comes next.
    window.setTimeout(() => {
      if (modeRef.current === 'pounce') {
        setModeBoth('idle')
        timeRef.current.nextDecision = performance.now() + 1800
      }
    }, 360)
  }

  const p = PALETTES[breed] || PALETTES.blue

  return (
    <div
      ref={wrapRef}
      data-cat-live="1"
      className={`catlive catlive--${mode}`}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: catW,
        height: catH,
        willChange: 'transform',
        cursor: 'pointer',
        touchAction: 'manipulation',
      }}
      onClick={onCatPointer}
    >
      <div
        ref={flipRef}
        className="catlive__flip"
        style={{
          width: '100%',
          height: '100%',
          transformOrigin: '50% 50%',
          transition: 'transform .25s ease',
        }}
      >
        <CatSvg p={p} mode={mode} />
      </div>
    </div>
  )
}
