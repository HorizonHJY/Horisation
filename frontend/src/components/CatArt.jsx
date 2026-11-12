import React from 'react'

/**
 * The ten cats, as one inline SVG.
 *
 *   <Cat breed="calico" stage="adult" size={200} />
 *
 * One body is drawn once (ears, head, body, paws, face); each breed is a
 * palette plus an optional pattern layer painted over it (stripes, patches,
 * a tuxedo bib, Siamese points, a Ragdoll cap). Flat fills only, no
 * gradients, readable on cream. The stage is not a second drawing: a kitten
 * is the same cat scaled down with a bigger head and the tail straight up,
 * an old friend sits a little larger with the tail wrapped round the paws
 * and the eyes half closed. Keep it that way — three drawings per breed
 * would be thirty files to keep in step.
 */

// ── Palettes ─────────────────────────────────────────────────────────────
// coat: main fill · belly: chest when lighter · ear: inner ear · eye: iris ·
// line: whisker/mouth ink · pattern: which overlay to paint · ink: its colours.
export const PALETTES = {
  blue:    { coat: '#8f98a8', belly: '#a6aebc', ear: '#d9b3b3', eye: '#d79a3a', line: '#4a505c' },
  calico:  { coat: '#f4eee3', belly: '#f4eee3', ear: '#e9b9b4', eye: '#5c9a64', line: '#6b625a', pattern: 'calico', ink: { a: '#e2934a', b: '#2b2b2b' } },
  golden:  { coat: '#dfb16a', belly: '#f2dcb0', ear: '#e8b4a4', eye: '#5c9a64', line: '#7a5a2a', pattern: 'shaded', ink: { a: '#b9843a' } },
  silver:  { coat: '#d8dce2', belly: '#f1f3f5', ear: '#e6b9b9', eye: '#5c9a64', line: '#6f7781', pattern: 'shaded', ink: { a: '#8e99a6' } },
  orange:  { coat: '#e39a4a', belly: '#f4d7a9', ear: '#e9b4a0', eye: '#d79a3a', line: '#7a4a1c', pattern: 'tabby',  ink: { a: '#b8641e' } },
  tuxedo:  { coat: '#2b2b2b', belly: '#f6f3ec', ear: '#8a6a6a', eye: '#d7c03a', line: '#8a8a8a', pattern: 'tuxedo', ink: { a: '#f6f3ec' } },
  black:   { coat: '#2b2b2b', belly: '#363636', ear: '#6e5555', eye: '#c9c93a', line: '#9a9a9a' },
  tabby:   { coat: '#a38a67', belly: '#d9c9a7', ear: '#e3b3a3', eye: '#9ab63a', line: '#4a3b2a', pattern: 'tabby',  ink: { a: '#4d3d2c' } },
  ragdoll: { coat: '#efe6d8', belly: '#f8f3ea', ear: '#8c7a6d', eye: '#5f93c9', line: '#6b5f55', pattern: 'ragdoll', ink: { a: '#8c7a6d' } },
  siamese: { coat: '#efe2cd', belly: '#efe2cd', ear: '#4a3a33', eye: '#5f93c9', line: '#4a3a33', pattern: 'siamese', ink: { a: '#4a3a33' } },
}

export const BREED_LIST = Object.keys(PALETTES)

// ── Geometry (adult, 200×200) ────────────────────────────────────────────
const BODY  = 'M62 152 C62 112 78 102 100 102 C122 102 138 112 138 152 C138 174 124 180 100 180 C76 180 62 174 62 152 Z'
const CHEST = 'M100 118 C84 118 76 134 78 154 C80 170 90 176 100 176 C110 176 120 170 122 154 C124 134 116 118 100 118 Z'
const HEAD  = 'M100 44 C76 44 62 62 62 84 C62 108 82 118 100 118 C118 118 138 108 138 84 C138 62 124 44 100 44 Z'
const EAR_L = 'M66 70 L70 26 L98 50 Z'
const EAR_R = 'M134 70 L130 26 L102 50 Z'
const EAR_L_IN = 'M72 66 L74 36 L92 52 Z'
const EAR_R_IN = 'M128 66 L126 36 L108 52 Z'

// Tails, one per stage. Behind the body (kitten up, adult out to the side)
// or in front of the paws (elder, wrapped round).
const TAILS = {
  kitten: { d: 'M128 150 C160 140 160 108 150 90', front: false },
  adult:  { d: 'M134 160 C176 164 180 130 168 118', front: false },
  elder:  { d: 'M60 170 C84 192 128 190 150 176 C166 166 160 150 152 146', front: true },
}

const STROKE = { fill: 'none', strokeLinecap: 'round', strokeLinejoin: 'round' }

/* The overlay for one breed, split into what sits on the body and what sits
   on the head, because the head is its own transformed group. */
function BodyPattern({ p }) {
  const ink = p.ink || {}
  switch (p.pattern) {
    case 'tabby':
      return (
        <g {...STROKE} stroke={ink.a} strokeWidth="3.4">
          <path d="M70 128 C76 132 78 140 74 148" />
          <path d="M70 150 C76 154 78 162 74 170" />
          <path d="M130 128 C124 132 122 140 126 148" />
          <path d="M130 150 C124 154 122 162 126 170" />
        </g>
      )
    case 'shaded':
      return (
        <g {...STROKE} stroke={ink.a} strokeWidth="3" opacity="0.85">
          <path d="M68 120 C66 134 66 150 70 166" />
          <path d="M132 120 C134 134 134 150 130 166" />
          <path d="M78 110 C76 120 76 130 78 140" />
          <path d="M122 110 C124 120 124 130 122 140" />
        </g>
      )
    case 'calico':
      return (
        <g>
          <path d="M64 140 C66 128 76 124 84 130 C90 138 84 154 72 156 C66 154 63 148 64 140 Z" fill={ink.a} />
          <path d="M112 112 C126 110 136 122 136 142 C136 152 128 158 118 154 C108 148 104 128 112 112 Z" fill={ink.b} />
        </g>
      )
    default:
      return null
  }
}

function HeadPattern({ p }) {
  const ink = p.ink || {}
  switch (p.pattern) {
    case 'tabby':
      return (
        <g {...STROKE} stroke={ink.a} strokeWidth="3.4">
          <path d="M86 62 L91 52 L96 62 L100 52 L104 62 L109 52 L114 62" />
          <path d="M70 84 C66 88 66 94 68 98" />
          <path d="M130 84 C134 88 134 94 132 98" />
        </g>
      )
    case 'shaded':
      return <path d="M86 56 C92 52 108 52 114 56" {...STROKE} stroke={ink.a} strokeWidth="3" opacity="0.85" />
    case 'calico':
      return (
        <g>
          {/* orange over the left ear and eye; black on the right ear */}
          <path d="M66 70 L70 26 L98 50 L96 62 C84 62 72 70 66 70 Z" fill={ink.a} />
          <path d="M66 72 C72 66 84 62 94 64 C92 76 80 86 66 84 Z" fill={ink.a} />
          <path d="M134 70 L130 26 L102 50 L106 62 C118 62 128 70 134 70 Z" fill={ink.b} />
          <path d={EAR_L_IN} fill={p.ear} opacity="0.7" /><path d={EAR_R_IN} fill={p.ear} opacity="0.5" />
        </g>
      )
    case 'tuxedo':
      return (
        <g fill={ink.a}>
          <path d="M100 86 C90 86 84 92 84 100 C84 110 92 116 100 116 C108 116 116 110 116 100 C116 92 110 86 100 86 Z" />
          <path d="M96 58 L100 46 L104 58 L102 82 L98 82 Z" />
        </g>
      )
    case 'siamese':
      return (
        <g fill={ink.a}>
          <path d="M100 66 C84 66 74 78 74 92 C74 108 86 116 100 116 C114 116 126 108 126 92 C126 78 116 66 100 66 Z" />
          <path d={EAR_L} /><path d={EAR_R} />
          <path d={EAR_L_IN} fill="#6a5248" /><path d={EAR_R_IN} fill="#6a5248" />
        </g>
      )
    case 'ragdoll':
      return (
        <g fill={ink.a}>
          <path d={EAR_L} /><path d={EAR_R} />
          <path d="M62 84 C62 62 76 44 100 44 C124 44 138 62 138 84 L134 86 C128 72 116 64 100 64 C84 64 72 72 66 86 Z" />
          <path d="M66 86 C72 72 84 64 100 64 L92 80 C84 82 76 86 70 92 Z" />
          <path d="M134 86 C128 72 116 64 100 64 L108 80 C116 82 124 86 130 92 Z" />
          <path d={EAR_L_IN} fill="#b39d8e" /><path d={EAR_R_IN} fill="#b39d8e" />
        </g>
      )
    default:
      return null
  }
}

function Face({ p, stage }) {
  const elder = stage === 'elder'
  const noseFill = p.pattern === 'siamese' ? '#2b2220' : '#d98a8a'
  const ink = p.pattern === 'siamese' ? '#d9cbb8' : p.line
  return (
    <g>
      {elder ? (
        <g {...STROKE} stroke={ink} strokeWidth="3.2">
          <path d="M80 86 C84 80 92 80 96 86" />
          <path d="M104 86 C108 80 116 80 120 86" />
        </g>
      ) : (
        <g>
          <ellipse cx="88" cy="84" rx="6.5" ry="7.5" fill={p.eye} />
          <ellipse cx="112" cy="84" rx="6.5" ry="7.5" fill={p.eye} />
          <ellipse cx="88" cy="84" rx="2.4" ry="6.4" fill="#1a1a1a" />
          <ellipse cx="112" cy="84" rx="2.4" ry="6.4" fill="#1a1a1a" />
          <circle cx="90" cy="81" r="1.5" fill="#ffffff" />
          <circle cx="114" cy="81" r="1.5" fill="#ffffff" />
        </g>
      )}
      <path d="M96 96 L104 96 L100 101 Z" fill={noseFill} />
      <path d="M100 101 L100 105 M100 105 C97 108 94 107 92 104 M100 105 C103 108 106 107 108 104"
            {...STROKE} stroke={ink} strokeWidth="2" />
      <g {...STROKE} stroke={ink} strokeWidth="1.6" opacity="0.9">
        <path d="M82 98 L58 94" /><path d="M82 102 L58 104" />
        <path d="M118 98 L142 94" /><path d="M118 102 L142 104" />
      </g>
    </g>
  )
}

export default function Cat({ breed = 'blue', stage = 'adult', size = 200, className, title }) {
  const p = PALETTES[breed] || PALETTES.blue
  const st = TAILS[stage] ? stage : 'adult'
  const tail = TAILS[st]
  const dark = breed === 'black' || breed === 'tuxedo'
  const pointInk = p.pattern === 'siamese' || p.pattern === 'ragdoll' ? p.ink.a : null
  const pawFill  = p.pattern === 'tuxedo' ? p.belly : p.pattern === 'siamese' ? pointInk : p.coat
  const tailFill = pointInk || p.coat

  // One scale for the whole cat; a kitten's head sits a little large on its
  // body. Both pivot on the baseline so the paws stay on the ground.
  const scale = st === 'kitten' ? 0.8 : st === 'elder' ? 1.04 : 1
  const headScale = st === 'kitten' ? 1.12 : 1

  const tailEl = <path d={tail.d} fill="none" stroke={tailFill} strokeWidth="14" strokeLinecap="round" />

  return (
    <svg
      className={className}
      width={size} height={size}
      viewBox="0 0 200 200"
      role="img" aria-label={title || `${breed} cat, ${st}`}
      style={{ display: 'block', overflow: 'visible' }}
    >
      <g transform={`translate(100 184) scale(${scale}) translate(-100 -184)`}>
        {!tail.front && tailEl}
        <path d={BODY} fill={p.coat} />
        {p.belly !== p.coat && <path d={CHEST} fill={p.belly} />}
        <BodyPattern p={p} />
        <ellipse cx="84" cy="176" rx="14" ry="7" fill={pawFill} />
        <ellipse cx="116" cy="176" rx="14" ry="7" fill={pawFill} />
        <path d="M80 178 L80 172 M88 178 L88 172 M112 178 L112 172 M120 178 L120 172"
              fill="none" stroke={dark ? '#5a5a5a' : p.line} strokeWidth="1.4" opacity="0.55" />
        {tail.front && tailEl}
        <g transform={`translate(100 100) scale(${headScale}) translate(-100 -100)`}>
          <path d={EAR_L} fill={p.coat} /><path d={EAR_R} fill={p.coat} />
          <path d={EAR_L_IN} fill={p.ear} /><path d={EAR_R_IN} fill={p.ear} />
          <path d={HEAD} fill={p.coat} />
          <HeadPattern p={p} />
          <Face p={p} stage={st} />
        </g>
      </g>
    </svg>
  )
}
