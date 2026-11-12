import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import HandLoader from '../components/HandLoader'
import Cat from '../components/CatArt'
import CatLive from '../components/CatLive'

/**
 * /pet — the virtual cat.
 *
 * Two states: no cat yet (pick one of ten breeds, give it a name, adopt) and
 * a cat (the big drawing, three meters, three chores). The breed is fixed at
 * adoption; the name can change from the little pencil beside it. Every
 * number on this page comes from the server — the decay is computed there
 * on read — so a reload is always the truth.
 */

// Mirrors Backend/Controller/pet_db.py :: BREEDS. The server validates.
const BREEDS = [
  { slug: 'blue',    zh: '蓝猫',   en: 'British Blue' },
  { slug: 'calico',  zh: '三花',   en: 'Calico' },
  { slug: 'golden',  zh: '金渐层', en: 'Golden Shaded' },
  { slug: 'silver',  zh: '银渐层', en: 'Silver Shaded' },
  { slug: 'orange',  zh: '橘猫',   en: 'Orange Tabby' },
  { slug: 'tuxedo',  zh: '奶牛猫', en: 'Tuxedo' },
  { slug: 'black',   zh: '黑猫',   en: 'Black Cat' },
  { slug: 'tabby',   zh: '狸花',   en: 'Li Hua' },
  { slug: 'ragdoll', zh: '布偶',   en: 'Ragdoll' },
  { slug: 'siamese', zh: '暹罗',   en: 'Siamese' },
]
const MAX_NAME = 12

const METERS = [
  { key: 'hunger', en: 'Full',  zh: '饱食' },
  { key: 'mood',   en: 'Mood',  zh: '心情' },
  { key: 'bond',   en: 'Bond',  zh: '亲密' },
]
const ACTIONS = [
  { key: 'feed',  en: 'Feed',  zh: '喂食', icon: 'fa-bowl-food' },
  { key: 'play',  en: 'Play',  zh: '陪玩', icon: 'fa-feather' },
  { key: 'brush', en: 'Brush', zh: '梳毛', icon: 'fa-brush' },
]

export function stageLabel(pet) {
  if (!pet) return ''
  return `${pet.stage_en} · ${pet.stage_zh}`
}

// ── Adoption ───────────────────────────────────────────────────────────────
function Adopt({ onAdopted }) {
  const [breed, setBreed] = useState(null)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const nameRef = useRef(null)

  const pick = (slug) => { setBreed(slug); setErr(''); setTimeout(() => nameRef.current?.focus(), 0) }

  const submit = async (e) => {
    e.preventDefault()
    setErr('')
    if (!breed) { setErr('Pick a breed first.'); return }
    const n = name.trim()
    if (!n) { setErr('Give it a name.'); nameRef.current?.focus(); return }
    setBusy(true)
    const d = await api.post('/api/pet/adopt', { breed, name: n })
    setBusy(false)
    if (!d.ok) { setErr(d.error || 'Could not adopt.'); return }
    onAdopted(d.pet)
  }

  return (
    <section className="pet-adopt" aria-labelledby="pet-adopt-title">
      <h2 id="pet-adopt-title" className="pet-h">Adopt a cat <span className="pet-h__zh">领养一只猫</span></h2>
      <p className="pet-lead">One cat each. Pick the breed now; it stays. The name you can change later.</p>

      <div className="pet-breeds" role="radiogroup" aria-label="Breed">
        {BREEDS.map(b => (
          <button
            key={b.slug}
            type="button"
            role="radio"
            aria-checked={breed === b.slug}
            className={'pet-breed' + (breed === b.slug ? ' is-picked' : '')}
            onClick={() => pick(b.slug)}
          >
            <span className="pet-breed__art"><Cat breed={b.slug} stage="kitten" size={88} title={b.en} /></span>
            <span className="pet-breed__en">{b.en}</span>
            <span className="pet-breed__zh">{b.zh}</span>
          </button>
        ))}
      </div>

      <form className="pet-adopt__form" onSubmit={submit}>
        <label className="pet-adopt__label" htmlFor="pet-name">Name</label>
        <div className="pet-adopt__row">
          <input
            id="pet-name"
            ref={nameRef}
            className="form-control pet-adopt__input"
            value={name}
            maxLength={MAX_NAME}
            placeholder="起个名字"
            autoComplete="off"
            onChange={e => { setName(e.target.value); setErr('') }}
          />
          <button type="submit" className="btn btn-primary pet-adopt__btn" disabled={busy}>
            {busy ? <span className="spinner-border spinner-border-sm" /> : 'Adopt'}
          </button>
        </div>
        <div className="pet-adopt__meta">
          <span className="pet-adopt__count">{name.trim().length}/{MAX_NAME}</span>
          {err && <span className="pet-error" role="alert">{err}</span>}
        </div>
      </form>
    </section>
  )
}

// ── The cat ────────────────────────────────────────────────────────────────
function Meter({ label, zh, value }) {
  const v = Math.max(0, Math.min(100, Number(value) || 0))
  return (
    <div className="pet-meter">
      <div className="pet-meter__row">
        <span className="pet-meter__label">{label} <span className="pet-meter__zh">{zh}</span></span>
        <span className="pet-meter__num tnum">{v}</span>
      </div>
      <div className="pet-meter__bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={v} aria-label={label}>
        <div className={'pet-meter__fill' + (v < 30 ? ' is-low' : '')} style={{ width: `${v}%` }} />
      </div>
    </div>
  )
}

function Home({ pet, setPet }) {
  const [busy, setBusy] = useState(null)       // action key while saving
  const [toast, setToast] = useState(null)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(pet.name)
  const [nameErr, setNameErr] = useState('')
  const [stageSize, setStageSize] = useState(140)
  const editRef = useRef(null)
  const stageRef = useRef(null)

  useEffect(() => {
    const pick = () => setStageSize(window.innerWidth < 576 ? 118 : 150)
    pick()
    window.addEventListener('resize', pick)
    return () => window.removeEventListener('resize', pick)
  }, [])

  const flash = (text) => { setToast(text); setTimeout(() => setToast(null), 2200) }

  const care = async (action) => {
    if (busy || pet.today?.[action]) return
    setBusy(action)
    const d = await api.post('/api/pet/care', { action })
    setBusy(null)
    if (!d.ok) { flash(d.error || 'Could not do that.'); return }
    setPet(d.pet)
    if (d.applied) flash(d.xp_added ? `+${d.xp_delta} 经验` : 'Done')
  }

  const startEdit = () => { setDraft(pet.name); setNameErr(''); setEditing(true); setTimeout(() => editRef.current?.focus(), 0) }
  const saveName = async (e) => {
    e.preventDefault()
    const n = draft.trim()
    if (!n) { setNameErr('Name cannot be empty.'); return }
    if (n === pet.name) { setEditing(false); return }
    const d = await api.patch('/api/pet/me', { name: n })
    if (!d.ok) { setNameErr(d.error || 'Could not rename.'); return }
    setPet(d.pet)
    setEditing(false)
  }

  const allDone = ACTIONS.every(a => pet.today?.[a.key])

  return (
    <section className="pet-home" aria-labelledby="pet-name-title">
      <div className="pet-stage" data-stage={pet.stage} ref={stageRef} aria-hidden="true">
        <CatLive breed={pet.breed} stageRef={stageRef} size={stageSize} />
      </div>

      <div className="pet-title">
        {editing ? (
          <form className="pet-title__edit" onSubmit={saveName}>
            <input
              ref={editRef}
              className="form-control pet-title__input"
              value={draft}
              maxLength={MAX_NAME}
              aria-label="Cat name"
              onChange={e => { setDraft(e.target.value); setNameErr('') }}
              onKeyDown={e => { if (e.key === 'Escape') setEditing(false) }}
            />
            <button type="submit" className="btn btn-primary btn-sm">Save</button>
            <button type="button" className="btn btn-outline-secondary btn-sm" onClick={() => setEditing(false)}>Cancel</button>
          </form>
        ) : (
          <h2 id="pet-name-title" className="pet-title__name">
            {pet.name}
            <button type="button" className="pet-title__pencil" onClick={startEdit} aria-label="Rename" title="Rename">
              <i className="fas fa-pen" aria-hidden="true" />
            </button>
          </h2>
        )}
        {nameErr && <div className="pet-error" role="alert">{nameErr}</div>}
        <div className="pet-title__meta">
          <span className="pet-pill">{stageLabel(pet)}</span>
          <span className="pet-title__breed">{pet.breed_en} · {pet.breed_zh}</span>
        </div>
        <p className="pet-line">{pet.line}</p>
      </div>

      <div className="pet-meters">
        {METERS.map(m => <Meter key={m.key} label={m.en} zh={m.zh} value={pet[m.key]} />)}
      </div>

      <div className="pet-actions">
        {ACTIONS.map(a => {
          const done = !!pet.today?.[a.key]
          return (
            <button
              key={a.key}
              type="button"
              className={'pet-action' + (done ? ' is-done' : '')}
              disabled={done || busy === a.key}
              aria-disabled={done}
              onClick={() => care(a.key)}
            >
              <i className={'fas ' + a.icon} aria-hidden="true" />
              <span className="pet-action__en">{a.en}</span>
              <span className="pet-action__zh">{done ? '今天已完成' : a.zh}</span>
            </button>
          )
        })}
      </div>
      {allDone && <p className="pet-done">All three done today. Come back tomorrow.</p>}

      {toast && <div className="pet-toast" role="status" aria-live="polite">{toast}</div>}
    </section>
  )
}

export default function Pet() {
  const [pet, setPet] = useState(null)
  const [state, setState] = useState('loading')   // loading | ok | error

  useEffect(() => {
    let cancelled = false
    api.get('/api/pet/me').then(d => {
      if (cancelled) return
      if (!d.ok) { setState('error'); return }
      setPet(d.pet)
      setState('ok')
    })
    return () => { cancelled = true }
  }, [])

  if (state === 'loading') return <HandLoader fullPage />

  return (
    <div className="pet-page">
      {state === 'error' && <div className="alert alert-danger py-2">Could not load your cat. Try again later.</div>}
      {state === 'ok' && (pet ? <Home pet={pet} setPet={setPet} /> : <Adopt onAdopted={setPet} />)}
    </div>
  )
}
