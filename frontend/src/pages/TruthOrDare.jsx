import React, { useState, useEffect, useRef } from 'react'
import { api } from '../api'
import HandLoader from '../components/HandLoader'

/* Truth or Dare — pick a level, get two face-down cards, flip left for a
   truth or right for a dare. A flip records the draw server-side; only a
   drawn prompt can be rated, and the first rating pays +2 XP (capped 5/day
   by xp_db). */

const LEVELS = [
  { key: 'ice',    label: '破冰', hint: '第一次见面也能答' },
  { key: 'normal', label: '正常', hint: '朋友之间的尺度' },
  { key: 'adult',  label: '18+',  hint: '暧昧、成人话题' },
  { key: 'random', label: '随机',  hint: '每张牌单独抽层' },
]

const TAGS = ['太普通', '有点尴尬', '很好玩']

function pickOne(pool, used, kind) {
  const free = pool.filter(p => p.kind === kind && !used.has(p.id))
  const list = free.length ? free : pool.filter(p => p.kind === kind)
  if (!list.length) return null
  return list[Math.floor(Math.random() * list.length)]
}

export default function TruthOrDare() {
  const [level, setLevel]       = useState(null)
  const [pool, setPool]         = useState([])
  const [myRatings, setMyRatings] = useState({})
  const [loading, setLoading]   = useState(false)
  const [cards, setCards]       = useState({ truth: null, dare: null })
  const [flipped, setFlipped]   = useState({ truth: false, dare: false })
  const [used, setUsed]         = useState(() => new Set())
  const [ratingFor, setRatingFor] = useState(null)
  const [stars, setStars]       = useState(0)
  const [tags, setTags]         = useState([])
  const [toast, setToast]       = useState(null)
  const [left, setLeft]         = useState(null)
  const tickRef = useRef(null)

  useEffect(() => () => clearInterval(tickRef.current), [])
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 2600)
    return () => clearTimeout(t)
  }, [toast])

  const deal = (src) => {
    const u = new Set()
    const t = pickOne(src, u, 'truth'); if (t) u.add(t.id)
    const d = pickOne(src, u, 'dare');  if (d) u.add(d.id)
    setCards({ truth: t, dare: d })
    setFlipped({ truth: false, dare: false })
    setUsed(u)
    setRatingFor(null); setStars(0); setTags([]); setLeft(null)
    clearInterval(tickRef.current)
  }

  const chooseLevel = async (key) => {
    setLevel(key); setLoading(true)
    const q = key === 'random' ? 'random' : key
    const d = await api.get(`/api/tod/prompts?level=${q}`)
    setLoading(false)
    if (!d.ok) { setToast(d.error || '加载失败'); return }
    setPool(d.prompts || [])
    setMyRatings(d.my_ratings || {})
    deal(d.prompts || [])
  }

  const startTimer = (prompt) => {
    clearInterval(tickRef.current)
    if (!prompt?.timer) { setLeft(null); return }
    setLeft(prompt.timer)
    tickRef.current = setInterval(() => {
      setLeft(s => {
        if (s === null) return null
        if (s <= 1) { clearInterval(tickRef.current); return 0 }
        return s - 1
      })
    }, 1000)
  }

  const flip = async (kind) => {
    if (flipped[kind] || !cards[kind]) return
    const prompt = cards[kind]
    setFlipped(f => ({ ...f, [kind]: true }))
    if (kind === 'dare') startTimer(prompt)
    // Record the draw so this card may be rated.
    api.post('/api/tod/draw', { prompt_id: prompt.id })
  }

  const swap = (kind) => {
    const next = pickOne(pool, used, kind)
    if (!next) { setToast('这一档的题都用过一轮了'); return }
    setCards(c => ({ ...c, [kind]: next }))
    setFlipped(f => ({ ...f, [kind]: false }))
    setUsed(u => new Set([...u, next.id]))
    setRatingFor(null); setStars(0); setTags([])
    if (kind === 'dare') { setLeft(null); clearInterval(tickRef.current) }
  }

  const openRating = (kind) => {
    setRatingFor(kind)
    setStars(myRatings[cards[kind].id] || 0)
    setTags([])
  }

  const submitRating = async () => {
    if (!ratingFor || !stars) return
    const prompt = cards[ratingFor]
    const d = await api.post('/api/tod/rate', { prompt_id: prompt.id, rating: stars, tags })
    if (!d.ok) { setToast(d.error || '评分失败'); return }
    setMyRatings(m => ({ ...m, [prompt.id]: stars }))
    setRatingFor(null)
    setToast(d.xp_awarded ? '已记录 · +2 经验' : '已更新评分')
  }

  if (!level) {
    return (
      <div className="tod-page">
        <header className="tod-head">
          <div className="tod-kicker">TRUTH OR DARE</div>
          <h1 className="tod-title">真心话 · 大冒险</h1>
          <p className="tod-sub">先选一档尺度。左边真心话，右边大冒险，翻牌才知道是什么。</p>
        </header>
        <div className="tod-levels">
          {LEVELS.map(l => (
            <button key={l.key} className="tod-level" onClick={() => chooseLevel(l.key)}>
              <span className="tod-level__name">{l.label}</span>
              <span className="tod-level__hint">{l.hint}</span>
            </button>
          ))}
        </div>
      </div>
    )
  }

  if (loading) return <HandLoader fullPage />

  return (
    <div className="tod-page">
      <header className="tod-head">
        <button className="tod-back" onClick={() => { setLevel(null); clearInterval(tickRef.current) }}>
          ← 换一档
        </button>
        <div className="tod-kicker">TRUTH OR DARE</div>
        <h1 className="tod-title">{LEVELS.find(l => l.key === level)?.label}</h1>
      </header>

      <div className="tod-table">
        {['truth', 'dare'].map(kind => {
          const p = cards[kind]
          const isFlipped = flipped[kind]
          return (
            <div className={'tod-card' + (isFlipped ? ' is-flipped' : '') + (kind === 'dare' ? ' tod-card--dare' : '')}
                 key={kind}>
              <button className="tod-card__inner" onClick={() => flip(kind)} disabled={!p}
                      aria-label={kind === 'truth' ? '翻真心话' : '翻大冒险'}>
                <span className="tod-card__face tod-card__back">
                  <span className="tod-card__mark">{kind === 'truth' ? '真' : '冒'}</span>
                  <span className="tod-card__label">{kind === 'truth' ? '真心话' : '大冒险'}</span>
                  <span className="tod-card__tap">点击翻牌</span>
                </span>
                <span className="tod-card__face tod-card__front">
                  <span className="tod-card__kind">{kind === 'truth' ? '真心话' : '大冒险'}</span>
                  <span className="tod-card__text">{p?.text}</span>
                  {kind === 'dare' && left !== null && (
                    <span className={'tod-timer' + (left === 0 ? ' is-done' : '')}>
                      {left === 0 ? '时间到' : `${left}s`}
                    </span>
                  )}
                </span>
              </button>

              {isFlipped && (
                <div className="tod-card__actions">
                  <button className="tod-mini" onClick={() => swap(kind)}>换一张</button>
                  <button className="tod-mini tod-mini--done" onClick={() => openRating(kind)}>完成</button>
                </div>
              )}

              {ratingFor === kind && (
                <div className="tod-rate">
                  <div className="tod-rate__q">这条怎么样？</div>
                  <div className="tod-rate__stars">
                    {[1, 2, 3, 4, 5].map(n => (
                      <button key={n} className={'tod-star' + (n <= stars ? ' is-on' : '')}
                              onClick={() => setStars(n)} aria-label={`${n} 星`}>★</button>
                    ))}
                  </div>
                  <div className="tod-rate__tags">
                    {TAGS.map(t => (
                      <button key={t}
                              className={'tod-tag' + (tags.includes(t) ? ' is-on' : '')}
                              onClick={() => setTags(v => v.includes(t) ? v.filter(x => x !== t) : [...v, t])}>
                        {t}
                      </button>
                    ))}
                  </div>
                  <button className="tod-submit" disabled={!stars} onClick={submitRating}>提交</button>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="tod-foot">
        <button className="tod-reset" onClick={() => deal(pool)}>重新发牌</button>
        <span className="tod-count">本局已用 {used.size} 条</span>
      </div>

      {toast && <div className="tod-toast">{toast}</div>}
    </div>
  )
}
