import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api'

/**
 * The reading of the three cards as one thing — asked for, not automatic.
 *
 * The cards are already face up and their meanings already shown. This is
 * the optional second step: write your question if you want the reading to
 * speak to it, or leave it blank and it reads the moment. The server
 * interprets the spread IT drew, addressed by reading_id; nothing about the
 * cards travels up from here.
 *
 * States: idle (the invitation) → asking (waiting on the model) → done (the
 * reading, then a chance to say how well it fit) → or an error that says
 * whether trying again will cost a turn.
 */

const MAX_QUESTION = 200
const MAX_NOTE = 100

export default function TarotReading({ readingId, positions }) {
  const [question, setQuestion] = useState('')
  const [state, setState] = useState('idle')          // idle | asking | done | error
  const [reading, setReading] = useState(null)         // { reading: {...} | null, reading_text, question, rating }
  const [quota, setQuota] = useState(null)
  const [err, setErr] = useState(null)                 // { message, retryable }
  const [rating, setRating] = useState(0)
  const [note, setNote] = useState('')
  const [rated, setRated] = useState(false)
  const [savingRating, setSavingRating] = useState(false)
  const topRef = useRef(null)

  // A fresh spread is a fresh reading.
  useEffect(() => {
    setQuestion(''); setState('idle'); setReading(null); setQuota(null)
    setErr(null); setRating(0); setNote(''); setRated(false)
  }, [readingId])

  async function ask() {
    if (state === 'asking' || !readingId) return
    setState('asking')
    setErr(null)
    const q = question.trim()
    const d = await api.post('/api/tarot/reading', { reading_id: readingId, question: q || undefined })
    if (!d.ok) {
      setErr({ message: d.error || '解读没有回来。', retryable: Boolean(d.retryable), kind: d.error_kind })
      if (d.quota) setQuota(d.quota)
      setState('error')
      return
    }
    setReading(d.reading)
    if (d.quota) setQuota(d.quota)
    if (d.reading?.rating) { setRating(d.reading.rating); setRated(true) }
    setState('done')
    // The reading lands below the fold on a phone; bring it into view.
    requestAnimationFrame(() => topRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  }

  async function submitRating(value) {
    if (!reading?.id || savingRating) return
    setRating(value)
    setSavingRating(true)
    const d = await api.post(`/api/tarot/readings/${reading.id}/rating`, {
      rating: value, note: note.trim() || undefined,
    })
    setSavingRating(false)
    if (d.ok) setRated(true)
  }

  const r = reading?.reading      // parsed structure, or null when the model wandered off the schema
  const posLabel = (key) => positions.find(p => p.key === key) || { label: key, label_zh: '' }

  return (
    <section className="tarot-read" ref={topRef} aria-label="Reading of the whole spread">
      <h2 className="tarot-read__title">
        Read the three together
        <span className="tarot-read__title-zh">整体解读</span>
      </h2>

      {state !== 'done' && (
        <div className="tarot-read__ask">
          <label className="tarot-read__label" htmlFor="tarot-question">
            Your question, if you want the reading to speak to it
            <span className="label-zh">想让解读贴合你的问题，可以写在这里（可选）</span>
          </label>
          <textarea
            id="tarot-question"
            className="tarot-read__input"
            rows={2}
            maxLength={MAX_QUESTION}
            value={question}
            onChange={e => setQuestion(e.target.value)}
            disabled={state === 'asking'}
            placeholder="留空也可以 — 就读此刻的处境"
          />
          <div className="tarot-read__row">
            <span className="tarot-read__hint">
              {question.length}/{MAX_QUESTION}
              {quota?.limit != null && (
                <> · {quota.remaining} of {quota.limit} left today <span className="label-zh">今天还剩 {quota.remaining} 次</span></>
              )}
            </span>
            <button
              type="button"
              className="tarot__btn"
              onClick={ask}
              disabled={state === 'asking'}
            >
              {state === 'asking' ? 'Reading… 解读中' : 'Read it 解读'}
            </button>
          </div>
          {/* Honest about where the words go, in one line. */}
          <p className="tarot-read__privacy">
            Your question and the three cards are sent to an AI model to write this. Nothing else is.
            <span className="label-zh">问题和三张牌会发给 AI 模型生成解读，别的不会。</span>
          </p>
        </div>
      )}

      {state === 'asking' && (
        <p className="tarot-read__wait" role="status">
          The reader is looking at your cards. <span className="label-zh">正在看牌，大约十秒。</span>
        </p>
      )}

      {state === 'error' && err && (
        <div className="tarot-read__error" role="alert">
          <p className="mb-2">{err.message}</p>
          {err.retryable && (
            <button type="button" className="tarot__btn tarot__btn--quiet" onClick={ask}>Try again 再试一次</button>
          )}
        </div>
      )}

      {state === 'done' && reading && (
        <div className="tarot-read__result">
          {reading.question && (
            <p className="tarot-read__q">
              <span className="tarot-read__q-label">You asked</span> {reading.question}
            </p>
          )}

          {r ? (
            <>
              <p className="tarot-read__summary">{r.summary}</p>
              <div className="tarot-read__positions">
                {['past', 'present', 'future'].map(key => {
                  const p = posLabel(key)
                  return (
                    <div className="tarot-read__pos" key={key}>
                      <div className="tarot-read__pos-label">
                        {p.label}<span className="label-zh">{p.label_zh}</span>
                      </div>
                      <p className="mb-0">{r[key]}</p>
                    </div>
                  )
                })}
              </div>
              <div className="tarot-read__next">
                <div className="tarot-read__pos-label">One small thing<span className="label-zh">明天可以做的一件小事</span></div>
                <p className="mb-0">{r.next_step}</p>
              </div>
            </>
          ) : (
            // Off-schema reply: show the model's text as it came, whole.
            <p className="tarot-read__summary" style={{ whiteSpace: 'pre-wrap' }}>{reading.reading_text}</p>
          )}

          <p className="tarot-read__ai-note">
            Written by an AI model from the three cards and your question. Take it as a mirror, not a verdict.
            <span className="label-zh">由 AI 根据三张牌和你的问题生成。当镜子看，别当判词。</span>
          </p>

          <div className="tarot-read__rate">
            <div className="tarot-read__rate-q">
              {rated ? 'Thanks — noted.' : 'Did it fit?'}
              <span className="label-zh">{rated ? '记下了。' : '贴合吗？'}</span>
            </div>
            <div className="tarot-read__stars" role="radiogroup" aria-label="How well the reading fit, 1 to 5">
              {[1, 2, 3, 4, 5].map(n => (
                <button
                  key={n}
                  type="button"
                  role="radio"
                  aria-checked={rating === n}
                  aria-label={`${n} of 5`}
                  className={`tarot-read__star${rating >= n ? ' is-on' : ''}`}
                  onClick={() => submitRating(n)}
                  disabled={savingRating}
                >
                  <i className="fas fa-star" aria-hidden="true" />
                </button>
              ))}
            </div>
            {rating > 0 && (
              <div className="tarot-read__note">
                <input
                  type="text"
                  className="tarot-read__input tarot-read__input--line"
                  maxLength={MAX_NOTE}
                  value={note}
                  onChange={e => setNote(e.target.value)}
                  placeholder="一句话，可选"
                  aria-label="A note about the rating, optional"
                />
                <button
                  type="button"
                  className="tarot__btn tarot__btn--quiet"
                  onClick={() => submitRating(rating)}
                  disabled={savingRating}
                >
                  Submit 提交
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
