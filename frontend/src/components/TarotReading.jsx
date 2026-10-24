import React, { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import ReadingChat from './ReadingChat'
import ReadingFeedback from './ReadingFeedback'

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

export default function TarotReading({ readingId, positions }) {
  const [question, setQuestion] = useState('')
  const [state, setState] = useState('idle')          // idle | asking | done | error
  const [reading, setReading] = useState(null)         // { reading: {...} | null, reading_text, question, rating }
  const [quota, setQuota] = useState(null)
  const [err, setErr] = useState(null)                 // { message, retryable }
  const topRef = useRef(null)

  // The questions live in their own drawer, not in this column.
  const [chatOpen, setChatOpen] = useState(false)
  const [turns, setTurns] = useState([])
  const [turnsLoaded, setTurnsLoaded] = useState(false)
  const timers = useRef({})
  const chatBusy = turns.some(t => t.state === 'pending')


  // A fresh spread is a fresh reading.
  useEffect(() => {
    setQuestion(''); setState('idle'); setReading(null); setQuota(null)
    setErr(null)
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
    setState('done')
    // The reading lands below the fold on a phone; bring it into view.
    requestAnimationFrame(() => topRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  }

  useEffect(() => () => Object.values(timers.current).forEach(clearTimeout), [])

  const watchTurn = useCallback((fid, started = Date.now()) => {
    clearTimeout(timers.current[fid])
    timers.current[fid] = setTimeout(async () => {
      const d = await api.get(`/api/tarot/followups/${fid}`)
      if (!d.ok) return
      setTurns(prev => prev.map(t => (t.id === fid ? d.followup : t)))
      if (d.followup.state === 'pending' && Date.now() - started < 4 * 60 * 1000) watchTurn(fid, started)
    }, 2500)
  }, [])

  // Opening the drawer the first time pulls what was already asked.
  useEffect(() => {
    if (!chatOpen || turnsLoaded || !readingId) return undefined
    let alive = true
    api.get(`/api/tarot/readings/${readingId}/followups`).then(d => {
      if (!alive || !d.ok) return
      setTurns(d.followups || [])
      setTurnsLoaded(true)
      ;(d.followups || []).filter(f => f.state === 'pending').forEach(f => watchTurn(f.id))
    })
    return () => { alive = false }
  }, [chatOpen, turnsLoaded, readingId, watchTurn])

  const sendQuestion = async (question) => {
    const d = await api.post(`/api/tarot/readings/${readingId}/ask`, { question })
    if (!d.ok) throw new Error(d.error || '没有发出去，再试一次。')
    setTurns(prev => [...prev, d.followup])
    watchTurn(d.followup.id)
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
                <> {quota.remaining} of {quota.limit} left today <span className="label-zh">今天还剩 {quota.remaining} 次</span></>
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

          <ReadingFeedback feature="tarot" readingId={reading.id} />

          <div className="ask-launch">
            <button type="button" className="ask-launch__btn" onClick={() => setChatOpen(true)}>
              <i className="fas fa-comments" aria-hidden="true" />
              追问<small>就这次的三张牌接着问</small>
            </button>
            {turns.length > 0 && <span className="ask-launch__count">已问 {turns.length} 个</span>}
          </div>
        </div>
      )}

      <ReadingChat
        open={chatOpen}
        onClose={() => setChatOpen(false)}
        title="追问"
        subtitle="就这次的三张牌接着问"
        turns={turns}
        onSend={sendQuestion}
        busy={chatBusy}
      />
    </section>
  )
}
