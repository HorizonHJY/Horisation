import React, { useEffect, useRef, useState } from 'react'

/**
 * The questions, kept off the reading.
 *
 * The result page is something you read; a conversation is something you sit
 * in. Sharing one column made the reading scroll away every time an answer
 * arrived. This is a drawer on a wide screen and a full-height sheet on a
 * phone - it covers the reading, but it never pushes it around.
 *
 * Turn shape, identical for tarot and bazi:
 *   { id, question, answer, state: 'pending' | 'done' | 'error', error }
 */

const MAX_Q = 300

export default function ReadingChat({
  open, onClose, title = '追问', subtitle, turns = [],
  onSend, busy = false, placeholder = '接着问…', footnote,
}) {
  const [q, setQ] = useState('')
  const [err, setErr] = useState('')
  const sending = useRef(false)
  const listRef = useRef(null)
  const inputRef = useRef(null)

  // A phone should not scroll the reading underneath the sheet.
  useEffect(() => {
    if (!open) return undefined
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = prev }
  }, [open])

  useEffect(() => {
    if (!open) return undefined
    const onKey = (e) => { if (e.key === 'Escape') onClose?.() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  // New answer landed: keep the newest turn in view.
  const lastState = turns[turns.length - 1]?.state
  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [turns.length, lastState])

  useEffect(() => {
    if (open) requestAnimationFrame(() => inputRef.current?.focus())
  }, [open])

  if (!open) return null

  async function send(e) {
    e?.preventDefault?.()
    const question = q.trim()
    // The ref closes the double-click gap before `busy` is re-rendered.
    if (!question || sending.current || busy) return
    sending.current = true
    setErr('')
    try {
      await onSend(question)
      setQ('')
    } catch (e2) {
      setErr(e2?.message || '没有发出去，再试一次。')
    } finally {
      sending.current = false
    }
  }

  return (
    <div className="rc" role="dialog" aria-modal="true" aria-label={title}>
      <button type="button" className="rc__scrim" aria-label="关闭" onClick={onClose} />
      <section className="rc__panel">
        <header className="rc__head">
          <div>
            <h2 className="rc__title">{title}</h2>
            {subtitle && <p className="rc__subtitle">{subtitle}</p>}
          </div>
          <button type="button" className="rc__close" onClick={onClose} aria-label="关闭">
            <i className="fas fa-xmark" aria-hidden="true" />
          </button>
        </header>

        <div className="rc__list" ref={listRef} aria-live="polite">
          {turns.length === 0 && (
            <p className="rc__empty">想问什么都可以。回答会参考你标的「准 / 不准」。</p>
          )}
          {turns.map(t => (
            <div className="rc__turn" key={t.id}>
              <p className="rc__q">{t.question}</p>
              {t.state === 'done' && <p className="rc__a">{t.answer}</p>}
              {t.state === 'pending' && (
                <p className="rc__a rc__a--wait">
                  <span className="rc__dots" aria-hidden="true"><i /><i /><i /></span>
                  正在看…
                </p>
              )}
              {t.state === 'error' && (
                <p className="rc__a rc__a--err">{t.error?.error || '没有答上来，换个说法再问一次。'}</p>
              )}
            </div>
          ))}
        </div>

        <form className="rc__form" onSubmit={send}>
          <textarea
            ref={inputRef}
            className="rc__input"
            rows={2}
            maxLength={MAX_Q}
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder={placeholder}
            aria-label="你的问题"
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) send(e) }}
          />
          <div className="rc__row">
            <span className="rc__hint">{q.length}/{MAX_Q}</span>
            <button type="submit" className="rc__send" disabled={busy || !q.trim()}>
              {busy ? '作答中…' : '发送'}
            </button>
          </div>
          {err && <p className="rc__a--err" role="alert">{err}</p>}
          {footnote && <p className="rc__foot">{footnote}</p>}
        </form>
      </section>
    </div>
  )
}
