import React, { useState } from 'react'

export default function ReviewComposer({ deal, tags, onCancel, onSubmit }) {
  const [rating, setRating] = useState(5)
  const [picked, setPicked] = useState([])
  const [note,   setNote]   = useState('')
  const [busy,   setBusy]   = useState(false)
  const [err,    setErr]    = useState('')

  function toggleTag(t) {
    setPicked(p => p.includes(t) ? p.filter(x => x !== t) : [...p, t])
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setBusy(true); setErr('')
    const res = await onSubmit({
      rating,
      tags: picked,
      note: note.trim() || undefined,
    })
    setBusy(false)
    if (res !== true) setErr(typeof res === 'string' ? res : 'Could not submit.')
  }

  return (
    <form
      className="mb-3 p-3"
      style={{ background: '#fff', border: '1px solid var(--border-soft, #e8e2d4)', borderRadius: 10 }}
      onSubmit={handleSubmit}
    >
      <div className="fw-semibold small mb-2">
        给 {deal.counterpart?.display_name || deal.counterpart?.username} 评价
        {deal.listing_title && <span className="text-muted">（{deal.listing_title}）</span>}
      </div>

      <div className="d-flex align-items-center gap-1 mb-2">
        {[1,2,3,4,5].map(n => (
          <button
            key={n}
            type="button"
            className="btn btn-link p-0"
            style={{ fontSize: '1.4rem', lineHeight: 1, color: n <= rating ? '#d6a73a' : '#ddd', textDecoration: 'none' }}
            onClick={() => setRating(n)}
            aria-label={`${n} star`}
          >★</button>
        ))}
        <span className="small text-muted ms-2">{rating}/5</span>
      </div>

      <div className="d-flex gap-1 flex-wrap mb-2">
        {tags.map(t => {
          const active = picked.includes(t)
          return (
            <button
              key={t}
              type="button"
              className={`btn btn-sm ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
              style={{ fontSize: '.78rem', padding: '.2rem .55rem' }}
              onClick={() => toggleTag(t)}
            >{t}</button>
          )
        })}
      </div>

      <textarea
        className="form-control"
        rows={2}
        maxLength={200}
        placeholder="可选:简单描述一下这次交易…"
        value={note}
        onChange={e => setNote(e.target.value)}
      />
      <div className="small text-muted mt-1">{note.length}/200</div>

      {err && <div className="small text-danger mt-2">{err}</div>}

      <div className="d-flex justify-content-end gap-2 mt-2">
        <button type="button" className="btn btn-outline-secondary btn-sm" onClick={onCancel} disabled={busy}>
          取消
        </button>
        <button type="submit" className="btn btn-primary btn-sm" disabled={busy}>
          {busy ? '提交中…' : '提交评价'}
        </button>
      </div>
    </form>
  )
}
