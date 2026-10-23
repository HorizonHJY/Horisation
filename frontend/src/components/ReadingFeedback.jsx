import React, { useEffect, useState } from 'react'
import { api } from '../api'

/**
 * Was it right?
 *
 * One tap for 准, one tap plus a reason for 不准. The reason chips are what
 * steer the next reading (Backend/Service/ai/feedback.py reads the tallies);
 * the free-text note stays in the table for the owner, and never reaches a
 * prompt — a note can name someone's family, job or illness, and the lesson is
 * shown to the model while it writes for a different person.
 */

const REASONS = ['太笼统', '时间点对不上', '和我的情况不符', '语气不喜欢', '别的']
const MAX_NOTE = 200

export default function ReadingFeedback({ feature, readingId, initial = null }) {
  const [verdict, setVerdict] = useState(initial?.verdict || null)
  const [reasons, setReasons] = useState(initial?.reasons || [])
  const [note, setNote] = useState(initial?.note || '')
  const [saved, setSaved] = useState(Boolean(initial?.verdict))
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  function adopt(f) {
    setVerdict(f?.verdict || null)
    setReasons(f?.reasons || [])
    setNote(f?.note || '')
    setSaved(Boolean(f?.verdict))
  }

  // The reading payload does not carry the vote; ask for it. Reopening a
  // reading should show what you said, not a blank question again.
  useEffect(() => {
    adopt(initial)
    if (!feature || !readingId || initial?.verdict) return undefined
    let alive = true
    api.get('/api/reading-feedback?feature=' + encodeURIComponent(feature)
            + '&reading_id=' + encodeURIComponent(readingId))
      .then(d => { if (alive && d.ok && d.feedback) adopt(d.feedback) })
      .catch(() => {})
    return () => { alive = false }
  }, [feature, readingId]) // eslint-disable-line react-hooks/exhaustive-deps

  async function save(nextVerdict, nextReasons = reasons, nextNote = note) {
    if (!feature || !readingId || busy) return
    setBusy(true)
    setErr('')
    const d = await api.post('/api/reading-feedback', {
      feature, reading_id: readingId, verdict: nextVerdict,
      reasons: nextVerdict === 'down' ? nextReasons : [],
      note: nextNote.trim() || undefined,
    })
    setBusy(false)
    if (d.ok) {
      setVerdict(nextVerdict)
      setSaved(true)
    } else {
      setErr(d.error || '没有存上，再试一次。')
    }
  }

  function toggleReason(reason) {
    setReasons(prev => prev.includes(reason) ? prev.filter(r => r !== reason) : [...prev, reason])
  }

  function pickNo() {
    setVerdict('down')
    setSaved(false)   // the chips are the answer; nothing is stored until 提交
  }

  return (
    <section className="rf" aria-label="这次说得准吗">
      <div className="rf__ask">
        <span className="rf__label">这次说得准吗？</span>
        <div className="rf__verdict">
          <button
            type="button"
            className={'rf__btn' + (verdict === 'up' ? ' is-on' : '')}
            aria-pressed={verdict === 'up'}
            disabled={busy}
            onClick={() => save('up', [], note)}
          >
            <i className="fas fa-check" aria-hidden="true" /> 准
          </button>
          <button
            type="button"
            className={'rf__btn rf__btn--no' + (verdict === 'down' ? ' is-on' : '')}
            aria-pressed={verdict === 'down'}
            disabled={busy}
            onClick={pickNo}
          >
            <i className="fas fa-xmark" aria-hidden="true" /> 不准
          </button>
        </div>
      </div>

      {verdict === 'down' && (
        <div className="rf__why">
          <p className="rf__why-label">哪里不准？可以多选，也可以不选。</p>
          <div className="rf__chips">
            {REASONS.map(r => (
              <button
                key={r}
                type="button"
                className={'rf__chip' + (reasons.includes(r) ? ' is-on' : '')}
                aria-pressed={reasons.includes(r)}
                onClick={() => toggleReason(r)}
              >
                {r}
              </button>
            ))}
          </div>
          <input
            type="text"
            className="rf__note"
            maxLength={MAX_NOTE}
            value={note}
            onChange={e => setNote(e.target.value)}
            placeholder="想说得具体一点，可以不填"
            aria-label="补充说明，可选"
          />
          <div className="rf__row">
            <span className="rf__hint">{note.length}/{MAX_NOTE}</span>
            <button
              type="button"
              className="rf__submit"
              onClick={() => save('down')}
              disabled={busy}
            >
              {busy ? '保存中…' : saved ? '更新' : '提交'}
            </button>
          </div>
        </div>
      )}

      {saved && verdict && (
        <p className="rf__done" role="status">
          记下了。<span className="rf__done-zh">
            {verdict === 'down' ? '下几次解读会避开这些问题。' : '谢谢。'}
          </span>
        </p>
      )}
      {err && <p className="rf__err" role="alert">{err}</p>}
    </section>
  )
}
