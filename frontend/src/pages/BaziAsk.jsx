import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import HandLoader from '../components/HandLoader'

const POLL_GIVE_UP_MS = 4 * 60 * 1000
const MAX_Q = 300

export default function BaziAsk() {
  const [items, setItems] = useState(null)   // read charts, newest first
  const [sel, setSel] = useState(null)        // the chosen chart one-liner
  const [turns, setTurns] = useState([])
  const [error, setError] = useState('')
  const [q, setQ] = useState('')
  const [sending, setSending] = useState(false)
  const timers = useRef({})
  const listRef = useRef(null)

  const watch = useCallback((fid, started = Date.now()) => {
    clearTimeout(timers.current[fid])
    timers.current[fid] = setTimeout(async () => {
      const d = await api.get('/api/bazi/followups/' + fid)
      if (!d.ok) return
      setTurns(prev => prev.map(f => (f.id === fid ? d.followup : f)))
      if (d.followup.state === 'pending' && Date.now() - started < POLL_GIVE_UP_MS) watch(fid, started)
    }, 2500)
  }, [])

  const openChart = async (h) => {
    const d = await api.get('/api/bazi/readings/' + h.id)
    if (!d.item) { setError(d.error); return }
    setSel(h)
    setTurns(d.item.followups || [])
    ;(d.item.followups || []).filter(f => f.state === 'pending').forEach(f => watch(f.id))
  }

  useEffect(() => {
    api.get('/api/bazi/readings').then(d => {
      if (d.ok) {
        const read = (d.items || []).filter(h => h.state === 'done')
        setItems(read)
        if (read.length) openChart(read[0])
      } else setError(d.error || '没有读到你的命盘。')
    })
    return () => Object.values(timers.current).forEach(clearTimeout)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const busy = turns.some(t => t.state === 'pending')

  const send = async (e) => {
    e.preventDefault()
    const question = q.trim()
    if (!question || sending || busy || !sel) return
    setSending(true)
    setError('')
    try {
      const d = await api.post('/api/bazi/readings/' + sel.id + '/ask', { question })
      if (!d.ok) { setError(d.error); return }
      setQ('')
      setTurns(prev => [...prev, d.followup])
      watch(d.followup.id)
    } finally { setSending(false) }
  }

  const lastKey = turns.length + ':' + (turns[turns.length - 1]?.state || '')
  useEffect(() => { const el = listRef.current; if (el) el.scrollTop = el.scrollHeight }, [lastKey])

  if (items === null) return <div className='text-center py-5'><HandLoader /></div>

  if (items.length === 0) {
    return <p className='bz-fine'>还没有解读过的命盘。先去「算盘」排一盘、请先生解读，再回来这里接着问。</p>
  }

  const birthDate = (s) => (s || '').slice(0, 10)

  return (
    <div className='bza'>
      {error && <div className='bz-alert' role='alert'>{error}</div>}

      <div className='bza-chips' role='tablist' aria-label='选择要提问的命盘'>
        {items.map(h => (
          <button
            key={h.id}
            type='button'
            role='tab'
            aria-selected={sel?.id === h.id}
            className={'bza-chip' + (sel?.id === h.id ? ' is-on' : '')}
            onClick={() => openChart(h)}
          >
            <span className='bza-chip__date'>{birthDate(h.birth)}</span>
            {h.place ? <span className='bza-chip__place'>{h.place}</span> : null}
          </button>
        ))}
      </div>

      {sel && (
        <>
          <div className='bza-ctx'>
            <div className='bza-ctx__pillars'>{sel.bazi}</div>
            <div className='bza-ctx__meta'>{sel.gender_label} {sel.birth} {sel.place}</div>
            {sel.verdict && <div className='bza-ctx__verdict'>{sel.verdict}</div>}
          </div>

          <div className='bza-chat'>
            <div className='bza-chat__list' ref={listRef} aria-live='polite'>
              {turns.length === 0 && (
                <p className='bza-chat__empty'>就这张命盘接着问。回答会参考你标的「准 / 不准」。</p>
              )}
              {turns.map(t => (
                <div className='bza-turn' key={t.id}>
                  <p className='bza-turn__q'>{t.question}</p>
                  {t.state === 'done' && <p className='bza-turn__a'>{t.answer}</p>}
                  {t.state === 'pending' && <p className='bza-turn__a bza-turn__a--wait'>正在看…</p>}
                  {t.state === 'error' && <p className='bza-turn__a bza-turn__a--err'>{t.error?.error || '没有答上来，换个说法再问一次。'}</p>}
                </div>
              ))}
            </div>

            <form className='bza-chat__form' onSubmit={send}>
              <textarea
                className='bza-chat__input'
                rows={2}
                maxLength={MAX_Q}
                value={q}
                onChange={e => setQ(e.target.value)}
                placeholder='就这张命盘接着问…'
                aria-label='你的问题'
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) send(e) }}
              />
              <div className='bza-chat__row'>
                <span className='bza-chat__hint'>{q.length}/{MAX_Q}</span>
                <button type='submit' className='bza-chat__send' disabled={busy || sending || !q.trim()}>
                  {busy || sending ? '作答中…' : '发送'}
                </button>
              </div>
            </form>
          </div>
        </>
      )}
    </div>
  )
}
