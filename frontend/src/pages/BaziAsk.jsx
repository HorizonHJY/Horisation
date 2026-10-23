import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import HandLoader from '../components/HandLoader'
import ReadingChat from '../components/ReadingChat'

const POLL_GIVE_UP_MS = 4 * 60 * 1000

export default function BaziAsk() {
  const [items, setItems] = useState(null)   // read charts, newest first
  const [error, setError] = useState('')
  const [meta, setMeta] = useState(null)      // the chosen chart's one-liner
  const [turns, setTurns] = useState([])
  const [open, setOpen] = useState(false)
  const timers = useRef({})

  useEffect(() => {
    api.get('/api/bazi/readings').then(d => {
      if (d.ok) setItems((d.items || []).filter(h => h.state === 'done'))
      else setError(d.error || '没有读到你的命盘。')
    })
    return () => Object.values(timers.current).forEach(clearTimeout)
  }, [])

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
    setMeta(h)
    setTurns(d.item.followups || [])
    setOpen(true)
    ;(d.item.followups || []).filter(f => f.state === 'pending').forEach(f => watch(f.id))
  }

  const ask = async (question) => {
    const d = await api.post('/api/bazi/readings/' + meta.id + '/ask', { question })
    if (!d.ok) throw new Error(d.error || '没有发出去，再试一次。')
    setTurns(prev => [...prev, d.followup])
    watch(d.followup.id)
  }

  if (items === null) return <div className='text-center py-5'><HandLoader /></div>

  return (
    <>
      {error && <div className='bz-alert' role='alert'>{error}</div>}

      {items.length === 0 ? (
        <p className='bz-fine'>还没有解读过的命盘。先去「算盘」排一盘、请先生解读，再回来这里接着问。</p>
      ) : (
        <div className='bz-ask-tab'>
          <p className='bz-kicker bz-kicker--rule'>选一张命盘 <small>点开就能接着问</small></p>
          <ul className='bz-hist'>
            {items.map(h => (
              <li key={h.id} className='bz-hist__row'>
                <button type='button' className='bz-hist__open' onClick={() => openChart(h)}>
                  <span className='bz-hist__bazi'>{h.bazi}</span>
                  <span className='bz-hist__meta'>{h.gender_label} · {h.birth} · {h.place}</span>
                  {h.verdict && <span className='bz-hist__verdict'>{h.verdict}</span>}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {meta && (
        <ReadingChat
          open={open}
          onClose={() => setOpen(false)}
          title='追问'
          subtitle={meta.gender_label + ' · ' + meta.birth + ' · ' + meta.place}
          turns={turns}
          onSend={ask}
          busy={turns.some(t => t.state === 'pending')}
          placeholder='就这张命盘接着问…'
        />
      )}
    </>
  )
}
