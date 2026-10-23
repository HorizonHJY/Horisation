import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import HandLoader from '../components/HandLoader'
import BaziCompat from './BaziCompat'
import ReadingChat from '../components/ReadingChat'
import ReadingFeedback from '../components/ReadingFeedback'

/**
 * 四柱八字 — cast a chart, then have it read.
 *
 * The chart is arithmetic and comes from the server (Backend/Service/bazi/
 * chart.py, true solar time included). The reading is the AI layer's, asked
 * for by reading_id only, and it runs in the background: up to two minutes,
 * longer than nginx holds a request, so this page starts it and polls.
 *
 * The page is its own world, as tarot is: rice paper, ink, one cinnabar red.
 * The reading is laid out, not printed — a seal for the verdict, four slips
 * for the gods, a track for the luck pillars, a thread for the years, and a
 * checklist the reader ticks, because the point of ④ is to be checked.
 */

const EL_CLASS = { 木: 'wood', 火: 'fire', 土: 'earth', 金: 'metal', 水: 'water' }
const EL_ORDER = ['木', '火', '土', '金', '水']
const ROWS = [
  ['主星', p => p.main_star],
  ['藏干', p => p.hidden.map(h => <span key={h.gan} className={`bz-el--${EL_CLASS[h.element]}`}>{h.gan}{h.element}</span>)],
  ['副星', p => p.hidden.map(h => <span key={h.gan}>{h.star}</span>)],
  ['星运', p => p.di_shi],
  ['自坐', p => p.zi_zuo],
  ['空亡', p => p.xun_kong],
  ['纳音', p => p.na_yin],
  ['神煞', p => (p.shen_sha.length ? p.shen_sha.map(s => <span key={s}>{s}</span>) : '—'), 'bz-row--sha'],
]
const POLL_MS = 3000
const POLL_GIVE_UP_MS = 4 * 60 * 1000

function elClass(_ch, el) { return `bz-el--${EL_CLASS[el] || ''}` }

// ── Form ───────────────────────────────────────────────────────────
function CastForm({ cities, initial, onCast, busy }) {
  const [f, setF] = useState(initial || { date: '', time: '', place: '', gender: '', longitude: '', utc_offset: '8' })
  const known = cities.some(c => c.name === f.place)
  const set = (k) => (e) => setF(prev => ({ ...prev, [k]: e.target.value }))

  const submit = (e) => {
    e.preventDefault()
    const body = { date: f.date, time: f.time, place: f.place.trim(), gender: f.gender, utc_offset: Number(f.utc_offset || 8) }
    if (!known) body.longitude = Number(f.longitude)
    onCast(body, f)
  }

  return (
    <form className="bz-form" onSubmit={submit}>
      <div className="bz-form__grid">
        <label className="bz-field">
          <span>出生日期<small>阳历</small></span>
          <input type="date" required value={f.date} onChange={set('date')} max={new Date().toISOString().slice(0, 10)} />
        </label>
        <label className="bz-field">
          <span>出生时间<small>出生证明上的钟表时间</small></span>
          <input type="time" required value={f.time} onChange={set('time')} />
        </label>
        <label className="bz-field">
          <span>出生地<small>用来校正真太阳时</small></span>
          <input list="bz-cities" required value={f.place} onChange={set('place')} placeholder="如：广州" autoComplete="off" />
          <datalist id="bz-cities">{cities.map(c => <option key={c.name} value={c.name} />)}</datalist>
        </label>
        <fieldset className="bz-field bz-field--gender">
          <legend>性别<small>大运顺逆看它</small></legend>
          <div className="bz-seg" role="radiogroup">
            {[['male', '男 · 乾造'], ['female', '女 · 坤造']].map(([v, label]) => (
              <label key={v} className={`bz-seg__opt${f.gender === v ? ' is-on' : ''}`}>
                <input type="radio" name="gender" value={v} checked={f.gender === v} onChange={set('gender')} required />
                {label}
              </label>
            ))}
          </div>
        </fieldset>
      </div>

      {f.place && !known && (
        <div className="bz-form__extra">
          <p>列表里没有「{f.place}」。填它的经度就行（东经为正，西经为负）。</p>
          <div className="bz-form__grid">
            <label className="bz-field"><span>经度</span>
              <input type="number" step="0.1" min="-180" max="180" required value={f.longitude} onChange={set('longitude')} placeholder="如：86.0" />
            </label>
            <label className="bz-field"><span>出生时用的时区<small>中国填 8</small></span>
              <input type="number" step="0.5" min="-12" max="14" value={f.utc_offset} onChange={set('utc_offset')} />
            </label>
          </div>
        </div>
      )}

      <div className="bz-form__foot">
        <button type="submit" className="bz-btn" disabled={busy}>{busy ? '排盘中…' : '排盘'}</button>
        <p className="bz-fine">命盘和解读只有你自己看得到。你可以随时从自己的记录里删除；网站会保留存档，用于分析和改进解读。</p>
      </div>
    </form>
  )
}

// ── Chart ──────────────────────────────────────────────────────────
function Chart({ chart }) {
  const st = chart.solar_time
  return (
    <section className="bz-chart" aria-label="命盘">
      <header className="bz-chart__head">
        <div className="bz-chart__seal" aria-hidden="true">{chart.gender_label}</div>
        <div>
          <div className="bz-chart__lunar">农历 {chart.lunar}</div>
          <div className="bz-chart__meta">
            阳历 {st.clock}（{chart.input.place}）· 真太阳时 <b>{st.true.slice(11)}</b>
            {st.dst_applied && ' · 已扣夏令时'}
          </div>
        </div>
      </header>

      <div className="bz-table" role="table">
        <div className="bz-row bz-row--head" role="row">
          <div role="columnheader" />
          {chart.pillars.map(p => <div role="columnheader" key={p.key}>{p.label}</div>)}
        </div>
        <div className="bz-row bz-row--big" role="row">
          <div role="rowheader">天干</div>
          {chart.pillars.map(p => <div key={p.key} className={`bz-glyph ${elClass(p.gan, p.gan_element)}`}>{p.gan}</div>)}
        </div>
        <div className="bz-row bz-row--big" role="row">
          <div role="rowheader">地支</div>
          {chart.pillars.map(p => <div key={p.key} className={`bz-glyph ${elClass(p.zhi, p.zhi_element)}`}>{p.zhi}</div>)}
        </div>
        {ROWS.map(([label, get, cls]) => (
          <div className={`bz-row ${cls || ''}`} role="row" key={label}>
            <div role="rowheader">{label}</div>
            {chart.pillars.map(p => <div key={p.key} className="bz-cell">{get(p)}</div>)}
          </div>
        ))}
      </div>

      <Elements elements={chart.elements} />

      <div className="bz-yun">
        <div className="bz-yun__label">
          大运 <small>起运 {chart.yun.start.date}，出生后 {chart.yun.start.years} 年 {chart.yun.start.months} 个月</small>
        </div>
        <ol className="bz-yun__track">
          <li className="bz-yun__step">
            <span className="bz-yun__age">{chart.yun.xiao_yun.from_age}–{chart.yun.xiao_yun.to_age}岁</span>
            <span className="bz-yun__gz">小运</span>
            <span className="bz-yun__year">{chart.yun.xiao_yun.from_year}</span>
          </li>
          {chart.yun.da_yun.slice(0, 8).map(d => (
            <li key={d.ganzhi} className={`bz-yun__step${d.current ? ' is-now' : ''}`} aria-current={d.current ? 'step' : undefined}>
              <span className="bz-yun__age">{d.start_age}岁</span>
              <span className="bz-yun__gz">
                <i className={elClass(d.ganzhi[0], d.gan_element)}>{d.ganzhi[0]}</i>
                <i className={elClass(d.ganzhi[1], d.zhi_element)}>{d.ganzhi[1]}</i>
              </span>
              <span className="bz-yun__year">{d.start_year}</span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}

function Elements({ elements }) {
  const total = EL_ORDER.reduce((s, e) => s + elements[e], 0) || 1
  return (
    <div className="bz-five" aria-label="五行比例">
      <div className="bz-five__bar">
        {EL_ORDER.map(e => elements[e] > 0 && (
          <span key={e} className={`bz-five__seg bz-bg--${EL_CLASS[e]}`} style={{ flexGrow: elements[e] }} title={`${e} ${elements[e]}`} />
        ))}
      </div>
      <ul className="bz-five__legend">
        {EL_ORDER.map(e => (
          <li key={e}><i className={`bz-bg--${EL_CLASS[e]}`} />{e}<b>{Math.round(elements[e] / total * 100)}%</b></li>
        ))}
      </ul>
    </div>
  )
}

// ── Reading ────────────────────────────────────────────────────────
function firstEl(s) { return [...(s || '')].find(ch => EL_CLASS[ch]) }

function Reading({ item, chart }) {
  const r = item.reading
  // Kept on the server, so they follow the reader to another device and the
  // follow-up answers can lean on them.
  const [marks, setMarks] = useState(item.verify || {})
  useEffect(() => { setMarks(item.verify || {}) }, [item.id]) // eslint-disable-line react-hooks/exhaustive-deps
  const mark = (i, v) => {
    const next = { ...marks }
    if (next[i] === v) delete next[i]; else next[i] = v
    setMarks(next)
    api.put(`/api/bazi/readings/${item.id}/verify`, { marks: next })
  }

  if (!r) {
    return <section className="bz-read"><p className="bz-read__raw">{item.reading_text}</p></section>
  }
  const yes = Object.values(marks).filter(v => v === 'yes').length
  const answered = Object.values(marks).filter(Boolean).length
  const themes = Object.fromEntries((r.da_yun || []).map(d => [d.ganzhi, d]))
  const slips = [['use', '用神'], ['like', '喜神'], ['avoid', '忌神'], ['idle', '闲神']].filter(([k]) => r.gods[k]?.element)

  return (
    <section className="bz-read" aria-label="命盘解读">
      {/* 判词: the whole reading in one line, under a seal. */}
      <div className="bz-verdict">
        <div className="bz-verdict__seal" aria-label={r.strength.level}>
          {[...r.strength.level].map((ch, i) => <span key={i}>{ch}</span>)}
        </div>
        <div>
          <div className="bz-kicker">判词</div>
          <h2 className="bz-verdict__text">{r.verdict}</h2>
          <div className="bz-verdict__pattern"><b>{r.pattern.name}</b>　{r.pattern.explain}</div>
        </div>
      </div>
      <ul className="bz-reasons">
        {r.strength.reasons.map((x, i) => <li key={i}>{x}</li>)}
      </ul>

      {/* 用喜忌闲: four slips, the element large. */}
      <div className="bz-kicker bz-kicker--rule">用神 · 喜忌</div>
      <div className="bz-slips">
        {slips.map(([k, label]) => {
          const g = r.gods[k]
          return (
            <div key={k} className={`bz-slip bz-slip--${k}`}>
              <div className="bz-slip__label">{label}</div>
              <div className={`bz-slip__el bz-el--${EL_CLASS[firstEl(g.element)] || ''}`}>{g.element}</div>
              {g.why && <p className="bz-slip__why">{g.why}</p>}
            </div>
          )
        })}
      </div>

      {/* 大运: the track again, now with what each ten years means. */}
      <div className="bz-kicker bz-kicker--rule">大运</div>
      <ol className="bz-runs">
        {chart.yun.da_yun.slice(0, 6).map(d => {
          const t = themes[d.ganzhi]
          return (
            <li key={d.ganzhi} className={`bz-run${d.current ? ' is-now' : ''}`}>
              <div className="bz-run__gz">
                <i className={elClass(d.ganzhi[0], d.gan_element)}>{d.ganzhi[0]}</i>
                <i className={elClass(d.ganzhi[1], d.zhi_element)}>{d.ganzhi[1]}</i>
              </div>
              <div className="bz-run__years">{d.start_year}–{d.end_year}{d.current && <span className="bz-now">当下</span>}</div>
              {t?.theme && <div className="bz-run__theme">{t.theme}</div>}
              {t?.reading && <p className="bz-run__text">{t.reading}</p>}
            </li>
          )
        })}
      </ol>

      {/* 流年: a thread down the page, one knot per year that mattered. */}
      <div className="bz-kicker bz-kicker--rule">流年回溯</div>
      <ol className="bz-years">
        {r.liu_nian.map(y => (
          <li key={y.year} className="bz-year">
            <div className="bz-year__when"><b>{y.year}</b><span>{y.ganzhi}</span></div>
            <div className="bz-year__body">
              {y.tag && <span className="bz-tag">{y.tag}</span>}
              <p>{y.event}</p>
            </div>
          </li>
        ))}
      </ol>

      {/* ④ is for checking — so it is a checklist. */}
      <div className="bz-kicker bz-kicker--rule">验证 <small>{answered ? `${yes} / ${answered} 准` : '逐条点一下'}</small></div>
      <ul className="bz-verify">
        {r.verify.map((q, i) => (
          <li key={i} className={`bz-check${marks[i] ? ` is-${marks[i]}` : ''}`}>
            <p>{q}</p>
            <div className="bz-check__btns" role="group" aria-label="准不准">
              <button type="button" aria-pressed={marks[i] === 'yes'} onClick={() => mark(i, 'yes')}>准</button>
              <button type="button" aria-pressed={marks[i] === 'no'} onClick={() => mark(i, 'no')}>不准</button>
            </div>
          </li>
        ))}
      </ul>

      <div className="bz-summary">
        <div className="bz-kicker">总评</div>
        <p>{r.summary}</p>
      </div>
      <p className="bz-fine">由 AI 根据程序排出的命盘推演生成。流年回溯是推演，不是事实，所以请逐条验证。</p>

      <ReadingFeedback feature="bazi" readingId={item.id} />

      <AskBox item={item} />
    </section>
  )
}

// ── Follow-up questions ────────────────────────────────────────────
const MAX_Q = 300

function AskBox({ item }) {
  const [list, setList] = useState(item.followups || [])
  const [open, setOpen] = useState(false)
  const timers = useRef({})

  useEffect(() => { setList(item.followups || []) }, [item.id]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => Object.values(timers.current).forEach(clearTimeout), [])

  const watch = useCallback((fid, started = Date.now()) => {
    clearTimeout(timers.current[fid])
    timers.current[fid] = setTimeout(async () => {
      const d = await api.get(`/api/bazi/followups/${fid}`)
      if (!d.ok) return
      setList(prev => prev.map(f => (f.id === fid ? d.followup : f)))
      if (d.followup.state === 'pending' && Date.now() - started < POLL_GIVE_UP_MS) watch(fid, started)
    }, 2500)
  }, [])

  // Questions still being answered when the page opened.
  useEffect(() => {
    list.filter(f => f.state === 'pending' && !timers.current[f.id]).forEach(f => watch(f.id))
  }, [list, watch])

  const busy = list.some(f => f.state === 'pending')

  const ask = async (question) => {
    const d = await api.post(`/api/bazi/readings/${item.id}/ask`, { question })
    if (!d.ok) throw new Error(d.error || '没有发出去，再试一次。')
    setList(prev => [...prev, d.followup])
    watch(d.followup.id)
  }

  return (
    <>
      <div className="ask-launch">
        <button type="button" className="ask-launch__btn" onClick={() => setOpen(true)}>
          <i className="fas fa-comments" aria-hidden="true" />
          追问<small>就这张命盘接着问</small>
        </button>
        {list.length > 0 && <span className="ask-launch__count">已问 {list.length} 个</span>}
      </div>
      <ReadingChat
        open={open}
        onClose={() => setOpen(false)}
        title="追问"
        subtitle="就这张命盘接着问"
        turns={list}
        onSend={ask}
        busy={busy}
        placeholder="比如：今年适合换工作吗？"
      />
    </>
  )
}

// ── Past charts ────────────────────────────────────────────────────
function History({ items, currentId, onOpen, onHide }) {
  if (!items.length) return <p className="bz-fine">还没有排过盘。</p>
  return (
    <ul className="bz-hist">
      {items.map(h => (
        <li key={h.id} className={`bz-hist__row${h.id === currentId ? ' is-current' : ''}`}>
          <button type="button" className="bz-hist__open" onClick={() => onOpen(h.id)}>
            <span className="bz-hist__bazi">{h.bazi}</span>
            <span className="bz-hist__meta">{h.gender_label} · {h.birth} · {h.place}</span>
            {h.verdict && <span className="bz-hist__verdict">{h.verdict}</span>}
          </button>
          <button type="button" className="bz-hist__del" onClick={() => onHide(h.id)} aria-label={`删除 ${h.bazi}`}>删除</button>
        </li>
      ))}
    </ul>
  )
}

// ── Page ───────────────────────────────────────────────────────────
export default function Bazi() {
  const [tab, setTab] = useState('cast')
  return (
    <div className="bazi">
      <div className="bazi__inner">
        <header className="bazi__head">
          <h1 className="bazi__title">八字<span>BaZi</span></h1>
          <p className="bazi__sub">四柱排盘 · 真太阳时 · 子平法解读</p>
        </header>
        <div className="bz-tabs" role="tablist">
          <button type="button" role="tab" aria-selected={tab === 'cast'}
            className={`bz-tab${tab === 'cast' ? ' is-on' : ''}`} onClick={() => setTab('cast')}>算盘</button>
          <button type="button" role="tab" aria-selected={tab === 'compat'}
            className={`bz-tab${tab === 'compat' ? ' is-on' : ''}`} onClick={() => setTab('compat')}>合盘</button>
        </div>
        {tab === 'cast' ? <BaziCast /> : <BaziCompat />}
      </div>
    </div>
  )
}

function BaziCast() {
  const [cities, setCities] = useState([])
  const [item, setItem] = useState(null)        // { id, chart, state, reading, error }
  const [loading, setLoading] = useState(true)
  const [casting, setCasting] = useState(false)
  const [editing, setEditing] = useState(false)
  const [formInit, setFormInit] = useState(null)
  const [error, setError] = useState('')
  const [history, setHistory] = useState(null)   // null until first opened
  const [showHistory, setShowHistory] = useState(false)
  const pollRef = useRef(null)
  const pollStart = useRef(0)
  const reading = useRef(false)

  useEffect(() => {
    Promise.all([api.get('/api/bazi/cities'), api.get('/api/bazi/latest')]).then(([c, l]) => {
      if (c.ok) setCities(c.cities)
      if (l.ok && l.item) setItem(l.item)
      setLoading(false)
    })
    return () => clearTimeout(pollRef.current)
  }, [])

  const poll = useCallback((rid) => {
    clearTimeout(pollRef.current)
    pollRef.current = setTimeout(async () => {
      const d = await api.get(`/api/bazi/reading/${rid}`)
      if (d.item) setItem(d.item)
      if (d.item?.state === 'pending' && Date.now() - pollStart.current < POLL_GIVE_UP_MS) poll(rid)
    }, POLL_MS)
  }, [])

  // Coming back to a page whose reading is still running picks the poll up again.
  useEffect(() => {
    if (item?.state === 'pending' && !pollRef.current) { pollStart.current = Date.now(); poll(item.id) }
  }, [item, poll])

  const stopPoll = () => { clearTimeout(pollRef.current); pollRef.current = null }

  const loadHistory = async () => {
    const d = await api.get('/api/bazi/readings')
    if (d.ok) setHistory(d.items)
  }
  const toggleHistory = () => {
    if (!showHistory) loadHistory()
    setShowHistory(v => !v)
  }
  const open = async (rid) => {
    const d = await api.get(`/api/bazi/readings/${rid}`)
    if (!d.item) { setError(d.error); return }
    stopPoll()
    setItem(d.item); setEditing(false); setShowHistory(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }
  const hide = async (rid) => {
    if (!window.confirm('从你的记录里删除这张命盘？')) return
    const d = await api.delete(`/api/bazi/readings/${rid}`)
    if (!d.ok) return
    setHistory(h => h.filter(x => x.id !== rid))
    if (item?.id === rid) { stopPoll(); setItem(null) }
  }

  const cast = async (body, raw) => {
    setCasting(true); setError('')
    const d = await api.post('/api/bazi/chart', body)
    setCasting(false)
    if (!d.ok) { setError(d.error); return }
    stopPoll()
    setFormInit(raw)
    setItem({ id: d.reading_id, chart: d.chart, state: 'none', reading: null, verify: {}, followups: [] })
    setEditing(false)
    setHistory(null)
  }

  const read = async () => {
    // Blocks the double-click gap before `item.state` flips to 'pending'.
    if (reading.current) return
    reading.current = true
    setError('')
    try {
      const d = await api.post('/api/bazi/reading', { reading_id: item.id })
      if (!d.ok) { setError(d.error); return }
      setItem(d.item)
      if (d.item.state === 'pending') { pollStart.current = Date.now(); poll(item.id) }
    } finally {
      reading.current = false
    }
  }

  const forget = async () => {
    if (!window.confirm('从你的记录里删除所有命盘、解读和追问？')) return
    const d = await api.delete('/api/bazi/mine')
    if (d.ok) { stopPoll(); setItem(null); setFormInit(null); setHistory([]) }
  }

  if (loading) return <div className="text-center py-5"><HandLoader /></div>

  const showForm = !item || editing
  const err = item?.state === 'error' ? item.error : null

  return (
    <>
      {error && <div className="bz-alert" role="alert">{error}</div>}

      <div className="bz-recast">
        <button type="button" className="bz-btn bz-btn--quiet" onClick={toggleHistory} aria-expanded={showHistory}>
          {showHistory ? '收起历史' : '历史命盘'}
        </button>
        {!showForm && (
          <button type="button" className="bz-btn bz-btn--quiet" onClick={() => setEditing(true)}>新排一盘</button>
        )}
        {editing && item && (
          <button type="button" className="bz-btn bz-btn--quiet" onClick={() => setEditing(false)}>回到命盘</button>
        )}
      </div>
      {showHistory && (
        <section className="bz-histbox" aria-label="历史命盘">
          {history ? <History items={history} currentId={item?.id} onOpen={open} onHide={hide} /> : <HandLoader />}
        </section>
      )}

      {showForm && <CastForm cities={cities} initial={formInit} onCast={cast} busy={casting} />}

      {item && !editing && (
        <>
          <Chart chart={item.chart} />

          {item.state === 'none' && (
            <div className="bz-ask">
              <button type="button" className="bz-btn" onClick={read}>请先生解读</button>
              <p className="bz-fine">命盘（不含姓名）会发给 AI 模型生成解读，大约需要一分钟。</p>
            </div>
          )}
          {item.state === 'pending' && (
            <div className="bz-wait" role="status">
              <span className="bz-wait__brush" aria-hidden="true" />
              正在看盘，大约一分钟。可以先离开，回来结果还在。
            </div>
          )}
          {err && (
            <div className="bz-alert" role="alert">
              {err.error}
              {err.retryable && <button type="button" className="bz-btn bz-btn--quiet" onClick={read}>再试一次</button>}
            </div>
          )}
          {item.state === 'done' && <Reading item={item} chart={item.chart} />}

          <p className="bz-forget">
            <button type="button" onClick={forget}>删除我的全部命盘记录</button>
            <span className="bz-fine">删除后你在这里看不到这些记录；网站会保留存档，用于分析和改进解读。</span>
          </p>
        </>
      )}
    </>
  )
}
