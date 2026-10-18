import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import HandLoader from '../components/HandLoader'

/**
 * 八字合盘 — pick two of your own charts, see how they fit.
 *
 * Left side is 本命 (self), right side is 对象 (other). The arithmetic — day
 * pillar合冲、生肖、五行互补、十神 — comes from the server (compat.py). The
 * reading is the AI layer's (feature bazi_compat), run in the background and
 * polled, same shape as the 八字 reading.
 */

const EL_CLASS = { 木: 'wood', 火: 'fire', 土: 'earth', 金: 'metal', 水: 'water' }
const EL_ORDER = ['木', '火', '土', '金', '水']
const POLL_MS = 3000
const POLL_GIVE_UP_MS = 4 * 60 * 1000

const LEVEL_TINT = {
  上等: 'auspicious', 中上: 'favorable', 中等: 'neutral',
  中下: 'challenging', 下等: 'adverse',
}
const DAY_LEVEL_LABEL = {
  auspicious_pair: '天合地合',
  favorable: '偏合',
  neutral: '中性',
  challenging: '偏克',
  adverse_pair: '天克地冲',
}

function fmtBirth(c) {
  return `${c.gender_label} · ${c.solar_time.clock}`
}

function bazi(c) {
  return c.pillars.map(p => p.gan + p.zhi).join(' ')
}

// ── Chart picker (one side) ───────────────────────────────────────
function PickerCard({ side, label, mine, picked, onPick }) {
  return (
    <div className="bzc-pick">
      <div className="bzc-pick__title">{label}<small>{side === 'self' ? '左' : '右'}</small></div>
      {!mine.length ? (
        <p className="bz-fine">还没有命盘。先去「算盘」排一张。</p>
      ) : (
        <ul className="bzc-pick__list">
          {mine.map(h => (
            <li key={h.id}>
              <button type="button"
                className={`bzc-pick__item${picked === h.id ? ' is-on' : ''}`}
                onClick={() => onPick(h.id)}>
                <span className="bzc-pick__bazi">{h.bazi}</span>
                <span className="bzc-pick__meta">{h.gender_label} · {h.birth}</span>
                <span className="bzc-pick__place">{h.place}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ── Compat result ─────────────────────────────────────────────────
function MiniChart({ chart, label }) {
  return (
    <div className="bzc-mini" aria-label={`${label} 命盘`}>
      <div className="bzc-mini__label">{label}</div>
      <div className="bzc-mini__pillars">
        {chart.pillars.map(p => (
          <div key={p.key} className="bzc-mini__pillar">
            <div className="bzc-mini__col">{p.label}</div>
            <div className={`bzc-mini__gan bz-el--${EL_CLASS[p.gan_element]}`}>{p.gan}</div>
            <div className={`bzc-mini__zhi bz-el--${EL_CLASS[p.zhi_element]}`}>{p.zhi}</div>
          </div>
        ))}
      </div>
      <div className="bzc-mini__sub">{fmtBirth(chart)} · {chart.input.place}</div>
    </div>
  )
}

function ElementsCompare({ elements }) {
  const total_a = EL_ORDER.reduce((s, e) => s + elements.self[e], 0) || 1
  const total_b = EL_ORDER.reduce((s, e) => s + elements.other[e], 0) || 1
  return (
    <div className="bzc-elcmp">
      {EL_ORDER.map(e => {
        const a = elements.self[e] || 0
        const b = elements.other[e] || 0
        const gift_to_self = (elements.gift_to_self || {})[e]
        const gift_to_other = (elements.gift_to_other || {})[e]
        return (
          <div key={e} className="bzc-elrow">
            <div className="bzc-elrow__label">
              <i className={`bz-bg--${EL_CLASS[e]}`} />{e}
            </div>
            <div className="bzc-elrow__bars">
              <div className="bzc-elrow__bar bzc-elrow__bar--self">
                <span className={`bz-bg--${EL_CLASS[e]}`} style={{ width: `${a / total_a * 100}%` }} />
                <small>{Math.round(a / total_a * 100)}%</small>
              </div>
              <div className="bzc-elrow__bar bzc-elrow__bar--other">
                <span className={`bz-bg--${EL_CLASS[e]}`} style={{ width: `${b / total_b * 100}%` }} />
                <small>{Math.round(b / total_b * 100)}%</small>
              </div>
            </div>
            <div className="bzc-elrow__gift">
              {gift_to_self && <span className="bzc-gift bzc-gift--in">对方 → 本命 +{gift_to_self}</span>}
              {gift_to_other && <span className="bzc-gift bzc-gift--out">本命 → 对方 +{gift_to_other}</span>}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function DayPairCard({ day }) {
  const relLabels = { he: '合', ke: '克', liu_he: '六合', chong: '冲', xing: '刑', hai: '害' }
  const relClass = { he: 'good', liu_he: 'good', ke: 'bad', chong: 'bad', xing: 'bad', hai: 'bad' }
  return (
    <div className={`bzc-day bzc-day--${day.level}`}>
      <div className="bzc-day__badge">{DAY_LEVEL_LABEL[day.level] || '日柱配对'}</div>
      <div className="bzc-day__cards">
        <div className="bzc-day__card">
          <div className="bzc-day__who">本命</div>
          <div className="bzc-day__gz">{day.self.label}</div>
        </div>
        <div className="bzc-day__center">
          {[...day.gan_rel, ...day.zhi_rel].map((r, i) => (
            <span key={i} className={`bzc-day__rel bzc-day__rel--${relClass[r.kind] || 'neutral'}`}>
              {relLabels[r.kind] || r.kind}
            </span>
          ))}
          {!day.gan_rel.length && !day.zhi_rel.length && <span className="bzc-day__rel">—</span>}
        </div>
        <div className="bzc-day__card">
          <div className="bzc-day__who">对象</div>
          <div className="bzc-day__gz">{day.other.label}</div>
        </div>
      </div>
      <ul className="bzc-day__details">
        {day.gan_rel.map((r, i) => <li key={`g${i}`}>{r.detail}</li>)}
        {day.zhi_rel.map((r, i) => <li key={`z${i}`}>{r.detail}</li>)}
      </ul>
    </div>
  )
}

function CompatReading({ item }) {
  const r = item.reading
  const compat = item.compat
  const [marks, setMarks] = useState(item.verify || {})
  useEffect(() => { setMarks(item.verify || {}) }, [item.id]) // eslint-disable-line
  const mark = (i, v) => {
    const next = { ...marks }
    if (next[i] === v) delete next[i]; else next[i] = v
    setMarks(next)
    api.put(`/api/bazi/compat/readings/${item.id}/verify`, { marks: next })
  }

  if (!r) {
    return <section className="bz-read"><p className="bz-read__raw">{item.reading_text}</p></section>
  }
  const yes = Object.values(marks).filter(v => v === 'yes').length
  const answered = Object.values(marks).filter(Boolean).length
  const tint = LEVEL_TINT[r.level] || 'neutral'

  return (
    <section className="bz-read bzc-read" aria-label="合盘解读">
      {/* 判词 + 分数 */}
      <div className={`bz-verdict bzc-verdict bzc-verdict--${tint}`}>
        <div className="bz-verdict__seal" aria-label={r.level}>
          <div className="bzc-score">{compat.score}</div>
          <div className="bzc-level">{r.level}</div>
        </div>
        <div>
          <div className="bz-kicker">合盘判词</div>
          <h2 className="bz-verdict__text">{r.verdict}</h2>
          <div className="bz-verdict__pattern">{r.chemistry.summary}</div>
        </div>
      </div>
      <ul className="bz-reasons">
        {r.chemistry.reasons.map((x, i) => <li key={i}>{x}</li>)}
      </ul>

      {/* 日柱配对 */}
      <div className="bz-kicker bz-kicker--rule">日柱配对<small>夫妻宫 · 感情位</small></div>
      <DayPairCard day={compat.day_pair} />
      <p className="bzc-text">{r.day_pillar}</p>

      {/* 五行互补 */}
      <div className="bz-kicker bz-kicker--rule">五行互补</div>
      <ElementsCompare elements={compat.elements} />
      <p className="bzc-text">{r.complement}</p>

      {/* 十神关系 */}
      <div className="bz-kicker bz-kicker--rule">十神<small>对方日干 → 本命 {compat.ten_gods.self_me_label}:{compat.ten_gods.spouse_god}</small></div>
      <div className="bzc-tengods">
        {compat.ten_gods.cells.map((c, i) => (
          <div key={i} className="bzc-tengod">
            <div className="bzc-tengod__pillar">{c.pillar}</div>
            <div className="bzc-tengod__gan">{c.gan}</div>
            <div className="bzc-tengod__god">{c.ten_god}</div>
          </div>
        ))}
      </div>
      <p className="bzc-text">{r.ten_god}</p>

      {/* 生肖 */}
      <div className="bz-kicker bz-kicker--rule">生肖</div>
      <div className="bzc-zodiac">
        <div className="bzc-zodiac__one"><b>{compat.zodiac.self.animal}</b><small>{compat.zodiac.self.zhi}</small></div>
        <div className="bzc-zodiac__rel">
          {compat.zodiac.rel.length ? compat.zodiac.rel.map((r, i) => <span key={i}>{r.detail}</span>) : <span className="bz-fine">生肖中性</span>}
        </div>
        <div className="bzc-zodiac__one"><b>{compat.zodiac.other.animal}</b><small>{compat.zodiac.other.zhi}</small></div>
      </div>
      <p className="bzc-text">{r.zodiac}</p>

      {/* 优势 & 要注意 */}
      <div className="bzc-twocol">
        <div>
          <div className="bz-kicker">优势</div>
          <ul className="bzc-list bzc-list--up">{r.strengths.map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
        <div>
          <div className="bz-kicker">要注意</div>
          <ul className="bzc-list bzc-list--down">{r.cautions.map((x, i) => <li key={i}>{x}</li>)}</ul>
        </div>
      </div>

      {/* 建议 */}
      <div className="bzc-advice">
        <div className="bz-kicker">相处建议</div>
        <p>{r.advice}</p>
      </div>

      {/* 验证 */}
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
      <p className="bz-fine">由 AI 根据程序排出的合盘推演生成，请逐条验证。</p>

      <AskBox item={item} />
    </section>
  )
}

const MAX_Q = 300

function AskBox({ item }) {
  const [list, setList] = useState(item.followups || [])
  const [q, setQ] = useState('')
  const [err, setErr] = useState('')
  const timers = useRef({})
  const sending = useRef(false)

  useEffect(() => { setList(item.followups || []) }, [item.id]) // eslint-disable-line
  useEffect(() => () => Object.values(timers.current).forEach(clearTimeout), [])

  const watch = useCallback((fid, started = Date.now()) => {
    clearTimeout(timers.current[fid])
    timers.current[fid] = setTimeout(async () => {
      const d = await api.get(`/api/bazi/compat/followups/${fid}`)
      if (!d.ok) return
      setList(prev => prev.map(f => (f.id === fid ? d.followup : f)))
      if (d.followup.state === 'pending' && Date.now() - started < POLL_GIVE_UP_MS) watch(fid, started)
    }, 2500)
  }, [])

  useEffect(() => {
    list.filter(f => f.state === 'pending' && !timers.current[f.id]).forEach(f => watch(f.id))
  }, [list, watch])

  const busy = list.some(f => f.state === 'pending')

  const send = async (e) => {
    e.preventDefault()
    const question = q.trim()
    if (!question || sending.current || busy) return
    sending.current = true
    setErr('')
    try {
      const d = await api.post(`/api/bazi/compat/readings/${item.id}/ask`, { question })
      if (!d.ok) { setErr(d.error); return }
      setQ('')
      setList(prev => [...prev, d.followup])
      watch(d.followup.id)
    } finally {
      sending.current = false
    }
  }

  return (
    <div className="bz-askbox">
      <div className="bz-kicker bz-kicker--rule">追问 <small>就这段关系接着问</small></div>
      {list.length > 0 && (
        <ol className="bz-qa" aria-live="polite">
          {list.map(f => (
            <li key={f.id} className="bz-qa__turn">
              <p className="bz-qa__q">{f.question}</p>
              {f.state === 'done' && <p className="bz-qa__a">{f.answer}</p>}
              {f.state === 'pending' && (
                <p className="bz-qa__a bz-qa__a--wait"><span className="bz-wait__brush" aria-hidden="true" />正在看盘…</p>
              )}
              {f.state === 'error' && <p className="bz-qa__a bz-qa__a--err">{f.error?.error || '没有答上来，请换个说法再问一次。'}</p>}
            </li>
          ))}
        </ol>
      )}
      <form className="bz-askbox__form" onSubmit={send}>
        <textarea
          className="bz-askbox__input"
          rows={2}
          maxLength={MAX_Q}
          value={q}
          onChange={e => setQ(e.target.value)}
          placeholder={list.length ? '接着问…' : '比如：我们适合长期走下去吗？冲突主要在哪里？'}
          aria-label="追问合盘"
          onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send(e) }}
        />
        <div className="bz-askbox__row">
          <span className="bz-fine">{q.length}/{MAX_Q}</span>
          <button type="submit" className="bz-btn" disabled={busy || !q.trim()}>{busy ? '作答中…' : '提问'}</button>
        </div>
        {err && <p className="bz-qa__a--err" role="alert">{err}</p>}
      </form>
    </div>
  )
}

// ── History ───────────────────────────────────────────────────────
function History({ items, currentId, onOpen, onHide }) {
  if (!items.length) return <p className="bz-fine">还没有合过盘。</p>
  return (
    <ul className="bz-hist bzc-hist">
      {items.map(h => (
        <li key={h.id} className={`bz-hist__row${h.id === currentId ? ' is-current' : ''}`}>
          <button type="button" className="bz-hist__open" onClick={() => onOpen(h.id)}>
            <span className="bzc-hist__score">{h.score ?? '—'}</span>
            <span className="bz-hist__bazi">{h.self_bazi} ⇔ {h.other_bazi}</span>
            <span className="bz-hist__meta">{h.self_birth} · {h.other_birth}</span>
            {h.verdict && <span className="bz-hist__verdict">{h.verdict}</span>}
          </button>
          <button type="button" className="bz-hist__del" onClick={() => onHide(h.id)} aria-label="删除合盘">删除</button>
        </li>
      ))}
    </ul>
  )
}

// ── Main ──────────────────────────────────────────────────────────
export default function BaziCompat() {
  const [mine, setMine] = useState([])          // 自己历史命盘 list_mine
  const [history, setHistory] = useState(null)  // 合盘历史 (null = 未加载)
  const [showHistory, setShowHistory] = useState(false)
  const [selfId, setSelfId] = useState('')
  const [otherId, setOtherId] = useState('')
  const [item, setItem] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [composing, setComposing] = useState(false)
  const pollRef = useRef(null)
  const pollStart = useRef(0)
  const readingRef = useRef(false)
  const composingRef = useRef(false)

  useEffect(() => {
    api.get('/api/bazi/readings').then(d => {
      if (d.ok) setMine(d.items || [])
      setLoading(false)
    })
    return () => clearTimeout(pollRef.current)
  }, [])

  const poll = useCallback((rid) => {
    clearTimeout(pollRef.current)
    pollRef.current = setTimeout(async () => {
      const d = await api.get(`/api/bazi/compat/reading/${rid}`)
      if (d.item) setItem(d.item)
      if (d.item?.state === 'pending' && Date.now() - pollStart.current < POLL_GIVE_UP_MS) poll(rid)
    }, POLL_MS)
  }, [])

  useEffect(() => {
    if (item?.state === 'pending' && !pollRef.current) { pollStart.current = Date.now(); poll(item.id) }
  }, [item, poll])

  const stopPoll = () => { clearTimeout(pollRef.current); pollRef.current = null }

  const loadHistory = async () => {
    const d = await api.get('/api/bazi/compat/readings')
    if (d.ok) setHistory(d.items)
  }
  const toggleHistory = () => {
    if (!showHistory) loadHistory()
    setShowHistory(v => !v)
  }

  const open = async (rid) => {
    const d = await api.get(`/api/bazi/compat/readings/${rid}`)
    if (!d.item) { setError(d.error); return }
    stopPoll()
    setItem(d.item); setShowHistory(false)
    setSelfId(''); setOtherId('')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const hide = async (rid) => {
    if (!window.confirm('从你的记录里删除这张合盘？')) return
    const d = await api.delete(`/api/bazi/compat/readings/${rid}`)
    if (!d.ok) return
    setHistory(h => h.filter(x => x.id !== rid))
    if (item?.id === rid) { stopPoll(); setItem(null) }
  }

  const compose = async () => {
    if (composingRef.current) return
    if (!selfId || !otherId) { setError('请从两边各选一张命盘。'); return }
    if (selfId === otherId) { setError('两张命盘不能是同一张。'); return }
    composingRef.current = true
    setComposing(true); setError('')
    try {
      const d = await api.post('/api/bazi/compat', { chart_a_id: selfId, chart_b_id: otherId })
      if (!d.ok) { setError(d.error); return }
      stopPoll()
      setItem(d.item)
      setHistory(null)
    } finally {
      setComposing(false)
      composingRef.current = false
    }
  }

  const read = async () => {
    if (readingRef.current) return
    readingRef.current = true
    setError('')
    try {
      const d = await api.post('/api/bazi/compat/reading', { reading_id: item.id })
      if (!d.ok) { setError(d.error); return }
      setItem(d.item)
      if (d.item.state === 'pending') { pollStart.current = Date.now(); poll(item.id) }
    } finally {
      readingRef.current = false
    }
  }

  const forget = async () => {
    if (!window.confirm('从你的记录里删除所有合盘？')) return
    const d = await api.delete('/api/bazi/compat/mine')
    if (d.ok) { stopPoll(); setItem(null); setHistory([]) }
  }

  const reset = () => { stopPoll(); setItem(null); setSelfId(''); setOtherId('') }

  if (loading) return <div className="bazi__inner text-center py-5"><HandLoader /></div>

  const err = item?.state === 'error' ? item.error : null

  return (
    <div className="bzc">
      {error && <div className="bz-alert" role="alert">{error}</div>}

      <div className="bz-recast">
        <button type="button" className="bz-btn bz-btn--quiet" onClick={toggleHistory} aria-expanded={showHistory}>
          {showHistory ? '收起历史' : '历史合盘'}
        </button>
        {item && (
          <button type="button" className="bz-btn bz-btn--quiet" onClick={reset}>重新选盘</button>
        )}
      </div>

      {showHistory && (
        <section className="bz-histbox" aria-label="合盘历史">
          {history ? <History items={history} currentId={item?.id} onOpen={open} onHide={hide} /> : <HandLoader />}
        </section>
      )}

      {/* 选盘态 */}
      {!item && (
        <>
          {mine.length < 2 ? (
            <div className="bzc-empty">
              <p>合盘需要你自己先排过至少两张命盘。</p>
              <p className="bz-fine">去「算盘」排一张自己的，再排一张对象的，再回来合盘。</p>
            </div>
          ) : (
            <>
              <div className="bzc-pickers">
                <PickerCard side="self"  label="本命" mine={mine} picked={selfId}  onPick={setSelfId} />
                <PickerCard side="other" label="对象" mine={mine} picked={otherId} onPick={setOtherId} />
              </div>
              <div className="bzc-compose">
                <button type="button" className="bz-btn" disabled={composing || !selfId || !otherId} onClick={compose}>
                  {composing ? '算合盘中…' : '合盘'}
                </button>
                <p className="bz-fine">点合盘后系统先把两盘的合冲算好，再请先生做整体解读。</p>
              </div>
            </>
          )}
        </>
      )}

      {/* 结果态 */}
      {item && (
        <>
          <div className="bzc-pair">
            <MiniChart chart={item.chart_a} label="本命" />
            <div className="bzc-pair__amp">⇔</div>
            <MiniChart chart={item.chart_b} label="对象" />
          </div>

          {item.state === 'none' && (
            <div className="bz-ask">
              <button type="button" className="bz-btn" onClick={read}>请先生解读</button>
              <p className="bz-fine">两张命盘（不含姓名）会发给 AI 模型生成合盘解读，大约需要一分钟。</p>
            </div>
          )}
          {item.state === 'pending' && (
            <div className="bz-wait" role="status">
              <span className="bz-wait__brush" aria-hidden="true" />
              正在合盘，大约一分钟。可以先离开，回来结果还在。
            </div>
          )}
          {err && (
            <div className="bz-alert" role="alert">
              {err.error}
              {err.retryable && <button type="button" className="bz-btn bz-btn--quiet" onClick={read}>再试一次</button>}
            </div>
          )}
          {item.state === 'done' && <CompatReading item={item} />}

          <p className="bz-forget">
            <button type="button" onClick={forget}>删除我的全部合盘记录</button>
            <span className="bz-fine">删除后你在这里看不到这些记录；网站会保留存档，用于分析和改进解读。</span>
          </p>
        </>
      )}
    </div>
  )
}
