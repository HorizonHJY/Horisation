import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'
import Modal, { ConfirmDialog } from '../components/Modal'
import HandLoader from '../components/HandLoader'

/* FX — 换汇 info board + live USD↔CNY rate.
   The platform does not broker trades — every page surface carries a
   disclaimer. Backend endpoints: /api/fx/{rate,history} and
   /api/market/fx/* for the posts. */

const DIRECTIONS = [
  { value: 'usd_to_cny', label: '我用美元换人民币', short: 'USD → CNY' },
  { value: 'cny_to_usd', label: '我用人民币换美元', short: 'CNY → USD' },
]
const DIR_BY = Object.fromEntries(DIRECTIONS.map(o => [o.value, o]))

const METHODS = [
  { value: 'cash',     label: '当面现金' },
  { value: 'transfer', label: '转账' },
  { value: 'either',   label: '都可以' },
]
const METHOD_BY = Object.fromEntries(METHODS.map(o => [o.value, o]))

const EMPTY_FORM = {
  direction: 'usd_to_cny', amount: '', rate: '',
  method: 'either', area: '', note: '',
}

const fmtMoney = (n, currency) => {
  const v = Number(n) || 0
  const str = currency === 'CNY' ? `¥${v.toFixed(2)}` : `$${v.toFixed(2)}`
  return str
}

const fmtRate = (r) => (r == null || isNaN(Number(r))) ? '—' : Number(r).toFixed(4)

const fmtWhen = (iso) => {
  if (!iso) return ''
  try {
    const d = new Date(iso)
    return d.toLocaleString('zh-CN', { timeZone: 'America/Chicago',
      month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
  } catch { return iso }
}

/* Owner avatar — same shape as Sublets. */
function OwnerAvatar({ username, displayName, avatarUrl, size = 28 }) {
  const style = { width: size, height: size, borderRadius: '50%', flexShrink: 0 }
  const name = displayName || username || '?'
  if (avatarUrl) return <img src={avatarUrl} alt="" style={{ ...style, objectFit: 'cover' }} />
  return (
    <span aria-hidden="true" style={{
      ...style, background: 'var(--accent-text, #6b9cdb)', color: '#fff',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontWeight: 700, fontSize: size * 0.42,
    }}>{name[0]?.toUpperCase() || '?'}</span>
  )
}

/* ── Rate card + converter ──────────────────────────────────────── */
function RateCard({ rate, history, loading, onRefresh }) {
  const [side, setSide] = useState('usd')      // which input the user drives
  const [usd, setUsd]   = useState('100')
  const [cny, setCny]   = useState('')

  const current = rate?.rate
  const stale   = Boolean(rate?.stale)

  useEffect(() => {
    if (!current) return
    if (side === 'usd') {
      const u = Number(usd); setCny(isNaN(u) ? '' : (u * current).toFixed(2))
    } else {
      const c = Number(cny); setUsd(isNaN(c) ? '' : (c / current).toFixed(2))
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current])

  const onUsd = (v) => {
    setSide('usd'); setUsd(v)
    const n = Number(v)
    if (current && !isNaN(n)) setCny((n * current).toFixed(2))
    if (v === '') setCny('')
  }
  const onCny = (v) => {
    setSide('cny'); setCny(v)
    const n = Number(v)
    if (current && !isNaN(n)) setUsd((n / current).toFixed(2))
    if (v === '') setUsd('')
  }

  return (
    <section className="fx-card" aria-label="USD to CNY rate">
      <header className="fx-card__head">
        <div>
          <div className="fx-card__label">美元 → 人民币</div>
          <div className="fx-card__rate">
            {current ? current.toFixed(4) : (loading ? '…' : '—')}
            <small className="fx-card__unit"> CNY / USD</small>
          </div>
          <div className="fx-card__meta">
            {rate?.ok ? (
              <>
                数据源:Frankfurter / 欧洲央行
                {rate.date && <> · 牌价日期 {rate.date}</>}
                {rate.updated_at && <> · 拉取时间 {fmtWhen(rate.updated_at)}</>}
                {stale && <span className="fx-card__stale"> · 暂用缓存值</span>}
              </>
            ) : (loading ? '加载中…' : '实时汇率暂不可用,请稍后再试。')}
          </div>
        </div>
        <button
          type="button"
          className="btn btn-outline-secondary btn-sm fx-card__refresh"
          onClick={onRefresh} disabled={loading}
          aria-label="刷新汇率"
        >
          <i className="fas fa-rotate" aria-hidden="true" /> 刷新
        </button>
      </header>

      <div className="fx-converter" role="group" aria-label="双向换算器">
        <div className="fx-converter__box">
          <label className="fx-converter__lab" htmlFor="fx-usd">美元</label>
          <div className="fx-converter__input">
            <span className="fx-converter__sym">$</span>
            <input id="fx-usd" type="number" inputMode="decimal" min="0"
                   value={usd} onChange={e => onUsd(e.target.value)}
                   placeholder="0.00" disabled={!current} />
            <span className="fx-converter__cur">USD</span>
          </div>
        </div>
        <div className="fx-converter__sep" aria-hidden="true">
          <i className="fas fa-right-left" />
        </div>
        <div className="fx-converter__box">
          <label className="fx-converter__lab" htmlFor="fx-cny">人民币</label>
          <div className="fx-converter__input">
            <span className="fx-converter__sym">¥</span>
            <input id="fx-cny" type="number" inputMode="decimal" min="0"
                   value={cny} onChange={e => onCny(e.target.value)}
                   placeholder="0.00" disabled={!current} />
            <span className="fx-converter__cur">CNY</span>
          </div>
        </div>
      </div>

      <TrendSpark points={history?.points} loading={loading} />
    </section>
  )
}

/* Tiny inline SVG spark-line for the last ~7 days — no deps.
   Y axis uses a "nice" padded range so the min/mid/max ticks stay stable
   when rates jitter by a few basis points. The SVG uses preserveAspectRatio
   via width:100% so it fits any container down to ~300px wide. */
function TrendSpark({ points, loading }) {
  if (!points || points.length < 2) {
    return (
      <div className="fx-spark fx-spark--empty" aria-hidden="true">
        {loading ? '走势加载中…' : '暂无走势数据'}
      </div>
    )
  }

  // Viewbox and padding. Left padding leaves room for Y tick labels (e.g. "7.12").
  const W = 360, H = 90
  const PAD_T = 14, PAD_B = 18, PAD_L = 36, PAD_R = 10
  const plotW = W - PAD_L - PAD_R
  const plotH = H - PAD_T - PAD_B

  // Build a padded, two-decimal axis. If min==max we invent a small band so the
  // line sits on the mid-tick instead of the bottom edge.
  const values = points.map(p => p.rate)
  const rawMin = Math.min(...values), rawMax = Math.max(...values)
  let lo, hi
  if (rawMin === rawMax) {
    lo = +(rawMin - 0.02).toFixed(2)
    hi = +(rawMax + 0.02).toFixed(2)
  } else {
    const span = rawMax - rawMin
    const pad  = Math.max(span * 0.15, 0.01)
    // Round down/up to 2 decimals so tick labels align with the grid.
    lo = Math.floor((rawMin - pad) * 100) / 100
    hi = Math.ceil((rawMax + pad) * 100) / 100
    if (hi === lo) hi = lo + 0.01
  }
  const mid = +((lo + hi) / 2).toFixed(2)
  const ticks = [hi, mid, lo]  // top -> bottom

  const yFor = (v) => PAD_T + (1 - (v - lo) / (hi - lo)) * plotH
  const xStep = plotW / (points.length - 1)
  const coords = points.map((p, i) => [PAD_L + i * xStep, yFor(p.rate)])
  const d = coords.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const area = `${d} L${(PAD_L + plotW).toFixed(1)},${(PAD_T + plotH).toFixed(1)} L${PAD_L.toFixed(1)},${(PAD_T + plotH).toFixed(1)} Z`

  const first = points[0], last = points[points.length - 1]
  const trendUp = last.rate >= first.rate
  const strokeC = trendUp ? 'var(--fx-up, #d26e5a)' : 'var(--fx-down, #4f8f5b)'

  return (
    <div className="fx-spark" aria-label="近 7 天 USD→CNY 走势">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-hidden="true"
           style={{ width: '100%', height: 'auto', display: 'block' }}>
        {/* Unit label, top-right */}
        <text x={W - PAD_R} y="10" textAnchor="end"
              fontSize="9" fill="currentColor" opacity="0.55">CNY / USD</text>

        {/* Grid + Y ticks */}
        {ticks.map((t) => {
          const y = yFor(t)
          return (
            <g key={t}>
              <line x1={PAD_L} x2={W - PAD_R} y1={y} y2={y}
                    stroke="currentColor" strokeWidth="0.5" opacity="0.12" />
              <text x={PAD_L - 4} y={y + 3} textAnchor="end"
                    fontSize="9" fill="currentColor" opacity="0.7">
                {t.toFixed(2)}
              </text>
            </g>
          )
        })}

        {/* Line + fill */}
        <path d={area} fill={strokeC} opacity="0.08" />
        <path d={d} fill="none" stroke={strokeC} strokeWidth="1.8"
              strokeLinecap="round" strokeLinejoin="round" />
        {coords.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={i === coords.length - 1 ? 2.8 : 1.6}
                  fill={strokeC} />
        ))}
      </svg>
      <div className="fx-spark__legend">
        <span>{first.date}:{first.rate.toFixed(4)}</span>
        <span>{last.date}:{last.rate.toFixed(4)}</span>
      </div>
    </div>
  )
}

/* ── Disclaimer ────────────────────────────────────────────────── */
function Disclaimer({ compact = false }) {
  if (compact) {
    return (
      <p className="fx-disclaimer fx-disclaimer--compact small">
        <i className="fas fa-triangle-exclamation" aria-hidden="true" />
        {' '}发布即表示你已知晓本平台不参与交易,仅提供信息发布。
      </p>
    )
  }
  return (
    <div className="fx-disclaimer" role="note">
      <i className="fas fa-triangle-exclamation" aria-hidden="true" />
      <div>
        <strong>本平台只提供信息发布,不参与任何换汇交易,也不担保对方身份与资金安全。</strong>
        {' '}请自行核实对方身份,风险自担。
      </div>
    </div>
  )
}

/* ── Post card ─────────────────────────────────────────────────── */
function FxCard({ p, onOpen }) {
  const dir  = DIR_BY[p.direction] || DIRECTIONS[0]
  const meth = METHOD_BY[p.method] || METHODS[2]
  const amountCurrency = p.direction === 'usd_to_cny' ? 'USD' : 'CNY'
  const isDone = p.status === 'done'
  return (
    <button
      type="button"
      className={`fx-post${isDone ? ' fx-post--done' : ''}`}
      onClick={() => onOpen(p)}
      aria-label={`打开换汇帖:${dir.label} ${fmtMoney(p.amount, amountCurrency)}`}
    >
      <span className="fx-post__dir">
        <i className={`fas ${p.direction === 'usd_to_cny' ? 'fa-arrow-right' : 'fa-arrow-left'}`} aria-hidden="true" />
        {dir.short}
      </span>
      <span className="fx-post__amount">{fmtMoney(p.amount, amountCurrency)}</span>
      <span className="fx-post__rate">
        {p.rate != null ? <>汇率 <strong>{fmtRate(p.rate)}</strong></> : <em>按实时汇率</em>}
      </span>
      <span className="fx-post__meta">
        <span className="fx-pill">{meth.label}</span>
        {p.area && <span className="fx-pill fx-pill--soft">{p.area}</span>}
        {isDone && <span className="fx-pill fx-pill--warn">已完成</span>}
      </span>
      <span className="fx-post__foot">
        <OwnerAvatar
          username={p.owner_username}
          displayName={p.owner_display}
          avatarUrl={p.owner_avatar}
          size={22}
        />
        <span className="fx-post__name">{p.owner_display || p.owner_username}</span>
        <span className="fx-post__time">{fmtWhen(p.created_at)}</span>
      </span>
    </button>
  )
}

/* ── Detail modal ──────────────────────────────────────────────── */
function FxDetailModal({ post, me, onClose, onEdit, onDone, onReopen, onDelete }) {
  const navigate = useNavigate()
  const dir  = DIR_BY[post.direction] || DIRECTIONS[0]
  const meth = METHOD_BY[post.method] || METHODS[2]
  const amountCurrency = post.direction === 'usd_to_cny' ? 'USD' : 'CNY'
  const isMine = me && me === post.owner_username
  const isDone = post.status === 'done'
  const dm = () => {
    onClose()
    navigate('/friends', { state: { openChat: post.owner_username } })
  }
  return (
    <Modal onClose={onClose} title="换汇详情" size="modal-lg">
      {({ titleId }) => (
        <>
          <div className="modal-header">
            <h5 className="modal-title fw-semibold" id={titleId}>换汇详情</h5>
            <button type="button" className="btn-close" aria-label="Close" onClick={onClose} />
          </div>
          <div className="modal-body">
            <div className="fx-detail__head">
              <span className="fx-detail__dir">{dir.label}</span>
              <span className="fx-detail__amount">{fmtMoney(post.amount, amountCurrency)}</span>
              {isDone && <span className="fx-pill fx-pill--warn">已完成</span>}
            </div>
            <div className="fx-detail__grid">
              <div><strong>方向</strong><span>{dir.short}</span></div>
              <div><strong>金额</strong><span>{fmtMoney(post.amount, amountCurrency)}</span></div>
              <div><strong>期望汇率</strong><span>{post.rate != null ? fmtRate(post.rate) : '按实时汇率'}</span></div>
              <div><strong>方式</strong><span>{meth.label}</span></div>
              <div><strong>区域</strong><span>{post.area || '—'}</span></div>
              <div><strong>发布时间</strong><span>{fmtWhen(post.created_at)}</span></div>
            </div>
            {post.note && (
              <>
                <hr />
                <p className="fx-detail__note">{post.note}</p>
              </>
            )}
            <hr />
            <div className="d-flex align-items-center gap-2">
              <OwnerAvatar
                username={post.owner_username}
                displayName={post.owner_display}
                avatarUrl={post.owner_avatar} size={34}
              />
              <div>
                <div style={{ fontWeight: 600 }}>{post.owner_display || post.owner_username}</div>
                <Link to={`/u/${post.owner_username}`} style={{ fontSize: '.8rem' }}>
                  @{post.owner_username}
                </Link>
              </div>
            </div>
            <p className="fx-disclaimer fx-disclaimer--inline small mt-3">
              <i className="fas fa-circle-info" aria-hidden="true" />
              {' '}平台不参与交易,也不担保对方身份。请自行核实,风险自担。
            </p>
          </div>
          <div className="modal-footer" style={{ gap: 8 }}>
            {isMine ? (
              <>
                {!isDone && (
                  <>
                    <button className="btn btn-outline-primary btn-sm"
                            onClick={() => { onEdit(post); onClose() }}>
                      <i className="fas fa-pen me-1" aria-hidden="true" />编辑
                    </button>
                    <button className="btn btn-outline-success btn-sm"
                            onClick={() => { onDone(post); onClose() }}>
                      <i className="fas fa-check me-1" aria-hidden="true" />标记完成
                    </button>
                  </>
                )}
                {isDone && (
                  <button className="btn btn-outline-secondary btn-sm"
                          onClick={() => { onReopen(post); onClose() }}>
                    <i className="fas fa-rotate-left me-1" aria-hidden="true" />重新开放
                  </button>
                )}
                <button className="btn btn-outline-danger btn-sm" onClick={() => onDelete(post)}>
                  <i className="fas fa-trash me-1" aria-hidden="true" />删除
                </button>
              </>
            ) : (
              me ? (
                <button className="btn btn-primary btn-sm" onClick={dm}>
                  <i className="fas fa-message me-1" aria-hidden="true" />私信发布者
                </button>
              ) : (
                <Link to="/login" className="btn btn-sm btn-primary">登录后私信</Link>
              )
            )}
          </div>
        </>
      )}
    </Modal>
  )
}

/* ── Post / Edit form ──────────────────────────────────────────── */
function FxForm({ initial, onSubmit, onCancel, busy }) {
  const [f, setF] = useState(initial || EMPTY_FORM)
  const [err, setErr] = useState('')
  const set = (k) => (e) => setF(p => ({ ...p, [k]: e.target.value }))

  const submit = (e) => {
    e.preventDefault(); setErr('')
    const amount = Number(f.amount)
    if (!(amount > 0)) return setErr('金额必须大于 0。')
    if (f.rate !== '' && f.rate != null) {
      const r = Number(f.rate)
      if (!(r > 0)) return setErr('汇率必须大于 0,留空表示按实时汇率。')
    }
    if ((f.note || '').length > 200) return setErr('备注不能超过 200 字。')

    onSubmit({
      direction: f.direction,
      amount,
      rate: (f.rate === '' || f.rate == null) ? null : Number(f.rate),
      method: f.method,
      area: (f.area || '').trim(),
      note: (f.note || '').trim(),
    })
  }

  const amountCurrency = f.direction === 'usd_to_cny' ? 'USD' : 'CNY'
  const amountSym      = f.direction === 'usd_to_cny' ? '$' : '¥'

  return (
    <form className="fx-form" onSubmit={submit}>
      {err && <div className="alert alert-danger py-2 small" role="alert">{err}</div>}
      <div className="mb-3">
        <label className="form-label">方向</label>
        <div className="fx-form__dir">
          {DIRECTIONS.map(d => (
            <label className={`fx-form__dirbtn${f.direction === d.value ? ' is-active' : ''}`} key={d.value}>
              <input type="radio" name="fx-dir" value={d.value}
                     checked={f.direction === d.value}
                     onChange={() => setF(p => ({ ...p, direction: d.value }))} />
              <span>{d.label}</span>
              <small>{d.short}</small>
            </label>
          ))}
        </div>
      </div>
      <div className="row g-3">
        <div className="col-6">
          <label className="form-label">金额 ({amountCurrency})</label>
          <div className="fx-converter__input">
            <span className="fx-converter__sym">{amountSym}</span>
            <input className="form-control" type="number" min="0" step="0.01" required
                   value={f.amount} onChange={set('amount')} placeholder="1000" />
            <span className="fx-converter__cur">{amountCurrency}</span>
          </div>
        </div>
        <div className="col-6">
          <label className="form-label">期望汇率 <small className="text-muted">(留空=按实时)</small></label>
          <input className="form-control" type="number" min="0" step="0.0001"
                 value={f.rate} onChange={set('rate')} placeholder="7.1000" />
        </div>
      </div>
      <div className="row g-3 mt-1">
        <div className="col-6">
          <label className="form-label">方式</label>
          <select className="form-select" value={f.method} onChange={set('method')}>
            {METHODS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
        </div>
        <div className="col-6">
          <label className="form-label">区域 <small className="text-muted">(可选)</small></label>
          <input className="form-control" maxLength={100}
                 value={f.area} onChange={set('area')} placeholder="e.g. Chicago / Loop" />
        </div>
      </div>
      <div className="mb-3 mt-3">
        <label className="form-label">备注 <small className="text-muted">(≤ 200 字,可选)</small></label>
        <textarea className="form-control" rows={3} maxLength={200}
                  value={f.note} onChange={set('note')}
                  placeholder="时间、首选付款方式、能走多少等……" />
      </div>
      <Disclaimer compact />
      <div className="d-flex gap-2 justify-content-end mt-3">
        {onCancel && (
          <button type="button" className="btn btn-outline-secondary" onClick={onCancel} disabled={busy}>
            取消
          </button>
        )}
        <button type="submit" className="btn btn-primary" disabled={busy}>
          {busy ? '保存中…' : '发布'}
        </button>
      </div>
    </form>
  )
}

/* ── Page ──────────────────────────────────────────────────────── */
export default function Fx() {
  const { user, requireAuth } = useAuth()
  const me = user?.username

  const SUB_TABS = useMemo(() => [
    { key: 'browse', label: 'Browse 浏览' },
    user && { key: 'mine',   label: 'My Posts 我的' },
    user && { key: 'create', label: 'Post 发布' },
  ].filter(Boolean), [user])

  const [subTab, setSubTab]       = useState('browse')
  const [rate, setRate]           = useState(null)
  const [history, setHistory]     = useState(null)
  const [rateLoading, setRL]      = useState(true)
  const [items, setItems]         = useState([])
  const [myItems, setMyItems]     = useState([])
  const [loading, setLoading]     = useState(true)
  const [posting, setPosting]     = useState(false)
  const [error, setError]         = useState('')
  const [detail, setDetail]       = useState(null)
  const [editing, setEditing]     = useState(null)
  const [confirmDel, setConfirmDel] = useState(null)

  const loadRate = useCallback(async () => {
    setRL(true)
    const [r, h] = await Promise.all([
      api.get('/api/fx/rate'),
      api.get('/api/fx/history'),
    ])
    setRate(r); setHistory(h); setRL(false)
  }, [])

  const loadBrowse = useCallback(async () => {
    setLoading(true); setError('')
    const d = await api.get('/api/market/fx')
    if (d.ok) setItems(d.items || []); else setError(d.error || '加载失败')
    setLoading(false)
  }, [])

  const loadMine = useCallback(async () => {
    if (!user) return
    setLoading(true); setError('')
    const d = await api.get('/api/market/fx/mine')
    if (d.ok) setMyItems(d.items || []); else setError(d.error || '加载失败')
    setLoading(false)
  }, [user])

  useEffect(() => { loadRate() }, [loadRate])
  useEffect(() => {
    if (subTab === 'browse') loadBrowse()
    if (subTab === 'mine')   loadMine()
  }, [subTab, loadBrowse, loadMine])

  const submitNew = async (payload) => {
    if (!requireAuth('发布换汇信息')) return
    setPosting(true); setError('')
    const d = await api.post('/api/market/fx', payload)
    setPosting(false)
    if (!d.ok) { setError(d.error || '发布失败'); return }
    setSubTab('mine'); loadMine()
  }

  const saveEdit = async (payload) => {
    if (!editing) return
    setPosting(true); setError('')
    const d = await api.put(`/api/market/fx/${editing.id}`, payload)
    setPosting(false)
    if (!d.ok) { setError(d.error || '保存失败'); return }
    setEditing(null); loadMine(); if (subTab === 'browse') loadBrowse()
  }

  const markDone = async (p) => {
    const d = await api.post(`/api/market/fx/${p.id}/done`)
    if (!d.ok) { setError(d.error); return }
    loadMine(); if (subTab === 'browse') loadBrowse()
  }
  const reopen = async (p) => {
    const d = await api.post(`/api/market/fx/${p.id}/reopen`)
    if (!d.ok) { setError(d.error); return }
    loadMine(); if (subTab === 'browse') loadBrowse()
  }
  const askDelete = (p) => { setDetail(null); setConfirmDel(p) }
  const doDelete = async () => {
    const p = confirmDel; setConfirmDel(null)
    if (!p) return
    const d = await api.delete(`/api/market/fx/${p.id}`)
    if (!d.ok) { setError(d.error); return }
    loadMine(); if (subTab === 'browse') loadBrowse()
  }

  const list = subTab === 'mine' ? myItems : items

  return (
    <div className="fx-page container py-4">
      <header className="fx-header mb-3">
        <h2 className="fx-title">FX <small className="text-muted">换汇</small></h2>
        <p className="fx-sub text-muted">
          实时汇率 + 换汇信息板。发布你的换汇意向,直接私信联系对方。
        </p>
      </header>

      <RateCard rate={rate} history={history} loading={rateLoading} onRefresh={loadRate} />

      <Disclaimer />

      <div className="radio-inputs my-4">
        {SUB_TABS.map(t => (
          <label className="radio" key={t.key}>
            <input type="radio" name="fx-subtab"
                   checked={subTab === t.key}
                   onChange={() => setSubTab(t.key)} />
            <span className="name">{t.label}</span>
          </label>
        ))}
      </div>

      {error && <div className="alert alert-danger py-2" role="alert">{error}</div>}

      {subTab === 'create' ? (
        <FxForm onSubmit={submitNew} busy={posting} onCancel={() => setSubTab('browse')} />
      ) : loading ? (
        <div className="text-center py-5"><HandLoader /></div>
      ) : list.length === 0 ? (
        <div className="fx-empty text-center py-5">
          <i className="fas fa-money-bill-transfer fa-2x mb-3 text-muted" aria-hidden="true" />
          <p className="text-muted mb-3">
            {subTab === 'mine' ? '你还没有发布任何换汇信息。' : '暂无换汇帖。'}
          </p>
          {user && (
            <button className="btn btn-primary" onClick={() => setSubTab('create')}>
              发布换汇
            </button>
          )}
          {!user && <Link to="/login" className="btn btn-primary">登录后发布</Link>}
        </div>
      ) : (
        <div className="fx-grid">
          {list.map(p => <FxCard key={p.id} p={p} onOpen={setDetail} />)}
        </div>
      )}

      {detail && (
        <FxDetailModal
          post={detail} me={me}
          onClose={() => setDetail(null)}
          onEdit={(p) => { setDetail(null); setEditing(p) }}
          onDone={markDone} onReopen={reopen} onDelete={askDelete}
        />
      )}

      {editing && (
        <Modal onClose={() => setEditing(null)} title="编辑换汇" size="modal-lg">
          {({ titleId }) => (
            <>
              <div className="modal-header">
                <h5 className="modal-title fw-semibold" id={titleId}>编辑换汇</h5>
                <button type="button" className="btn-close" aria-label="Close" onClick={() => setEditing(null)} />
              </div>
              <div className="modal-body">
                <FxForm
                  initial={{
                    direction: editing.direction,
                    amount: String(editing.amount ?? ''),
                    rate:   editing.rate == null ? '' : String(editing.rate),
                    method: editing.method,
                    area:   editing.area || '',
                    note:   editing.note || '',
                  }}
                  onSubmit={saveEdit} busy={posting}
                  onCancel={() => setEditing(null)}
                />
              </div>
            </>
          )}
        </Modal>
      )}

      {confirmDel && (
        <ConfirmDialog
          title="删除这条换汇?"
          itemName={`${DIR_BY[confirmDel.direction]?.short || ''} ${fmtMoney(confirmDel.amount, confirmDel.direction === 'usd_to_cny' ? 'USD' : 'CNY')}`}
          message="删除后无法恢复。"
          confirmLabel="删除"
          destructive
          onConfirm={doDelete}
          onClose={() => setConfirmDel(null)}
        />
      )}
    </div>
  )
}
