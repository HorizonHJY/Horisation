import React, { useState, useEffect, useCallback } from 'react'

// ── localStorage helpers ──────────────────────────────────────────────────────
const LS_KEY = 'horisation_bills'

function loadBills() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || [] } catch { return [] }
}
function saveBills(bills) {
  localStorage.setItem(LS_KEY, JSON.stringify(bills))
}
function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
}

// ── Settlement algorithm ──────────────────────────────────────────────────────
// Returns [{from, to, amount}] with minimum transactions.
function calcSettlement(participants, expenses) {
  const net = {}
  participants.forEach(p => (net[p] = 0))

  expenses.forEach(({ paidBy, amount, splitAmong }) => {
    const amt = parseFloat(amount) || 0
    const among = splitAmong.filter(p => participants.includes(p))
    if (!among.length) return
    const share = amt / among.length
    net[paidBy] = (net[paidBy] || 0) + amt
    among.forEach(p => { net[p] = (net[p] || 0) - share })
  })

  // Greedy min-transactions
  const creditors = []
  const debtors   = []
  Object.entries(net).forEach(([name, bal]) => {
    const b = Math.round(bal * 100) / 100
    if (b > 0.01)  creditors.push({ name, bal: b })
    if (b < -0.01) debtors.push({ name, bal: b })
  })
  creditors.sort((a, b) => b.bal - a.bal)
  debtors.sort((a, b) => a.bal - b.bal)

  const txns = []
  let ci = 0, di = 0
  while (ci < creditors.length && di < debtors.length) {
    const c = creditors[ci]
    const d = debtors[di]
    const transfer = Math.min(c.bal, -d.bal)
    txns.push({ from: d.name, to: c.name, amount: Math.round(transfer * 100) / 100 })
    c.bal -= transfer
    d.bal += transfer
    if (Math.abs(c.bal) < 0.01) ci++
    if (Math.abs(d.bal) < 0.01) di++
  }
  return txns
}

// ── Sub-components ────────────────────────────────────────────────────────────
function Toast({ msg, type }) {
  if (!msg) return null
  return (
    <div
      className={`alert alert-${type} position-fixed top-0 end-0 m-3 shadow`}
      style={{ zIndex: 9999, minWidth: 220, fontSize: '.875rem' }}
    >
      {msg}
    </div>
  )
}

function CurrencyBadge({ amount, positive }) {
  const color = positive ? '#10b981' : '#ef4444'
  return (
    <span style={{ fontFamily: 'monospace', color, fontWeight: 700 }}>
      {positive ? '+' : ''}{Number(amount).toFixed(2)}
    </span>
  )
}

// ── Participants panel ────────────────────────────────────────────────────────
function ParticipantsPanel({ bill, onChange }) {
  const [name, setName] = useState('')

  function add() {
    const n = name.trim()
    if (!n || bill.participants.includes(n)) return
    onChange({ ...bill, participants: [...bill.participants, n] })
    setName('')
  }

  function remove(p) {
    // Guard: don't remove if used in expenses
    const used = bill.expenses.some(e => e.paidBy === p || e.splitAmong.includes(p))
    if (used) {
      alert(`"${p}" 已在账单记录中，无法删除。`)
      return
    }
    onChange({ ...bill, participants: bill.participants.filter(x => x !== p) })
  }

  return (
    <div>
      <div className="d-flex gap-2 mb-3">
        <input
          className="form-control form-control-sm"
          placeholder="添加成员姓名…"
          value={name}
          onChange={e => setName(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && add()}
          style={{ maxWidth: 220 }}
        />
        <button className="btn btn-sm btn-primary" onClick={add}>
          <i className="fas fa-plus me-1" />添加
        </button>
      </div>

      {bill.participants.length === 0 ? (
        <p className="text-muted small">还没有成员，先添加几个人吧。</p>
      ) : (
        <div className="d-flex flex-wrap gap-2">
          {bill.participants.map(p => (
            <span key={p} className="badge d-flex align-items-center gap-1"
              style={{ background: '#e2e8f0', color: '#334155', fontSize: '.85rem', padding: '6px 10px', borderRadius: 20 }}>
              <i className="fas fa-user" style={{ fontSize: '.7rem' }} />
              {p}
              <button
                onClick={() => remove(p)}
                className="btn-close btn-close-sm ms-1"
                style={{ fontSize: '.5rem', opacity: 0.5 }}
              />
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Expense form ──────────────────────────────────────────────────────────────
function ExpenseForm({ participants, onAdd, onCancel }) {
  const [form, setForm] = useState({
    desc: '', amount: '', paidBy: participants[0] || '', splitAmong: [...participants]
  })

  function toggleSplit(p) {
    setForm(f => ({
      ...f,
      splitAmong: f.splitAmong.includes(p)
        ? f.splitAmong.filter(x => x !== p)
        : [...f.splitAmong, p]
    }))
  }

  function selectAll() { setForm(f => ({ ...f, splitAmong: [...participants] })) }
  function clearAll()  { setForm(f => ({ ...f, splitAmong: [] })) }

  function submit() {
    if (!form.desc.trim()) return alert('请填写描述')
    const amt = parseFloat(form.amount)
    if (!amt || amt <= 0) return alert('请填写有效金额')
    if (!form.paidBy) return alert('请选择付款人')
    if (!form.splitAmong.length) return alert('请至少选择一位分摊人')
    onAdd({ id: uid(), ...form, amount: amt })
  }

  return (
    <div className="card border-0 shadow-sm mb-3" style={{ borderRadius: 12 }}>
      <div className="card-body">
        <div className="row g-2 mb-2">
          <div className="col-7">
            <label className="form-label small fw-semibold mb-1">描述</label>
            <input className="form-control form-control-sm" placeholder="如：晚饭、门票…"
              value={form.desc} onChange={e => setForm(f => ({ ...f, desc: e.target.value }))} />
          </div>
          <div className="col-5">
            <label className="form-label small fw-semibold mb-1">金额 ($)</label>
            <input className="form-control form-control-sm" type="number" min="0" step="0.01" placeholder="0.00"
              value={form.amount} onChange={e => setForm(f => ({ ...f, amount: e.target.value }))} />
          </div>
        </div>

        <div className="mb-2">
          <label className="form-label small fw-semibold mb-1">付款人</label>
          <select className="form-select form-select-sm"
            value={form.paidBy} onChange={e => setForm(f => ({ ...f, paidBy: e.target.value }))}>
            {participants.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <div className="mb-3">
          <div className="d-flex align-items-center gap-2 mb-1">
            <label className="form-label small fw-semibold mb-0">分摊人</label>
            <button className="btn btn-link btn-sm p-0 text-decoration-none" style={{ fontSize: '.75rem' }} onClick={selectAll}>全选</button>
            <button className="btn btn-link btn-sm p-0 text-decoration-none" style={{ fontSize: '.75rem' }} onClick={clearAll}>清空</button>
          </div>
          <div className="d-flex flex-wrap gap-2">
            {participants.map(p => (
              <label key={p} className="d-flex align-items-center gap-1 small" style={{ cursor: 'pointer' }}>
                <input type="checkbox" checked={form.splitAmong.includes(p)}
                  onChange={() => toggleSplit(p)} />
                {p}
              </label>
            ))}
          </div>
        </div>

        <div className="d-flex gap-2">
          <button className="btn btn-primary btn-sm" onClick={submit}>
            <i className="fas fa-check me-1" />确认添加
          </button>
          <button className="btn btn-outline-secondary btn-sm" onClick={onCancel}>取消</button>
        </div>
      </div>
    </div>
  )
}

// ── Expenses panel ────────────────────────────────────────────────────────────
function ExpensesPanel({ bill, onChange }) {
  const [adding, setAdding] = useState(false)

  function addExpense(expense) {
    onChange({ ...bill, expenses: [...bill.expenses, expense] })
    setAdding(false)
  }

  function deleteExpense(id) {
    onChange({ ...bill, expenses: bill.expenses.filter(e => e.id !== id) })
  }

  const total = bill.expenses.reduce((s, e) => s + (parseFloat(e.amount) || 0), 0)

  return (
    <div>
      {bill.participants.length < 2 && (
        <div className="alert alert-warning py-2 small mb-3">请先在"成员"标签里添加至少 2 位成员。</div>
      )}

      <div className="d-flex justify-content-between align-items-center mb-3">
        <span className="text-muted small">
          共 {bill.expenses.length} 笔 · 总计{' '}
          <strong style={{ color: '#1a1a1a' }}>${total.toFixed(2)}</strong>
        </span>
        {bill.participants.length >= 2 && !adding && (
          <button className="btn btn-sm btn-primary" onClick={() => setAdding(true)}>
            <i className="fas fa-plus me-1" />记一笔
          </button>
        )}
      </div>

      {adding && (
        <ExpenseForm participants={bill.participants} onAdd={addExpense} onCancel={() => setAdding(false)} />
      )}

      {bill.expenses.length === 0 ? (
        <p className="text-muted small">还没有账单记录。</p>
      ) : (
        <div className="d-flex flex-column gap-2">
          {bill.expenses.map(e => {
            const share = (parseFloat(e.amount) / (e.splitAmong.length || 1)).toFixed(2)
            return (
              <div key={e.id} className="card border-0 shadow-sm" style={{ borderRadius: 10 }}>
                <div className="card-body py-2 px-3 d-flex align-items-start gap-3">
                  <div style={{ flex: 1 }}>
                    <div className="d-flex align-items-center gap-2">
                      <strong style={{ fontSize: '.9rem' }}>{e.desc}</strong>
                    </div>
                    <div className="text-muted small mt-1">
                      <span className="me-2"><i className="fas fa-user me-1" />{e.paidBy} 付款</span>
                      <span><i className="fas fa-users me-1" />{e.splitAmong.join('、')} 分摊（每人 ${share}）</span>
                    </div>
                  </div>
                  <div className="d-flex flex-column align-items-end">
                    <strong style={{ fontSize: '1rem', color: '#3b82f6' }}>${parseFloat(e.amount).toFixed(2)}</strong>
                    <button className="btn btn-link btn-sm p-0 text-danger" style={{ fontSize: '.75rem' }}
                      onClick={() => deleteExpense(e.id)}>
                      <i className="fas fa-trash" />
                    </button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Settlement panel ──────────────────────────────────────────────────────────
function SettlementPanel({ bill }) {
  if (bill.expenses.length === 0) {
    return <p className="text-muted small">添加账单记录后才能计算结算方案。</p>
  }

  const txns = calcSettlement(bill.participants, bill.expenses)

  // Also compute per-person summary
  const net = {}
  bill.participants.forEach(p => (net[p] = 0))
  bill.expenses.forEach(({ paidBy, amount, splitAmong }) => {
    const amt = parseFloat(amount) || 0
    const among = splitAmong.filter(p => bill.participants.includes(p))
    if (!among.length) return
    const share = amt / among.length
    net[paidBy] += amt
    among.forEach(p => { net[p] -= share })
  })

  return (
    <div>
      {/* Per-person balance */}
      <h6 className="fw-semibold mb-2" style={{ fontSize: '.85rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '.06em' }}>
        各人余额
      </h6>
      <div className="d-flex flex-wrap gap-2 mb-4">
        {bill.participants.map(p => {
          const bal = Math.round(net[p] * 100) / 100
          return (
            <div key={p} className="card border-0 shadow-sm text-center"
              style={{ borderRadius: 10, minWidth: 100, padding: '8px 14px' }}>
              <div style={{ fontSize: '.8rem', color: '#64748b' }}>{p}</div>
              <CurrencyBadge amount={Math.abs(bal)} positive={bal >= 0} />
              <div style={{ fontSize: '.7rem', color: '#94a3b8' }}>
                {Math.abs(bal) < 0.01 ? '已平衡' : bal > 0 ? '应收' : '应付'}
              </div>
            </div>
          )
        })}
      </div>

      {/* Transactions */}
      <h6 className="fw-semibold mb-2" style={{ fontSize: '.85rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '.06em' }}>
        最简结算方案（{txns.length} 笔转账）
      </h6>
      {txns.length === 0 ? (
        <div className="alert alert-success py-2 small">
          <i className="fas fa-check-circle me-1" />大家已经平衡，无需转账！
        </div>
      ) : (
        <div className="d-flex flex-column gap-2">
          {txns.map((t, i) => (
            <div key={i} className="card border-0 shadow-sm" style={{ borderRadius: 10 }}>
              <div className="card-body py-2 px-3 d-flex align-items-center gap-3">
                <span className="badge rounded-pill bg-light text-secondary" style={{ fontSize: '.75rem', minWidth: 24 }}>{i + 1}</span>
                <div style={{ flex: 1 }}>
                  <strong style={{ color: '#ef4444' }}>{t.from}</strong>
                  <span className="text-muted mx-2">→</span>
                  <strong style={{ color: '#10b981' }}>{t.to}</strong>
                </div>
                <strong style={{ fontSize: '1.05rem', color: '#3b82f6' }}>${t.amount.toFixed(2)}</strong>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function BillSplit() {
  const [bills, setBills]         = useState(loadBills)
  const [selectedId, setSelected] = useState(null)
  const [tab, setTab]             = useState('expenses')
  const [newName, setNewName]     = useState('')
  const [creating, setCreating]   = useState(false)
  const [toast, setToast]         = useState(null)

  // Persist to localStorage whenever bills change
  useEffect(() => { saveBills(bills) }, [bills])

  const showToast = useCallback((msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 2500)
  }, [])

  const selected = bills.find(b => b.id === selectedId) || null

  function createBill() {
    const name = newName.trim()
    if (!name) return
    const bill = { id: uid(), name, createdAt: new Date().toISOString(), participants: [], expenses: [] }
    const next = [bill, ...bills]
    setBills(next)
    setSelected(bill.id)
    setNewName('')
    setCreating(false)
    setTab('participants')
    showToast(`"${name}" 已创建`)
  }

  function updateBill(updated) {
    setBills(prev => prev.map(b => b.id === updated.id ? updated : b))
  }

  function deleteBill(id) {
    if (!window.confirm('确认删除这个账单？')) return
    setBills(prev => prev.filter(b => b.id !== id))
    if (selectedId === id) setSelected(null)
  }

  // ── Bill list view ──────────────────────────────────────────────────────────
  if (!selected) {
    return (
      <div className="page-content" style={{ maxWidth: 640 }}>
        <Toast {...(toast || { msg: null, type: 'success' })} />

        <div className="d-flex align-items-center justify-content-between mb-4">
          <div>
            <h2 className="fw-bold mb-0" style={{ fontSize: '1.6rem' }}>
              <i className="fas fa-receipt me-2" style={{ color: '#3b82f6' }} />Bill Split
            </h2>
            <p className="text-muted small mb-0">智能分账，最少转账次数结清</p>
          </div>
          <button className="btn btn-primary btn-sm" onClick={() => setCreating(c => !c)}>
            <i className="fas fa-plus me-1" />新建账单
          </button>
        </div>

        {creating && (
          <div className="card border-0 shadow-sm mb-3" style={{ borderRadius: 12 }}>
            <div className="card-body d-flex gap-2">
              <input className="form-control form-control-sm" placeholder="账单名称，如：成都旅游、AA聚餐…"
                value={newName} onChange={e => setNewName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && createBill()} autoFocus />
              <button className="btn btn-primary btn-sm" onClick={createBill}>创建</button>
              <button className="btn btn-outline-secondary btn-sm" onClick={() => setCreating(false)}>取消</button>
            </div>
          </div>
        )}

        {bills.length === 0 ? (
          <div className="text-center py-5 text-muted">
            <i className="fas fa-receipt fa-2x mb-3 d-block opacity-25" />
            <p className="small">还没有账单，点击"新建账单"开始吧</p>
          </div>
        ) : (
          <div className="d-flex flex-column gap-2">
            {bills.map(b => {
              const total = b.expenses.reduce((s, e) => s + (parseFloat(e.amount) || 0), 0)
              const txns  = calcSettlement(b.participants, b.expenses)
              return (
                <div key={b.id} className="card border-0 shadow-sm"
                  style={{ borderRadius: 12, cursor: 'pointer' }}
                  onClick={() => { setSelected(b.id); setTab('expenses') }}>
                  <div className="card-body d-flex align-items-center gap-3 py-3">
                    <div className="d-flex align-items-center justify-content-center rounded-3"
                      style={{ width: 40, height: 40, background: '#dbeafe', flexShrink: 0 }}>
                      <i className="fas fa-file-invoice-dollar" style={{ color: '#3b82f6' }} />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div className="fw-semibold">{b.name}</div>
                      <div className="text-muted small">
                        {b.participants.length} 人 · {b.expenses.length} 笔 · 总计 ${total.toFixed(2)}
                        {txns.length > 0 && (
                          <span className="ms-2 badge bg-warning text-dark" style={{ fontSize: '.65rem' }}>
                            待结算 {txns.length} 笔
                          </span>
                        )}
                        {txns.length === 0 && b.expenses.length > 0 && (
                          <span className="ms-2 badge bg-success" style={{ fontSize: '.65rem' }}>已平衡</span>
                        )}
                      </div>
                    </div>
                    <button className="btn btn-link btn-sm text-danger p-0" onClick={ev => { ev.stopPropagation(); deleteBill(b.id) }}>
                      <i className="fas fa-trash" />
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    )
  }

  // ── Bill detail view ────────────────────────────────────────────────────────
  const tabs = [
    { key: 'participants', label: '成员', icon: 'fa-users' },
    { key: 'expenses',     label: '账单', icon: 'fa-list-ul' },
    { key: 'settlement',   label: '结算', icon: 'fa-balance-scale' },
  ]

  return (
    <div className="page-content" style={{ maxWidth: 640 }}>
      <Toast {...(toast || { msg: null, type: 'success' })} />

      {/* Header */}
      <div className="d-flex align-items-center gap-2 mb-4">
        <button className="btn btn-link btn-sm p-0 text-secondary" onClick={() => setSelected(null)}>
          <i className="fas fa-arrow-left" />
        </button>
        <div style={{ flex: 1 }}>
          <h2 className="fw-bold mb-0" style={{ fontSize: '1.4rem' }}>{selected.name}</h2>
          <span className="text-muted small">
            {selected.participants.length} 人 · {selected.expenses.length} 笔账单
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div className="d-flex gap-1 mb-4 p-1 rounded-3" style={{ background: '#f1f5f9' }}>
        {tabs.map(t => (
          <button key={t.key}
            className={`btn btn-sm flex-fill ${tab === t.key ? 'btn-white shadow-sm' : ''}`}
            style={{ borderRadius: 8, fontWeight: tab === t.key ? 600 : 400, background: tab === t.key ? '#fff' : 'transparent', border: 'none' }}
            onClick={() => setTab(t.key)}>
            <i className={`fas ${t.icon} me-1`} />{t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'participants' && (
        <ParticipantsPanel bill={selected} onChange={updateBill} />
      )}
      {tab === 'expenses' && (
        <ExpensesPanel bill={selected} onChange={updateBill} />
      )}
      {tab === 'settlement' && (
        <SettlementPanel bill={selected} />
      )}
    </div>
  )
}
