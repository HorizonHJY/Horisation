import React, { useState } from 'react'

/* One expense form for both places that record money: the bill split page and
   a travel plan's expense tab. The parent owns the endpoint (they differ), so
   `onSubmit` receives the payload and returns { ok, error }. */

const todayISO = () => {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

export default function ExpenseForm({ participants, expense, onSubmit, onCancel }) {
  const [form, setForm] = useState(() => expense ? {
    desc: expense.desc || '',
    amount: String(expense.amount ?? ''),
    paidBy: expense.paidBy || participants[0] || '',
    splitAmong: [...(expense.splitAmong || participants)],
    spentAt: expense.spentAt || todayISO(),
  } : {
    desc: '', amount: '', paidBy: participants[0] || '', splitAmong: [...participants],
    spentAt: todayISO(),
  })
  const [saving, setSaving] = useState(false)
  const [error, setError]   = useState('')

  const set = (key) => (e) => setForm(f => ({ ...f, [key]: e.target.value }))
  const toggleSplit = (p) => setForm(f => ({
    ...f,
    splitAmong: f.splitAmong.includes(p)
      ? f.splitAmong.filter(x => x !== p)
      : [...f.splitAmong, p],
  }))

  async function submit() {
    setError('')
    if (!form.desc.trim()) return setError('请填写描述')
    const amt = parseFloat(form.amount)
    if (!amt || amt <= 0) return setError('请填写有效金额')
    if (!form.paidBy) return setError('请选择付款人')
    if (!form.splitAmong.length) return setError('请至少选择一位分摊人')

    setSaving(true)
    const res = await onSubmit({
      desc: form.desc.trim(), amount: amt, paidBy: form.paidBy, splitAmong: form.splitAmong,
      spentAt: form.spentAt,
    })
    setSaving(false)
    if (!res?.ok) setError(res?.error || '保存失败')
  }

  return (
    <div className="card border-0 shadow-sm mb-3" style={{ borderRadius: 12 }}>
      <div className="card-body">
        <div className="row g-2 mb-2">
          <div className="col-7">
            <label className="form-label small fw-semibold mb-1">描述</label>
            <input className="form-control form-control-sm" placeholder="如：晚饭、门票…"
              value={form.desc} onChange={set('desc')} />
          </div>
          <div className="col-5">
            <label className="form-label small fw-semibold mb-1">金额 ($)</label>
            <input className="form-control form-control-sm" type="number" min="0" step="0.01" placeholder="0.00"
              value={form.amount} onChange={set('amount')} />
          </div>
        </div>
        <div className="mb-2" style={{ maxWidth: 190 }}>
          <label className="form-label small fw-semibold mb-1">日期</label>
          <input className="form-control form-control-sm" type="date"
            value={form.spentAt} onChange={set('spentAt')} />
        </div>
        <div className="mb-2">
          <label className="form-label small fw-semibold mb-1">付款人</label>
          <select className="form-select form-select-sm" value={form.paidBy} onChange={set('paidBy')}>
            {participants.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div className="mb-3">
          <div className="d-flex align-items-center gap-2 mb-1">
            <label className="form-label small fw-semibold mb-0">分摊人</label>
            <button type="button" className="btn btn-link btn-sm p-0 text-decoration-none" style={{ fontSize: '.75rem' }}
              onClick={() => setForm(f => ({ ...f, splitAmong: [...participants] }))}>全选</button>
            <button type="button" className="btn btn-link btn-sm p-0 text-decoration-none" style={{ fontSize: '.75rem' }}
              onClick={() => setForm(f => ({ ...f, splitAmong: [] }))}>清空</button>
          </div>
          <div className="d-flex flex-wrap gap-2">
            {participants.map(p => (
              <label key={p} className="d-flex align-items-center gap-1 small" style={{ cursor: 'pointer' }}>
                <input type="checkbox" checked={form.splitAmong.includes(p)} onChange={() => toggleSplit(p)} />
                {p}
              </label>
            ))}
          </div>
        </div>
        {error && <div className="alert alert-danger py-1 px-2 small mb-2">{error}</div>}
        <div className="d-flex gap-2">
          <button type="button" className="btn btn-primary btn-sm" onClick={submit} disabled={saving}>
            {saving
              ? <span className="spinner-border spinner-border-sm me-1" />
              : <i className="fas fa-check me-1" />}
            {expense ? '保存修改' : '确认添加'}
          </button>
          <button type="button" className="btn btn-outline-secondary btn-sm" onClick={onCancel}>取消</button>
        </div>
      </div>
    </div>
  )
}
