import React, { useState, useEffect } from 'react'
import { api } from '../api'
import {
  DndContext,
  closestCenter,
  PointerSensor,
  KeyboardSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core'
import {
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable,
  arrayMove,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'

// ── Entry type config ─────────────────────────────────────────────────────────
const ENTRY_TYPES = [
  { value: 'breakfast',  label: '早饭',   icon: 'fa-coffee',       color: '#f59e0b', bg: '#fef3c7' },
  { value: 'brunch',     label: 'Brunch', icon: 'fa-egg',          color: '#f97316', bg: '#ffedd5' },
  { value: 'lunch',      label: '午饭',   icon: 'fa-utensils',     color: '#10b981', bg: '#d1fae5' },
  { value: 'dinner',     label: '晚饭',   icon: 'fa-moon',         color: '#6366f1', bg: '#ede9fe' },
  { value: 'hiking',     label: 'Hiking', icon: 'fa-hiking',       color: '#059669', bg: '#ecfdf5' },
  { value: 'shopping',   label: '逛街',   icon: 'fa-shopping-bag', color: '#ec4899', bg: '#fce7f3' },
  { value: 'attraction', label: '景点',   icon: 'fa-landmark',     color: '#3b82f6', bg: '#dbeafe' },
  { value: 'transport',  label: '交通',   icon: 'fa-car',          color: '#64748b', bg: '#f1f5f9' },
  { value: 'hotel',      label: '酒店',   icon: 'fa-hotel',        color: '#8b5cf6', bg: '#ede9fe' },
  { value: 'other',      label: '其他',   icon: 'fa-circle',       color: '#94a3b8', bg: '#f8fafc' },
]
const TYPE_MAP = Object.fromEntries(ENTRY_TYPES.map(t => [t.value, t]))
const EMPTY_ENTRY = { type: 'attraction', name: '', time_start: '', time_end: '', address: '', notes: '' }

// ── Helpers ───────────────────────────────────────────────────────────────────
function Toast({ toast }) {
  if (!toast) return null
  return (
    <div
      className={`alert alert-${toast.type} position-fixed top-0 end-0 m-3 shadow`}
      style={{ zIndex: 9999, minWidth: 240 }}
    >
      {toast.msg}
    </div>
  )
}

function TypeBadge({ type }) {
  const t = TYPE_MAP[type] || TYPE_MAP['other']
  return (
    <span
      className="badge rounded-pill d-inline-flex align-items-center gap-1"
      style={{ background: t.bg, color: t.color, fontWeight: 600, fontSize: '0.75rem', padding: '4px 8px' }}
    >
      <i className={`fas ${t.icon}`} style={{ fontSize: '0.65rem' }} />
      {t.label}
    </span>
  )
}

function CopyBadge({ planId }) {
  const [copied, setCopied] = useState(false)
  function copy() {
    const url = window.location.origin + '/travel?join=' + planId
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    })
  }
  return (
    <span
      onClick={copy}
      className="badge bg-secondary ms-2 user-select-none"
      style={{ cursor: 'pointer', fontSize: '0.85rem', letterSpacing: 2, padding: '5px 10px' }}
      title="复制分享链接"
    >
      <i className={`fas ${copied ? 'fa-check' : 'fa-share-alt'} me-1`} style={{ fontSize: '0.7rem' }} />
      {copied ? '已复制链接' : planId}
    </span>
  )
}

// ── Sortable table row ────────────────────────────────────────────────────────
function SortableRow({ entry, onEdit, onDelete }) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: entry.id })

  const rowStyle = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity:    isDragging ? 0.45 : 1,
    background: isDragging ? '#f0f9ff' : undefined,
    position:   'relative',
    zIndex:     isDragging ? 10 : undefined,
  }

  return (
    <tr ref={setNodeRef} style={rowStyle}>
      {/* Drag handle */}
      <td style={{ width: 28, padding: '6px 4px 6px 8px' }}>
        <span
          {...attributes}
          {...listeners}
          style={{
            cursor: isDragging ? 'grabbing' : 'grab',
            color: '#cbd5e1',
            display: 'inline-flex',
            alignItems: 'center',
            padding: '2px 4px',
            borderRadius: 4,
            touchAction: 'none',   // required for mobile
          }}
          title="拖拽排序"
        >
          <i className="fas fa-grip-vertical" style={{ fontSize: '0.8rem' }} />
        </span>
      </td>

      <td className="text-muted small text-nowrap" style={{ minWidth: 72 }}>
        {entry.time_start
          ? (
            <>
              <span>{entry.time_start}</span>
              {entry.time_end && (
                <><br /><span style={{ opacity: 0.55 }}>{entry.time_end}</span></>
              )}
            </>
          )
          : <span style={{ opacity: 0.35 }}>—</span>
        }
      </td>

      <td style={{ width: 90 }}>
        <TypeBadge type={entry.type} />
      </td>

      <td className="fw-medium" style={{ minWidth: 130 }}>
        {entry.name}
      </td>

      <td className="text-muted small d-none d-md-table-cell" style={{ minWidth: 150 }}>
        {entry.address
          ? <span title={entry.address}>{entry.address.length > 30 ? entry.address.slice(0, 28) + '…' : entry.address}</span>
          : <span style={{ opacity: 0.35 }}>—</span>
        }
      </td>

      <td className="text-muted small d-none d-lg-table-cell" style={{ minWidth: 150 }}>
        {entry.notes
          ? <span title={entry.notes}>{entry.notes.length > 35 ? entry.notes.slice(0, 33) + '…' : entry.notes}</span>
          : <span style={{ opacity: 0.35 }}>—</span>
        }
      </td>

      <td style={{ width: 72 }}>
        <div className="d-flex gap-1">
          <button
            className="btn btn-xs btn-outline-secondary"
            style={{ fontSize: '.72rem', padding: '2px 7px' }}
            onClick={() => onEdit(entry)}
          ><i className="fas fa-pen" /></button>
          <button
            className="btn btn-xs btn-outline-danger"
            style={{ fontSize: '.72rem', padding: '2px 7px' }}
            onClick={() => onDelete(entry.id)}
          ><i className="fas fa-trash" /></button>
        </div>
      </td>
    </tr>
  )
}

// ── Day Table with DnD ────────────────────────────────────────────────────────
function DayTable({ entries, onEdit, onDelete, onReorder }) {
  const sensors = useSensors(
    useSensor(PointerSensor, {
      // Require 5px movement before drag starts — prevents accidental drags on click
      activationConstraint: { distance: 5 },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  )

  function handleDragEnd({ active, over }) {
    if (!over || active.id === over.id) return
    const oldIdx = entries.findIndex(e => e.id === active.id)
    const newIdx = entries.findIndex(e => e.id === over.id)
    onReorder(arrayMove(entries, oldIdx, newIdx))
  }

  if (entries.length === 0) {
    return (
      <div className="text-center py-5 text-muted">
        <i className="fas fa-map-marked-alt fa-2x mb-2 d-block" style={{ opacity: 0.2 }} />
        <span className="small">这天还没有安排，点击下方添加第一个项目</span>
      </div>
    )
  }

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragEnd={handleDragEnd}
    >
      <div className="table-responsive">
        <table className="table table-hover align-middle mb-0" style={{ fontSize: '.875rem' }}>
          <thead className="table-light">
            <tr>
              <th style={{ width: 28 }}></th>
              <th style={{ width: 100 }}>时间</th>
              <th style={{ width: 90 }}>类型</th>
              <th style={{ minWidth: 130 }}>名称</th>
              <th style={{ minWidth: 150 }} className="d-none d-md-table-cell">地址</th>
              <th style={{ minWidth: 150 }} className="d-none d-lg-table-cell">备注</th>
              <th style={{ width: 72 }}></th>
            </tr>
          </thead>
          <SortableContext
            items={entries.map(e => e.id)}
            strategy={verticalListSortingStrategy}
          >
            <tbody>
              {entries.map(e => (
                <SortableRow
                  key={e.id}
                  entry={e}
                  onEdit={onEdit}
                  onDelete={onDelete}
                />
              ))}
            </tbody>
          </SortableContext>
        </table>
      </div>
    </DndContext>
  )
}

// ── Entry Form Modal ──────────────────────────────────────────────────────────
function EntryModal({ planId, dayNumber, entry, onClose, onSaved, showToast }) {
  const isEdit = !!entry
  const [form, setForm]     = useState(
    isEdit
      ? { type: entry.type, name: entry.name, time_start: entry.time_start || '', time_end: entry.time_end || '', address: entry.address || '', notes: entry.notes || '' }
      : { ...EMPTY_ENTRY }
  )
  const [saving, setSaving] = useState(false)

  function set(k, v) { setForm(f => ({ ...f, [k]: v })) }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!form.name.trim()) { showToast('名称不能为空', 'danger'); return }
    setSaving(true)
    const payload = { ...form, day_number: dayNumber, name: form.name.trim() }
    const d = isEdit
      ? await api.put(`/api/travel/plans/${planId}/entries/${entry.id}`, payload)
      : await api.post(`/api/travel/plans/${planId}/entries`, payload)
    setSaving(false)
    if (d.ok) { onSaved(d.entry); onClose() }
    else showToast(d.error || '保存失败', 'danger')
  }

  return (
    <div className="modal d-block" style={{ background: 'rgba(0,0,0,.45)' }} onClick={onClose}>
      <div className="modal-dialog modal-dialog-centered" style={{ maxWidth: 520 }} onClick={e => e.stopPropagation()}>
        <div className="modal-content shadow-lg">
          <div className="modal-header border-0 pb-0">
            <h5 className="modal-title fw-bold">
              <i className="fas fa-map-marker-alt me-2 text-primary" />
              {isEdit ? '编辑项目' : '添加项目'}
            </h5>
            <button className="btn-close" onClick={onClose} />
          </div>
          <form onSubmit={handleSubmit}>
            <div className="modal-body pt-2">
              {/* Type */}
              <div className="mb-3">
                <label className="form-label fw-medium small">类型</label>
                <div className="d-flex flex-wrap gap-1">
                  {ENTRY_TYPES.map(t => (
                    <button
                      key={t.value} type="button" className="btn btn-sm"
                      style={form.type === t.value
                        ? { background: t.color, color: '#fff', border: `1.5px solid ${t.color}` }
                        : { background: t.bg, color: t.color, border: `1.5px solid ${t.bg}` }
                      }
                      onClick={() => set('type', t.value)}
                    >
                      <i className={`fas ${t.icon} me-1`} style={{ fontSize: '0.7rem' }} />
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>
              {/* Name */}
              <div className="mb-3">
                <label className="form-label fw-medium small">名称 <span className="text-danger">*</span></label>
                <input
                  className="form-control" autoFocus
                  placeholder="如：秋叶原 / 寿司小仓 / 新干线"
                  value={form.name} onChange={e => set('name', e.target.value)}
                />
              </div>
              {/* Time */}
              <div className="mb-3">
                <label className="form-label fw-medium small">时间</label>
                <div className="d-flex align-items-center gap-2">
                  <input type="time" className="form-control" value={form.time_start} onChange={e => set('time_start', e.target.value)} />
                  <span className="text-muted small">→</span>
                  <input type="time" className="form-control" value={form.time_end} onChange={e => set('time_end', e.target.value)} />
                </div>
              </div>
              {/* Address */}
              <div className="mb-3">
                <label className="form-label fw-medium small">地址</label>
                <input className="form-control" placeholder="详细地址或地标" value={form.address} onChange={e => set('address', e.target.value)} />
              </div>
              {/* Notes */}
              <div className="mb-1">
                <label className="form-label fw-medium small">备注</label>
                <textarea className="form-control" rows={2} placeholder="注意事项、预约信息、推荐菜品…" value={form.notes} onChange={e => set('notes', e.target.value)} />
              </div>
            </div>
            <div className="modal-footer border-0 pt-0">
              <button type="button" className="btn btn-outline-secondary" onClick={onClose}>取消</button>
              <button type="submit" className="btn btn-primary" disabled={saving}>
                {saving
                  ? <><span className="spinner-border spinner-border-sm me-1" />保存中…</>
                  : <><i className="fas fa-check me-1" />{isEdit ? '保存修改' : '添加'}</>
                }
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

// ── Plan View ─────────────────────────────────────────────────────────────────
function PlanView({ plan: initPlan, onBack, showToast }) {
  const [plan, setPlan]               = useState(initPlan)
  const [activeDay, setActiveDay]     = useState(1)
  const [entryModal, setEntryModal]   = useState(null)
  const [editingName, setEditingName] = useState(false)
  const [nameInput, setNameInput]     = useState(initPlan.name)
  const [saving, setSaving]           = useState(false)
  const [reordering, setReordering]   = useState(false)
  // 'plan' = the day-by-day itinerary, 'expenses' = the trip's shared pot.
  const [view, setView]               = useState('plan')

  // Sort: display_order first (set after a drag), then time_start for ties
  const dayEntries = plan.entries
    .filter(e => e.day_number === activeDay)
    .slice()
    .sort((a, b) => {
      if (a.display_order !== b.display_order) return a.display_order - b.display_order
      if (a.time_start && b.time_start) return a.time_start.localeCompare(b.time_start)
      if (a.time_start) return -1
      if (b.time_start) return 1
      return 0
    })

  async function saveName() {
    if (!nameInput.trim()) return
    setSaving(true)
    const d = await api.put(`/api/travel/plans/${plan.id}`, { name: nameInput.trim() })
    setSaving(false)
    if (d.ok) { setPlan(p => ({ ...p, name: nameInput.trim() })); setEditingName(false) }
    else showToast(d.error || '保存失败', 'danger')
  }

  async function addDay() {
    const newNum = plan.num_days + 1
    const d = await api.put(`/api/travel/plans/${plan.id}`, { num_days: newNum })
    if (d.ok) { setPlan(d.plan); setActiveDay(newNum) }
    else showToast(d.error || '添加失败', 'danger')
  }

  async function removeDay(day) {
    if (plan.num_days <= 1) {
      showToast('至少保留一天', 'warning')
      return
    }
    // 删除该天所有条目
    const toDelete = plan.entries.filter(e => e.day_number === day)
    for (const entry of toDelete) {
      await api.delete(`/api/travel/plans/${plan.id}/entries/${entry.id}`)
    }
    // 更新天数
    const newNum = plan.num_days - 1
    const d = await api.put(`/api/travel/plans/${plan.id}`, { num_days: newNum })
    if (d.ok) {
      // 把后面天的条目往前挪一天
      for (const e of plan.entries) {
        if (e.day_number > day) {
          await api.put(`/api/travel/plans/${plan.id}/entries/${e.id}`, { day_number: e.day_number - 1 })
        }
      }
      setPlan(d.plan)
      if (activeDay > newNum) setActiveDay(newNum)
      else if (activeDay === day && activeDay > 1) setActiveDay(activeDay - 1)
      showToast('已删除第 ' + day + ' 天', 'success')
    } else {
      showToast('删除失败', 'danger')
    }
  }

  function onEntrySaved(entry) {
    setPlan(p => {
      const exists = p.entries.find(e => e.id === entry.id)
      return {
        ...p,
        entries: exists
          ? p.entries.map(e => e.id === entry.id ? entry : e)
          : [...p.entries, entry],
      }
    })
    showToast(entryModal?.editing ? '已更新' : '已添加', 'success')
  }

  async function deleteEntry(entryId) {
    if (!window.confirm('确认删除此项目？')) return
    const d = await api.delete(`/api/travel/plans/${plan.id}/entries/${entryId}`)
    if (d.ok) {
      setPlan(p => ({ ...p, entries: p.entries.filter(e => e.id !== entryId) }))
      showToast('已删除', 'success')
    } else {
      showToast(d.error || '删除失败', 'danger')
    }
  }

  // Called by DayTable when drag ends — receives new ordered array for the active day
  async function handleReorder(newDayEntries) {
    // Assign fresh display_order values based on new positions
    const updated = newDayEntries.map((e, i) => ({ ...e, display_order: i }))

    // Optimistic update so the UI feels instant
    setPlan(p => ({
      ...p,
      entries: [
        ...p.entries.filter(e => e.day_number !== activeDay),
        ...updated,
      ],
    }))

    // Persist to DB
    setReordering(true)
    const orders = updated.map(e => ({ id: e.id, display_order: e.display_order }))
    const d = await api.put(`/api/travel/plans/${plan.id}/entries/reorder`, { orders })
    setReordering(false)
    if (!d.ok) showToast('顺序保存失败', 'danger')
  }

  return (
    <div>
      {/* Plan header */}
      <div className="d-flex align-items-center flex-wrap gap-2 mb-3">
        <button className="btn btn-sm btn-outline-secondary" onClick={onBack}>
          <i className="fas fa-arrow-left me-1" />我的行程
        </button>

        {editingName ? (
          <div className="d-flex align-items-center gap-2 flex-grow-1">
            <input
              className="form-control form-control-sm fw-bold"
              style={{ maxWidth: 260, fontSize: '1.1rem' }}
              value={nameInput}
              autoFocus
              onChange={e => setNameInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') saveName(); if (e.key === 'Escape') setEditingName(false) }}
            />
            <button className="btn btn-sm btn-primary" onClick={saveName} disabled={saving}>
              {saving ? <span className="spinner-border spinner-border-sm" /> : <i className="fas fa-check" />}
            </button>
            <button className="btn btn-sm btn-outline-secondary" onClick={() => setEditingName(false)}>
              <i className="fas fa-times" />
            </button>
          </div>
        ) : (
          <h5
            className="mb-0 fw-bold d-flex align-items-center gap-1"
            style={{ cursor: 'pointer' }}
            onClick={() => setEditingName(true)}
            title="点击编辑行程名称"
          >
            {plan.name}
            <i className="fas fa-pen text-muted ms-1" style={{ fontSize: '0.7rem', opacity: 0.5 }} />
          </h5>
        )}

        <CopyBadge planId={plan.id} />

        {reordering && (
          <span className="text-muted small ms-2">
            <span className="spinner-border spinner-border-sm me-1" style={{ width: 10, height: 10 }} />
            保存顺序…
          </span>
        )}

      </div>

      {/* Plan / Expenses switch. The day tabs below belong to Plan only. */}
      <ul className="nav nav-pills nav-fill mb-3" style={{ maxWidth: 320 }}>
        {[{ k: 'plan', label: '行程', icon: 'fa-route' }, { k: 'expenses', label: '费用', icon: 'fa-receipt' }].map(t => (
          <li className="nav-item" key={t.k}>
            <button
              className={`nav-link py-1 ${view === t.k ? 'active' : ''}`}
              style={{ fontSize: '.85rem' }}
              onClick={() => setView(t.k)}
            >
              <i className={`fas ${t.icon} me-1`} />{t.label}
            </button>
          </li>
        ))}
      </ul>

      {view === 'expenses' ? (
        <ExpenseTab plan={plan} showToast={showToast} />
      ) : (
      <>
      {/* Day tabs */}
      <div className="d-flex align-items-center gap-1 flex-wrap mb-0">
        {Array.from({ length: plan.num_days }, (_, i) => i + 1).map(d => {
          const count = plan.entries.filter(e => e.day_number === d).length
          return (
            <div
              key={d}
              className={`btn btn-sm position-relative ${activeDay === d ? 'btn-primary' : 'btn-outline-secondary'}`}
              onClick={() => setActiveDay(d)}
              style={{ minWidth: 72, paddingRight: plan.num_days > 1 ? '22px' : undefined }}
            >
              Day {d}
              {count > 0 && (
                <span
                  className="badge ms-1"
                  style={{
                    background: activeDay === d ? 'rgba(255,255,255,.3)' : '#e2e8f0',
                    color: activeDay === d ? '#fff' : '#64748b',
                    fontSize: '0.65rem',
                  }}
                >
                  {count}
                </span>
              )}
              {plan.num_days > 1 && (
                <span
                  className="position-absolute"
                  style={{
                    top: -3, right: -3, width: 16, height: 16,
                    background: '#ef4444', color: '#fff',
                    borderRadius: '50%', display: 'flex',
                    alignItems: 'center', justifyContent: 'center',
                    cursor: 'pointer', fontSize: '0.55rem',
                    lineHeight: 1, zIndex: 2,
                    boxShadow: '0 1px 3px rgba(0,0,0,.3)',
                  }}
                  onClick={e => {
                    e.stopPropagation()
                    if (window.confirm(`确定要删除 Day ${d}${count > 0 ? '（含 ' + count + ' 个项目）' : ''}吗？`)) {
                      removeDay(d)
                    }
                  }}
                  title="删除这一天"
                >
                  <i className="fas fa-times" style={{ fontSize: '0.5rem' }} />
                </span>
              )}
            </div>
          )
        })}
        {plan.num_days < 30 && (
          <button className="btn btn-sm btn-outline-primary" onClick={addDay} title="添加一天">
            <i className="fas fa-plus" />
          </button>
        )}
      </div>

      {/* Day content card */}
      <div className="card shadow-sm border-0 mt-0">
        <div className="card-body p-0">
          <DayTable
            entries={dayEntries}
            onEdit={e => setEntryModal({ editing: e })}
            onDelete={deleteEntry}
            onReorder={handleReorder}
          />
        </div>
        <div className="card-footer bg-transparent border-top-0 pt-0 pb-3 px-3">
          <button
            className="btn btn-sm btn-outline-primary"
            onClick={() => setEntryModal({ editing: null })}
          >
            <i className="fas fa-plus me-1" />添加项目
          </button>
        </div>
      </div>

      {entryModal && (
        <EntryModal
          planId={plan.id}
          dayNumber={activeDay}
          entry={entryModal.editing}
          onClose={() => setEntryModal(null)}
          onSaved={onEntrySaved}
          showToast={showToast}
        />
      )}
      </>
      )}
    </div>
  )
}

// ── Plan Expenses (the plan's own bill) ──────────────────────────
// Same math the Bill Split page uses — one settlement algorithm, so a trip's
// numbers and a standalone bill's numbers cannot disagree.
function calcSettlement(participants, expenses) {
  const actualSet = new Set()
  expenses.forEach(({ paidBy, splitAmong }) => {
    if (paidBy) actualSet.add(paidBy)
    splitAmong.forEach(p => actualSet.add(p))
  })
  const active = participants.filter(p => actualSet.has(p))
  const net = {}
  active.forEach(p => (net[p] = 0))
  expenses.forEach(({ paidBy, amount, splitAmong }) => {
    const amt = parseFloat(amount) || 0
    const among = splitAmong.filter(p => active.includes(p))
    if (!among.length) return
    const share = amt / among.length
    net[paidBy] = (net[paidBy] || 0) + amt
    among.forEach(p => { net[p] = (net[p] || 0) - share })
  })
  const creditors = [], debtors = []
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
    const c = creditors[ci], d = debtors[di]
    const transfer = Math.min(c.bal, -d.bal)
    txns.push({ from: d.name, to: c.name, amount: Math.round(transfer * 100) / 100 })
    c.bal -= transfer; d.bal += transfer
    if (Math.abs(c.bal) < 0.01) ci++
    if (Math.abs(d.bal) < 0.01) di++
  }
  return txns
}

/* Participant picker: a text input backed by a friend datalist. Typing a name
   that is not on the list still works — Horizon's call, because a trip is not
   only friends (妈妈, a colleague, a plus-one). */
function ParticipantPicker({ planId, participants, friends, onRefresh, showToast }) {
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)

  async function add() {
    const n = name.trim()
    if (!n) return
    if (participants.includes(n)) { showToast('这个名字已在列表中', 'warning'); return }
    setBusy(true)
    const d = await api.post(`/api/travel/plans/${planId}/bill/participants`, { name: n })
    setBusy(false)
    if (d.ok) { setName(''); onRefresh(d.bill) }
    else showToast(d.error || '添加失败', 'danger')
  }

  async function remove(p) {
    const d = await api.delete(`/api/travel/plans/${planId}/bill/participants/${encodeURIComponent(p)}`)
    if (d.ok) onRefresh(d.bill)
    else showToast(d.error || '删除失败', 'danger')
  }

  return (
    <div className="d-flex flex-wrap align-items-center gap-2 mb-3">
      {participants.map(p => (
        <span key={p} className="badge d-inline-flex align-items-center gap-1"
          style={{ background: '#eef2f7', color: '#334155', fontWeight: 500, fontSize: '.8rem', padding: '.4em .6em' }}>
          {p}
          <button className="btn btn-link btn-sm p-0 text-danger" style={{ lineHeight: 1 }}
            onClick={() => remove(p)} title={`移除 ${p}`}>
            <i className="fas fa-times" style={{ fontSize: '.6rem' }} />
          </button>
        </span>
      ))}
      <div className="input-group input-group-sm" style={{ maxWidth: 220 }}>
        <input
          className="form-control" list={`plan-friends-${planId}`}
          placeholder="输入名字，或从好友里选"
          value={name}
          onChange={e => setName(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); add() } }}
        />
        <datalist id={`plan-friends-${planId}`}>
          {friends.filter(f => !participants.includes(f.display_name) && !participants.includes(f.username))
            .map(f => <option key={f.username} value={f.display_name || f.username} />)}
        </datalist>
        <button className="btn btn-outline-primary" onClick={add} disabled={busy}>
          <i className="fas fa-plus" />
        </button>
      </div>
    </div>
  )
}

function ExpenseTab({ plan, showToast }) {
  const [bill, setBill]       = useState(null)
  const [friends, setFriends] = useState([])
  const [loading, setLoading] = useState(true)
  const [adding, setAdding]   = useState(false)
  const [form, setForm]       = useState({ desc: '', amount: '', paidBy: '', splitAmong: [] })
  const [saving, setSaving]   = useState(false)

  useEffect(() => {
    let alive = true
    api.get(`/api/travel/plans/${plan.id}/bill`).then(d => {
      if (!alive) return
      if (d.ok) {
        setBill(d.bill)
        setForm(f => ({ ...f, paidBy: d.bill.participants[0] || '', splitAmong: [...d.bill.participants] }))
      } else showToast(d.error || '费用加载失败', 'danger')
      setLoading(false)
    })
    api.get('/api/travel/friends').then(d => { if (alive && d.ok) setFriends(d.friends) })
    return () => { alive = false }
  }, [plan.id])

  function applyBill(b) {
    setBill(b)
    setForm(f => ({
      ...f,
      paidBy: b.participants.includes(f.paidBy) ? f.paidBy : (b.participants[0] || ''),
      splitAmong: f.splitAmong.filter(p => b.participants.includes(p)),
    }))
  }

  function toggleSplit(p) {
    setForm(f => ({
      ...f,
      splitAmong: f.splitAmong.includes(p) ? f.splitAmong.filter(x => x !== p) : [...f.splitAmong, p],
    }))
  }

  async function submit() {
    if (!form.desc.trim()) { showToast('请填写描述', 'danger'); return }
    const amt = parseFloat(form.amount)
    if (!amt || amt <= 0) { showToast('请填写有效金额', 'danger'); return }
    if (!form.paidBy) { showToast('请选择付款人', 'danger'); return }
    if (!form.splitAmong.length) { showToast('请至少选择一位分摊人', 'danger'); return }
    setSaving(true)
    const d = await api.post(`/api/travel/plans/${plan.id}/bill/expenses`, {
      desc: form.desc.trim(), amount: amt, paidBy: form.paidBy, splitAmong: form.splitAmong,
    })
    setSaving(false)
    if (d.ok) {
      applyBill(d.bill)
      setForm(f => ({ ...f, desc: '', amount: '' }))
      setAdding(false)
      showToast('已记录', 'success')
    } else showToast(d.error || '添加失败', 'danger')
  }

  async function deleteExpense(id) {
    const d = await api.delete(`/api/travel/plans/${plan.id}/bill/expenses/${id}`)
    if (d.ok) applyBill(d.bill)
    else showToast(d.error || '删除失败', 'danger')
  }

  if (loading) return <p className="text-muted small">加载中…</p>
  if (!bill) return <p className="text-muted small">费用记录暂不可用。</p>

  const total = bill.expenses.reduce((s, e) => s + (parseFloat(e.amount) || 0), 0)
  const txns  = calcSettlement(bill.participants, bill.expenses)

  return (
    <div>
      <div className="d-flex align-items-center flex-wrap gap-2 mb-2">
        <span className="fw-semibold" style={{ fontSize: '.9rem' }}>参与人</span>
      </div>
      <ParticipantPicker
        planId={plan.id}
        participants={bill.participants}
        friends={friends}
        onRefresh={applyBill}
        showToast={showToast}
      />

      <div className="d-flex justify-content-between align-items-center mb-2">
        <span className="text-muted small">
          共 {bill.expenses.length} 笔 总计 <strong style={{ color: '#1a1a1a' }}>${total.toFixed(2)}</strong>
        </span>
        {bill.participants.length >= 2 && !adding && (
          <button className="btn btn-sm btn-primary" onClick={() => setAdding(true)}>
            <i className="fas fa-plus me-1" />记一笔
          </button>
        )}
      </div>

      {bill.participants.length < 2 && (
        <div className="alert alert-warning py-2 small mb-3">先添加至少 2 位参与人，才能记账。</div>
      )}

      {adding && (
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
                {bill.participants.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div className="mb-3">
              <div className="d-flex align-items-center gap-2 mb-1">
                <label className="form-label small fw-semibold mb-0">分摊人</label>
                <button className="btn btn-link btn-sm p-0 text-decoration-none" style={{ fontSize: '.75rem' }}
                  onClick={() => setForm(f => ({ ...f, splitAmong: [...bill.participants] }))}>全选</button>
                <button className="btn btn-link btn-sm p-0 text-decoration-none" style={{ fontSize: '.75rem' }}
                  onClick={() => setForm(f => ({ ...f, splitAmong: [] }))}>清空</button>
              </div>
              <div className="d-flex flex-wrap gap-2">
                {bill.participants.map(p => (
                  <label key={p} className="d-flex align-items-center gap-1 small" style={{ cursor: 'pointer' }}>
                    <input type="checkbox" checked={form.splitAmong.includes(p)} onChange={() => toggleSplit(p)} />
                    {p}
                  </label>
                ))}
              </div>
            </div>
            <div className="d-flex gap-2">
              <button className="btn btn-primary btn-sm" onClick={submit} disabled={saving}>
                {saving ? <span className="spinner-border spinner-border-sm me-1" /> : <i className="fas fa-check me-1" />}确认添加
              </button>
              <button className="btn btn-outline-secondary btn-sm" onClick={() => setAdding(false)}>取消</button>
            </div>
          </div>
        </div>
      )}

      {bill.expenses.length === 0 ? (
        <p className="text-muted small">还没有费用记录。</p>
      ) : (
        <div className="d-flex flex-column gap-2">
          {bill.expenses.map(e => {
            const share = (parseFloat(e.amount) / (e.splitAmong.length || 1)).toFixed(2)
            return (
              <div key={e.id} className="card border-0 shadow-sm" style={{ borderRadius: 10 }}>
                <div className="card-body py-2 px-3 d-flex align-items-start gap-3">
                  <div style={{ flex: 1 }}>
                    <strong style={{ fontSize: '.9rem' }}>{e.desc}</strong>
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

      {txns.length > 0 && (
        <div className="mt-4">
          <div className="fw-semibold mb-2" style={{ fontSize: '.9rem' }}>结算方案</div>
          <div className="d-flex flex-column gap-1">
            {txns.map((t, i) => (
              <div key={i} className="d-flex align-items-center gap-2 small">
                <span className="badge" style={{ background: '#fee2e2', color: '#b91c1c' }}>{t.from}</span>
                <i className="fas fa-arrow-right text-muted" style={{ fontSize: '.7rem' }} />
                <span className="badge" style={{ background: '#dcfce7', color: '#15803d' }}>{t.to}</span>
                <strong className="ms-auto">${t.amount.toFixed(2)}</strong>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  )
}

// ── Home (landing) ────────────────────────────────────────────────────────────
function PlanHome({ onOpenPlan, showToast }) {
  const [myPlans, setMyPlans]         = useState([])
  const [loadingMy, setLoadingMy]     = useState(true)
  const [creating, setCreating]       = useState(false)
  const [newName, setNewName]         = useState('')
  const [loadId, setLoadId]           = useState('')
  const [showCreate, setShowCreate]   = useState(false)
  const [loadingPlan, setLoadingPlan] = useState(false)

  useEffect(() => {
    api.get('/api/travel/my').then(d => {
      if (d.ok) setMyPlans(d.plans)
      setLoadingMy(false)
    })
  }, [])

  async function handleCreate(e) {
    e.preventDefault()
    if (!newName.trim()) return
    setCreating(true)
    const d = await api.post('/api/travel/plans', { name: newName.trim() })
    setCreating(false)
    if (d.ok) onOpenPlan(d.plan)
    else showToast(d.error || '创建失败', 'danger')
  }

  async function handleLoad(e) {
    e.preventDefault()
    const id = loadId.trim().toUpperCase()
    if (!id) return
    setLoadingPlan(true)
    const d = await api.get(`/api/travel/plans/${id}`)
    setLoadingPlan(false)
    if (d.ok) onOpenPlan(d.plan)
    else showToast('找不到该行程 ID，请检查后重试', 'danger')
  }

  async function deletePlan(plan) {
    if (!window.confirm(`确认删除行程「${plan.name}」？此操作不可撤销。`)) return
    const d = await api.delete(`/api/travel/plans/${plan.id}`)
    if (d.ok) {
      setMyPlans(prev => prev.filter(p => p.id !== plan.id))
      showToast('已删除', 'success')
    } else {
      showToast(d.error || '删除失败', 'danger')
    }
  }

  return (
    <div className="row g-3">
      <div className="col-12 col-md-5">
        <div className="card shadow-sm h-100 border-0">
          <div className="card-body">
            <h6 className="fw-bold mb-3">
              <i className="fas fa-plus-circle text-primary me-2" />创建新行程
            </h6>
            {showCreate ? (
              <form onSubmit={handleCreate} className="d-flex flex-column gap-2">
                <input
                  className="form-control" autoFocus
                  placeholder="行程名称，如：日本9天"
                  value={newName} onChange={e => setNewName(e.target.value)}
                />
                <div className="d-flex gap-2">
                  <button className="btn btn-primary flex-grow-1" type="submit" disabled={creating}>
                    {creating ? <span className="spinner-border spinner-border-sm" /> : <><i className="fas fa-map me-1" />创建</>}
                  </button>
                  <button className="btn btn-outline-secondary" type="button" onClick={() => { setShowCreate(false); setNewName('') }}>取消</button>
                </div>
              </form>
            ) : (
              <button className="btn btn-primary w-100" onClick={() => setShowCreate(true)}>
                <i className="fas fa-plus me-2" />新建行程
              </button>
            )}

            <hr />

            <h6 className="fw-bold mb-3">
              <i className="fas fa-share-alt text-success me-2" />加载共享行程
            </h6>
            <form onSubmit={handleLoad} className="d-flex gap-2">
              <input
                className="form-control font-monospace text-uppercase"
                placeholder="输入行程 ID（如 ABC123）"
                value={loadId}
                onChange={e => setLoadId(e.target.value.toUpperCase())}
                maxLength={8}
                style={{ letterSpacing: 2 }}
              />
              <button className="btn btn-success" type="submit" disabled={loadingPlan || !loadId.trim()}>
                {loadingPlan ? <span className="spinner-border spinner-border-sm" /> : <i className="fas fa-arrow-right" />}
              </button>
            </form>
          </div>
        </div>
      </div>

      <div className="col-12 col-md-7">
        <div className="card shadow-sm border-0 h-100">
          <div className="card-header bg-transparent fw-semibold border-0 pb-0">
            <i className="fas fa-suitcase text-primary me-2" />我的行程
          </div>
          <div className="card-body pt-2 p-0">
            {loadingMy ? (
              <div className="text-center py-4 text-muted"><span className="spinner-border spinner-border-sm" /></div>
            ) : myPlans.length === 0 ? (
              <div className="text-center py-5 text-muted small">
                <i className="fas fa-plane fa-2x mb-2 d-block" style={{ opacity: 0.2 }} />
                还没有行程，创建你的第一个旅行计划吧
              </div>
            ) : (
              <div className="list-group list-group-flush">
                {myPlans.map(p => (
                  <div
                    key={p.id}
                    className="list-group-item list-group-item-action d-flex align-items-center gap-2 py-2 px-3"
                    style={{ cursor: 'pointer' }}
                    onClick={() => onOpenPlan(p)}
                  >
                    <div className="flex-grow-1 min-w-0">
                      <div className="fw-medium text-truncate">{p.name}</div>
                      <div className="text-muted small d-flex align-items-center gap-2">
                        <span><i className="fas fa-calendar-alt me-1" style={{ opacity: 0.4 }} />{p.num_days} 天</span>
                        <span><i className="fas fa-key me-1" style={{ opacity: 0.4 }} /><code style={{ fontSize: '0.7rem', letterSpacing: 1 }}>{p.id}</code></span>
                      </div>
                    </div>
                    <button
                      className="btn btn-xs btn-outline-danger flex-shrink-0"
                      style={{ fontSize: '.7rem', padding: '2px 7px' }}
                      onClick={e => { e.stopPropagation(); deletePlan(p) }}
                    ><i className="fas fa-trash" /></button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function TravelPlanner() {
  const [view, setView]   = useState('home')
  const [plan, setPlan]   = useState(null)
  const [toast, setToast] = useState(null)

  function showToast(msg, type = 'success') {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 2600)
  }

  function openPlan(p) {
    // Always fetch from API — my-plans list returns entries:[] regardless of real data
    api.get(`/api/travel/plans/${p.id}`).then(d => {
      if (d.ok) { setPlan(d.plan); setView('plan') }
      else showToast('加载失败', 'danger')
    })
  }

  // Auto-load if URL contains ?join=CODE
  useEffect(() => {
    const code = new URLSearchParams(window.location.search).get('join')
    if (code) {
      openPlan({ id: code.toUpperCase() })
      // Clean the URL without triggering a navigation
      window.history.replaceState({}, '', window.location.pathname)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 960 }}>
      <Toast toast={toast} />
      <div className="d-flex align-items-center mb-4">
        <i className="fas fa-route fa-lg me-2 text-primary" />
        <h4 className="mb-0 fw-bold">旅行规划</h4>
      </div>
      {view === 'home' && <PlanHome onOpenPlan={openPlan} showToast={showToast} />}
      {view === 'plan' && plan && (
        <PlanView plan={plan} onBack={() => { setView('home'); setPlan(null) }} showToast={showToast} />
      )}
    </div>
  )
}
