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
    navigator.clipboard.writeText(planId).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    })
  }
  return (
    <span
      onClick={copy}
      className="badge bg-secondary ms-2 user-select-none"
      style={{ cursor: 'pointer', fontSize: '0.85rem', letterSpacing: 2, padding: '5px 10px' }}
      title="Click to copy Plan ID"
    >
      <i className={`fas ${copied ? 'fa-check' : 'fa-copy'} me-1`} style={{ fontSize: '0.7rem' }} />
      {planId}
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

        <span className="text-muted small ms-auto d-none d-sm-inline">
          分享此 ID 可与好友协作编辑
        </span>
      </div>

      {/* Day tabs */}
      <div className="d-flex align-items-center gap-1 flex-wrap mb-0">
        {Array.from({ length: plan.num_days }, (_, i) => i + 1).map(d => {
          const count = plan.entries.filter(e => e.day_number === d).length
          return (
            <button
              key={d}
              className={`btn btn-sm ${activeDay === d ? 'btn-primary' : 'btn-outline-secondary'}`}
              onClick={() => setActiveDay(d)}
              style={{ minWidth: 72 }}
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
            </button>
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
            <p className="text-muted small mt-2 mb-0">
              <i className="fas fa-info-circle me-1" />
              输入他人分享的行程 ID，即可查看并共同编辑
            </p>
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
    if (!p.entries) {
      api.get(`/api/travel/plans/${p.id}`).then(d => {
        if (d.ok) { setPlan(d.plan); setView('plan') }
        else showToast('加载失败', 'danger')
      })
    } else {
      setPlan(p)
      setView('plan')
    }
  }

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
}p.id}
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
    if (!p.entries) {
      api.get(`/api/travel/plans/${p.id}`).then(d => {
        if (d.ok) { setPlan(d.plan); setView('plan') }
        else showToast('加载失败', 'danger')
      })
    } else {
      setPlan(p)
      setView('plan')
    }
  }

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
p.id}
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
    if (!p.entries) {
      api.get(`/api/travel/plans/${p.id}`).then(d => {
        if (d.ok) { setPlan(d.plan); setView('plan') }
        else showToast('加载失败', 'danger')
      })
    } else {
      setPlan(p)
      setView('plan')
    }
  }

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
