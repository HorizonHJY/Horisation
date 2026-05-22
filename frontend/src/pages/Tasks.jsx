import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'
import HandLoader from '../components/HandLoader'

/* ═══════════════════════════════════════════════════════════════════
   Task/Bounty categories with icons and colors
   ═══════════════════════════════════════════════════════════════════ */
const TASK_CATEGORIES = [
  { slug: 'grocery',      label: '逛超市',   icon: 'fa-shopping-basket',  color: '#27ae60' },
  { slug: 'airport',      label: '接送机',   icon: 'fa-plane-departure',  color: '#2d7dd2' },
  { slug: 'delivery',     label: '跑腿代取',  icon: 'fa-box',              color: '#e67e22' },
  { slug: 'pet',          label: '遛狗/宠物',  icon: 'fa-paw',             color: '#8e44ad' },
  { slug: 'moving',       label: '搬家搬运',  icon: 'fa-truck-moving',     color: '#c0392b' },
  { slug: 'tech_support', label: '技术支援',  icon: 'fa-wrench',           color: '#2c3e50' },
  { slug: 'tutoring',     label: '辅导/教学',  icon: 'fa-chalkboard-teacher', color: '#16a085' },
  { slug: 'other',        label: '其他',      icon: 'fa-ellipsis-h',       color: '#7f8c8d' },
]

const CATEGORY_MAP = Object.fromEntries(TASK_CATEGORIES.map(c => [c.slug, c]))

const EMPTY_FORM = {
  title: '', description: '', category: 'grocery',
  bounty: '', location: '', due_date: '',
}

const STATUS_CONFIG = {
  open:         { label: 'Open',    color: '#27ae60', bg: 'rgba(39,174,96,0.08)' },
  in_progress:  { label: '进行中',   color: '#e67e22', bg: 'rgba(230,126,34,0.08)' },
  completed:    { label: '已完成',   color: '#7f8c8d', bg: 'rgba(127,140,141,0.08)' },
  cancelled:    { label: '已取消',   color: '#c0392b', bg: 'rgba(192,57,43,0.08)' },
}


/* ── Toast ──────────────────────────────────────────────────────────── */
function useToast() {
  const [toast, setToast] = useState(null)
  const show = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 2800)
  }
  return [toast, show]
}


/* ── Category Badge ────────────────────────────────────────────────── */
function CategoryBadge({ slug, style }) {
  const cat = CATEGORY_MAP[slug] || CATEGORY_MAP.other
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 3,
      fontSize: '.7rem', fontWeight: 500,
      padding: '2px 10px', borderRadius: 20,
      background: `${cat.color}18`,
      color: cat.color,
      ...style,
    }}>
      <i className={`fas ${cat.icon}`} style={{ fontSize: '.65rem' }} />
      {cat.label}
    </span>
  )
}


/* ── Avatar ──────────────────────────────────────────────────────────── */
function PosterAvatar({ username, displayName, avatarUrl, size = 28, onClick }) {
  const style = {
    width: size, height: size, borderRadius: '50%', flexShrink: 0,
    cursor: onClick ? 'pointer' : 'default',
  }
  if (avatarUrl) return (
    <img src={avatarUrl} alt={displayName} style={{ ...style, objectFit: 'cover' }} onClick={onClick} />
  )
  return (
    <div style={{
      ...style,
      background: '#6b9cdb', color: '#fff',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontWeight: 700, fontSize: size * 0.4,
    }} onClick={onClick}>
      {(displayName || username)?.[0]?.toUpperCase() || '?'}
    </div>
  )
}


/* ── Task Card ──────────────────────────────────────────────────────── */
function TaskCard({ task, currentUser, onEdit, onDelete, onInProgress, onComplete, onPosterClick }) {
  const isMine = task.poster_username === currentUser
  const cat    = CATEGORY_MAP[task.category] || CATEGORY_MAP.other
  const statusCfg = STATUS_CONFIG[task.status] || STATUS_CONFIG.open

  return (
    <div className="market-card" style={{ position: 'relative' }}>
      {/* Status badge */}
      <span style={{
        position: 'absolute', top: 8, right: 8,
        fontSize: '.65rem', fontWeight: 600,
        padding: '2px 8px', borderRadius: 12,
        background: statusCfg.bg, color: statusCfg.color,
        zIndex: 2,
      }}>
        {statusCfg.label}
      </span>

      {/* Icon header */}
      <div style={{
        width: '100%',
        aspectRatio: '4 / 3',
        background: `${cat.color}0d`,
        borderRadius: 'var(--radius-sm)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: '1px solid var(--border-soft)',
        position: 'relative', overflow: 'hidden',
      }}>
        <i className={`fas ${cat.icon}`} style={{
          fontSize: '2.8rem', color: `${cat.color}40`,
        }} />
      </div>

      {/* Title */}
      <div className="market-card__title" title={task.title}>
        {task.title}
      </div>

      {/* Category + Bounty */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <CategoryBadge slug={task.category} />
        <span style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          fontSize: '1.15rem',
          color: cat.color,
          letterSpacing: '-0.01em',
        }}>
          ${task.bounty}
        </span>
      </div>

      {/* Description */}
      <p className="market-card__desc">{task.description}</p>

      {/* Location / Due date */}
      {(task.location || task.due_date) && (
        <div style={{ fontSize: '.7rem', color: 'var(--text-muted)', display: 'flex', gap: 12, marginBottom: 4 }}>
          {task.location && (
            <span><i className="fas fa-map-marker-alt me-1" />{task.location}</span>
          )}
          {task.due_date && (
            <span><i className="far fa-calendar-alt me-1" />{task.due_date}</span>
          )}
        </div>
      )}

      <hr className="market-card__divider" style={{ margin: '6px 0' }} />

      {/* Poster */}
      <div
        style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}
        onClick={() => onPosterClick?.(task.poster_username)}
      >
        <PosterAvatar
          username={task.poster_username}
          displayName={task.poster_display}
          avatarUrl={task.poster_avatar}
          size={22}
        />
        <span style={{ fontSize: '.72rem', fontWeight: 500 }}>
          {task.poster_display || task.poster_username}
        </span>
        <span style={{ fontSize: '.65rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          {new Date(task.created_at).toLocaleDateString()}
        </span>
      </div>

      {/* Actions */}
      <div className="market-card__action">
        {isMine && task.status === 'open' && (
          <>
            <button className="market-card__btn" onClick={() => onInProgress(task.id)}>
              <i className="fas fa-play" />开始
            </button>
            <button className="market-card__btn market-card__btn--edit" onClick={() => onEdit(task)}>
              <i className="fas fa-pen" />
            </button>
            <button className="market-card__btn market-card__btn--danger" onClick={() => onDelete(task.id)}>
              <i className="fas fa-trash" />
            </button>
          </>
        )}
        {isMine && task.status === 'in_progress' && (
          <>
            <button className="market-card__btn" style={{ color: '#27ae60', borderColor: 'rgba(39,174,96,0.3)' }}
              onClick={() => onComplete(task.id)}>
              <i className="fas fa-check-circle" />完成
            </button>
            <button className="market-card__btn market-card__btn--danger" onClick={() => onDelete(task.id)}>
              <i className="fas fa-trash" />
            </button>
          </>
        )}
        {isMine && task.status === 'completed' && (
          <span style={{ fontSize: '.7rem', color: 'var(--text-muted)', padding: '4px 0' }}>
            <i className="fas fa-check-circle me-1" style={{ color: '#27ae60' }} />已结束
          </span>
        )}
      </div>
    </div>
  )
}


/* ── Edit/Detail Modal ──────────────────────────────────────────────── */
function TaskEditModal({ task, onClose, onSave, categories }) {
  const [form, setForm] = useState({
    title:       task.title,
    description: task.description,
    category:    task.category,
    bounty:      String(task.bounty),
    location:    task.location || '',
    due_date:    task.due_date || '',
  })
  const [saving, setSaving] = useState(false)
  const [err, setErr]       = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.title.trim())       return setErr('Title is required.')
    if (!form.description.trim()) return setErr('Description is required.')
    if (!form.bounty || isNaN(form.bounty) || Number(form.bounty) < 0)
      return setErr('Enter a valid bounty.')

    setSaving(true)
    await onSave(task.id, {
      title:       form.title.trim(),
      description: form.description.trim(),
      category:    form.category,
      bounty:      Number(form.bounty),
      location:    form.location.trim(),
      due_date:    form.due_date.trim(),
    })
    setSaving(false)
  }

  return (
    <div className="modal show d-block" style={{ background: 'rgba(0,0,0,.45)' }} onClick={onClose}>
      <div className="modal-dialog modal-dialog-centered" onClick={e => e.stopPropagation()}>
        <div className="modal-content">
          <form onSubmit={handleSubmit}>
            <div className="modal-header">
              <h5 className="modal-title fw-semibold">
                <i className="fas fa-edit me-2" style={{ color: '#e67e22' }} />Edit Task
              </h5>
              <button type="button" className="btn-close" onClick={onClose} />
            </div>
            <div className="modal-body">
              {err && <div className="alert alert-danger py-2 small">{err}</div>}

              <div className="mb-3">
                <label className="form-label fw-medium">Title</label>
                <input className="form-control" maxLength={200} value={form.title}
                  onChange={e => setForm(f => ({ ...f, title: e.target.value }))} required />
              </div>

              <div className="mb-3">
                <label className="form-label fw-medium">Description</label>
                <textarea className="form-control" rows={3} value={form.description}
                  onChange={e => setForm(f => ({ ...f, description: e.target.value }))} required />
              </div>

              <div className="row mb-3">
                <div className="col">
                  <label className="form-label fw-medium">Category</label>
                  <select className="form-select" value={form.category}
                    onChange={e => setForm(f => ({ ...f, category: e.target.value }))}>
                    {categories.map(c => (
                      <option key={c.slug} value={c.slug}>{c.label}</option>
                    ))}
                  </select>
                </div>
                <div className="col">
                  <label className="form-label fw-medium">
                    悬赏金额 <span className="text-muted small">(Bounty $)</span>
                  </label>
                  <input type="number" className="form-control" min={0} step="0.01"
                    value={form.bounty}
                    onChange={e => setForm(f => ({ ...f, bounty: e.target.value }))} required />
                </div>
              </div>

              <div className="row mb-3">
                <div className="col">
                  <label className="form-label fw-medium">地点 (Location)</label>
                  <input className="form-control" placeholder="e.g. Delmar Loop"
                    value={form.location}
                    onChange={e => setForm(f => ({ ...f, location: e.target.value }))} />
                </div>
                <div className="col">
                  <label className="form-label fw-medium">截止日期 (Due date)</label>
                  <input type="date" className="form-control"
                    value={form.due_date}
                    onChange={e => setForm(f => ({ ...f, due_date: e.target.value }))} />
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
              <button type="submit" className="btn btn-primary" disabled={saving}>
                {saving ? <span className="spinner-border spinner-border-sm me-1" /> : null}
                Save
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}


/* ── Create Form ────────────────────────────────────────────────────── */
function TaskCreateForm({ categories, onSubmit, submitting }) {
  const [form, setForm] = useState(EMPTY_FORM)
  const [err, setErr]   = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setErr('')
    if (!form.title.trim())       return setErr('请填写标题')
    if (!form.description.trim()) return setErr('请填写描述')
    if (!form.bounty || isNaN(form.bounty) || Number(form.bounty) < 0)
      return setErr('请输入有效的悬赏金额')
    if (Number(form.bounty) === 0 && !window.confirm('悬赏金额为 $0？确定要发布免费任务吗？'))
      return

    const ok = await onSubmit({
      title:       form.title.trim(),
      description: form.description.trim(),
      category:    form.category,
      bounty:      Number(form.bounty),
      location:    form.location.trim(),
      due_date:    form.due_date.trim(),
    })
    if (ok) {
      setForm(EMPTY_FORM)
    }
  }

  const cat = CATEGORY_MAP[form.category] || CATEGORY_MAP.other

  return (
    <div className="row justify-content-center">
      <div className="col-lg-7">
        <div className="card shadow-sm">
          <div className="card-body p-4">
            <h5 className="card-title mb-3 fw-semibold">
              <i className="fas fa-bullhorn me-2" style={{ color: '#e67e22' }} />发布悬赏
            </h5>
            <p className="text-muted small mb-4">
              发布任务并设定悬赏金额，其他小伙伴可以来接单！
            </p>

            <form onSubmit={handleSubmit}>
              {err && <div className="alert alert-danger py-2 small">{err}</div>}

              <div className="mb-3">
                <label className="form-label fw-medium">任务分类</label>
                <div className="d-flex gap-2 flex-wrap">
                  {categories.map(c => {
                    const active = form.category === c.slug
                    return (
                      <button
                        key={c.slug}
                        type="button"
                        onClick={() => setForm(f => ({ ...f, category: c.slug }))}
                        style={{
                          display: 'inline-flex', alignItems: 'center', gap: 5,
                          padding: '6px 14px', borderRadius: 20,
                          border: active ? `2px solid ${cat.color}` : '1px solid var(--border-medium)',
                          background: active ? `${cat.color}12` : 'var(--bg-surface)',
                          color: active ? cat.color : 'var(--text-secondary)',
                          fontWeight: active ? 600 : 400,
                          fontSize: '.82rem',
                          cursor: 'pointer', fontFamily: 'var(--font-body)',
                          transition: 'all .15s',
                        }}
                      >
                        <i className={`fas ${c.icon}`} />
                        {c.label}
                      </button>
                    )
                  })}
                </div>
              </div>

              <div className="mb-3">
                <label className="form-label fw-medium">
                  标题 <span className="text-danger">*</span>
                </label>
                <input
                  className="form-control"
                  maxLength={200}
                  placeholder="e.g. 明早去Costco帮我带些水果"
                  value={form.title}
                  onChange={e => setForm(f => ({ ...f, title: e.target.value }))}
                />
              </div>

              <div className="mb-3">
                <label className="form-label fw-medium">
                  描述 <span className="text-danger">*</span>
                </label>
                <textarea
                  className="form-control"
                  rows={3}
                  placeholder="详细说明需要做什么、有什么要求..."
                  value={form.description}
                  onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                />
              </div>

              <div className="row mb-3">
                <div className="col-md-4">
                  <label className="form-label fw-medium">
                    <i className="fas fa-dollar-sign me-1" />悬赏金额 <span className="text-danger">*</span>
                  </label>
                  <input
                    type="number"
                    className="form-control"
                    min={0}
                    step="0.01"
                    placeholder="0.00"
                    value={form.bounty}
                    onChange={e => setForm(f => ({ ...f, bounty: e.target.value }))}
                  />
                </div>
                <div className="col-md-4">
                  <label className="form-label fw-medium">
                    <i className="fas fa-map-marker-alt me-1" />地点
                  </label>
                  <input
                    className="form-control"
                    placeholder="e.g. Delmar Loop"
                    value={form.location}
                    onChange={e => setForm(f => ({ ...f, location: e.target.value }))}
                  />
                </div>
                <div className="col-md-4">
                  <label className="form-label fw-medium">
                    <i className="far fa-calendar-alt me-1" />截止日期
                  </label>
                  <input
                    type="date"
                    className="form-control"
                    value={form.due_date}
                    onChange={e => setForm(f => ({ ...f, due_date: e.target.value }))}
                  />
                </div>
              </div>

              <button
                type="submit"
                className="btn btn-primary w-100"
                disabled={submitting}
              >
                {submitting
                  ? <><span className="spinner-border spinner-border-sm me-2" />发布中…</>
                  : <><i className="fas fa-paper-plane me-2" />发布悬赏</>
                }
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}


/* ══════════════════════════════════════════════════════════════════════
   Main Market Tasks Component
   ══════════════════════════════════════════════════════════════════════ */
export default function Tasks() {
  const { user } = useAuth()
  const navigate = useNavigate()

  const [tab, setTab] = useState('browse')
  const [tasks, setTasks] = useState([])
  const [myTasks, setMyTasks] = useState([])
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [categoryFilter, setCategoryFilter] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [toast, showToast] = useToast()
  const [editTask, setEditTask] = useState(null)
  const [friendsMap, setFriendsMap] = useState({})
  const tabRef = useRef(tab)
  useEffect(() => { tabRef.current = tab }, [tab])

  const categories = TASK_CATEGORIES

  // Load tasks
  useEffect(() => {
    if (tab === 'browse') { loadBrowse(); setSearchQuery(''); setCategoryFilter('') }
    if (tab === 'my') loadMine()
  }, [tab])

  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState !== 'visible') return
      if (tabRef.current === 'browse') loadBrowse()
      else if (tabRef.current === 'my') loadMine()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [])

  async function loadBrowse() {
    setLoading(true)
    const [tasksRes, friendsRes] = await Promise.all([
      api.get('/api/market/tasks/'),
      api.get('/api/friends/list'),
    ])
    if (friendsRes.ok) {
      setFriendsMap(Object.fromEntries(friendsRes.friends.map(f => [f.username, f])))
    }
    if (tasksRes.ok) setTasks(tasksRes.tasks)
    setLoading(false)
  }

  async function loadMine() {
    setLoading(true)
    const d = await api.get('/api/market/tasks/my')
    if (d.ok) setMyTasks(d.tasks)
    setLoading(false)
  }

  async function handleCreate(data) {
    setSubmitting(true)
    const d = await api.post('/api/market/tasks/', data)
    setSubmitting(false)
    if (d.ok) {
      showToast('悬赏发布成功！🎉')
      setTab('browse')
      return true
    }
    showToast(d.error || '发布失败', 'danger')
    return false
  }

  async function handleEditSave(taskId, data) {
    const d = await api.put(`/api/market/tasks/${taskId}`, data)
    if (d.ok) {
      showToast('任务已更新')
      setEditTask(null)
      loadBrowse()
      loadMine()
    } else {
      showToast(d.error, 'danger')
    }
  }

  async function handleInProgress(taskId) {
    const d = await api.post(`/api/market/tasks/${taskId}/in-progress`)
    if (d.ok) {
      showToast('状态已更新')
      loadBrowse()
      loadMine()
    } else {
      showToast(d.error, 'danger')
    }
  }

  async function handleComplete(taskId) {
    if (!window.confirm('确认此任务已完成？（悬赏将被视为已支付）')) return
    const d = await api.post(`/api/market/tasks/${taskId}/complete`)
    if (d.ok) {
      showToast('任务已完成！✅')
      loadBrowse()
      loadMine()
    } else {
      showToast(d.error, 'danger')
    }
  }

  async function handleDelete(taskId) {
    if (!window.confirm('确定删除此任务？')) return
    const d = await api.delete(`/api/market/tasks/${taskId}`)
    if (d.ok) {
      showToast('任务已删除')
      loadBrowse()
      loadMine()
    } else {
      showToast(d.error, 'danger')
    }
  }

  function handlePosterClick(username) {
    if (username === user.username) return
    const friend = friendsMap[username]
    const partner = friend ?? { username, display_name: username, avatar_url: null }
    navigate('/friends', { state: { openChat: partner } })
  }

  // Filter tasks
  const filteredTasks = tasks.filter(t => {
    const matchCat = !categoryFilter || t.category === categoryFilter
    const q = searchQuery.trim().toLowerCase()
    const matchSearch = !q ||
      t.title.toLowerCase().includes(q) ||
      t.description.toLowerCase().includes(q)
    return matchCat && matchSearch
  })

  const activeTasks = tab === 'my' ? myTasks : filteredTasks

  return (
    <>
      {/* Toast */}
      {toast && (
        <div className={`alert alert-${toast.type} alert-dismissible position-fixed top-0 end-0 m-3`}
             style={{ zIndex: 9999 }}>
          {toast.msg}
        </div>
      )}

      {/* Edit modal */}
      {editTask && (
        <TaskEditModal
          task={editTask}
          categories={categories}
          onClose={() => setEditTask(null)}
          onSave={handleEditSave}
        />
      )}

      {/* Tabs */}
      <div className="radio-inputs mb-4">
        {[
          { key: 'browse', label: 'Browse Tasks' },
          { key: 'my',     label: 'My Tasks' },
          { key: 'create', label: 'Post Task' },
        ].map(t => (
          <label className="radio" key={t.key}>
            <input
              type="radio"
              name="task-tab"
              checked={tab === t.key}
              onChange={() => setTab(t.key)}
            />
            <span className="name">{t.label}</span>
          </label>
        ))}
      </div>

      {/* ── Browse / My Tasks ── */}
      {(tab === 'browse' || tab === 'my') && (
        <>
          {tab === 'browse' && (
            <div className="mb-4">
              {/* Search */}
              <div className="search mb-3">
                <input
                  className="search__input"
                  placeholder="搜索任务…"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
                <button className="search__button">
                  <i className="fas fa-search" />
                </button>
                {searchQuery && (
                  <button className="search__clear" onClick={() => setSearchQuery('')}>
                    <i className="fas fa-times" />
                  </button>
                )}
              </div>

              {/* Category pills */}
              <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
                <div className="d-flex gap-2 align-items-center" style={{ width: 'max-content', paddingBottom: 2 }}>
                  <span className="text-muted small me-1" style={{ whiteSpace: 'nowrap' }}>分类</span>
                  <button
                    onClick={() => setCategoryFilter('')}
                    className={`btn btn-sm ${!categoryFilter ? 'btn-primary' : 'btn-outline-secondary'}`}
                    style={{ borderRadius: 20, whiteSpace: 'nowrap' }}
                  >
                    全部
                  </button>
                  {categories.map(c => {
                    const active = categoryFilter === c.slug
                    return (
                      <button
                        key={c.slug}
                        onClick={() => setCategoryFilter(active ? '' : c.slug)}
                        className={`btn btn-sm ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
                        style={{
                          borderRadius: 20, whiteSpace: 'nowrap',
                          ...(active ? {} : { borderColor: `${c.color}40`, color: c.color }),
                        }}
                      >
                        <i className={`fas ${c.icon} me-1`} />{c.label}
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>
          )}

          {loading ? (
            <div className="text-center py-5"><HandLoader /></div>
          ) : activeTasks.length === 0 ? (
            <div className="text-center py-5 text-muted">
              <i className="fas fa-tasks fa-3x mb-3" />
              {tab === 'browse' && (searchQuery || categoryFilter) ? (
                <>
                  <p>没有找到匹配的任务。</p>
                  <button className="btn btn-outline-secondary btn-sm" onClick={() => { setSearchQuery(''); setCategoryFilter('') }}>
                    Clear filters
                  </button>
                </>
              ) : (
                <>
                  <p>{tab === 'my' ? '你还没有发布过任务。' : '还没有人发布任务。抢先发一个？'}</p>
                  {tab === 'my' && (
                    <button className="btn btn-primary" onClick={() => setTab('create')}>
                      <i className="fas fa-plus me-1" />发布悬赏
                    </button>
                  )}
                </>
              )}
            </div>
          ) : (
            <div className="row row-cols-2 row-cols-sm-3 row-cols-lg-4 row-cols-xl-5 g-2">
              {activeTasks.map(t => (
                <div className="col" key={t.id}>
                  <TaskCard
                    task={t}
                    currentUser={user.username}
                    onEdit={setEditTask}
                    onDelete={handleDelete}
                    onInProgress={handleInProgress}
                    onComplete={handleComplete}
                    onPosterClick={handlePosterClick}
                  />
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* ── Create ── */}
      {tab === 'create' && (
        <TaskCreateForm
          categories={categories}
          onSubmit={handleCreate}
          submitting={submitting}
        />
      )}
    </>
  )
}
