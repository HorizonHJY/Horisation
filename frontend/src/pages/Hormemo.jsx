import React, { useState, useEffect } from 'react'
import { useAuth } from '../App'
import { api } from '../api'

const TYPES     = ['general', 'todo', 'reminder', 'idea']
const PRIORITIES = ['low', 'normal', 'high']
const FILTERS   = ['all', 'active', 'completed']

const PRIORITY_COLORS = { high: 'danger', normal: 'primary', low: 'secondary' }
const TYPE_ICONS      = { general: 'fa-circle', todo: 'fa-check-square', reminder: 'fa-bell', idea: 'fa-lightbulb' }

export default function Hormemo() {
  const { user } = useAuth()
  const [memos, setMemos]     = useState([])
  const [stats, setStats]     = useState(null)
  const [filter, setFilter]   = useState('all')
  const [loading, setLoading] = useState(true)
  const [editing, setEditing] = useState(null)   // memo being edited
  const [form, setForm]       = useState({ content: '', type: 'general', priority: 'normal' })
  const [adding, setAdding]   = useState(false)
  const [toast, setToast]     = useState(null)

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 2500)
  }

  const loadMemos = async (status) => {
    setLoading(true)
    const params = status && status !== 'all' ? `?status=${status}` : ''
    const d = await api.get(`/api/memos/${params}`)
    if (d.ok) setMemos(d.memos)
    setLoading(false)
  }

  const loadStats = async () => {
    const d = await api.get('/api/memos/statistics')
    if (d.ok) setStats(d.statistics)
  }

  useEffect(() => { loadMemos(filter); loadStats() }, [filter])

  const createMemo = async () => {
    if (!form.content.trim()) { showToast('Content is required.', 'danger'); return }
    setAdding(true)
    const d = await api.post('/api/memos/', form)
    if (d.ok) {
      setForm({ content: '', type: 'general', priority: 'normal' })
      showToast('Memo added!')
      loadMemos(filter)
      loadStats()
    } else {
      showToast(d.error, 'danger')
    }
    setAdding(false)
  }

  const completeMemo = async (id) => {
    await api.post(`/api/memos/${id}/complete`)
    showToast('Marked as completed!')
    loadMemos(filter)
    loadStats()
  }

  const deleteMemo = async (id) => {
    await api.delete(`/api/memos/${id}`)
    showToast('Memo deleted.')
    loadMemos(filter)
    loadStats()
  }

  const saveEdit = async () => {
    if (!editing) return
    await api.put(`/api/memos/${editing.id}`, { content: editing.content, priority: editing.priority })
    showToast('Memo updated.')
    setEditing(null)
    loadMemos(filter)
  }

  return (
    <>
      {/* Toast */}
      {toast && (
        <div
          className={`alert alert-${toast.type} position-fixed py-2 px-3`}
          style={{ top: 80, right: 24, zIndex: 9999, minWidth: 220, boxShadow: '0 4px 12px rgba(0,0,0,.15)' }}
        >
          {toast.msg}
        </div>
      )}

      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h4 className="fw-bold mb-0">{user?.display_name}'s Hormemo</h4>
          <span className="text-muted small">Your personal memo board</span>
        </div>
        <button className="btn btn-sm btn-outline-secondary" onClick={() => { loadMemos(filter); loadStats() }}>
          <i className="fas fa-sync-alt" />
        </button>
      </div>

      {/* Stats */}
      {stats && (
        <div className="row g-3 mb-4">
          {[
            ['Total',     stats.total_memos,                          'fa-clipboard',     'primary'],
            ['Active',    stats.status_stats?.active ?? 0,            'fa-spinner',       'warning'],
            ['Completed', stats.status_stats?.completed ?? 0,         'fa-check-circle',  'success'],
            ['High Pri',  stats.priority_stats?.high ?? 0,            'fa-exclamation',   'danger'],
          ].map(([label, val, icon, color]) => (
            <div key={label} className="col-6 col-md-3">
              <div className="stat-card">
                <i className={`fas ${icon} text-${color} mb-1`} style={{ fontSize: '1.2rem' }} />
                <div className="stat-number">{val}</div>
                <div className="stat-label">{label}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="row g-4">
        {/* Add memo */}
        <div className="col-12 col-md-4">
          <div className="card p-3">
            <h6 className="fw-semibold mb-3">Add Memo</h6>
            <textarea
              className="form-control mb-2"
              rows={3}
              placeholder="What's on your mind?"
              value={form.content}
              onChange={e => setForm(f => ({ ...f, content: e.target.value }))}
            />
            <div className="row g-2 mb-3">
              <div className="col-6">
                <select className="form-select form-select-sm" value={form.type} onChange={e => setForm(f => ({ ...f, type: e.target.value }))}>
                  {TYPES.map(t => <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>)}
                </select>
              </div>
              <div className="col-6">
                <select className="form-select form-select-sm" value={form.priority} onChange={e => setForm(f => ({ ...f, priority: e.target.value }))}>
                  {PRIORITIES.map(p => <option key={p} value={p}>{p.charAt(0).toUpperCase() + p.slice(1)}</option>)}
                </select>
              </div>
            </div>
            <button className="btn btn-primary w-100" onClick={createMemo} disabled={adding}>
              {adding ? <span className="spinner-border spinner-border-sm" /> : <><i className="fas fa-plus me-2" />Add</>}
            </button>
          </div>
        </div>

        {/* Memo list */}
        <div className="col-12 col-md-8">
          {/* Filter tabs */}
          <div className="d-flex gap-2 mb-3">
            {FILTERS.map(f => (
              <button
                key={f}
                className={`btn btn-sm ${filter === f ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setFilter(f)}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>

          {loading ? (
            <div className="text-center p-5"><div className="spinner-border text-primary" /></div>
          ) : memos.length === 0 ? (
            <div className="card text-center p-5 text-muted">
              <i className="fas fa-inbox mb-2" style={{ fontSize: '2rem', opacity: .3 }} />
              <div>No memos yet. Add one!</div>
            </div>
          ) : (
            <div className="d-flex flex-column gap-2">
              {memos.map(m => (
                <div
                  key={m.id}
                  className={`card p-3 ${m.status === 'completed' ? 'opacity-65' : ''}`}
                  style={{ borderLeft: `4px solid var(--bs-${PRIORITY_COLORS[m.priority] ?? 'secondary'})` }}
                >
                  {editing?.id === m.id ? (
                    <>
                      <textarea
                        className="form-control form-control-sm mb-2"
                        rows={2}
                        value={editing.content}
                        onChange={e => setEditing(ed => ({ ...ed, content: e.target.value }))}
                      />
                      <div className="d-flex gap-2 align-items-center">
                        <select
                          className="form-select form-select-sm w-auto"
                          value={editing.priority}
                          onChange={e => setEditing(ed => ({ ...ed, priority: e.target.value }))}
                        >
                          {PRIORITIES.map(p => <option key={p} value={p}>{p}</option>)}
                        </select>
                        <button className="btn btn-success btn-sm" onClick={saveEdit}>Save</button>
                        <button className="btn btn-secondary btn-sm" onClick={() => setEditing(null)}>Cancel</button>
                      </div>
                    </>
                  ) : (
                    <div className="d-flex justify-content-between align-items-start gap-2">
                      <div className="flex-grow-1">
                        <div className="d-flex align-items-center gap-2 mb-1">
                          <i className={`fas ${TYPE_ICONS[m.type] ?? 'fa-circle'} text-muted`} style={{ fontSize: '.75rem' }} />
                          <span
                            className={`badge bg-${PRIORITY_COLORS[m.priority]}-subtle text-${PRIORITY_COLORS[m.priority]}`}
                            style={{ fontSize: '.65rem' }}
                          >
                            {m.priority}
                          </span>
                          <span className="text-muted" style={{ fontSize: '.72rem' }}>{m.type}</span>
                        </div>
                        <p className={`mb-0 ${m.status === 'completed' ? 'text-decoration-line-through text-muted' : ''}`}>
                          {m.content}
                        </p>
                      </div>
                      <div className="d-flex gap-1 flex-shrink-0">
                        {m.status !== 'completed' && (
                          <button className="btn btn-sm btn-outline-success" title="Complete" onClick={() => completeMemo(m.id)}>
                            <i className="fas fa-check" />
                          </button>
                        )}
                        <button className="btn btn-sm btn-outline-secondary" title="Edit" onClick={() => setEditing({ ...m })}>
                          <i className="fas fa-edit" />
                        </button>
                        <button className="btn btn-sm btn-outline-danger" title="Delete" onClick={() => deleteMemo(m.id)}>
                          <i className="fas fa-trash" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  )
}
