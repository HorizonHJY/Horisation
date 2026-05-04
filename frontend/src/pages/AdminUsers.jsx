import React, { useState, useEffect } from 'react'
import { useAuth } from '../App'
import { useNavigate } from 'react-router-dom'
import HandLoader from '../components/HandLoader'
import { api } from '../api'
import { validatePassword } from '../utils'

const ROLES = ['horizon', 'horizonadmin', 'vip1', 'vip2', 'vip3', 'test', 'user']
const ROLE_COLORS = {
  horizon: 'role-horizon', horizonadmin: 'role-horizonadmin',
  vip1: 'role-vip1', vip2: 'role-vip2', vip3: 'role-vip3', user: 'role-user',
}

const EMPTY_NEW       = { username: '', password: '', role: 'user', email: '', display_name: '' }
const EMPTY_CODE_FORM = { code: '', valid_from: '', valid_to: '' }

function todayStr() {
  return new Date().toISOString().slice(0, 10)
}
function plusDaysStr(n) {
  const d = new Date(); d.setDate(d.getDate() + n)
  return d.toISOString().slice(0, 10)
}

export default function AdminUsers() {
  const { user } = useAuth()
  const navigate  = useNavigate()

  const [users, setUsers]     = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch]   = useState('')
  const [msg, setMsg]         = useState(null)

  // Create user modal state
  const [newUser, setNewUser]   = useState(EMPTY_NEW)
  const [creating, setCreating] = useState(false)

  // Edit modal state
  const [editTarget, setEditTarget] = useState(null)
  const [editForm, setEditForm]     = useState({ display_name: '', email: '', password: '' })
  const [saving, setSaving]         = useState(false)

  // Invite codes state (horizon only)
  const [codes, setCodes]           = useState([])
  const [codesLoading, setCodesLoading] = useState(false)
  const [codeForm, setCodeForm]     = useState({ ...EMPTY_CODE_FORM, valid_from: todayStr(), valid_to: plusDaysStr(30) })
  const [creatingCode, setCreatingCode] = useState(false)

  const isAdmin   = user?.role_info?.permissions?.includes('admin')
  const isHorizon = user?.username === 'horizon'

  useEffect(() => {
    if (!isAdmin) { navigate('/home'); return }
    load()
    if (isHorizon) loadCodes()
  }, [])

  const load = () =>
    api.get('/api/auth/users')
       .then(d => { if (d.ok) setUsers(d.users) })
       .finally(() => setLoading(false))

  const loadCodes = () => {
    setCodesLoading(true)
    api.get('/api/auth/invite-codes')
       .then(d => { if (d.ok) setCodes(d.codes) })
       .finally(() => setCodesLoading(false))
  }

  const createCode = async (e) => {
    e.preventDefault()
    setCreatingCode(true)
    const d = await api.post('/api/auth/invite-codes', codeForm)
    setCreatingCode(false)
    if (d.ok) {
      flash('Invite code created.')
      setCodeForm({ ...EMPTY_CODE_FORM, valid_from: todayStr(), valid_to: plusDaysStr(30) })
      document.getElementById('codeModal').querySelector('[data-bs-dismiss="modal"]').click()
      loadCodes()
    } else {
      flash(d.error, 'danger')
    }
  }

  const deleteCode = async (id) => {
    if (!window.confirm('Delete this invite code?')) return
    const d = await api.delete(`/api/auth/invite-codes/${id}`)
    if (d.ok) { flash('Invite code deleted.'); loadCodes() }
    else flash(d.error, 'danger')
  }

  const flash = (text, type = 'success') => {
    setMsg({ type, text })
    setTimeout(() => setMsg(null), 3000)
  }

  // ── Role ──────────────────────────────────────────────────────────
  const updateRole = async (username, role) => {
    const d = await api.put(`/api/auth/users/${username}/role`, { role })
    if (d.ok) { flash(`${username} role updated.`); load() }
    else flash(d.error, 'danger')
  }

  // ── Activate / Deactivate ─────────────────────────────────────────
  const toggleStatus = async (username, is_active) => {
    const d = await api.put(`/api/auth/users/${username}/status`, { is_active })
    if (d.ok) load()
    else flash(d.error, 'danger')
  }

  // ── Delete ────────────────────────────────────────────────────────
  const deleteUser = async (username) => {
    if (!window.confirm(`Delete user "${username}"? This cannot be undone.`)) return
    const d = await api.delete(`/api/auth/users/${username}`)
    if (d.ok) { flash(`${username} deleted.`); load() }
    else flash(d.error, 'danger')
  }

  // ── Create ────────────────────────────────────────────────────────
  const createUser = async (e) => {
    e.preventDefault()
    const pwErr = validatePassword(newUser.password)
    if (pwErr) { flash(pwErr, 'danger'); return }
    setCreating(true)
    const d = await api.post('/api/auth/register', newUser)
    if (d.ok) {
      flash('User created.')
      setNewUser(EMPTY_NEW)
      document.getElementById('createModal').querySelector('[data-bs-dismiss="modal"]').click()
      load()
    } else {
      flash(d.error, 'danger')
    }
    setCreating(false)
  }

  // ── Edit (open modal) ─────────────────────────────────────────────
  const openEdit = (u) => {
    setEditTarget(u)
    setEditForm({ display_name: u.display_name, email: u.email, password: '' })
  }

  // ── Edit (save) ───────────────────────────────────────────────────
  const saveEdit = async (e) => {
    e.preventDefault()
    if (!editTarget) return
    setSaving(true)

    const username = editTarget.username
    let ok = true

    // Update profile (display_name + email)
    const profileChanged =
      editForm.display_name !== editTarget.display_name ||
      editForm.email        !== editTarget.email

    if (profileChanged) {
      const d = await api.put(`/api/auth/users/${username}/profile`, {
        display_name: editForm.display_name,
        email:        editForm.email,
      })
      if (!d.ok) { flash(d.error, 'danger'); ok = false }
    }

    // Reset password (only if filled in)
    if (ok && editForm.password) {
      const pwErr = validatePassword(editForm.password)
      if (pwErr) { flash(pwErr, 'danger'); setSaving(false); return }
      const d = await api.put(`/api/auth/users/${username}/password`, {
        password: editForm.password,
      })
      if (!d.ok) { flash(d.error, 'danger'); ok = false }
    }

    setSaving(false)
    if (ok) {
      flash(`${username} updated.`)
      document.getElementById('editModal').querySelector('[data-bs-dismiss="modal"]').click()
      load()
    }
  }

  const filtered = users.filter(u =>
    u.username.toLowerCase().includes(search.toLowerCase()) ||
    u.display_name.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <>
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h4 className="fw-bold mb-0">User Management</h4>
        <button className="btn btn-primary btn-sm" data-bs-toggle="modal" data-bs-target="#createModal">
          <i className="fas fa-plus me-1" />Add User
        </button>
      </div>

      {msg && (
        <div className={`alert alert-${msg.type} alert-dismissible py-2`}>
          {msg.text}
          <button type="button" className="btn-close" onClick={() => setMsg(null)} />
        </div>
      )}

      {/* Stats */}
      <div className="row g-3 mb-4">
        {[
          ['Total Users', users.length,                                                             'fa-users',      'primary'],
          ['Active',      users.filter(u => u.is_active).length,                                    'fa-user-check', 'success'],
          ['Admins',      users.filter(u => ['horizon','horizonadmin'].includes(u.role)).length,     'fa-user-shield','warning'],
        ].map(([label, val, icon, color]) => (
          <div key={label} className="col-4">
            <div className="stat-card">
              <i className={`fas ${icon} text-${color} mb-2`} style={{ fontSize: '1.4rem' }} />
              <div className="stat-number">{val}</div>
              <div className="stat-label">{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Search */}
      <div className="mb-3">
        <input
          className="form-control"
          placeholder="Search users..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {/* User list */}
      <div className="card">
        {loading ? (
          <div className="text-center p-5"><HandLoader /></div>
        ) : (
          <div className="list-group list-group-flush">
            {filtered.map(u => (
              <div key={u.username} className="list-group-item d-flex align-items-center gap-3 py-3">
                <div
                  className="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
                  style={{ width: 40, height: 40, background: '#6b9cdb1a', color: '#6b9cdb', fontWeight: 700 }}
                >
                  {u.display_name?.[0]?.toUpperCase()}
                </div>

                <div className="flex-grow-1">
                  <div className="fw-semibold">
                    {u.display_name} <span className="text-muted fw-normal">@{u.username}</span>
                  </div>
                  <span className={`role-badge ${ROLE_COLORS[u.role] ?? 'role-user'}`}>{u.role}</span>
                </div>

                <div className="d-flex align-items-center gap-2 flex-shrink-0">
                  {/* Role selector */}
                  <select
                    className="form-select form-select-sm"
                    style={{ width: 130 }}
                    value={u.role}
                    onChange={e => updateRole(u.username, e.target.value)}
                  >
                    {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
                  </select>

                  {/* Edit button */}
                  <button
                    className="btn btn-sm btn-outline-primary"
                    data-bs-toggle="modal"
                    data-bs-target="#editModal"
                    onClick={() => openEdit(u)}
                  >
                    <i className="fas fa-pen" />
                  </button>

                  {/* Activate / Deactivate */}
                  <button
                    className={`btn btn-sm ${u.is_active ? 'btn-outline-warning' : 'btn-outline-success'}`}
                    onClick={() => toggleStatus(u.username, !u.is_active)}
                  >
                    {u.is_active ? 'Deactivate' : 'Activate'}
                  </button>

                  {/* Delete (hide for root admin) */}
                  {u.username !== 'horizon' && (
                    <button
                      className="btn btn-sm btn-outline-danger"
                      onClick={() => deleteUser(u.username)}
                    >
                      <i className="fas fa-trash" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Invite Codes Section (horizon only) ── */}
      {isHorizon && (
        <div className="mt-5">
          <div className="d-flex justify-content-between align-items-center mb-3">
            <h5 className="fw-bold mb-0">
              <i className="fas fa-ticket-alt me-2 text-primary" />Invite Codes
            </h5>
            <button className="btn btn-primary btn-sm" data-bs-toggle="modal" data-bs-target="#codeModal">
              <i className="fas fa-plus me-1" />New Code
            </button>
          </div>

          <div className="card">
            {codesLoading ? (
              <div className="text-center p-4"><HandLoader /></div>
            ) : codes.length === 0 ? (
              <div className="text-center py-4 text-muted small">No invite codes yet.</div>
            ) : (
              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Code</th>
                      <th>Valid From</th>
                      <th>Valid To</th>
                      <th>Status</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {codes.map(c => {
                      const now   = new Date().toISOString().slice(0, 10)
                      const live  = c.valid_from <= now && now <= c.valid_to
                      const past  = now > c.valid_to
                      return (
                        <tr key={c.id}>
                          <td><code className="fw-bold">{c.code}</code></td>
                          <td>{c.valid_from}</td>
                          <td>{c.valid_to}</td>
                          <td>
                            {live
                              ? <span className="badge bg-success">Active</span>
                              : past
                              ? <span className="badge bg-secondary">Expired</span>
                              : <span className="badge bg-warning text-dark">Upcoming</span>
                            }
                          </td>
                          <td className="text-end">
                            <button
                              className="btn btn-sm btn-outline-danger"
                              onClick={() => deleteCode(c.id)}
                            >
                              <i className="fas fa-trash" />
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Create User Modal ── */}
      <div className="modal fade" id="createModal" tabIndex="-1">
        <div className="modal-dialog">
          <div className="modal-content">
            <form onSubmit={createUser}>
              <div className="modal-header">
                <h5 className="modal-title">Add New User</h5>
                <button type="button" className="btn-close" data-bs-dismiss="modal" />
              </div>
              <div className="modal-body">
                {[
                  ['Username',     'username',     'text',     'Username'],
                  ['Password',     'password',     'password', 'Min 6 characters'],
                  ['Display Name', 'display_name', 'text',     'Display name'],
                  ['Email',        'email',        'email',    'Email (optional)'],
                ].map(([label, key, type, placeholder]) => (
                  <div className="mb-3" key={key}>
                    <label className="form-label">{label}</label>
                    <input
                      type={type}
                      className="form-control"
                      placeholder={placeholder}
                      value={newUser[key]}
                      onChange={e => setNewUser(u => ({ ...u, [key]: e.target.value }))}
                      required={key !== 'email'}
                    />
                  </div>
                ))}
                <div className="mb-3">
                  <label className="form-label">Role</label>
                  <select
                    className="form-select"
                    value={newUser.role}
                    onChange={e => setNewUser(u => ({ ...u, role: e.target.value }))}
                  >
                    {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
                  </select>
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={creating}>
                  {creating ? <span className="spinner-border spinner-border-sm" /> : 'Create User'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>

      {/* ── Edit User Modal ── */}
      <div className="modal fade" id="editModal" tabIndex="-1">
        <div className="modal-dialog">
          <div className="modal-content">
            <form onSubmit={saveEdit}>
              <div className="modal-header">
                <h5 className="modal-title">
                  Edit User — <span className="text-muted fw-normal">@{editTarget?.username}</span>
                </h5>
                <button type="button" className="btn-close" data-bs-dismiss="modal" />
              </div>
              <div className="modal-body">
                <div className="mb-3">
                  <label className="form-label">Display Name</label>
                  <input
                    type="text"
                    className="form-control"
                    value={editForm.display_name}
                    onChange={e => setEditForm(f => ({ ...f, display_name: e.target.value }))}
                    required
                  />
                </div>
                <div className="mb-3">
                  <label className="form-label">Email</label>
                  <input
                    type="email"
                    className="form-control"
                    value={editForm.email}
                    onChange={e => setEditForm(f => ({ ...f, email: e.target.value }))}
                  />
                </div>
                <div className="mb-3">
                  <label className="form-label">
                    New Password <span className="text-muted fw-normal">(leave blank to keep current)</span>
                  </label>
                  <input
                    type="password"
                    className="form-control"
                    placeholder="Min 6 characters"
                    value={editForm.password}
                    onChange={e => setEditForm(f => ({ ...f, password: e.target.value }))}
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={saving}>
                  {saving ? <span className="spinner-border spinner-border-sm" /> : 'Save Changes'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
      {/* ── New Invite Code Modal ── */}
      <div className="modal fade" id="codeModal" tabIndex="-1">
        <div className="modal-dialog">
          <div className="modal-content">
            <form onSubmit={createCode}>
              <div className="modal-header">
                <h5 className="modal-title">New Invite Code</h5>
                <button type="button" className="btn-close" data-bs-dismiss="modal" />
              </div>
              <div className="modal-body">
                <div className="mb-3">
                  <label className="form-label">Code</label>
                  <input
                    type="text"
                    className="form-control"
                    placeholder="e.g. summer2025"
                    value={codeForm.code}
                    onChange={e => setCodeForm(f => ({ ...f, code: e.target.value }))}
                    required
                  />
                </div>
                <div className="row g-2">
                  <div className="col-6">
                    <label className="form-label">Valid From</label>
                    <input
                      type="date"
                      className="form-control"
                      value={codeForm.valid_from}
                      onChange={e => setCodeForm(f => ({ ...f, valid_from: e.target.value }))}
                      required
                    />
                  </div>
                  <div className="col-6">
                    <label className="form-label">Valid To</label>
                    <input
                      type="date"
                      className="form-control"
                      value={codeForm.valid_to}
                      onChange={e => setCodeForm(f => ({ ...f, valid_to: e.target.value }))}
                      required
                    />
                  </div>
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={creatingCode}>
                  {creatingCode ? <span className="spinner-border spinner-border-sm" /> : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </>
  )
}
