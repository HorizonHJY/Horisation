import React, { useState, useEffect } from 'react'
import { api } from '../api'

const ICON_MAP = {
  clothing: 'fa-tshirt', furniture: 'fa-couch', kitchen: 'fa-utensils',
  electronics: 'fa-laptop', beauty: 'fa-spa', books: 'fa-book', other: 'fa-box',
}

function Toast({ toast }) {
  if (!toast) return null
  return (
    <div className={`alert alert-${toast.type} position-fixed top-0 end-0 m-3`} style={{ zIndex: 9999 }}>
      {toast.msg}
    </div>
  )
}

export default function SystemManagement() {
  const [categories, setCategories]   = useState([])
  const [loading, setLoading]         = useState(true)
  const [saving, setSaving]           = useState(null)   // slug being saved
  const [deleting, setDeleting]       = useState(null)   // slug being deleted
  const [toast, setToast]             = useState(null)
  const [newRow, setNewRow]           = useState({ slug: '', label: '', order: 0, active: true })
  const [addingNew, setAddingNew]     = useState(false)

  function showToast(msg, type = 'success') {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 2800)
  }

  async function loadCategories() {
    setLoading(true)
    const d = await api.get('/api/market/categories/all')
    if (d.ok) setCategories(d.categories)
    setLoading(false)
  }

  useEffect(() => { loadCategories() }, [])

  function updateLocal(slug, field, value) {
    setCategories(prev => prev.map(c => c.slug === slug ? { ...c, [field]: value } : c))
  }

  async function handleSave(cat) {
    setSaving(cat.slug)
    const d = await api.put(`/api/market/categories/${cat.slug}`, {
      label: cat.label, order: Number(cat.order), active: cat.active,
    })
    setSaving(null)
    if (d.ok) showToast(`"${cat.label}" saved.`)
    else showToast(d.error || 'Save failed.', 'danger')
  }

  async function handleDelete(cat) {
    if (!window.confirm(`Delete category "${cat.label}" (${cat.slug})? Existing listings with this category will keep the slug but it won't appear in filters.`)) return
    setDeleting(cat.slug)
    const d = await api.delete(`/api/market/categories/${cat.slug}`)
    setDeleting(null)
    if (d.ok) { showToast(`"${cat.label}" deleted.`); loadCategories() }
    else showToast(d.error || 'Delete failed.', 'danger')
  }

  async function handleAddNew() {
    const { slug, label, order, active } = newRow
    if (!slug.trim() || !label.trim()) { showToast('Slug and label are required.', 'warning'); return }
    const d = await api.post('/api/market/categories', { slug: slug.trim(), label: label.trim(), order: Number(order), active })
    if (d.ok) {
      showToast(`Category "${label}" created.`)
      setNewRow({ slug: '', label: '', order: 0, active: true })
      setAddingNew(false)
      loadCategories()
    } else {
      showToast(d.error || 'Create failed.', 'danger')
    }
  }

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 860 }}>
      <Toast toast={toast} />

      <div className="d-flex align-items-center mb-4">
        <i className="fas fa-cogs fa-lg me-2 text-primary" />
        <h4 className="mb-0 fw-bold">System Management</h4>
      </div>

      {/* ── Categories ── */}
      <div className="card shadow-sm">
        <div className="card-header d-flex align-items-center justify-content-between py-2">
          <span className="fw-semibold">
            <i className="fas fa-tags me-2 text-primary" />Market Categories
          </span>
          <button
            className="btn btn-sm btn-primary"
            onClick={() => setAddingNew(v => !v)}
          >
            <i className={`fas ${addingNew ? 'fa-times' : 'fa-plus'} me-1`} />
            {addingNew ? 'Cancel' : 'Add Category'}
          </button>
        </div>

        <div className="card-body p-0">
          {/* New row form */}
          {addingNew && (
            <div className="p-3 border-bottom bg-light">
              <div className="row g-2 align-items-end">
                <div className="col-12 col-sm-3">
                  <label className="form-label small fw-medium mb-1">Slug <span className="text-danger">*</span></label>
                  <input
                    className="form-control form-control-sm font-monospace"
                    placeholder="e.g. sports"
                    value={newRow.slug}
                    onChange={e => setNewRow(r => ({ ...r, slug: e.target.value.toLowerCase().replace(/\s/g, '_') }))}
                  />
                </div>
                <div className="col-12 col-sm-3">
                  <label className="form-label small fw-medium mb-1">Label <span className="text-danger">*</span></label>
                  <input
                    className="form-control form-control-sm"
                    placeholder="Display name"
                    value={newRow.label}
                    onChange={e => setNewRow(r => ({ ...r, label: e.target.value }))}
                  />
                </div>
                <div className="col-6 col-sm-2">
                  <label className="form-label small fw-medium mb-1">Order</label>
                  <input
                    type="number" className="form-control form-control-sm"
                    value={newRow.order}
                    onChange={e => setNewRow(r => ({ ...r, order: e.target.value }))}
                  />
                </div>
                <div className="col-6 col-sm-2 d-flex align-items-center gap-2 pt-3">
                  <div className="form-check form-switch mb-0">
                    <input
                      className="form-check-input" type="checkbox"
                      checked={newRow.active}
                      onChange={e => setNewRow(r => ({ ...r, active: e.target.checked }))}
                      id="new-active"
                    />
                    <label className="form-check-label small" htmlFor="new-active">Active</label>
                  </div>
                </div>
                <div className="col-12 col-sm-2">
                  <button className="btn btn-sm btn-success w-100" onClick={handleAddNew}>
                    <i className="fas fa-check me-1" />Save
                  </button>
                </div>
              </div>
            </div>
          )}

          {loading ? (
            <div className="text-center py-5 text-muted">Loading…</div>
          ) : (
            <div className="table-responsive">
              <table className="table table-hover mb-0" style={{ fontSize: '.85rem' }}>
                <thead className="table-light">
                  <tr>
                    <th style={{ width: 36 }}></th>
                    <th style={{ width: 130 }}>Slug</th>
                    <th>Label</th>
                    <th style={{ width: 80 }}>Order</th>
                    <th style={{ width: 80 }}>Active</th>
                    <th style={{ width: 100 }}></th>
                  </tr>
                </thead>
                <tbody>
                  {categories.map(cat => (
                    <tr key={cat.slug} style={{ opacity: cat.active ? 1 : 0.5 }}>
                      <td className="text-center align-middle text-muted">
                        <i className={`fas ${ICON_MAP[cat.slug] || 'fa-tag'}`} />
                      </td>
                      <td className="align-middle">
                        <code style={{ fontSize: '.78rem' }}>{cat.slug}</code>
                      </td>
                      <td className="align-middle">
                        <input
                          className="form-control form-control-sm"
                          value={cat.label}
                          onChange={e => updateLocal(cat.slug, 'label', e.target.value)}
                        />
                      </td>
                      <td className="align-middle">
                        <input
                          type="number"
                          className="form-control form-control-sm"
                          value={cat.order}
                          onChange={e => updateLocal(cat.slug, 'order', e.target.value)}
                          style={{ width: 64 }}
                        />
                      </td>
                      <td className="align-middle">
                        <div className="form-check form-switch mb-0">
                          <input
                            className="form-check-input" type="checkbox"
                            checked={cat.active}
                            onChange={e => updateLocal(cat.slug, 'active', e.target.checked)}
                            id={`active-${cat.slug}`}
                          />
                          <label className="form-check-label" htmlFor={`active-${cat.slug}`} />
                        </div>
                      </td>
                      <td className="align-middle">
                        <div className="d-flex gap-1">
                          <button
                            className="btn btn-xs btn-primary"
                            style={{ fontSize: '.72rem', padding: '2px 8px' }}
                            disabled={saving === cat.slug}
                            onClick={() => handleSave(cat)}
                          >
                            {saving === cat.slug
                              ? <span className="spinner-border spinner-border-sm" />
                              : <><i className="fas fa-save me-1" />Save</>
                            }
                          </button>
                          <button
                            className="btn btn-xs btn-outline-danger"
                            style={{ fontSize: '.72rem', padding: '2px 8px' }}
                            disabled={deleting === cat.slug}
                            onClick={() => handleDelete(cat)}
                          >
                            {deleting === cat.slug
                              ? <span className="spinner-border spinner-border-sm" />
                              : <i className="fas fa-trash" />
                            }
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="card-footer text-muted small">
          <i className="fas fa-info-circle me-1" />
          Inactive categories are hidden from the Market form but preserved on existing listings.
          Slug cannot be changed after creation.
        </div>
      </div>
    </div>
  )
}
