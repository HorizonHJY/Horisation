import React, { useState, useEffect } from 'react'
import { api } from '../api'

// Predefined icon options for category picker
const ICON_OPTIONS = [
  { value: 'fa-tag',          label: 'Default (Tag)' },
  { value: 'fa-tshirt',       label: 'Clothing / 衣服' },
  { value: 'fa-couch',        label: 'Furniture / 家具' },
  { value: 'fa-utensils',     label: 'Kitchen / 厨具' },
  { value: 'fa-laptop',       label: 'Electronics' },
  { value: 'fa-spa',          label: 'Beauty / 美妆' },
  { value: 'fa-book',         label: 'Books' },
  { value: 'fa-box',          label: 'Other / 其他' },
  { value: 'fa-home',         label: 'Home Goods / 生活用品' },
  { value: 'fa-baby',         label: 'Baby & Kids / 母婴' },
  { value: 'fa-dumbbell',     label: 'Sports / 运动' },
  { value: 'fa-car',          label: 'Automotive / 汽车' },
  { value: 'fa-camera',       label: 'Camera / 摄影' },
  { value: 'fa-gamepad',      label: 'Gaming / 游戏' },
  { value: 'fa-paw',          label: 'Pets / 宠物' },
  { value: 'fa-tools',        label: 'Tools / 工具' },
  { value: 'fa-graduation-cap', label: 'Education / 教育' },
  { value: 'fa-music',        label: 'Music / 音乐' },
]

function IconSelect({ value, onChange, id }) {
  return (
    <select
      className="form-select form-select-sm"
      value={value}
      onChange={e => onChange(e.target.value)}
      id={id}
    >
      {ICON_OPTIONS.map(opt => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  )
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
  const [saving, setSaving]           = useState(null)
  const [deleting, setDeleting]       = useState(null)
  const [toast, setToast]             = useState(null)
  const [newRow, setNewRow]           = useState({ slug: '', label: '', order: 0, active: true, icon: 'fa-tag' })
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
      label: cat.label, order: Number(cat.order), active: cat.active, icon: cat.icon || 'fa-tag',
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
    const { slug, label, order, active, icon } = newRow
    if (!slug.trim() || !label.trim()) { showToast('Slug and label are required.', 'warning'); return }
    const d = await api.post('/api/market/categories', {
      slug: slug.trim(), label: label.trim(), order: Number(order), active, icon: icon || 'fa-tag',
    })
    if (d.ok) {
      showToast(`Category "${label}" created.`)
      setNewRow({ slug: '', label: '', order: 0, active: true, icon: 'fa-tag' })
      setAddingNew(false)
      loadCategories()
    } else {
      showToast(d.error || 'Create failed.', 'danger')
    }
  }

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 900 }}>
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
                <div className="col-12 col-sm-3">
                  <label className="form-label small fw-medium mb-1">Icon</label>
                  <div className="d-flex align-items-center gap-2">
                    <i className={`fas ${newRow.icon || 'fa-tag'} text-muted`} style={{ width: 16, textAlign: 'center' }} />
                    <div className="flex-grow-1">
                      <IconSelect
                        value={newRow.icon || 'fa-tag'}
                        onChange={v => setNewRow(r => ({ ...r, icon: v }))}
                        id="new-icon"
                      />
                    </div>
                  </div>
                </div>
                <div className="col-4 col-sm-1">
                  <label className="form-label small fw-medium mb-1">Order</label>
                  <input
                    type="number" className="form-control form-control-sm"
                    value={newRow.order}
                    onChange={e => setNewRow(r => ({ ...r, order: e.target.value }))}
                  />
                </div>
                <div className="col-4 col-sm-1 d-flex align-items-center pt-3">
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
                <div className="col-4 col-sm-1">
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
                    <th style={{ width: 120 }}>Slug</th>
                    <th style={{ minWidth: 120 }}>Label</th>
                    <th style={{ minWidth: 160 }}>Icon</th>
                    <th style={{ width: 72 }}>Order</th>
                    <th style={{ width: 72 }}>Active</th>
                    <th style={{ width: 96 }}></th>
                  </tr>
                </thead>
                <tbody>
                  {categories.map(cat => (
                    <tr key={cat.slug} style={{ opacity: cat.active ? 1 : 0.5 }}>
                      <td className="text-center align-middle text-muted">
                        <i className={`fas ${cat.icon || 'fa-tag'}`} />
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
                        <IconSelect
                          value={cat.icon || 'fa-tag'}
                          onChange={v => updateLocal(cat.slug, 'icon', v)}
                          id={`icon-${cat.slug}`}
                        />
                      </td>
                      <td className="align-middle">
                        <input
                          type="number"
                          className="form-control form-control-sm"
                          value={cat.order}
                          onChange={e => updateLocal(cat.slug, 'order', e.target.value)}
                          style={{ width: 56 }}
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
