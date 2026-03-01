import React, { useState, useRef, useCallback } from 'react'
import { api } from '../api'

const ENCODINGS = [
  { value: '', label: 'Auto-detect' },
  { value: 'utf-8', label: 'UTF-8' },
  { value: 'utf-8-sig', label: 'UTF-8 BOM' },
  { value: 'gbk', label: 'GBK (Chinese)' },
  { value: 'gb2312', label: 'GB2312' },
  { value: 'latin1', label: 'Latin-1' },
  { value: 'cp1252', label: 'CP1252' },
]

const DTYPE_COLORS = { numeric: '#3a7bd5', date: '#27ae60', text: '#888', unknown: '#aaa' }

export default function CSV() {
  const [file, setFile]         = useState(null)
  const [dragging, setDragging] = useState(false)
  const [rows, setRows]         = useState(null)   // { columns, rows }
  const [summary, setSummary]   = useState(null)
  const [status, setStatus]     = useState(null)   // { type, msg }
  const [tab, setTab]           = useState('preview')
  const [loading, setLoading]   = useState('')
  const [rowsN, setRowsN]       = useState(10)
  const [encoding, setEncoding] = useState('')
  const [sep, setSep]           = useState('')
  const fileRef = useRef()

  const pickFile = (f) => {
    if (!f) return
    setFile(f)
    setStatus(null)
    setRows(null)
    setSummary(null)
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer?.files?.[0]
    if (f) pickFile(f)
  }, [])

  const buildParams = () => {
    const p = new URLSearchParams({ n: Math.min(2000, Math.max(1, rowsN)) })
    if (encoding) p.append('encoding', encoding)
    if (sep && sep !== ',') p.append('sep', sep)
    return p.toString()
  }

  const doPreview = async () => {
    if (!file) { setStatus({ type: 'danger', msg: 'Please select a file first.' }); return }
    setLoading('preview')
    const fd = new FormData()
    fd.append('file', file, file.name)
    try {
      const res = await fetch(`/api/csv/preview?${buildParams()}`, {
        method: 'POST', credentials: 'include', body: fd
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error)
      setRows({ columns: data.columns, rows: data.rows })
      setTab('preview')
      setStatus({ type: 'success', msg: `Preview: ${data.columns.length} columns, ${data.rows.length} rows` })
    } catch (e) {
      setStatus({ type: 'danger', msg: `Parse failed: ${e.message}` })
    } finally { setLoading('') }
  }

  const doSummary = async () => {
    if (!file) { setStatus({ type: 'danger', msg: 'Please select a file first.' }); return }
    setLoading('summary')
    const fd = new FormData()
    fd.append('file', file, file.name)
    try {
      const res = await fetch(`/api/csv/summary?${buildParams()}`, {
        method: 'POST', credentials: 'include', body: fd
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error)
      setSummary(data.summary)
      setTab('summary')
      setStatus({ type: 'success', msg: 'Summary generated.' })
    } catch (e) {
      setStatus({ type: 'danger', msg: `Failed: ${e.message}` })
    } finally { setLoading('') }
  }

  return (
    <>
      <h4 className="fw-bold mb-1">CSV Workspace</h4>
      <p className="text-muted mb-4">Upload a CSV or Excel file to preview and summarise.</p>

      <div className="row g-3">
        {/* Left: upload + controls */}
        <div className="col-12 col-lg-4">
          <div className="card p-3 mb-3">
            {/* Dropzone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileRef.current?.click()}
              style={{
                border: `2px dashed ${dragging ? '#3a7bd5' : '#cdd5e0'}`,
                borderRadius: 8, padding: 24, textAlign: 'center',
                cursor: 'pointer', background: dragging ? '#eef3fc' : '#fafbfc',
                transition: 'all .15s',
              }}
            >
              <i className="fas fa-cloud-upload-alt mb-2" style={{ fontSize: '2rem', color: '#3a7bd5' }} />
              <div className="fw-semibold">{file ? `📄 ${file.name}` : 'Drop file or click to browse'}</div>
              <div className="text-muted small mt-1">CSV, XLS, XLSX (max 100MB)</div>
              <input
                ref={fileRef}
                type="file"
                accept=".csv,.xls,.xlsx"
                className="d-none"
                onChange={e => pickFile(e.target.files?.[0])}
              />
            </div>
          </div>

          {/* Options */}
          <div className="card p-3 mb-3">
            <div className="mb-2">
              <label className="form-label fw-semibold small">Preview rows</label>
              <input
                type="number" className="form-control form-control-sm"
                value={rowsN} min={1} max={2000}
                onChange={e => setRowsN(+e.target.value)}
              />
            </div>
            <div className="mb-2">
              <label className="form-label fw-semibold small">Encoding</label>
              <select className="form-select form-select-sm" value={encoding} onChange={e => setEncoding(e.target.value)}>
                {ENCODINGS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            </div>
            <div>
              <label className="form-label fw-semibold small">Separator</label>
              <input
                type="text" className="form-control form-control-sm" placeholder="Default: comma"
                value={sep} maxLength={1}
                onChange={e => setSep(e.target.value)}
              />
            </div>
          </div>

          {/* Actions */}
          <div className="d-flex flex-column gap-2">
            <button className="btn btn-primary" onClick={doPreview} disabled={!!loading}>
              {loading === 'preview' ? <span className="spinner-border spinner-border-sm me-1" /> : <i className="fas fa-eye me-2" />}
              Preview
            </button>
            <button className="btn btn-outline-secondary" onClick={doSummary} disabled={!!loading}>
              {loading === 'summary' ? <span className="spinner-border spinner-border-sm me-1" /> : <i className="fas fa-chart-bar me-2" />}
              Summary
            </button>
          </div>

          {/* Status */}
          {status && (
            <div className={`alert alert-${status.type} mt-3 py-2 small mb-0`}>{status.msg}</div>
          )}
        </div>

        {/* Right: results */}
        <div className="col-12 col-lg-8">
          {(rows || summary) ? (
            <div className="card">
              <div className="card-header d-flex gap-2">
                {rows && (
                  <button
                    className={`btn btn-sm ${tab === 'preview' ? 'btn-primary' : 'btn-outline-secondary'}`}
                    onClick={() => setTab('preview')}
                  >Preview</button>
                )}
                {summary && (
                  <button
                    className={`btn btn-sm ${tab === 'summary' ? 'btn-primary' : 'btn-outline-secondary'}`}
                    onClick={() => setTab('summary')}
                  >Summary</button>
                )}
              </div>

              {tab === 'preview' && rows && (
                <div className="overflow-auto" style={{ maxHeight: 500 }}>
                  <table className="table table-sm table-striped table-hover mb-0" style={{ fontSize: '.82rem' }}>
                    <thead className="table-dark sticky-top">
                      <tr>{rows.columns.map(c => <th key={c}>{c}</th>)}</tr>
                    </thead>
                    <tbody>
                      {rows.rows.map((r, i) => (
                        <tr key={i}>
                          {rows.columns.map(c => <td key={c}>{r[c] ?? ''}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {tab === 'summary' && summary && (
                <div className="p-3">
                  <div className="row g-3 mb-3">
                    {[
                      ['Rows',          summary.rows],
                      ['Columns',       summary.cols],
                      ['Missing cells', Object.values(summary.na_count || {}).reduce((a, b) => a + b, 0)],
                    ].map(([label, val]) => (
                      <div key={label} className="col-4">
                        <div className="stat-card"><div className="stat-number">{val}</div><div className="stat-label">{label}</div></div>
                      </div>
                    ))}
                  </div>

                  <h6 className="fw-semibold mb-2">Columns</h6>
                  <div className="d-flex flex-wrap gap-2 mb-3">
                    {summary.columns?.map(col => (
                      <span
                        key={col}
                        className="badge"
                        style={{ background: DTYPE_COLORS[summary.dtypes?.[col]] + '22', color: DTYPE_COLORS[summary.dtypes?.[col]], fontSize: '.8rem' }}
                      >
                        {col} <span className="opacity-75 ms-1">{summary.dtypes?.[col]}</span>
                      </span>
                    ))}
                  </div>

                  {Object.entries(summary.na_count || {}).some(([, v]) => v > 0) && (
                    <>
                      <h6 className="fw-semibold mb-2">Missing Values</h6>
                      <div className="row g-2">
                        {Object.entries(summary.na_count).filter(([, v]) => v > 0).map(([col, count]) => (
                          <div key={col} className="col-6 col-md-4">
                            <div className="d-flex justify-content-between small">
                              <span className="text-truncate">{col}</span>
                              <span className="text-danger fw-semibold">{count}</span>
                            </div>
                            <div className="progress" style={{ height: 4 }}>
                              <div
                                className="progress-bar bg-danger"
                                style={{ width: `${((count / summary.rows) * 100).toFixed(1)}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="card d-flex align-items-center justify-content-center" style={{ minHeight: 300 }}>
              <div className="text-center text-muted p-4">
                <i className="fas fa-table mb-3" style={{ fontSize: '3rem', opacity: .3 }} />
                <div>Upload a file and click <strong>Preview</strong> or <strong>Summary</strong></div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
