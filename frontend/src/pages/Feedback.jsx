import React, { useState, useEffect, useRef } from 'react'
import { api } from '../api'
import { useAuth } from '../App'

function timeAgo(isoStr) {
  const diff = Math.floor((Date.now() - new Date(isoStr)) / 1000)
  if (diff < 60)    return 'just now'
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return new Date(isoStr).toLocaleDateString()
}

export default function Feedback() {
  const { user }            = useAuth()
  const [messages, setMsgs] = useState([])
  const [content, setContent] = useState('')
  const [posting, setPosting] = useState(false)
  const [toast, setToast]   = useState(null)
  const bottomRef           = useRef()

  useEffect(() => { load() }, [])

  const load = async () => {
    const d = await api.get('/api/feedback/messages')
    if (d.ok) setMsgs(d.messages)
  }

  const flash = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 2500)
  }

  const submit = async (e) => {
    e.preventDefault()
    if (!content.trim()) return
    setPosting(true)
    const d = await api.post('/api/feedback/messages', { content: content.trim() })
    setPosting(false)
    if (d.ok) {
      setContent('')
      setMsgs(prev => [d.message, ...prev])
    } else {
      flash(d.error, 'danger')
    }
  }

  const remove = async (id) => {
    const d = await api.delete(`/api/feedback/messages/${id}`)
    if (d.ok) setMsgs(prev => prev.filter(m => m.id !== id))
    else flash(d.error, 'danger')
  }

  const isAdmin = user?.role_info?.permissions?.includes('admin')

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 720 }}>

      {toast && (
        <div className={`alert alert-${toast.type} position-fixed top-0 end-0 m-3`} style={{ zIndex: 9999 }}>
          {toast.msg}
        </div>
      )}

      <div className="d-flex align-items-center mb-4">
        <i className="fas fa-comments fa-lg me-2 text-primary" />
        <h4 className="mb-0 fw-bold">Message Board</h4>
      </div>

      {/* Post box */}
      <div className="card p-3 mb-4 shadow-sm">
        <form onSubmit={submit}>
          <textarea
            className="form-control border-0 mb-2"
            rows={3}
            placeholder="Leave a message…"
            maxLength={500}
            value={content}
            onChange={e => setContent(e.target.value)}
            style={{ resize: 'none', background: '#f8f9fa', borderRadius: 8 }}
          />
          <div className="d-flex justify-content-between align-items-center">
            <span className="text-muted small">{content.length} / 500</span>
            <button className="btn btn-primary btn-sm px-4" disabled={posting || !content.trim()}>
              {posting
                ? <span className="spinner-border spinner-border-sm" />
                : <><i className="fas fa-paper-plane me-1" />Post</>
              }
            </button>
          </div>
        </form>
      </div>

      {/* Message list */}
      {messages.length === 0 ? (
        <div className="text-center text-muted py-5">
          <i className="fas fa-comment-slash fa-3x mb-3" />
          <p>No messages yet. Be the first to say something!</p>
        </div>
      ) : (
        <div className="d-flex flex-column gap-3">
          {messages.map(m => {
            const isMine = m.username === user.username
            return (
              <div key={m.id} className="card shadow-sm px-4 py-3">
                <div className="d-flex justify-content-between align-items-start">
                  <div className="d-flex align-items-center gap-2">
                    {/* Avatar initial */}
                    <div
                      className="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
                      style={{ width: 36, height: 36, background: '#3a7bd51a', color: '#3a7bd5', fontWeight: 700, fontSize: '0.9rem' }}
                    >
                      {m.display_name?.[0]?.toUpperCase()}
                    </div>
                    <div>
                      <span className="fw-semibold">{m.display_name}</span>
                      <span className="text-muted ms-1 small">@{m.username}</span>
                    </div>
                  </div>
                  <div className="d-flex align-items-center gap-2">
                    <span className="text-muted small">{timeAgo(m.created_at)}</span>
                    {(isMine || isAdmin) && (
                      <button
                        className="btn btn-sm btn-link text-danger p-0"
                        onClick={() => remove(m.id)}
                        title="Delete"
                      >
                        <i className="fas fa-trash-alt" />
                      </button>
                    )}
                  </div>
                </div>
                <p className="mb-0 mt-2" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                  {m.content}
                </p>
                <div className="text-muted mt-1" style={{ fontSize: '0.72rem' }}>
                  {new Date(m.created_at).toLocaleString()}
                </div>
              </div>
            )
          })}
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
