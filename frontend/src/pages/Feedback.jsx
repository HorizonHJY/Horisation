import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'

const CT_ZONE = 'America/Chicago'

function timeAgo(isoStr) {
  const diff = Math.floor((Date.now() - new Date(isoStr)) / 1000)
  if (diff < 60)    return 'just now'
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return new Date(isoStr).toLocaleDateString('en-US', { timeZone: CT_ZONE })
}

function formatCT(isoStr) {
  return new Date(isoStr).toLocaleString('en-US', {
    timeZone:  CT_ZONE,
    month:     'short',
    day:       'numeric',
    hour:      'numeric',
    minute:    '2-digit',
    hour12:    true,
  })
}

export default function Feedback() {
  const { user }                = useAuth()
  const navigate                = useNavigate()
  const [messages, setMsgs]     = useState([])
  const [content, setContent]   = useState('')
  const [posting, setPosting]   = useState(false)
  const [toast, setToast]       = useState(null)
  const [pendingDel, setPending] = useState(null)   // { id, msg, timerId }
  const bottomRef               = useRef()

  useEffect(() => { load() }, [])

  // Clean up undo timer on unmount
  useEffect(() => () => {
    if (pendingDel) {
      clearTimeout(pendingDel.timerId)
      api.delete(`/api/feedback/messages/${pendingDel.id}`)
    }
  }, []) // eslint-disable-line

  const load = async () => {
    const d = await api.get('/api/feedback/messages')
    if (d.ok) setMsgs(d.messages)
  }

  const flash = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
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

  // ── Delete with undo ──────────────────────────────────────────────────────────
  const remove = (id) => {
    const msg = messages.find(m => m.id === id)
    if (!msg) return

    // If there's already a pending delete, commit it immediately before starting new one
    if (pendingDel) {
      clearTimeout(pendingDel.timerId)
      api.delete(`/api/feedback/messages/${pendingDel.id}`)
      setPending(null)
    }

    // Optimistically remove from UI
    setMsgs(prev => prev.filter(m => m.id !== id))

    // Schedule the actual API call after 5 s
    const timerId = setTimeout(() => {
      api.delete(`/api/feedback/messages/${id}`)
      setPending(null)
    }, 5000)

    setPending({ id, msg, timerId })
  }

  const undoDelete = () => {
    if (!pendingDel) return
    clearTimeout(pendingDel.timerId)
    // Re-insert at the correct position (newest-first order)
    setMsgs(prev => {
      const next = [...prev, pendingDel.msg]
      next.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      return next
    })
    setPending(null)
  }

  const isAdmin = user?.role_info?.permissions?.includes('admin')

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 720 }}>

      {/* Regular toast */}
      {toast && (
        <div className={`alert alert-${toast.type} position-fixed top-0 end-0 m-3`} style={{ zIndex: 9999 }}>
          {toast.msg}
        </div>
      )}

      {/* Undo toast */}
      {pendingDel && (
        <div
          className="alert alert-dark d-flex align-items-center gap-3 position-fixed bottom-0 start-50 translate-middle-x mb-4"
          style={{ zIndex: 9999, minWidth: 280, boxShadow: '0 4px 16px rgba(0,0,0,0.2)' }}
        >
          <i className="fas fa-trash-alt" />
          <span className="flex-grow-1">Message deleted</span>
          <button
            className="btn btn-sm btn-outline-light fw-semibold"
            onClick={undoDelete}
          >
            Undo
          </button>
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
                  <div
                    className="d-flex 