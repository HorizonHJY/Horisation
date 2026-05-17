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
    timeZone: CT_ZONE,
    month:    'short',
    day:      'numeric',
    hour:     'numeric',
    minute:   '2-digit',
    hour12:   true,
  })
}

export default function Feedback() {
  const { user }                 = useAuth()
  const navigate                 = useNavigate()
  const textareaRef              = useRef()
  const [messages, setMsgs]      = useState([])
  const [content, setContent]    = useState('')
  const [posting, setPosting]    = useState(false)
  const [replyTo, setReplyTo]    = useState(null)
  const [toast, setToast]        = useState(null)
  const [pendingDel, setPending] = useState(null)

  useEffect(() => { load() }, [])

  useEffect(() => () => {
    if (pendingDel) {
      clearTimeout(pendingDel.timerId)
      api.delete('/api/feedback/messages/' + pendingDel.id)
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

  const startReply = (m) => {
    setReplyTo({ id: m.id, username: m.username, display_name: m.display_name, content: m.content })
    textareaRef.current?.focus()
  }

  const cancelReply = () => setReplyTo(null)

  const submit = async (e) => {
    e.preventDefault()
    if (!content.trim()) return
    setPosting(true)
    const d = await api.post('/api/feedback/messages', {
      content:     content.trim(),
      reply_to_id: replyTo ? replyTo.id : null,
    })
    setPosting(false)
    if (d.ok) {
      const newMsg = { ...d.message, reply_to: replyTo || null }
      setContent('')
      setReplyTo(null)
      setMsgs(prev => [newMsg, ...prev])
    } else {
      flash(d.error, 'danger')
    }
  }

  const remove = (id) => {
    const msg = messages.find(m => m.id === id)
    if (!msg) return
    if (pendingDel) {
      clearTimeout(pendingDel.timerId)
      api.delete('/api/feedback/messages/' + pendingDel.id)
      setPending(null)
    }
    setMsgs(prev => prev.filter(m => m.id !== id))
    const timerId = setTimeout(() => {
      api.delete('/api/feedback/messages/' + id)
      setPending(null)
    }, 5000)
    setPending({ id, msg, timerId })
  }

  const undoDelete = () => {
    if (!pendingDel) return
    clearTimeout(pendingDel.timerId)
    setMsgs(prev => {
      const next = [...prev, pendingDel.msg]
      next.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
      return next
    })
    setPending(null)
  }

  const isAdmin = user && user.role_info && user.role_info.permissions &&
                  user.role_info.permissions.includes('admin')

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 720 }}>

      {toast && (
        <div
          className={'alert alert-' + toast.type + ' position-fixed top-0 end-0 m-3'}
          style={{ zIndex: 9999 }}
        >
          {toast.msg}
        </div>
      )}

      {pendingDel && (
        <div
          className="alert alert-dark d-flex align-items-center gap-3 position-fixed bottom-0 start-50 translate-middle-x mb-4"
          style={{ zIndex: 9999, minWidth: 280, boxShadow: '0 4px 16px rgba(0,0,0,0.2)' }}
        >
          <i className="fas fa-trash-alt" />
          <span className="flex-grow-1">Message deleted</span>
          <button className="btn btn-sm btn-outline-light fw-semibold" onClick={undoDelete}>
            Undo
          </button>
        </div>
      )}

      <div className="d-flex align-items-center mb-4">
        <i className="fas fa-comments fa-lg me-2 text-primary" />
        <h4 className="mb-0 fw-bold">Message Board</h4>
      </div>

      <div className="card p-3 mb-4 shadow-sm">
        {replyTo && (
          <div
            className="d-flex align-items-start gap-2 rounded px-3 py-2 mb-2"
            style={{ background: '#f0f4ff', borderLeft: '3px solid #6b9cdb', fontSize: '.85rem' }}
          >
            <div className="flex-grow-1 overflow-hidden">
              <span className="fw-semibold text-primary">
                {'↩ Replying to ' + replyTo.display_name}
              </span>
              <p className="mb-0 text-muted text-truncate" style={{ maxWidth: '100%' }}>
                {replyTo.content}
              </p>
            </div>
            <button
              className="btn btn-sm btn-link text-muted p-0 flex-shrink-0"
              onClick={cancelReply}
              title="Cancel reply"
            >
              <i className="fas fa-times" />
            </button>
          </div>
        )}

        <form onSubmit={submit}>
          <textarea
            ref={textareaRef}
            className="form-control border-0 mb-2"
            rows={3}
            placeholder={replyTo ? ('Reply to ' + replyTo.display_name + '…') : 'Leave a message…'}
            maxLength={500}
            value={content}
            onChange={e => setContent(e.target.value)}
            style={{ resize: 'none', background: '#f8f9fa', borderRadius: 8 }}
          />
          <div className="d-flex justify-content-between align-items-center">
            <span className="text-muted small">{content.length} / 500</span>
            <button
              className="btn btn-primary btn-sm px-4"
              disabled={posting || !content.trim()}
            >
              {posting
                ? <span className="spinner-border spinner-border-sm" />
                : (
                  <span>
                    <i className={'fas ' + (replyTo ? 'fa-reply' : 'fa-paper-plane') + ' me-1'} />
                    {replyTo ? 'Reply' : 'Post'}
                  </span>
                )
              }
            </button>
          </div>
        </form>
      </div>

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
                    className="d-flex align-items-center gap-2"
                    style={{ cursor: 'pointer' }}
                    onClick={() => navigate('/u/' + m.username)}
                    title={'View ' + m.display_name + "'s profile"}
                  >
                    {m.avatar_url ? (
                      <img
                        src={m.avatar_url}
                        alt={m.display_name}
                        className="rounded-circle flex-shrink-0"
                        style={{ width: 36, height: 36, objectFit: 'cover' }}
                      />
                    ) : (
                      <div
                        className="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
                        style={{
                          width: 36, height: 36,
                          background: '#6b9cdb1a', color: '#6b9cdb',
                          fontWeight: 700, fontSize: '0.9rem',
                        }}
                      >
                        {m.display_name && m.display_name[0] ? m.display_name[0].toUpperCase() : '?'}
                      </div>
                    )}
                    <div>
                      <span className="fw-semibold">{m.display_name}</span>
                      <span className="text-muted ms-1 small">{'@' + m.username}</span>
                    </div>
                  </div>

                  <div className="d-flex align-items-center gap-2">
                    <span className="text-muted small">{timeAgo(m.created_at)}</span>
                    <button
                      className="btn btn-sm btn-link text-muted p-0"
                      onClick={() => startReply(m)}
                      title="Reply"
                      style={{ fontSize: '.8rem' }}
                    >
                      <i className="fas fa-reply" />
                    </button>
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

                {m.reply_to && (
                  <div
                    className="rounded px-3 py-2 mt-2"
                    style={{
                      background: '#f0f4ff',
                      borderLeft: '3px solid #6b9cdb',
                      fontSize: '.82rem',
                    }}
                  >
                    <span className="fw-semibold text-primary">{m.reply_to.display_name}</span>
                    <p
                      className="mb-0 text-muted"
                      style={{
                        whiteSpace: 'pre-wrap',
                        overflow: 'hidden',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                      }}
                    >
                      {m.reply_to.content}
                    </p>
                  </div>
                )}

                <p className="mb-0 mt-2" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                  {m.content}
                </p>
                <div className="text-muted mt-1" style={{ fontSize: '0.72rem' }}>
                  {formatCT(m.created_at)} CT
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
