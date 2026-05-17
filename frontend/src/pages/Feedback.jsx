import React, { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'

const CT_ZONE = 'America/Chicago'
const PER_PAGE = 5

function timeAgo(isoStr) {
  const diff = Math.floor((Date.now() - new Date(isoStr)) / 1000)
  if (diff < 60)    return 'just now'
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return new Date(isoStr).toLocaleDateString('en-US', { timeZone: CT_ZONE })
}

function UserAvatar({ displayName, avatarUrl, size = 36 }) {
  if (avatarUrl) {
    return (
      <img src={avatarUrl} alt={displayName}
        className="rounded-circle flex-shrink-0"
        style={{ width: size, height: size, objectFit: 'cover' }} />
    )
  }
  return (
    <div className="rounded-circle d-flex align-items-center justify-content-center flex-shrink-0"
      style={{
        width: size, height: size,
        background: '#6b9cdb1a', color: '#6b9cdb',
        fontWeight: 700, fontSize: size * 0.38,
      }}>
      {displayName?.[0]?.toUpperCase() || '?'}
    </div>
  )
}

function QuoteBlock({ replyTo }) {
  return (
    <div className="rounded px-2 py-1 mb-2"
      style={{ background: '#f0f4ff', borderLeft: '3px solid #6b9cdb', fontSize: '.8rem' }}>
      <span className="fw-semibold text-primary">{replyTo.display_name}</span>
      <p className="mb-0 text-muted" style={{
        overflow: 'hidden', display: '-webkit-box',
        WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
        whiteSpace: 'pre-wrap',
      }}>
        {replyTo.content}
      </p>
    </div>
  )
}

function LikeBtn({ liked, count, onClick }) {
  return (
    <button className="btn btn-sm btn-link p-0 d-flex align-items-center gap-1"
      style={{ color: liked ? '#0d6efd' : '#6c757d', fontSize: '.82rem', textDecoration: 'none' }}
      onClick={onClick}>
      <i className={liked ? 'fas fa-thumbs-up' : 'far fa-thumbs-up'} />
      {count > 0 && <span>{count}</span>}
    </button>
  )
}

function MsgCard({ m, topLevelId, currentUser, isAdmin, isReply, onLike, onReply, onDelete, navigate }) {
  const isMine = m.username === currentUser
  return (
    <div style={{ padding: isReply ? '8px 12px' : undefined }}>
      {m.reply_to && <QuoteBlock replyTo={m.reply_to} />}

      <div className="d-flex align-items-center gap-2 mb-1">
        <div style={{ cursor: 'pointer' }} onClick={() => navigate('/u/' + m.username)}>
          <UserAvatar displayName={m.display_name} avatarUrl={m.avatar_url} size={isReply ? 28 : 36} />
        </div>
        <div className="flex-grow-1 overflow-hidden">
          <span className="fw-semibold" style={{ fontSize: isReply ? '.88rem' : '1rem' }}>
            {m.display_name}
          </span>
          <span className="text-muted ms-1" style={{ fontSize: '.78rem' }}>@{m.username}</span>
        </div>
        <span className="text-muted" style={{ fontSize: '.75rem', flexShrink: 0 }}>
          {timeAgo(m.created_at)}
        </span>
      </div>

      <p className="mb-1 ms-1" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: isReply ? '.9rem' : '1rem' }}>
        {m.content}
      </p>

      <div className="d-flex align-items-center gap-3 ms-1">
        <LikeBtn liked={m.liked_by_me} count={m.like_count} onClick={() => onLike(m.id, topLevelId)} />
        <button className="btn btn-sm btn-link p-0 text-muted"
          style={{ fontSize: '.82rem', textDecoration: 'none' }}
          onClick={() => onReply(m, topLevelId)}>
          <i className="fas fa-reply me-1" />Reply
        </button>
        {(isMine || isAdmin) && (
          <button className="btn btn-sm btn-link p-0 text-danger"
            style={{ fontSize: '.82rem', textDecoration: 'none' }}
            onClick={() => onDelete(m.id, topLevelId)}>
            <i className="fas fa-trash-alt" />
          </button>
        )}
      </div>
    </div>
  )
}

export default function Feedback() {
  const { user }               = useAuth()
  const navigate               = useNavigate()
  const textareaRef            = useRef()
  const [messages, setMsgs]    = useState([])
  const [page, setPage]        = useState(1)
  const [total, setTotal]      = useState(0)
  const [expanded, setExpanded] = useState({})   // { [msgId]: Reply[] | 'loading' }
  const [content, setContent]  = useState('')
  const [replyTo, setReplyTo]  = useState(null)  // { id, display_name, content, topLevelId }
  const [posting, setPosting]  = useState(false)
  const [pendingDel, setPending] = useState(null)
  const [toast, setToast]      = useState(null)

  const totalPages = Math.ceil(total / PER_PAGE)
  const isAdmin = user?.role_info?.permissions?.includes('admin')

  useEffect(() => { load(1) }, [])

  useEffect(() => () => {
    if (pendingDel) {
      clearTimeout(pendingDel.timerId)
      api.delete('/api/feedback/messages/' + pendingDel.id)
    }
  }, []) // eslint-disable-line

  const load = async (p) => {
    const d = await api.get('/api/feedback/messages?page=' + p)
    if (d.ok) {
      setMsgs(d.messages)
      setTotal(d.total)
      setPage(p)
      setExpanded({})
    }
  }

  const flash = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  // ── Reply ─────────────────────────────────────────────────────────────────────
  const startReply = (m, topLevelId) => {
    setReplyTo({ id: m.id, display_name: m.display_name, content: m.content, topLevelId })
    textareaRef.current?.focus()
  }

  // ── Submit ────────────────────────────────────────────────────────────────────
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
      setContent('')
      const capturedReply = replyTo
      setReplyTo(null)
      if (!capturedReply) {
        // New top-level: prepend and adjust total
        setMsgs(prev => [{ ...d.message, reply_count: 0, top_replies: [] }, ...prev.slice(0, PER_PAGE - 1)])
        setTotal(t => t + 1)
      } else {
        // Reply: reload page to get updated reply counts + ordering
        await load(page)
        // If the thread was expanded, reload its replies too
        const tid = capturedReply.topLevelId
        if (expanded[tid] && expanded[tid] !== 'loading') {
          const r = await api.get('/api/feedback/messages/' + tid + '/replies')
          if (r.ok) setExpanded(prev => ({ ...prev, [tid]: r.replies }))
        }
      }
    } else {
      flash(d.error, 'danger')
    }
  }

  // ── Like ──────────────────────────────────────────────────────────────────────
  const toggleLike = async (msgId, topLevelId) => {
    const d = await api.post('/api/feedback/messages/' + msgId + '/like')
    if (!d.ok) return
    const patch = (m) => m.id === msgId ? { ...m, like_count: d.like_count, liked_by_me: d.liked } : m

    if (!topLevelId) {
      // top-level message
      setMsgs(prev => prev.map(patch))
    } else {
      // reply — update in top_replies and/or expanded
      setMsgs(prev => prev.map(m =>
        m.id === topLevelId
          ? { ...m, top_replies: m.top_replies.map(patch) }
          : m
      ))
      setExpanded(prev => {
        const reps = prev[topLevelId]
        if (!reps || reps === 'loading') return prev
        return { ...prev, [topLevelId]: reps.map(patch) }
      })
    }
  }

  // ── Delete with undo ──────────────────────────────────────────────────────────
  const remove = (msgId, topLevelId) => {
    // Find the message
    let msg = null
    if (!topLevelId) {
      msg = messages.find(m => m.id === msgId)
    } else {
      const parent = messages.find(m => m.id === topLevelId)
      msg = parent?.top_replies?.find(r => r.id === msgId)
        || (expanded[topLevelId] !== 'loading' && expanded[topLevelId]?.find(r => r.id === msgId))
    }
    if (!msg) return

    if (pendingDel) {
      clearTimeout(pendingDel.timerId)
      api.delete('/api/feedback/messages/' + pendingDel.id)
      setPending(null)
    }

    // Optimistic remove
    if (!topLevelId) {
      setMsgs(prev => prev.filter(m => m.id !== msgId))
      setTotal(t => t - 1)
    } else {
      setMsgs(prev => prev.map(m =>
        m.id === topLevelId
          ? { ...m,
              reply_count: Math.max(0, m.reply_count - 1),
              top_replies: m.top_replies.filter(r => r.id !== msgId) }
          : m
      ))
      setExpanded(prev => {
        const reps = prev[topLevelId]
        if (!reps || reps === 'loading') return prev
        return { ...prev, [topLevelId]: reps.filter(r => r.id !== msgId) }
      })
    }

    const timerId = setTimeout(() => {
      api.delete('/api/feedback/messages/' + msgId)
      setPending(null)
    }, 5000)
    setPending({ id: msgId, msg, timerId, topLevelId })
  }

  const undoDelete = () => {
    if (!pendingDel) return
    clearTimeout(pendingDel.timerId)
    const { msg, topLevelId } = pendingDel
    if (!topLevelId) {
      setMsgs(prev => {
        const next = [...prev, msg]
        next.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
        return next
      })
      setTotal(t => t + 1)
    } else {
      setMsgs(prev => prev.map(m => {
        if (m.id !== topLevelId) return m
        const next = [...m.top_replies, msg]
        next.sort((a, b) => (b.like_count - a.like_count) || new Date(a.created_at) - new Date(b.created_at))
        return { ...m, reply_count: m.reply_count + 1, top_replies: next.slice(0, 2) }
      }))
    }
    setPending(null)
  }

  // ── Expand / collapse replies ─────────────────────────────────────────────────
  const toggleExpand = async (msgId) => {
    if (expanded[msgId]) {
      setExpanded(prev => { const n = { ...prev }; delete n[msgId]; return n })
      return
    }
    setExpanded(prev => ({ ...prev, [msgId]: 'loading' }))
    const d = await api.get('/api/feedback/messages/' + msgId + '/replies')
    if (d.ok) setExpanded(prev => ({ ...prev, [msgId]: d.replies }))
    else setExpanded(prev => { const n = { ...prev }; delete n[msgId]; return n })
  }

  // ── Pagination ────────────────────────────────────────────────────────────────
  function Pagination() {
    if (totalPages <= 1) return null
    return (
      <div className="d-flex justify-content-center align-items-center gap-2 mt-4">
        <button className="btn btn-sm btn-outline-secondary" disabled={page <= 1}
          onClick={() => load(page - 1)}>
          <i className="fas fa-chevron-left" />
        </button>
        {Array.from({ length: totalPages }, (_, i) => i + 1).map(p => (
          <button key={p}
            className={'btn btn-sm ' + (p === page ? 'btn-primary' : 'btn-outline-secondary')}
            onClick={() => load(p)}>
            {p}
          </button>
        ))}
        <button className="btn btn-sm btn-outline-secondary" disabled={page >= totalPages}
          onClick={() => load(page + 1)}>
          <i className="fas fa-chevron-right" />
        </button>
      </div>
    )
  }

  const sharedProps = { currentUser: user?.username, isAdmin, navigate }

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 720 }}>

      {toast && (
        <div className={'alert alert-' + toast.type + ' position-fixed top-0 end-0 m-3'}
          style={{ zIndex: 9999 }}>{toast.msg}</div>
      )}

      {pendingDel && (
        <div className="alert alert-dark d-flex align-items-center gap-3 position-fixed bottom-0 start-50 translate-middle-x mb-4"
          style={{ zIndex: 9999, minWidth: 280, boxShadow: '0 4px 16px rgba(0,0,0,.2)' }}>
          <i className="fas fa-trash-alt" />
          <span className="flex-grow-1">Message deleted</span>
          <button className="btn btn-sm btn-outline-light fw-semibold" onClick={undoDelete}>Undo</button>
        </div>
      )}

      <div className="d-flex align-items-center mb-4">
        <i className="fas fa-comments fa-lg me-2 text-primary" />
        <h4 className="mb-0 fw-bold">Message Board</h4>
        {total > 0 && <span className="badge bg-secondary ms-2" style={{ fontSize: '.75rem' }}>{total}</span>}
      </div>

      {/* ── Post box ── */}
      <div className="card p-3 mb-4 shadow-sm">
        {replyTo && (
          <div className="d-flex align-items-start gap-2 rounded px-3 py-2 mb-2"
            style={{ background: '#f0f4ff', borderLeft: '3px solid #6b9cdb', fontSize: '.85rem' }}>
            <div className="flex-grow-1 overflow-hidden">
              <span className="fw-semibold text-primary">{'↩ Replying to ' + replyTo.display_name}</span>
              <p className="mb-0 text-muted text-truncate">{replyTo.content}</p>
            </div>
            <button className="btn btn-sm btn-link text-muted p-0 flex-shrink-0"
              onClick={() => setReplyTo(null)} title="Cancel reply">
              <i className="fas fa-times" />
            </button>
          </div>
        )}
        <form onSubmit={submit}>
          <textarea ref={textareaRef}
            className="form-control border-0 mb-2" rows={3}
            placeholder={replyTo ? 'Reply to ' + replyTo.display_name + '…' : 'Leave a message…'}
            maxLength={500} value={content}
            onChange={e => setContent(e.target.value)}
            style={{ resize: 'none', background: '#f8f9fa', borderRadius: 8 }} />
          <div className="d-flex justify-content-between align-items-center">
            <span className="text-muted small">{content.length} / 500</span>
            <button className="btn btn-primary btn-sm px-4"
              disabled={posting || !content.trim()}>
              {posting
                ? <span className="spinner-border spinner-border-sm" />
                : <span><i className={'fas ' + (replyTo ? 'fa-reply' : 'fa-paper-plane') + ' me-1'} />{replyTo ? 'Reply' : 'Post'}</span>
              }
            </button>
          </div>
        </form>
      </div>

      {/* ── Message list ── */}
      {messages.length === 0 ? (
        <div className="text-center text-muted py-5">
          <i className="fas fa-comment-slash fa-3x mb-3" />
          <p>No messages yet. Be the first to say something!</p>
        </div>
      ) : (
        <div className="d-flex flex-column gap-3">
          {messages.map(m => {
            const reps     = expanded[m.id]
            const isLoading = reps === 'loading'
            const allReps  = Array.isArray(reps) ? reps : null
            const showMore = m.reply_count > 2 && !allReps && !isLoading
            const showCollapse = Array.isArray(reps)

            return (
              <div key={m.id} className="card shadow-sm px-4 py-3">
                <MsgCard m={m} topLevelId={null}
                  onLike={toggleLike} onReply={startReply} onDelete={remove}
                  {...sharedProps} />

                {/* ── Reply section ── */}
                {(m.top_replies.length > 0 || m.reply_count > 0) && (
                  <div className="mt-2 ms-1" style={{ borderLeft: '2px solid #e9ecef', paddingLeft: 12 }}>
                    {/* show either expanded or top 2 */}
                    {(allReps || m.top_replies).map(r => (
                      <div key={r.id} className="py-2"
                        style={{ borderBottom: '1px solid #f0f0f0' }}>
                        <MsgCard m={r} topLevelId={m.id} isReply
                          onLike={toggleLike} onReply={startReply} onDelete={remove}
                          {...sharedProps} />
                      </div>
                    ))}

                    {isLoading && (
                      <div className="text-center py-2 text-muted small">
                        <span className="spinner-border spinner-border-sm me-1" />Loading…
                      </div>
                    )}

                    {showMore && (
                      <button className="btn btn-link btn-sm p-0 mt-1 text-muted"
                        style={{ fontSize: '.82rem', textDecoration: 'none' }}
                        onClick={() => toggleExpand(m.id)}>
                        <i className="fas fa-chevron-down me-1" />
                        View {m.reply_count - 2} more {m.reply_count - 2 === 1 ? 'reply' : 'replies'}
                      </button>
                    )}

                    {showCollapse && (
                      <button className="btn btn-link btn-sm p-0 mt-1 text-muted"
                        style={{ fontSize: '.82rem', textDecoration: 'none' }}
                        onClick={() => toggleExpand(m.id)}>
                        <i className="fas fa-chevron-up me-1" />Collapse replies
                      </button>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      <Pagination />
    </div>
  )
}
