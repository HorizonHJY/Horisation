import React, { useState, useEffect, useRef } from 'react'
import { io } from 'socket.io-client'
import { useLocation, useNavigate } from 'react-router-dom'
import { api } from '../api'
import HandLoader from '../components/HandLoader'
import { useAuth, useUnread } from '../App'

function Avatar({ display, avatar, size = 40 }) {
  if (avatar) return (
    <img src={avatar} alt={display}
      style={{ width: size, height: size, borderRadius: '50%', objectFit: 'cover', flexShrink: 0 }} />
  )
  return (
    <div style={{
      width: size, height: size, borderRadius: '50%', background: '#6b9cdb', color: '#fff',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontWeight: 700, fontSize: size * 0.4, flexShrink: 0,
    }}>
      {display?.[0]?.toUpperCase() || '?'}
    </div>
  )
}

// ── Message rendering helpers ────────────────────────────────────────────────

// Format date label in Central Time (St. Louis)
function cstDateLabel(isoStr) {
  const d = new Date(isoStr)
  const opts = { timeZone: 'America/Chicago' }
  const dStr = d.toLocaleDateString('zh-CN', opts)
  const now = new Date()
  const nowStr = now.toLocaleDateString('zh-CN', opts)
  const yest = new Date(now); yest.setDate(yest.getDate() - 1)
  const yStr = yest.toLocaleDateString('zh-CN', opts)
  if (dStr === nowStr) return '今天'
  if (dStr === yStr) return '昨天'
  return d.toLocaleDateString('zh-CN', { timeZone: 'America/Chicago', month: 'long', day: 'numeric' })
}

// Detect travel/bill-split join URLs
function parseJoinUrl(text) {
  const m = text.match(/https?:\/\/[^\s]*(\/(travel|bill-split))\?join=([A-Z0-9]+)/i)
  if (!m) return null
  return { type: m[2], code: m[3].toUpperCase(), url: text.match(/https?:\/\/[^\s]+/)[0] }
}

// Render text with plain clickable links
function renderContent(text, isMe) {
  const urlRegex = /(https?:\/\/[^\s]+)/g
  const parts = text.split(urlRegex)
  return parts.map((part, i) =>
    urlRegex.test(part)
      ? <a key={i} href={part} target="_blank" rel="noopener noreferrer"
          style={{ color: isMe ? '#d4eaff' : '#3b82f6', textDecoration: 'underline', wordBreak: 'break-all' }}>
          {part}
        </a>
      : part
  )
}

// Render full message content — share card or plain text
function renderMessageContent(content, isMe) {
  const join = parseJoinUrl(content)
  if (join) {
    const isTravel = join.type === 'travel'
    return (
      <a href={join.url} style={{ textDecoration: 'none', display: 'block', minWidth: 200 }}>
        <div style={{
          background: isMe ? 'rgba(255,255,255,0.18)' : '#eff6ff',
          border: `1px solid ${isMe ? 'rgba(255,255,255,0.35)' : '#bfdbfe'}`,
          borderRadius: 10, padding: '8px 12px',
        }}>
          <div className="d-flex align-items-center gap-2">
            <i className={`fas ${isTravel ? 'fa-route' : 'fa-receipt'}`}
               style={{ color: isMe ? '#fff' : '#3b82f6', fontSize: '1.2rem', flexShrink: 0 }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, fontSize: '.85rem', color: isMe ? '#fff' : '#1e3a5f' }}>
                {isTravel ? '旅行计划邀请' : '分账邀请'}
              </div>
              <div style={{ fontSize: '.72rem', opacity: 0.65, fontFamily: 'monospace', letterSpacing: '.08em' }}>
                {join.code}
              </div>
            </div>
            <span style={{ fontSize: '.72rem', color: isMe ? '#d4eaff' : '#3b82f6', flexShrink: 0 }}>
              点击加入 →
            </span>
          </div>
        </div>
      </a>
    )
  }
  return renderContent(content, isMe)
}

export default function Friends() {
  const { user } = useAuth()
  const { unreadMap, clearUnread, bumpUnread } = useUnread()
  const location      = useLocation()
  const navigate      = useNavigate()
  const socketRef     = useRef(null)
  const chatEndRef    = useRef(null)
  const activeChatRef = useRef(null)   // mirror of activeChat for socket handler
  const tabRef        = useRef('friends')

  const [tab, setTab]               = useState('friends')
  const [friends, setFriends]       = useState([])
  const [pending, setPending]       = useState([])
  const [searchQuery, setSearchQuery]   = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searching, setSearching]   = useState(false)
  const [sentSet, setSentSet]       = useState(new Set())
  const [activeChat, setActiveChat] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [chatInput, setChatInput]   = useState('')
  const [onlineSet, setOnlineSet]   = useState(new Set())
  const [contactModal, setContactModal]   = useState(null)
  const [sharedContacts, setSharedContacts] = useState([])  // approved contact reqs where I am to_user
  // contactStatusMap: { [username]: 'pending' | 'approved' | 'declined' }
  const [contactStatusMap, setContactStatusMap] = useState({})
  const [contactReqs, setContactReqs]     = useState([])  // incoming contact requests
  const [toast, setToast]           = useState(null)
  const [loading, setLoading]       = useState(false)

  const flash = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  // ── Socket ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    const socket = io({ withCredentials: true })
    socketRef.current = socket

    socket.on('connect', () => socket.emit('friends_get_online'))
    socket.on('online_list', ({ online }) => setOnlineSet(new Set(online)))

    socket.on('friend_request_incoming', (req) => {
      setPending(prev => [req, ...prev])
      flash('New friend request!', 'info')
    })
    socket.on('contact_request_incoming', (req) => {
      setContactReqs(prev => [req, ...prev])
      flash('New contact request!', 'info')
    })
    socket.on('friend_accepted', ({ from_user }) => {
      flash(`${from_user} accepted your friend request!`, 'success')
      loadFriends()
    })
    socket.on('chat_message', (msg) => {
      const chat = activeChatRef.current
      if (chat) {
        const [ua, ub] = [user.username, chat.username].sort()
        if (msg.room_key === `${ua}:${ub}`) {
          setChatHistory(h => [...h, msg])
          if (msg.sender !== user.username) {
            api.post(`/api/friends/${chat.username}/read`)
          }
          return
        }
      }
      // Message not in currently open chat — bump unread badge
      if (msg.sender !== user.username) {
        bumpUnread(msg.sender)
      }
    })
    socket.on('chat_error', ({ message }) => flash(message, 'danger'))

    return () => socket.disconnect()
  }, [])

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory])

  // Load on tab change
  useEffect(() => {
    tabRef.current = tab
    if (tab === 'friends') loadFriends()
    if (tab === 'pending') loadPending()
    if (tab === 'add')     loadSentRequests()
    else { setSearchQuery(''); setSearchResults([]) }
  }, [tab])

  // Refresh when user returns to this tab (skip if chat is open — already real-time)
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState !== 'visible' || activeChatRef.current) return
      if (tabRef.current === 'friends') loadFriends()
      else if (tabRef.current === 'pending') loadPending()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [])

  // Auto-open chat if navigated here from Market with state
  useEffect(() => {
    const state = location.state
    if (!state?.openChat) return
    const friend = state.openChat
    const initialMsg = state.initialMessage || ''
    // clear navigation state so back-navigation doesn't re-trigger
    window.history.replaceState({}, '')
    openChat(friend).then(() => {
      if (initialMsg) setChatInput(initialMsg)
    })
  }, [])

  // ── Data loaders ────────────────────────────────────────────────────────────
  async function loadFriends() {
    setLoading(true)
    const [fRes, cRes, sRes] = await Promise.all([
      api.get('/api/friends/list'),
      api.get('/api/friends/contact/sent'),
      api.get('/api/friends/contact/shared'),
    ])
    if (fRes.ok) setFriends(fRes.friends)
    if (cRes.ok) {
      const map = {}
      // requests are ordered created_at DESC — first entry per user is the latest; skip older duplicates
      cRes.requests.forEach(r => {
        if (!(r.to_user in map)) map[r.to_user] = r.status
      })
      setContactStatusMap(map)
    }
    if (sRes.ok) setSharedContacts(sRes.requests)
    setLoading(false)
  }

  async function loadPending() {
    setLoading(true)
    const [fRes, cRes, sRes] = await Promise.all([
      api.get('/api/friends/requests/pending'),
      api.get('/api/friends/contact/requests'),
      api.get('/api/friends/contact/shared'),
    ])
    if (fRes.ok) setPending(fRes.requests)
    if (cRes.ok) setContactReqs(cRes.requests)
    if (sRes.ok) setSharedContacts(sRes.requests)
    setLoading(false)
  }

  async function loadSentRequests() {
    const d = await api.get('/api/friends/requests/sent')
    if (d.ok) {
      const pending = d.requests.filter(r => r.status === 'pending').map(r => r.to_user)
      setSentSet(new Set(pending))
    }
  }

  async function handleSearch(e) {
    e.preventDefault()
    if (searchQuery.trim().length < 2) return
    setSearching(true)
    const d = await api.get(`/api/friends/users?q=${encodeURIComponent(searchQuery.trim())}`)
    if (d.ok) setSearchResults(d.users)
    setSearching(false)
  }

  // ── Actions ─────────────────────────────────────────────────────────────────
  const sendRequest = async (toUser) => {
    const d = await api.post('/api/friends/requests', { to_user: toUser })
    if (d.ok) { setSentSet(s => new Set([...s, toUser])); flash('Friend request sent!') }
    else flash(d.error, 'danger')
  }

  const respond = async (reqId, action) => {
    const d = await api.put(`/api/friends/requests/${reqId}`, { action })
    if (d.ok) {
      setPending(prev => prev.filter(r => r.id !== reqId))
      if (action === 'accept') { flash('Friend added!'); loadFriends() }
    } else flash(d.error, 'danger')
  }

  const unfriend = async (username) => {
    if (!window.confirm(`Remove ${username} from friends?`)) return
    const d = await api.delete(`/api/friends/${username}`)
    if (d.ok) { setFriends(prev => prev.filter(f => f.username !== username)); flash('Removed from friends.') }
  }

  const openChat = async (friend) => {
    activeChatRef.current = friend
    setActiveChat(friend)
    setChatHistory([])
    clearUnread(friend.username)
    api.post(`/api/friends/${friend.username}/read`)
    const d = await api.get(`/api/friends/${friend.username}/history`)
    if (d.ok) setChatHistory(d.messages)
    return friend
  }

  const closeChat = () => {
    activeChatRef.current = null
    setActiveChat(null)
    setChatHistory([])
  }

  const sendMessage = () => {
    if (!chatInput.trim() || !activeChat) return
    socketRef.current?.emit('chat_send', { to_user: activeChat.username, content: chatInput.trim() })
    setChatInput('')
  }

  const showContact = async (friend) => {
    const d = await api.get(`/api/friends/${friend.username}/contact`)
    if (d.ok) setContactModal({
      name: friend.display_name,
      phone: d.phone, wechat: d.wechat,
      address: d.address, postal_code: d.postal_code,
    })
    else flash(d.error, 'danger')
  }

  const revokeContact = async (reqId, fromUser) => {
    const d = await api.put(`/api/friends/contact/requests/${reqId}`, { action: 'revoke' })
    if (d.ok) {
      setSharedContacts(prev => prev.filter(r => r.id !== reqId))
      flash(`Contact access revoked for ${fromUser}.`)
    } else flash(d.error, 'danger')
  }

  const requestContact = async (username) => {
    const d = await api.post(`/api/friends/${username}/contact/request`)
    if (d.ok) {
      setContactStatusMap(prev => ({ ...prev, [username]: 'pending' }))
      flash('Contact request sent!')
    } else {
      if (d.error === 'Contact is hidden') setContactStatusMap(prev => ({ ...prev, [username]: 'hidden' }))
      flash(d.error, 'danger')
    }
  }

  const respondContact = async (reqId, action, fromUser) => {
    const d = await api.put(`/api/friends/contact/requests/${reqId}`, { action })
    if (d.ok) {
      if (action === 'approve') {
        const req = contactReqs.find(r => r.id === reqId)
        if (req) setSharedContacts(prev => [...prev, req])
      }
      setContactReqs(prev => prev.filter(r => r.id !== reqId))
      flash(action === 'approve' ? 'Contact shared!' : 'Request declined.')
    } else flash(d.error, 'danger')
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="container-fluid py-3" style={{ maxWidth: 720 }}>

      {/* Toast */}
      {toast && (
        <div className={`alert alert-${toast.type} position-fixed top-0 end-0 m-3`}
          style={{ zIndex: 9999, minWidth: 240 }}>
          {toast.msg}
        </div>
      )}

      {/* Contact info modal */}
      {contactModal && (
        <div className="modal show d-block" style={{ background: 'rgba(0,0,0,.4)' }}
          onClick={() => setContactModal(null)}>
          <div className="modal-dialog modal-sm modal-dialog-centered"
            onClick={e => e.stopPropagation()}>
            <div className="modal-content">
              <div className="modal-header">
                <h6 className="modal-title">{contactModal.name}'s Contact</h6>
                <button className="btn-close" onClick={() => setContactModal(null)} />
              </div>
              <div className="modal-body px-3 py-2">
                {[
                  { icon: 'fas fa-phone',          color: '#6b9cdb', label: 'Phone',    value: contactModal.phone },
                  { icon: 'fab fa-weixin',          color: '#07c160', label: 'WeChat',   value: contactModal.wechat },
                  { icon: 'fas fa-map-marker-alt',  color: '#e74c3c', label: 'Address',
                    value: [contactModal.address, contactModal.postal_code].filter(Boolean).join('  ') },
                ].filter(row => row.value).map(row => (
                  <div key={row.label} className="d-flex align-items-center gap-3 py-3 border-bottom">
                    <i className={`${row.icon}`} style={{ color: row.color, width: 18, textAlign: 'center', fontSize: '1rem' }} />
                    <div className="flex-grow-1 overflow-hidden">
                      <div style={{ fontSize: '.7rem', color: '#888', marginBottom: 1 }}>{row.label}</div>
                      <div className="fw-semibold text-truncate" style={{ fontSize: '.95rem' }}>{row.value}</div>
                    </div>
                    <button
                      className="btn btn-sm btn-outline-secondary flex-shrink-0"
                      style={{ fontSize: '.75rem', padding: '2px 8px' }}
                      onClick={() => { navigator.clipboard.writeText(row.value); flash('Copied!') }}
                    >
                      <i className="fas fa-copy" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Chat view ─────────────────────────────────────────────────────── */}
      {activeChat ? (
        <div className="d-flex flex-column" style={{ height: 'calc(100vh - 110px)' }}>
          {/* Header */}
          <div className="d-flex align-items-center gap-3 mb-3">
            <button className="btn btn-outline-secondary btn-sm" onClick={closeChat}>
              <i className="fas fa-arrow-left" />
            </button>
            <Avatar display={activeChat.display_name} avatar={activeChat.avatar_url} size={36} />
            <div className="flex-grow-1">
              <div className="fw-semibold">{activeChat.display_name}</div>
              <div className="text-muted small" style={{ fontSize: '.75rem' }}>
                {onlineSet.has(activeChat.username) ? '🟢 Online' : '⚪ Offline'}
              </div>
            </div>
            {contactStatusMap[activeChat.username] === 'approved' ? (
              <button className="btn btn-outline-success btn-sm" onClick={() => showContact(activeChat)}>
                <i className="fas fa-id-card me-1" />Contact
              </button>
            ) : contactStatusMap[activeChat.username] === 'pending' ? (
              <span className="badge bg-warning text-dark">Contact Pending</span>
            ) : (
              <button className="btn btn-outline-primary btn-sm" onClick={() => requestContact(activeChat.username)}>
                <i className="fas fa-address-card me-1" />Request Contact
              </button>
            )}
          </div>

          {/* Messages */}
          <div className="flex-grow-1 overflow-auto border rounded p-3 d-flex flex-column gap-2"
            style={{ background: '#f8f9fa' }}>
            {chatHistory.length === 0 && (
              <div className="text-center text-muted my-auto" style={{ fontSize: '.875rem' }}>
                No messages yet. Say hello!
              </div>
            )}
            {(() => {
              let lastDate = null
              return chatHistory.flatMap(m => {
                const isMe = m.sender === user.username
                const dateLabel = cstDateLabel(m.created_at)
                const items = []
                if (dateLabel !== lastDate) {
                  lastDate = dateLabel
                  items.push(
                    <div key={`sep-${m.id}`} className="text-center my-1">
                      <span style={{ background: '#dde3ea', color: '#555', fontSize: '.7rem',
                        padding: '2px 12px', borderRadius: 10, userSelect: 'none' }}>
                        {dateLabel}
                      </span>
                    </div>
                  )
                }
                const isCard = !!parseJoinUrl(m.content)
                items.push(
                  <div key={m.id} className={`d-flex ${isMe ? 'justify-content-end' : 'justify-content-start'}`}>
                    <div style={{
                      maxWidth: '70%', padding: isCard ? '6px 8px' : '8px 14px',
                      borderRadius: isMe ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                      background: isMe ? '#6b9cdb' : '#fff',
                      color: isMe ? '#fff' : '#333',
                      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
                      fontSize: '.9rem', wordBreak: 'break-word',
                    }}>
                      {renderMessageContent(m.content, isMe)}
                      <div style={{ fontSize: '.65rem', opacity: .6, marginTop: 3, textAlign: 'right' }}>
                        {new Date(m.created_at).toLocaleTimeString('zh-CN', {
                          hour: '2-digit', minute: '2-digit', timeZone: 'America/Chicago'
                        })}
                      </div>
                    </div>
                  </div>
                )
                return items
              })
            })()}
            <div ref={chatEndRef} />
          </div>

          {/* Input bar */}
          <div className="d-flex gap-2 mt-3">
            <input
              className="form-control"
              placeholder="Type a message…"
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              maxLength={1000}
              autoFocus
            />
            <button className="btn btn-primary px-3" onClick={sendMessage}>
              <i className="fas fa-paper-plane" />
            </button>
          </div>
        </div>

      ) : (
        /* ── Tabs view ──────────────────────────────────────────────────────── */
        <>
          <div className="d-flex align-items-center gap-2 mb-4">
            <i className="fas fa-user-friends fa-lg text-primary" />
            <h4 className="mb-0 fw-bold">Friends</h4>
            {(pending.length + contactReqs.length) > 0 && <span className="badge bg-danger">{pending.length + contactReqs.length}</span>}
          </div>

          <div className="radio-inputs mb-4">
            {[
              { key: 'friends', label: 'Friends' },
              { key: 'pending', label: (pending.length + contactReqs.length) > 0 ? `Requests (${pending.length + contactReqs.length})` : 'Requests' },
              { key: 'add',     label: 'Add' },
            ].map(t => (
              <label className="radio" key={t.key}>
                <input
                  type="radio"
                  name="friends-tab"
                  checked={tab === t.key}
                  onChange={() => setTab(t.key)}
                />
                <span className="name">{t.label}</span>
              </label>
            ))}
          </div>

          {loading ? (
            <div className="text-center py-5"><HandLoader /></div>

          ) : tab === 'friends' ? (
            friends.length === 0 ? (
              <div className="text-center py-5 text-muted">
                <i className="fas fa-user-friends fa-3x mb-3 d-block opacity-25" />
                <p>No friends yet. Use the Add tab to connect!</p>
              </div>
            ) : (
              <div className="d-flex flex-column gap-2">
                {friends.map(f => {
                  const cStatus  = contactStatusMap[f.username]
                  const sharedReq = sharedContacts.find(r => r.from_user === f.username)
                  return (
                  <div key={f.username} className="card px-3 py-2 d-flex flex-row align-items-center gap-3">
                    <div
                      className="position-relative"
                      style={{ cursor: 'pointer' }}
                      onClick={() => navigate(`/u/${f.username}`)}
                      title={`View ${f.display_name}'s profile`}
                    >
                      <Avatar display={f.display_name} avatar={f.avatar_url} size={42} />
                      {onlineSet.has(f.username) && (
                        <span style={{
                          position: 'absolute', bottom: 1, right: 1,
                          width: 11, height: 11, borderRadius: '50%',
                          background: '#22c55e', border: '2px solid #fff',
                        }} />
                      )}
                    </div>
                    <div className="flex-grow-1 overflow-hidden">
                      <div className="d-flex align-items-center gap-2">
                        <span className="fw-semibold text-truncate">{f.display_name}</span>
                        {unreadMap[f.username] > 0 && (
                          <span className="badge bg-danger" style={{ fontSize: '.65rem', minWidth: 18 }}>
                            {unreadMap[f.username] > 99 ? '99+' : unreadMap[f.username]}
                          </span>
                        )}
                      </div>
                      <div className="text-muted small">{f.username}</div>
                    </div>
                    <div className="d-flex gap-1 flex-shrink-0 flex-wrap justify-content-end">
                      {/* Contact status */}
                      {cStatus === 'approved' ? (
                        <button className="btn btn-sm btn-outline-success" onClick={() => showContact(f)}
                          data-full="Contact" data-icon="fas fa-id-card">
                          <i className="fas fa-id-card" /> <span className="d-none d-sm-inline">Contact</span>
                        </button>
                      ) : cStatus === 'pending' ? (
                        <span className="badge bg-warning text-dark align-self-center px-1" title="Contact request pending">
                          <i className="fas fa-clock" />
                        </span>
                      ) : cStatus === 'hidden' ? (
                        <span className="badge bg-secondary align-self-center px-1" title="They hid their contact">
                          <i className="fas fa-eye-slash" />
                        </span>
                      ) : (
                        <button className="btn btn-sm btn-outline-primary px-1" onClick={() => requestContact(f.username)}
                          title="Request Contact">
                          <i className="fas fa-address-card" /> <span className="d-none d-sm-inline">Contact</span>
                        </button>
                      )}
                      <button className="btn btn-sm btn-primary px-2" onClick={() => openChat(f)}>
                        <i className="fas fa-comment-dots" /> <span className="d-none d-sm-inline">Chat</span>
                      </button>
                      {sharedReq && (
                        <button className="btn btn-sm btn-outline-warning px-2"
                          title="Withdraw contact access"
                          onClick={() => revokeContact(sharedReq.id, f.display_name)}>
                          <i className="fas fa-eye-slash" />
                        </button>
                      )}
                      <button className="btn btn-sm btn-outline-danger px-2" onClick={() => unfriend(f.username)}
                        title="Unfriend">
                        <i className="fas fa-user-minus" />
                      </button>
                    </div>
                  </div>
                )})}
              </div>
            )

          ) : tab === 'pending' ? (
            pending.length === 0 && contactReqs.length === 0 && sharedContacts.length === 0 ? (
              <div className="text-center py-5 text-muted">
                <i className="fas fa-bell fa-3x mb-3 d-block opacity-25" />
                <p>No pending requests.</p>
              </div>
            ) : (
              <div className="d-flex flex-column gap-3">
                {/* Friend requests */}
                {pending.length > 0 && (
                  <>
                    <div className="text-muted small fw-semibold text-uppercase" style={{ letterSpacing: '.06em' }}>
                      Friend Requests
                    </div>
                    {pending.map(r => (
                      <div key={r.id} className="card px-3 py-2 d-flex flex-row align-items-center gap-3">
                        <Avatar display={r.from_display || r.from_user} avatar={r.from_avatar} size={42} />
                        <div className="flex-grow-1">
                          <div className="fw-semibold">{r.from_display || r.from_user}</div>
                          <div className="text-muted small">
                            {r.message || 'wants to be your friend'}
                          </div>
                        </div>
                        <div className="d-flex gap-2 flex-shrink-0">
                          <button className="btn btn-sm btn-success" onClick={() => respond(r.id, 'accept')}>
                            <i className="fas fa-check me-1" />Accept
                          </button>
                          <button className="btn btn-sm btn-outline-secondary" onClick={() => respond(r.id, 'reject')}>
                            Decline
                          </button>
                        </div>
                      </div>
                    ))}
                  </>
                )}

                {/* Contact requests */}
                {contactReqs.length > 0 && (
                  <>
                    <div className="text-muted small fw-semibold text-uppercase mt-2" style={{ letterSpacing: '.06em' }}>
                      Contact Requests
                    </div>
                    {contactReqs.map(r => (
                      <div key={r.id} className="card px-3 py-2 d-flex flex-row align-items-center gap-3">
                        <Avatar display={r.from_display || r.from_user} avatar={r.from_avatar} size={42} />
                        <div className="flex-grow-1">
                          <div className="fw-semibold">{r.from_display || r.from_user}</div>
                          <div className="text-muted small">wants to see your contact info</div>
                        </div>
                        <div className="d-flex gap-2 flex-shrink-0">
                          <button className="btn btn-sm btn-success" onClick={() => respondContact(r.id, 'approve', r.from_user)}>
                            <i className="fas fa-check me-1" />Share
                          </button>
                          <button className="btn btn-sm btn-outline-secondary" onClick={() => respondContact(r.id, 'decline', r.from_user)}>
                            Decline
                          </button>
                        </div>
                      </div>
                    ))}
                  </>
                )}

                {/* Shared with */}
                {sharedContacts.length > 0 && (
                  <>
                    <div className="text-muted small fw-semibold text-uppercase mt-2" style={{ letterSpacing: '.06em' }}>
                      Shared With
                    </div>
                    {sharedContacts.map(r => (
                      <div key={r.id} className="card px-3 py-2 d-flex flex-row align-items-center gap-3">
                        <Avatar display={r.from_display || r.from_user} avatar={r.from_avatar} size={42} />
                        <div className="flex-grow-1">
                          <div className="fw-semibold">{r.from_display || r.from_user}</div>
                          <div className="text-muted small">can see your contact info</div>
                        </div>
                        <button className="btn btn-sm btn-outline-danger flex-shrink-0"
                          onClick={() => revokeContact(r.id, r.from_display || r.from_user)}>
                          <i className="fas fa-eye-slash me-1" />Revoke
                        </button>
                      </div>
                    ))}
                  </>
                )}
              </div>
            )

          ) : (
            /* Add tab */
            <div>
              <form className="d-flex gap-2 mb-4" onSubmit={handleSearch}>
                <input
                  className="form-control"
                  placeholder="Search by username or display name…"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  minLength={2}
                />
                <button className="btn btn-primary flex-shrink-0" type="submit" disabled={searching}>
                  {searching
                    ? <span className="spinner-border spinner-border-sm" />
                    : <i className="fas fa-search" />}
                </button>
              </form>

              {searchResults.length === 0 && !searching && (
                <div className="text-center py-4 text-muted" style={{ fontSize: '.9rem' }}>
                  {searchQuery.length >= 2
                    ? 'No users found.'
                    : 'Type at least 2 characters to search.'}
                </div>
              )}

              <div className="d-flex flex-column gap-2">
                {searchResults.map(u => {
                  const isFriend  = friends.some(f => f.username === u.username)
                  const isPending = sentSet.has(u.username)
                  return (
                    <div key={u.username} className="card px-3 py-2 d-flex flex-row align-items-center gap-3">
                      <Avatar display={u.display_name} avatar={u.avatar_url} size={42} />
                      <div className="flex-grow-1 overflow-hidden">
                        <div className="fw-semibold text-truncate">{u.display_name}</div>
                        <div className="text-muted small">{u.username}</div>
                      </div>
                      <div className="flex-shrink-0 d-flex align-items-center gap-1">
                        {isFriend && <span className="badge bg-success px-1" style={{fontSize:'.7rem'}}>Friends</span>}
                        {!isFriend && (isPending
                          ? <span className="badge bg-warning text-dark px-1" style={{fontSize:'.7rem'}}>Pending</span>
                          : <button className="btn btn-sm btn-outline-primary px-2" onClick={() => sendRequest(u.username)} title="Add friend">
                              <i className="fas fa-user-plus" />
                            </button>
                        )}
                        <button
                          className="btn btn-sm btn-primary px-2"
                          onClick={() => { setTab('friends'); openChat(u) }}
                          title="Send a message"
                        >
                          <i className="fas fa-comment-dots" />
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
