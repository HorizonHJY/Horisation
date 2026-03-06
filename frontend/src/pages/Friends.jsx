import React, { useState, useEffect, useRef } from 'react'
import { io } from 'socket.io-client'
import { api } from '../api'
import HandLoader from '../components/HandLoader'
import { useAuth } from '../App'

function Avatar({ display, avatar, size = 40 }) {
  if (avatar) return (
    <img src={avatar} alt={display}
      style={{ width: size, height: size, borderRadius: '50%', objectFit: 'cover', flexShrink: 0 }} />
  )
  return (
    <div style={{
      width: size, height: size, borderRadius: '50%', background: '#3a7bd5', color: '#fff',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontWeight: 700, fontSize: size * 0.4, flexShrink: 0,
    }}>
      {display?.[0]?.toUpperCase() || '?'}
    </div>
  )
}

export default function Friends() {
  const { user } = useAuth()
  const socketRef     = useRef(null)
  const chatEndRef    = useRef(null)
  const activeChatRef = useRef(null)   // mirror of activeChat for socket handler

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
    socket.on('friend_accepted', ({ from_user }) => {
      flash(`${from_user} accepted your friend request!`, 'success')
      loadFriends()
    })
    socket.on('chat_message', (msg) => {
      const chat = activeChatRef.current
      if (!chat) return
      const [ua, ub] = [user.username, chat.username].sort()
      if (msg.room_key === `${ua}:${ub}`) {
        setChatHistory(h => [...h, msg])
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
    if (tab === 'friends') loadFriends()
    if (tab === 'pending') loadPending()
    if (tab === 'add')     loadSentRequests()
    else { setSearchQuery(''); setSearchResults([]) }
  }, [tab])

  // ── Data loaders ────────────────────────────────────────────────────────────
  async function loadFriends() {
    setLoading(true)
    const [fRes, cRes] = await Promise.all([
      api.get('/api/friends/list'),
      api.get('/api/friends/contact/sent'),
    ])
    if (fRes.ok) setFriends(fRes.friends)
    if (cRes.ok) {
      const map = {}
      cRes.requests.forEach(r => { map[r.to_user] = r.status })
      setContactStatusMap(map)
    }
    setLoading(false)
  }

  async function loadPending() {
    setLoading(true)
    const [fRes, cRes] = await Promise.all([
      api.get('/api/friends/requests/pending'),
      api.get('/api/friends/contact/requests'),
    ])
    if (fRes.ok) setPending(fRes.requests)
    if (cRes.ok) setContactReqs(cRes.requests)
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
    const d = await api.get(`/api/friends/${friend.username}/history`)
    if (d.ok) setChatHistory(d.messages)
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
    if (d.ok) setContactModal({ name: friend.display_name, info: d.contact_info || '(not set)' })
    else flash(d.error, 'danger')
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
              <div className="modal-body text-center py-4">
                <i className="fas fa-id-card fa-2x text-primary mb-3 d-block" />
                <p className="fs-5 mb-0 fw-semibold">{contactModal.info}</p>
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
            {chatHistory.map(m => {
              const isMe = m.sender === user.username
              return (
                <div key={m.id} className={`d-flex ${isMe ? 'justify-content-end' : 'justify-content-start'}`}>
                  <div style={{
                    maxWidth: '70%', padding: '8px 14px',
                    borderRadius: isMe ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                    background: isMe ? '#3a7bd5' : '#fff',
                    color: isMe ? '#fff' : '#333',
                    boxShadow: '0 1px 3px rgba(0,0,0,.08)',
                    fontSize: '.9rem', wordBreak: 'break-word',
                  }}>
                    {m.content}
                    <div style={{ fontSize: '.65rem', opacity: .6, marginTop: 3, textAlign: 'right' }}>
                      {new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                </div>
              )
            })}
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

          <ul className="nav nav-tabs mb-4">
            {[
              { key: 'friends', label: 'Friends',  icon: 'fa-user-friends' },
              { key: 'pending', label: (pending.length + contactReqs.length) > 0 ? `Requests (${pending.length + contactReqs.length})` : 'Requests', icon: 'fa-bell' },
              { key: 'add',     label: 'Add',       icon: 'fa-user-plus' },
            ].map(t => (
              <li className="nav-item" key={t.key}>
                <button className={`nav-link ${tab === t.key ? 'active' : ''}`} onClick={() => setTab(t.key)}>
                  <i className={`fas ${t.icon} me-1`} />{t.label}
                </button>
              </li>
            ))}
          </ul>

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
                  const cStatus = contactStatusMap[f.username]
                  return (
                  <div key={f.username} className="card px-3 py-2 d-flex flex-row align-items-center gap-3">
                    <div className="position-relative">
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
                      <div className="fw-semibold text-truncate">{f.display_name}</div>
                      <div className="text-muted small">{f.username}</div>
                    </div>
                    <div className="d-flex gap-2 flex-shrink-0 flex-wrap justify-content-end">
                      {/* Contact status */}
                      {cStatus === 'approved' ? (
                        <button className="btn btn-sm btn-outline-success" onClick={() => showContact(f)}>
                          <i className="fas fa-id-card me-1" />Contact
                        </button>
                      ) : cStatus === 'pending' ? (
                        <span className="badge bg-warning text-dark align-self-center">
                          <i className="fas fa-clock me-1" />Pending
                        </span>
                      ) : cStatus === 'hidden' ? (
                        <span className="badge bg-secondary align-self-center" title="They hid their contact">
                          <i className="fas fa-eye-slash me-1" />Hidden
                        </span>
                      ) : (
                        <button className="btn btn-sm btn-outline-primary" onClick={() => requestContact(f.username)}>
                          <i className="fas fa-address-card me-1" />Request Contact
                        </button>
                      )}
                      <button className="btn btn-sm btn-primary" onClick={() => openChat(f)}>
                        <i className="fas fa-comment-dots me-1" />Chat
                      </button>
                      <button className="btn btn-sm btn-outline-danger" onClick={() => unfriend(f.username)}
                        title="Unfriend">
                        <i className="fas fa-user-minus" />
                      </button>
                    </div>
                  </div>
                )})}
              </div>
            )

          ) : tab === 'pending' ? (
            pending.length === 0 && contactReqs.length === 0 ? (
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
                      <div className="flex-shrink-0">
                        {isFriend ? (
                          <span className="badge bg-success">Friends</span>
                        ) : isPending ? (
                          <span className="badge bg-warning text-dark">Pending</span>
                        ) : (
                          <button className="btn btn-sm btn-primary" onClick={() => sendRequest(u.username)}>
                            <i className="fas fa-user-plus me-1" />Add
                          </button>
                        )}
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
