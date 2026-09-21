import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useSocket, useSocketEvent } from '../components/SocketProvider'
import HandLoader from '../components/HandLoader'
import Modal, { ConfirmDialog } from '../components/Modal'
import { useAuth, useNotifications } from '../App'

/**
 * Friends — really the messages page.
 *
 * Two panes on a desktop, like Teams: the left rail holds four tabs (Chats,
 * Friends, Requests, Add) and the right pane holds the open conversation. On
 * a phone it is one pane at a time — list, tap, chat with a back arrow.
 *
 * Chats is the default and lists every conversation, newest first, including
 * people who are not friends (Market's Reach Out lets anyone write). Before
 * 2026-09-21 a non-friend's message raised the badge but had no row anywhere
 * to open it from.
 *
 * The contact-sharing state is one text chip with four values. Everything
 * else about a person lives behind "···", written as a full sentence — the
 * old row had five unlabelled icons, two of them the same eye meaning
 * opposite things.
 */

function Avatar({ display, avatar, size = 40, online = false }) {
  const inner = avatar
    ? <img src={avatar} alt="" style={{ width: size, height: size, borderRadius: '50%', objectFit: 'cover' }} />
    : <div className="fr-av__fallback" style={{ width: size, height: size, fontSize: size * 0.4 }}>{display?.[0]?.toUpperCase() || '?'}</div>
  return (
    <span className="fr-av" style={{ width: size, height: size }}>
      {inner}
      {online && <span className="fr-av__dot" aria-label="online" />}
    </span>
  )
}

// ── Time helpers (St. Louis) ───────────────────────────────────────────────────
const CT = { timeZone: 'America/Chicago' }

function cstDateLabel(isoStr) {
  const d = new Date(isoStr)
  const dStr = d.toLocaleDateString('zh-CN', CT)
  const now = new Date()
  const nowStr = now.toLocaleDateString('zh-CN', CT)
  const yest = new Date(now); yest.setDate(yest.getDate() - 1)
  const yStr = yest.toLocaleDateString('zh-CN', CT)
  if (dStr === nowStr) return '今天'
  if (dStr === yStr) return '昨天'
  return d.toLocaleDateString('zh-CN', { ...CT, month: 'long', day: 'numeric' })
}

// The time column of the Chats list: today → "21:14", yesterday → "Yesterday", else "Sep 14".
function listTime(isoStr) {
  if (!isoStr) return ''
  const d = new Date(isoStr)
  const dStr = d.toLocaleDateString('en-US', CT)
  const now = new Date()
  if (dStr === now.toLocaleDateString('en-US', CT)) {
    return d.toLocaleTimeString('en-GB', { ...CT, hour: '2-digit', minute: '2-digit' })
  }
  const yest = new Date(now); yest.setDate(yest.getDate() - 1)
  if (dStr === yest.toLocaleDateString('en-US', CT)) return 'Yesterday'
  return d.toLocaleDateString('en-US', { ...CT, month: 'short', day: 'numeric' })
}

// ── Message rendering ──────────────────────────────────────────────────────────
function parseJoinUrl(text) {
  const m = text.match(/https?:\/\/[^\s]*(\/(travel|bill-split))\?join=([A-Z0-9]+)/i)
  if (!m) return null
  return { type: m[2], code: m[3].toUpperCase(), url: text.match(/https?:\/\/[^\s]+/)[0] }
}

function renderContent(text, isMe) {
  const urlRegex = /(https?:\/\/[^\s]+)/g
  const parts = text.split(urlRegex)
  return parts.map((part, i) =>
    urlRegex.test(part)
      ? <a key={i} href={part} target="_blank" rel="noopener noreferrer"
          style={{ color: isMe ? '#d4eaff' : 'var(--accent-text)', textDecoration: 'underline', wordBreak: 'break-all' }}>
          {part}
        </a>
      : part
  )
}

function renderMessageContent(content, isMe) {
  const join = parseJoinUrl(content)
  if (join) {
    const isTravel = join.type === 'travel'
    return (
      <a href={join.url} style={{ textDecoration: 'none', display: 'block', minWidth: 200 }}>
        <div style={{
          background: isMe ? 'rgba(255,255,255,0.18)' : 'var(--badge-info-bg)',
          border: `1px solid ${isMe ? 'rgba(255,255,255,0.35)' : 'var(--border-medium)'}`,
          borderRadius: 10, padding: '8px 12px',
        }}>
          <div className="d-flex align-items-center gap-2">
            <i className={`fas ${isTravel ? 'fa-route' : 'fa-receipt'}`}
               style={{ color: isMe ? '#fff' : 'var(--badge-info-fg)', fontSize: '1.2rem', flexShrink: 0 }} aria-hidden="true" />
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, fontSize: '.85rem', color: isMe ? '#fff' : 'var(--text-primary)' }}>
                {isTravel ? '旅行计划邀请' : '分账邀请'}
              </div>
              <div style={{ fontSize: '.72rem', opacity: 0.65, fontFamily: 'monospace', letterSpacing: '.08em' }}>
                {join.code}
              </div>
            </div>
            <span style={{ fontSize: '.72rem', color: isMe ? '#d4eaff' : 'var(--badge-info-fg)', flexShrink: 0 }}>
              点击加入 →
            </span>
          </div>
        </div>
      </a>
    )
  }
  return renderContent(content, isMe)
}

// One line for the Chats list. A share card reads as what it is, not as a URL.
function previewText(content, isMine) {
  const join = parseJoinUrl(content)
  const body = join ? (join.type === 'travel' ? '旅行计划邀请' : '分账邀请') : content
  return (isMine ? 'You: ' : '') + body
}

// ── "···" menu ─────────────────────────────────────────────────────────────────
/* A small popover of full sentences. Outside click and Escape close it; the
   items are real buttons so the keyboard reaches them. */
function RowMenu({ label, items, align = 'right' }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  useEffect(() => {
    if (!open) return
    const onDoc = (e) => { if (!ref.current?.contains(e.target)) setOpen(false) }
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey) }
  }, [open])
  return (
    <span className="fr-more" ref={ref}>
      <button type="button" className="fr-more__btn" aria-haspopup="menu" aria-expanded={open}
              aria-label={label} onClick={() => setOpen(o => !o)}>
        <i className="fas fa-ellipsis-h" aria-hidden="true" />
      </button>
      {open && (
        <div className="fr-menu" role="menu" style={align === 'left' ? { left: 0, right: 'auto' } : undefined}>
          {items.map((it, i) => it.section
            ? <small key={i} className="fr-menu__sec">{it.section}</small>
            : (
              <button key={i} type="button" role="menuitem"
                      className={`fr-menu__item${it.danger ? ' is-danger' : ''}`}
                      onClick={() => { setOpen(false); it.onClick() }}>
                <i className={`fas ${it.icon}`} aria-hidden="true" />{it.label}
              </button>
            ))}
        </div>
      )}
    </span>
  )
}

// The contact-sharing chip. One of four things, always in words.
function ContactChip({ status, onView, onRequest }) {
  if (status === 'approved') return (
    <button type="button" className="fr-chip fr-chip--good" onClick={onView}>
      <i className="fas fa-id-card" aria-hidden="true" />View contact
    </button>
  )
  if (status === 'pending') return (
    <span className="fr-chip fr-chip--warn"><i className="fas fa-clock" aria-hidden="true" />Requested · waiting</span>
  )
  if (status === 'hidden') return (
    <span className="fr-chip fr-chip--neutral" title="They keep their contact details private">
      <i className="fas fa-lock" aria-hidden="true" />Contact hidden
    </span>
  )
  return (
    <button type="button" className="fr-chip fr-chip--info" onClick={onRequest}>
      <i className="fas fa-address-card" aria-hidden="true" />Request contact
    </button>
  )
}

export default function Friends() {
  const { user } = useAuth()
  const {
    unreadMap, clearUnread,
    friendRequests: pending, contactRequests: contactReqs,
    dismissFriendRequest, dismissContactRequest, contactRequestFrom,
    refresh: refreshNotifications,
  } = useNotifications()
  const { socket } = useSocket()
  const location      = useLocation()
  const navigate      = useNavigate()
  const msgsRef       = useRef(null)
  const inputRef      = useRef(null)
  const activeChatRef = useRef(null)   // mirror of activeChat for socket handler
  const tabRef        = useRef('chats')

  const [tab, setTab]                 = useState('chats')
  const [conversations, setConversations] = useState([])
  const [friends, setFriends]         = useState([])
  const [friendFilter, setFriendFilter] = useState('')
  const [chatFilter, setChatFilter]   = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [searching, setSearching]     = useState(false)
  const [sentSet, setSentSet]         = useState(new Set())
  const [activeChat, setActiveChat]   = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [chatInput, setChatInput]     = useState('')
  const [onlineSet, setOnlineSet]     = useState(new Set())
  const [contactModal, setContactModal]   = useState(null)
  const [sharedContacts, setSharedContacts] = useState([])  // approved contact reqs where I am to_user
  // contactStatusMap: { [username]: 'pending' | 'approved' | 'declined' | 'hidden' }
  const [contactStatusMap, setContactStatusMap] = useState({})
  const [confirmUnfriend, setConfirmUnfriend] = useState(null)   // { username, display_name }
  const [toast, setToast]             = useState(null)
  const [loading, setLoading]         = useState(false)
  const toastTimer = useRef(null)

  const flash = useCallback((msg, type = 'success') => {
    if (toastTimer.current) clearTimeout(toastTimer.current)
    setToast({ msg, type })
    toastTimer.current = setTimeout(() => setToast(null), 3000)
  }, [])
  useEffect(() => () => { if (toastTimer.current) clearTimeout(toastTimer.current) }, [])

  const friendSet = new Set(friends.map(f => f.username))
  const isFriend  = (username) => friendSet.has(username)

  // ── Data loaders ────────────────────────────────────────────────────────────
  async function loadConversations() {
    const d = await api.get('/api/friends/conversations')
    if (d.ok) setConversations(d.conversations)
  }

  async function loadFriends() {
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
  }

  async function loadPending() {
    // Both request lists come from NotificationsProvider, which keeps them
    // live over the socket; only "shared with" is local to this page.
    const sRes = await api.get('/api/friends/contact/shared')
    if (sRes.ok) setSharedContacts(sRes.requests)
    refreshNotifications()
  }

  async function loadSentRequests() {
    const d = await api.get('/api/friends/requests/sent')
    if (d.ok) {
      const pendingTo = d.requests.filter(r => r.status === 'pending').map(r => r.to_user)
      setSentSet(new Set(pendingTo))
    }
  }

  // Friends and contact state are needed by every tab (the chip in the chat
  // header, the "not friends" banner), so they load once up front.
  useEffect(() => {
    setLoading(true)
    Promise.all([loadConversations(), loadFriends()]).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    tabRef.current = tab
    if (tab === 'chats')    loadConversations()
    if (tab === 'friends')  loadFriends()
    if (tab === 'requests') loadPending()
    if (tab === 'add')      loadSentRequests()
    else { setSearchQuery(''); setSearchResults([]) }
  }, [tab])

  // Refresh when the user comes back to the tab.
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState !== 'visible') return
      loadConversations()
      if (tabRef.current === 'friends') loadFriends()
      else if (tabRef.current === 'requests') loadPending()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [])

  // ── Socket ──────────────────────────────────────────────────────────────────
  /* The connection itself is owned by SocketProvider and lives for the whole
     session; this page only attaches the handlers it needs and detaches them
     on unmount. It must never disconnect the socket — other pages and the
     notification badge are listening on the same one.

     Friend and contact requests are handled in NotificationsProvider so they
     arrive wherever the user is; what stays here is presence, the open
     conversation, the Chats list, and send errors. */
  useEffect(() => {
    if (!socket) return

    const askOnline  = () => socket.emit('friends_get_online')
    const onOnline   = ({ online }) => setOnlineSet(new Set(online))
    const onAccepted = ({ from_user }) => {
      flash(`${from_user} accepted your friend request!`, 'success')
      loadFriends(); loadConversations()
    }
    const onChatMessage = (msg) => {
      const chat = activeChatRef.current
      const [sa, sb] = msg.room_key.split(':')
      const other = sa === user.username ? sb : sa
      // Keep the Chats list current without a refetch: move the row up and
      // rewrite its last line. A brand-new correspondent gets a refetch,
      // because the identity (name, avatar) is not on the message.
      setConversations(prev => {
        const idx = prev.findIndex(c => c.username === other)
        if (idx < 0) { loadConversations(); return prev }
        const isOpen = chat && chat.username === other
        const row = { ...prev[idx], last_sender: msg.sender, last_content: msg.content, last_at: msg.created_at,
                      unread: (msg.sender === user.username || isOpen) ? 0 : (prev[idx].unread || 0) + 1 }
        return [row, ...prev.filter((_, i) => i !== idx)]
      })
      if (!chat) return
      const [ua, ub] = [user.username, chat.username].sort()
      if (msg.room_key !== `${ua}:${ub}`) return
      setChatHistory(h => (h.some(m => m.id && m.id === msg.id) ? h : [...h, msg]))
      if (msg.sender !== user.username) {
        api.post(`/api/friends/${chat.username}/read`)
      }
    }
    const onChatError = ({ message }) => flash(message, 'danger')

    if (socket.connected) askOnline()
    socket.on('connect', askOnline)
    socket.on('online_list', onOnline)
    socket.on('friend_accepted', onAccepted)
    socket.on('chat_message', onChatMessage)
    socket.on('chat_error', onChatError)

    return () => {
      socket.off('connect', askOnline)
      socket.off('online_list', onOnline)
      socket.off('friend_accepted', onAccepted)
      socket.off('chat_message', onChatMessage)
      socket.off('chat_error', onChatError)
    }
  }, [socket, user.username])

  // Toast the requests that NotificationsProvider received, but only while
  // this page is the one on screen.
  useSocketEvent('friend_request_incoming', () => flash('New friend request!', 'info'))
  useSocketEvent('contact_request_incoming', () => flash('New contact request!', 'info'))

  // Auto-scroll the message column, not the page — scrollIntoView on a
  // two-pane layout drags the whole document down and hides the header.
  useEffect(() => {
    const el = msgsRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [chatHistory])

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

  // ── Actions ─────────────────────────────────────────────────────────────────
  const sendRequest = async (toUser) => {
    const d = await api.post('/api/friends/requests', { to_user: toUser })
    if (d.ok) { setSentSet(s => new Set([...s, toUser])); flash('Friend request sent!') }
    else flash(d.error, 'danger')
  }

  const respond = async (reqId, action) => {
    const d = await api.put(`/api/friends/requests/${reqId}`, { action })
    if (d.ok) {
      dismissFriendRequest(reqId)
      if (action === 'accept') { flash('Friend added!'); loadFriends(); loadConversations() }
    } else flash(d.error, 'danger')
  }

  const unfriend = async () => {
    const { username } = confirmUnfriend
    const d = await api.delete(`/api/friends/${username}`)
    setConfirmUnfriend(null)
    if (d.ok) {
      setFriends(prev => prev.filter(f => f.username !== username))
      setConversations(prev => prev.map(c => c.username === username ? { ...c, is_friend: false } : c))
      flash('Removed from friends.')
    } else flash(d.error, 'danger')
  }

  const openChat = async (person) => {
    activeChatRef.current = person
    // NotificationsProvider listens for every message app-wide; this tells it
    // not to raise a badge for the conversation already on screen.
    window.__hzActiveChatWith = person.username
    setActiveChat(person)
    setChatHistory([])
    setChatInput('')                 // a draft belongs to one conversation
    clearUnread(person.username)
    setConversations(prev => prev.map(c => c.username === person.username ? { ...c, unread: 0 } : c))
    api.post(`/api/friends/${person.username}/read`)
    const d = await api.get(`/api/friends/${person.username}/history`)
    if (d.ok) setChatHistory(d.messages)
    requestAnimationFrame(() => inputRef.current?.focus())
    return person
  }

  const closeChat = () => {
    activeChatRef.current = null
    window.__hzActiveChatWith = null
    setActiveChat(null)
    setChatHistory([])
  }

  // Leaving the page entirely counts as closing the conversation.
  useEffect(() => () => { window.__hzActiveChatWith = null }, [])

  const sendMessage = () => {
    if (!chatInput.trim() || !activeChat) return
    socket?.emit('chat_send', { to_user: activeChat.username, content: chatInput.trim() })
    setChatInput('')
  }

  const showContact = async (person) => {
    const d = await api.get(`/api/friends/${person.username}/contact`)
    if (d.ok) setContactModal({
      name: person.display_name,
      phone: d.phone, wechat: d.wechat,
      address: d.address, postal_code: d.postal_code,
    })
    else flash(d.error, 'danger')
  }

  const revokeContact = async (reqId, fromUser) => {
    const d = await api.put(`/api/friends/contact/requests/${reqId}`, { action: 'revoke' })
    if (d.ok) {
      setSharedContacts(prev => prev.filter(r => r.id !== reqId))
      flash(`${fromUser} can no longer see your contact.`)
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
      dismissContactRequest(reqId)
      flash(action === 'approve' ? 'Contact shared!' : 'Request declined.')
    } else flash(d.error, 'danger')
  }

  // The "···" for one person, shared by the Friends row and the chat header.
  const personMenu = (p, { inChat = false } = {}) => {
    const sharedReq = sharedContacts.find(r => r.from_user === p.username)
    const items = inChat ? [] : [{ label: 'Open chat', icon: 'fa-comment-dots', onClick: () => openChat(p) }]
    items.push({ label: 'View profile', icon: 'fa-user', onClick: () => navigate(`/u/${p.username}`) })
    if (isFriend(p.username)) {
      // The chat header's chip is hidden on a phone, so the menu carries the
      // contact action there too. On desktop it is simply a second way in.
      if (inChat) {
        const st = contactStatusMap[p.username]
        if (st === 'approved') items.push({ label: 'View contact', icon: 'fa-id-card', onClick: () => showContact(p) })
        else if (!st || st === 'declined') items.push({ label: 'Request contact', icon: 'fa-address-card', onClick: () => requestContact(p.username) })
      }
      if (sharedReq) {
        items.push({ section: 'Sharing' })
        items.push({ label: `Stop sharing my contact with ${p.display_name}`, icon: 'fa-eye-slash',
                     onClick: () => revokeContact(sharedReq.id, p.display_name) })
      }
      items.push({ label: 'Remove friend', icon: 'fa-user-minus', danger: true,
                   onClick: () => setConfirmUnfriend(p) })
    } else if (!sentSet.has(p.username)) {
      items.push({ label: 'Add friend', icon: 'fa-user-plus', onClick: () => sendRequest(p.username) })
    }
    return items
  }

  const requestCount = pending.length + contactReqs.length
  const totalUnread  = conversations.reduce((n, c) => n + (c.unread || 0), 0)

  const visibleConvos  = conversations.filter(c => !chatFilter ||
    `${c.display_name} ${c.username}`.toLowerCase().includes(chatFilter.toLowerCase()))
  const visibleFriends = friends.filter(f => !friendFilter ||
    `${f.display_name} ${f.username}`.toLowerCase().includes(friendFilter.toLowerCase()))

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="fr-page">
      {toast && (
        <div className={`alert alert-${toast.type} app-toast`} role="alert" aria-live="assertive">{toast.msg}</div>
      )}

      {contactModal && (
        <Modal onClose={() => setContactModal(null)} title={`${contactModal.name}'s contact`} scrollable={false}>
          {({ titleId }) => (
            <>
              <div className="modal-header">
                <h5 className="modal-title fw-semibold" id={titleId} style={{ fontSize: '1rem' }}>{contactModal.name}'s contact</h5>
                <button type="button" className="btn-close" aria-label="Close" onClick={() => setContactModal(null)} />
              </div>
              <div className="modal-body px-3 py-2">
                {[
                  { icon: 'fas fa-phone',          label: 'Phone',   value: contactModal.phone },
                  { icon: 'fab fa-weixin',          label: 'WeChat',  value: contactModal.wechat },
                  { icon: 'fas fa-map-marker-alt',  label: 'Address',
                    value: [contactModal.address, contactModal.postal_code].filter(Boolean).join('  ') },
                ].filter(row => row.value).map(row => (
                  <div key={row.label} className="d-flex align-items-center gap-3 py-3 border-bottom">
                    <i className={row.icon} style={{ color: 'var(--accent-text)', width: 18, textAlign: 'center' }} aria-hidden="true" />
                    <div className="flex-grow-1 overflow-hidden">
                      <div className="fr-sub">{row.label}</div>
                      <div className="fw-semibold text-truncate">{row.value}</div>
                    </div>
                    <button type="button" className="btn btn-sm btn-outline-secondary flex-shrink-0"
                            aria-label={`Copy ${row.label}`}
                            onClick={() => { navigator.clipboard.writeText(row.value); flash('Copied!') }}>
                      <i className="fas fa-copy" aria-hidden="true" />
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </Modal>
      )}

      {confirmUnfriend && (
        <ConfirmDialog
          title="Remove friend?"
          message={`${confirmUnfriend.display_name} will no longer see you as a friend. Your messages stay.`}
          confirmLabel="Remove"
          onConfirm={unfriend}
          onClose={() => setConfirmUnfriend(null)}
        />
      )}

      <div className="fr-head">
        <i className="fas fa-user-friends" aria-hidden="true" />
        <h1 className="fr-title">Friends</h1>
      </div>

      <div className={`fr-shell${activeChat ? ' is-chat' : ''}`}>
        {/* ── Left rail ─────────────────────────────────────────────────────── */}
        <div className="fr-rail">
          <div className="fr-tabs" role="tablist" aria-label="Friends sections">
            {[
              { key: 'chats',    label: 'Chats',    count: totalUnread },
              { key: 'friends',  label: 'Friends' },
              { key: 'requests', label: 'Requests', count: requestCount },
              { key: 'add',      label: 'Add' },
            ].map(t => (
              <button key={t.key} type="button" role="tab" aria-selected={tab === t.key}
                      className="fr-tab" onClick={() => setTab(t.key)}>
                {t.label}{t.count > 0 && <span className="fr-tab__dot">{t.count > 99 ? '99+' : t.count}</span>}
              </button>
            ))}
          </div>

          {loading ? (
            <div className="text-center py-5"><HandLoader /></div>

          ) : tab === 'chats' ? (
            <>
              {conversations.length > 4 && (
                <input className="fr-search" placeholder="Search chats" aria-label="Search chats"
                       value={chatFilter} onChange={e => setChatFilter(e.target.value)} />
              )}
              <div className="fr-list" role="list">
                {conversations.length === 0 ? (
                  <div className="fr-empty">
                    <i className="fas fa-comment-dots" aria-hidden="true" />
                    <p>No conversations yet.</p>
                    <p className="fr-sub">Open a friend and say hello, or wait for someone to Reach Out on Market.</p>
                  </div>
                ) : visibleConvos.map(c => (
                  <button key={c.username} type="button" role="listitem"
                          className={`fr-conv${c.unread > 0 ? ' is-unread' : ''}`}
                          aria-current={activeChat?.username === c.username ? 'true' : undefined}
                          onClick={() => openChat(c)}>
                    <Avatar display={c.display_name} avatar={c.avatar_url} size={44} online={onlineSet.has(c.username)} />
                    <span className="fr-conv__body">
                      <span className="fr-conv__top">
                        <span className="fr-conv__name">{c.display_name}</span>
                        <span className="fr-conv__time">{listTime(c.last_at)}</span>
                      </span>
                      <span className="fr-conv__last">{previewText(c.last_content, c.last_sender === user.username)}</span>
                      {!c.is_friend && (
                        <span className="fr-chip fr-chip--warn fr-chip--xs">
                          <i className="fas fa-user-plus" aria-hidden="true" />Not friends
                        </span>
                      )}
                    </span>
                    {c.unread > 0 && <span className="fr-conv__count">{c.unread > 99 ? '99+' : c.unread}</span>}
                  </button>
                ))}
              </div>
            </>

          ) : tab === 'friends' ? (
            <>
              {friends.length > 4 && (
                <input className="fr-search" placeholder="Search friends" aria-label="Search friends"
                       value={friendFilter} onChange={e => setFriendFilter(e.target.value)} />
              )}
              <div className="fr-list">
                {friends.length === 0 ? (
                  <div className="fr-empty">
                    <i className="fas fa-user-friends" aria-hidden="true" />
                    <p>No friends yet.</p>
                    <button type="button" className="btn btn-sm btn-outline-primary" onClick={() => setTab('add')}>Find someone</button>
                  </div>
                ) : visibleFriends.map(f => (
                  <div key={f.username} className="fr-row">
                    <button type="button" className="fr-row__hit" onClick={() => openChat(f)}
                            aria-label={`Open chat with ${f.display_name}`}>
                      <Avatar display={f.display_name} avatar={f.avatar_url} size={44} online={onlineSet.has(f.username)} />
                      <span className="fr-row__body">
                        <span className="fr-row__name">{f.display_name}</span>
                        <span className="fr-sub">{f.username}{onlineSet.has(f.username) ? ' · online' : ''}</span>
                      </span>
                    </button>
                    <span className="fr-row__acts">
                      <ContactChip status={contactStatusMap[f.username]}
                                   onView={() => showContact(f)} onRequest={() => requestContact(f.username)} />
                      <RowMenu label={`More for ${f.display_name}`} items={personMenu(f)} />
                    </span>
                  </div>
                ))}
              </div>
            </>

          ) : tab === 'requests' ? (
            <div className="fr-list">
              {pending.length === 0 && contactReqs.length === 0 && sharedContacts.length === 0 ? (
                <div className="fr-empty">
                  <i className="fas fa-bell" aria-hidden="true" />
                  <p>Nothing waiting on you.</p>
                </div>
              ) : (
                <>
                  {pending.length > 0 && (
                    <>
                      <div className="fr-sec">Friend requests</div>
                      {pending.map(r => (
                        <div key={r.id} className="fr-row fr-row--stack">
                          <span className="fr-row__hit fr-row__hit--static">
                            <Avatar display={r.from_display || r.from_user} avatar={r.from_avatar} size={44} />
                            <span className="fr-row__body">
                              <span className="fr-row__name">{r.from_display || r.from_user}</span>
                              <span className="fr-sub">{r.message || 'wants to be friends'}</span>
                            </span>
                          </span>
                          <span className="fr-row__acts">
                            <button type="button" className="btn btn-sm btn-primary" onClick={() => respond(r.id, 'accept')}>Accept</button>
                            <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => respond(r.id, 'reject')}>Decline</button>
                          </span>
                        </div>
                      ))}
                    </>
                  )}
                  {contactReqs.length > 0 && (
                    <>
                      <div className="fr-sec">Wants to see your contact</div>
                      {contactReqs.map(r => (
                        <div key={r.id} className="fr-row fr-row--stack">
                          <span className="fr-row__hit fr-row__hit--static">
                            <Avatar display={r.from_display || r.from_user} avatar={r.from_avatar} size={44} />
                            <span className="fr-row__body">
                              <span className="fr-row__name">{r.from_display || r.from_user}</span>
                              <span className="fr-sub">asked to see your phone, WeChat, address</span>
                            </span>
                          </span>
                          <span className="fr-row__acts">
                            <button type="button" className="btn btn-sm btn-primary" onClick={() => respondContact(r.id, 'approve', r.from_user)}>Share</button>
                            <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => respondContact(r.id, 'decline', r.from_user)}>Not now</button>
                          </span>
                        </div>
                      ))}
                    </>
                  )}
                  {sharedContacts.length > 0 && (
                    <>
                      <div className="fr-sec">People who can see your contact</div>
                      {sharedContacts.map(r => (
                        <div key={r.id} className="fr-row fr-row--stack">
                          <span className="fr-row__hit fr-row__hit--static">
                            <Avatar display={r.from_display || r.from_user} avatar={r.from_avatar} size={44} />
                            <span className="fr-row__body">
                              <span className="fr-row__name">{r.from_display || r.from_user}</span>
                              <span className="fr-sub">can see your contact details</span>
                            </span>
                          </span>
                          <span className="fr-row__acts">
                            <button type="button" className="btn btn-sm btn-outline-secondary"
                                    onClick={() => revokeContact(r.id, r.from_display || r.from_user)}>
                              <i className="fas fa-eye-slash me-1" aria-hidden="true" />Stop sharing
                            </button>
                          </span>
                        </div>
                      ))}
                    </>
                  )}
                </>
              )}
            </div>

          ) : (
            /* Add */
            <div className="fr-list">
              <form className="d-flex gap-2 mb-3" onSubmit={async (e) => {
                e.preventDefault()
                if (searchQuery.trim().length < 2) return
                setSearching(true)
                const d = await api.get(`/api/friends/users?q=${encodeURIComponent(searchQuery.trim())}`)
                if (d.ok) setSearchResults(d.users)
                setSearching(false)
              }}>
                <input className="fr-search mb-0" placeholder="Search by username or name" aria-label="Search people"
                       value={searchQuery} onChange={e => setSearchQuery(e.target.value)} minLength={2} />
                <button className="btn btn-primary flex-shrink-0" type="submit" disabled={searching} aria-label="Search">
                  {searching ? <span className="spinner-border spinner-border-sm" /> : <i className="fas fa-search" aria-hidden="true" />}
                </button>
              </form>
              {searchResults.length === 0 && !searching && (
                <div className="fr-empty">
                  <i className="fas fa-user-plus" aria-hidden="true" />
                  <p>{searchQuery.length >= 2 ? 'No one by that name.' : 'Type at least 2 characters.'}</p>
                </div>
              )}
              {searchResults.map(u => {
                const friend  = isFriend(u.username)
                const waiting = sentSet.has(u.username)
                return (
                  <div key={u.username} className="fr-row">
                    <button type="button" className="fr-row__hit" onClick={() => openChat(u)}
                            aria-label={`Open chat with ${u.display_name}`}>
                      <Avatar display={u.display_name} avatar={u.avatar_url} size={44} />
                      <span className="fr-row__body">
                        <span className="fr-row__name">{u.display_name}</span>
                        <span className="fr-sub">{u.username}</span>
                      </span>
                    </button>
                    <span className="fr-row__acts">
                      {friend
                        ? <span className="fr-chip fr-chip--good"><i className="fas fa-check" aria-hidden="true" />Friends</span>
                        : waiting
                          ? <span className="fr-chip fr-chip--warn"><i className="fas fa-clock" aria-hidden="true" />Request sent</span>
                          : <button type="button" className="fr-chip fr-chip--info" onClick={() => sendRequest(u.username)}>
                              <i className="fas fa-user-plus" aria-hidden="true" />Add friend
                            </button>}
                      <button type="button" className="btn btn-sm btn-outline-secondary" onClick={() => openChat(u)}
                              aria-label={`Message ${u.display_name}`}>
                        <i className="fas fa-comment-dots" aria-hidden="true" />
                      </button>
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* ── Right pane ────────────────────────────────────────────────────── */}
        <div className="fr-pane">
          {!activeChat ? (
            <div className="fr-empty fr-empty--pane">
              <i className="fas fa-comment-dots" aria-hidden="true" />
              <p>Pick a chat on the left.</p>
              <p className="fr-sub">选一个对话</p>
            </div>
          ) : (
            <>
              <div className="fr-ph">
                <button type="button" className="btn btn-sm fr-back" onClick={closeChat} aria-label="Back to chats">
                  <i className="fas fa-arrow-left" aria-hidden="true" />
                </button>
                <Avatar display={activeChat.display_name} avatar={activeChat.avatar_url} size={36} online={onlineSet.has(activeChat.username)} />
                <div className="fr-ph__who">
                  <b>{activeChat.display_name}</b>
                  <span className="fr-sub">
                    {activeChat.username} · {onlineSet.has(activeChat.username) ? 'online' : 'offline'}
                    {!isFriend(activeChat.username) && ' · not a friend'}
                  </span>
                </div>
                {isFriend(activeChat.username) && (
                  <ContactChip status={contactStatusMap[activeChat.username]}
                               onView={() => showContact(activeChat)} onRequest={() => requestContact(activeChat.username)} />
                )}
                <RowMenu label={`More for ${activeChat.display_name}`} items={personMenu(activeChat, { inChat: true })} />
              </div>

              {!isFriend(activeChat.username) && (
                <div className="fr-banner" role="status">
                  <i className="fas fa-user-plus" aria-hidden="true" />
                  <span>You're not friends yet. Messages still get through.
                    <span className="label-zh">还不是好友，消息照样能发</span></span>
                  {sentSet.has(activeChat.username)
                    ? <span className="fr-chip fr-chip--warn ms-auto"><i className="fas fa-clock" aria-hidden="true" />Request sent</span>
                    : <button type="button" className="btn btn-sm fr-banner__btn ms-auto" onClick={() => sendRequest(activeChat.username)}>Add friend</button>}
                </div>
              )}

              {/* Someone asked to see your contact details and is waiting. It used
                  to live only in the Requests tab, so you could be mid-conversation
                  with them and never know. Answerable right here. */}
              {(() => {
                const req = contactRequestFrom(activeChat.username)
                if (!req) return null
                return (
                  <div className="fr-banner" role="status">
                    <i className="fas fa-address-card" aria-hidden="true" />
                    <span><strong>{activeChat.display_name}</strong> asked to see your contact details
                      <span className="label-zh">想看你的联系方式</span></span>
                    <span className="d-flex gap-2 ms-auto flex-shrink-0">
                      <button type="button" className="btn btn-sm btn-primary"
                              onClick={() => respondContact(req.id, 'approve', req.from_user)}>Share</button>
                      <button type="button" className="btn btn-sm fr-banner__btn"
                              onClick={() => respondContact(req.id, 'decline', req.from_user)}>Not now</button>
                    </span>
                  </div>
                )
              })()}

              <div className="fr-msgs" ref={msgsRef}>
                {chatHistory.length === 0 && (
                  <div className="fr-empty my-auto"><p>No messages yet. Say hello!</p></div>
                )}
                {(() => {
                  let lastDate = null
                  return chatHistory.flatMap(m => {
                    const isMe = m.sender === user.username
                    const dateLabel = cstDateLabel(m.created_at)
                    const items = []
                    if (dateLabel !== lastDate) {
                      lastDate = dateLabel
                      items.push(<div key={`sep-${m.id}`} className="fr-day">{dateLabel}</div>)
                    }
                    const isCard = !!parseJoinUrl(m.content)
                    items.push(
                      <div key={m.id} className={`fr-msg${isMe ? ' is-me' : ''}`}
                           style={isCard ? { padding: '6px 8px' } : undefined}>
                        {renderMessageContent(m.content, isMe)}
                        <small>{new Date(m.created_at).toLocaleTimeString('zh-CN', { ...CT, hour: '2-digit', minute: '2-digit' })}</small>
                      </div>
                    )
                    return items
                  })
                })()}
              </div>

              <div className="fr-compose">
                <input
                  ref={inputRef}
                  className="fr-compose__in"
                  placeholder="Type a message…"
                  aria-label={`Message ${activeChat.display_name}`}
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
                  maxLength={1000}
                />
                <button type="button" className="btn btn-primary" onClick={sendMessage} disabled={!chatInput.trim()}>
                  <i className="fas fa-paper-plane" aria-hidden="true" /><span className="d-none d-sm-inline ms-1">Send</span>
                </button>
              </div>
              <div className="fr-compose__hint">Enter to send</div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
