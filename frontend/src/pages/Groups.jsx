import React, { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api'
import './Groups.css'

// ── Small shared bits ────────────────────────────────────────────────────────
function Avatar({ display, avatar, size = 38 }) {
  if (avatar) return (
    <img src={avatar} alt={display}
      style={{ width: size, height: size, borderRadius: '50%', objectFit: 'cover', flexShrink: 0 }} />
  )
  return (
    <div style={{
      width: size, height: size, borderRadius: '50%', background: '#6b9cdb', color: '#fff',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontWeight: 700, fontSize: size * 0.4, flexShrink: 0,
    }}>{display?.[0]?.toUpperCase() || '?'}</div>
  )
}

function timeLabel(isoStr) {
  const d = new Date(isoStr)
  return d.toLocaleTimeString('zh-CN', { timeZone: 'America/Chicago', hour: '2-digit', minute: '2-digit' })
}

function dateLabel(isoStr) {
  const d = new Date(isoStr)
  const opts = { timeZone: 'America/Chicago' }
  const now = new Date()
  if (d.toLocaleDateString('zh-CN', opts) === now.toLocaleDateString('zh-CN', opts)) return '今天'
  const yest = new Date(); yest.setDate(yest.getDate() - 1)
  if (d.toLocaleDateString('zh-CN', opts) === yest.toLocaleDateString('zh-CN', opts)) return '昨天'
  return d.toLocaleDateString('zh-CN', { timeZone: 'America/Chicago', month: 'long', day: 'numeric' })
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function Groups() {
  const [groups, setGroups] = useState([])
  const [active, setActive] = useState(null)          // group detail object
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(true)

  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [showAdd, setShowAdd] = useState(false)
  const [searchQ, setSearchQ] = useState('')
  const [searchRes, setSearchRes] = useState([])
  const [showRename, setShowRename] = useState(false)
  const [renName, setRenName] = useState('')

  const [draft, setDraft] = useState('')
  const [, forceTick] = useState(0)
  const msgEndRef = useRef(null)
  const me = useRef(null)
  try { me.current = me.current || JSON.parse(localStorage.getItem('horisation_user') || 'null')?.username } catch {}

  const loadGroups = useCallback(async () => {
    const r = await api.get('/api/groups')
    if (r.ok) setGroups(r.groups || [])
  }, [])

  const loadDetail = useCallback(async (gid) => {
    const r = await api.get(`/api/groups/${gid}`)
    if (r.ok) { setActive(r.group); return r.group }
    return null
  }, [])

  const loadMessages = useCallback(async (gid) => {
    const r = await api.get(`/api/groups/${gid}/messages`)
    if (r.ok) setMessages(r.messages || [])
  }, [])

  // init
  useEffect(() => {
    (async () => {
      await loadGroups()
      setLoading(false)
    })()
  }, [loadGroups])

  // open a group
  const openGroup = async (g) => {
    const detail = await loadDetail(g.id)
    if (detail) { setActive(detail); await loadMessages(detail.id) }
  }

  // poll messages while a group is open
  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => loadMessages(active.id), 3000)
    return () => clearInterval(timer)
  }, [active, loadMessages])

  useEffect(() => { msgEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, active])

  const createGroup = async () => {
    const r = await api.post('/api/groups', { name: newName.trim() })
    if (!r.ok) { alert(r.error || '创建失败'); return }
    setShowCreate(false); setNewName('')
    await loadGroups()
    await openGroup(r.group)
  }

  const sendMsg = async () => {
    const content = draft.trim()
    if (!content || !active) return
    const r = await api.post(`/api/groups/${active.id}/messages`, { content })
    if (r.ok) { setDraft(''); await loadMessages(active.id) }
    else alert(r.error)
  }

  const searchUsers = async (q) => {
    const r = await api.get(`/api/friends/users?q=${encodeURIComponent(q)}`)
    setSearchRes(r.users || [])
  }

  const addMember = async (username) => {
    const r = await api.post(`/api/groups/${active.id}/members`, { username })
    if (r.ok) { setShowAdd(false); setSearchQ(''); setSearchRes([]); await loadDetail(active.id) }
    else alert(r.error)
  }

  const renameGroup = async () => {
    const r = await api.put(`/api/groups/${active.id}`, { name: renName.trim() })
    if (r.ok) { setShowRename(false); await loadDetail(active.id); await loadGroups() }
    else alert(r.error)
  }

  const removeMember = async (username) => {
    const r = await api.delete(`/api/groups/${active.id}/members/${username}`)
    if (r.ok) { await loadDetail(active.id); await loadGroups() }
    else alert(r.error)
  }

  const deleteGroup = async () => {
    if (!confirm(`解散「${active.name}」？群聊记录会一并删除。`)) return
    const r = await api.delete(`/api/groups/${active.id}`)
    if (r.ok) { setActive(null); setMessages([]); await loadGroups() }
    else alert(r.error)
  }

  const isOwner = active && me.current && active.owner === me.current

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="groups-page">
      {/* ── Left: group list ── */}
      <div className="groups-list">
        <div className="groups-list-header">
          <h2>群组</h2>
          <button className="btn btn-primary btn-sm" onClick={() => setShowCreate(true)}>
            <i className="fas fa-plus" /> 建群
          </button>
        </div>
        {loading ? <div className="groups-empty">加载中…</div> : groups.length === 0 ? (
          <div className="groups-empty">还没有群，建一个开始吧</div>
        ) : (
          groups.map(g => (
            <div key={g.id}
              className={`group-item${active?.id === g.id ? ' active' : ''}`}
              onClick={() => openGroup(g)}>
              <Avatar display={g.name} size={38} />
              <div className="group-item-info">
                <div className="group-item-name">{g.name}</div>
                <div className="group-item-meta">{g.member_count} 位成员</div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* ── Right: chat ── */}
      <div className="groups-chat">
        {!active ? (
          <div className="groups-empty groups-chat-empty">选择一个群开始聊天</div>
        ) : (
          <>
            <div className="groups-chat-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <Avatar display={active.name} size={36} />
                <div>
                  <div className="groups-chat-title">{active.name}</div>
                  <div className="groups-chat-meta">群主 {active.owner} · {active.member_count} 人</div>
                </div>
              </div>
              <div className="groups-chat-actions d-flex gap-2">
                <button className="btn btn-sm btn-outline-secondary" onClick={() => setShowAdd(true)}>
                  <i className="fas fa-user-plus" /> 拉人
                </button>
                {isOwner && (
                  <>
                    <button className="btn btn-sm btn-outline-secondary" onClick={() => { setRenName(active.name); setShowRename(true) }}>
                      <i className="fas fa-pen" /> 改名
                    </button>
                    <button className="btn btn-sm btn-outline-danger" onClick={deleteGroup} title="解散群">
                      <i className="fas fa-trash" />
                    </button>
                  </>
                )}
                {!isOwner && (
                  <button className="btn btn-sm btn-outline-danger" onClick={() => removeMember(me.current)}>
                    <i className="fas fa-sign-out-alt" /> 退出
                  </button>
                )}
              </div>
            </div>

            <div className="groups-chat-members">
              {active.members?.map(m => (
                <span key={m.username} className="group-member-chip">
                  <Avatar display={m.display_name || m.username} avatar={m.avatar_url} size={20} />
                  {m.username}{m.role === 'owner' ? ' 👑' : ''}
                  {isOwner && m.role !== 'owner' && (
                    <button className="group-kick" title="移出群聊" onClick={() => removeMember(m.username)}>×</button>
                  )}
                </span>
              ))}
            </div>

            <div className="groups-msgs">
              {messages.length === 0 ? (
                <div className="groups-empty">还没有消息</div>
              ) : messages.map((m, i) => {
                const prev = messages[i - 1]
                const isMe = me.current && m.sender === me.current
                const showDate = !prev || dateLabel(prev.created_at) !== dateLabel(m.created_at)
                const showHeader = !prev || prev.sender !== m.sender
                return (
                  <div key={m.id}>
                    {showDate && <div className="group-date-divider">{dateLabel(m.created_at)}</div>}
                    <div className={`group-msg${isMe ? ' mine' : ''}`}>
                      {!isMe && showHeader && (
                        <div className="group-msg-avatar"><Avatar display={m.sender_display || m.sender} avatar={m.sender_avatar} size={34} /></div>
                      )}
                      <div className="group-msg-body">
                        {!isMe && showHeader && <div className="group-msg-sender">{m.sender_display || m.sender}</div>}
                        <div className="group-msg-bubble">{m.content}</div>
                      </div>
                      {isMe && <div className="group-msg-time me-time">{timeLabel(m.created_at)}</div>}
                      {!isMe && <div className="group-msg-time"><button className="btn-link-time" onClick={() => setDraft(`@${m.sender} `)}>回复</button></div>}
                    </div>
                  </div>
                )
              })}
              <div ref={msgEndRef} />
            </div>

            <div className="groups-input">
              <input
                value={draft}
                onChange={e => setDraft(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg() } }}
                placeholder="输入消息…" maxLength={1000} />
              <button className="btn btn-primary" onClick={sendMsg}><i className="fas fa-paper-plane" /></button>
            </div>
          </>
        )}
      </div>

      {/* ── Create modal ── */}
      {showCreate && (
        <div className="groups-modal-backdrop" onClick={() => setShowCreate(false)}>
          <div className="groups-modal" onClick={e => e.stopPropagation()}>
            <h3>创建群组</h3>
            <input autoFocus value={newName} maxLength={50} onChange={e => setNewName(e.target.value)}
              placeholder="群组名称（最多50字）" className="form-control mb-2" />
            <div className="d-flex justify-content-end gap-2">
              <button className="btn btn-secondary" onClick={() => setShowCreate(false)}>取消</button>
              <button className="btn btn-primary" disabled={!newName.trim()} onClick={createGroup}>创建</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Add member modal ── */}
      {showAdd && (
        <div className="groups-modal-backdrop" onClick={() => setShowAdd(false)}>
          <div className="groups-modal" onClick={e => e.stopPropagation()}>
            <h3>拉人进群</h3>
            <input autoFocus value={searchQ} onChange={e => { setSearchQ(e.target.value); if (e.target.value.trim().length >= 2) searchUsers(e.target.value.trim()) }}
              placeholder="搜索用户名（至少2个字）" className="form-control mb-2" />
            <div className="groups-search-res">
              {searchQ.trim().length < 2 ? <div className="groups-empty">输入用户名搜索</div> :
                searchRes.length === 0 ? <div className="groups-empty">没有找到用户</div> :
                searchRes.map(u => (
                  <div key={u.username} className="group-search-item">
                    <Avatar display={u.display_name} avatar={u.avatar_url} size={32} />
                    <div className="group-search-info">
                      <div>{u.display_name}</div>
                      <div className="group-search-username">@{u.username}</div>
                    </div>
                    <button className="btn btn-sm btn-primary" onClick={() => addMember(u.username)}>拉入</button>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Rename modal ── */}
      {showRename && (
        <div className="groups-modal-backdrop" onClick={() => setShowRename(false)}>
          <div className="groups-modal" onClick={e => e.stopPropagation()}>
            <h3>修改群名</h3>
            <input autoFocus value={renName} maxLength={50} onChange={e => setRenName(e.target.value)}
              className="form-control mb-2" />
            <div className="d-flex justify-content-end gap-2">
              <button className="btn btn-secondary" onClick={() => setShowRename(false)}>取消</button>
              <button className="btn btn-primary" disabled={!renName.trim()} onClick={renameGroup}>保存</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
