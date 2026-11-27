import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../App'

/* The access list on a bill or a trip: avatar, name, a link to the profile,
   and a one-tap friend request when the two are not connected yet. */
export default function MemberChips({ members, showToast, compact = false }) {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [friends, setFriends] = useState(null)
  const [busy, setBusy] = useState('')

  useEffect(() => {
    api.get('/api/friends/list').then(d => {
      if (d && d.ok) setFriends(new Set((d.friends || []).map(f => f.username)))
    })
  }, [])

  async function addFriend(username) {
    setBusy(username)
    const d = await api.post('/api/friends/requests', { to_user: username })
    setBusy('')
    if (d && d.ok) {
      setFriends(s => new Set([...(s || []), username]))
      showToast && showToast('好友请求已发送', 'success')
    } else {
      showToast && showToast((d && d.error) || '发送失败', 'danger')
    }
  }

  return (
    <div className="member-chips">
      {(members || []).map(m => {
        const isMe = m.username === user?.username
        const isFriend = friends ? friends.has(m.username) : false
        return (
          <span key={m.username} className="member-chip">
            <button type="button" className="member-chip__main"
              onClick={() => navigate(`/u/${encodeURIComponent(m.username)}`)}
              title="查看主页">
              <img className="member-chip__avatar" src={m.avatar_url || '/avatars/fox.svg'} alt="" />
              {!compact && (
                <span>{m.display_name || m.username}{m.is_owner ? ' · 创建者' : ''}</span>
              )}
            </button>
            {!m.is_owner && !isMe && friends && !isFriend && (
              <button type="button" className="member-chip__add" title="加好友"
                disabled={busy === m.username}
                onClick={() => addFriend(m.username)}>
                <i className="fas fa-user-plus" aria-hidden="true" />
              </button>
            )}
          </span>
        )
      })}
    </div>
  )
}
