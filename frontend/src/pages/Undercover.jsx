import React, { useState, useEffect, useCallback } from 'react'
import { api } from '../api'
import { useAuth } from '../App'

/* Who is the Undercover — v1: a room, a deal, a reveal.
   Everyone joins, the host starts, each person flips their own card.
   Voting and elimination are deliberately not here yet. */

export default function Undercover() {
  const { user } = useAuth()
  const [room, setRoom]       = useState(null)
  const [code, setCode]       = useState('')
  const [loading, setLoading] = useState(false)
  const [toast, setToast]     = useState(null)
  const [flipped, setFlipped] = useState(false)

  const flash = (text) => setToast(text)

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 2600)
    return () => clearTimeout(t)
  }, [toast])

  const load = useCallback(async (roomCode) => {
    const d = await api.get(`/api/undercover/rooms/${roomCode}`)
    if (d.ok) setRoom(d.room)
    else setRoom(null)
  }, [])

  // Poll while in a room; the roster and the phase change on other people's
  // actions, and v1 has no socket channel for this.
  useEffect(() => {
    if (!room?.code) return
    const t = setInterval(() => load(room.code), 3000)
    return () => clearInterval(t)
  }, [room?.code, load])

  const create = async () => {
    setLoading(true)
    const d = await api.post('/api/undercover/rooms', {})
    setLoading(false)
    if (!d.ok) return flash(d.error || '创建失败')
    setRoom(d.room); setFlipped(false)
  }

  const join = async () => {
    const c = code.trim().toUpperCase()
    if (!c) return flash('请输入房间码')
    setLoading(true)
    const d = await api.post(`/api/undercover/rooms/${c}/join`, {})
    setLoading(false)
    if (!d.ok) return flash(d.error || '加入失败')
    setRoom(d.room); setFlipped(false)
  }

  const start = async () => {
    const d = await api.post(`/api/undercover/rooms/${room.code}/start`, {})
    if (!d.ok) return flash(d.error || '开始失败')
    setRoom(d.room); setFlipped(false)
  }

  const reveal = async () => {
    const d = await api.post(`/api/undercover/rooms/${room.code}/reveal`, {})
    if (!d.ok) return flash(d.error || '揭晓失败')
    setRoom(d.room)
  }

  const leave = async () => {
    await api.post(`/api/undercover/rooms/${room.code}/leave`, {})
    setRoom(null); setCode(''); setFlipped(false)
  }

  if (!room) {
    return (
      <div className="uc-page">
        <header className="uc-head">
          <div className="uc-kicker">WHO IS THE UNDERCOVER</div>
          <h1 className="uc-title">谁是卧底</h1>
        </header>
        <div className="uc-lobby">
          <div className="uc-lobby__card">
            <h3>创建房间</h3>
            <p>拉朋友进来，3 人起。</p>
            <button className="uc-btn uc-btn--primary" onClick={create} disabled={loading}>
              创建房间
            </button>
          </div>
          <div className="uc-lobby__card">
            <h3>加入房间</h3>
            <p>输入房主给你的 6 位房间码。</p>
            <input className="uc-input" value={code} maxLength={8}
                   onChange={e => setCode(e.target.value.toUpperCase())}
                   placeholder="例如 K7M2QX" />
            <button className="uc-btn" onClick={join} disabled={loading || !code.trim()}>
              加入
            </button>
          </div>
        </div>
        {toast && <div className="uc-toast">{toast}</div>}
      </div>
    )
  }

  const isHost = room.host === user?.username
  const enough = room.players.length >= room.min_players
  const mine = room.me

  return (
    <div className="uc-page">
      <header className="uc-head">
        <button className="uc-back" onClick={leave}>← 离开房间</button>
        <div className="uc-kicker">WHO IS THE UNDERCOVER</div>
        <h1 className="uc-title">谁是卧底</h1>
        <div className="uc-code">
          房间码 <strong>{room.code}</strong>
          <button className="uc-mini" onClick={() => navigator.clipboard?.writeText(room.code).then(() => flash('已复制'))}>
            复制
          </button>
        </div>
      </header>

      {room.status === 'waiting' && (
        <>
          <div className="uc-players">
            {room.players.map(p => (
              <div className="uc-player" key={p.username}>
                <span className="uc-player__dot" />
                <span>{p.display_name}{p.username === user?.username ? '（你）' : ''}</span>
                {p.is_host && <span className="uc-tag">房主</span>}
              </div>
            ))}
          </div>
          <div className="uc-foot">
            {isHost
              ? <button className="uc-btn uc-btn--primary" onClick={start} disabled={!enough}>
                  {enough ? '开始游戏' : `还差 ${room.min_players - room.players.length} 人`}
                </button>
              : <p className="uc-wait">等待房主开始……</p>}
          </div>
        </>
      )}

      {room.status !== 'waiting' && (
        <>
          <div className={'uc-card' + (flipped ? ' is-flipped' : '')}>
            <button className="uc-card__inner" onClick={() => setFlipped(true)} disabled={flipped}>
              <span className="uc-card__back">
                <span className="uc-card__mark">词</span>
                <span>点击查看你的词</span>
              </span>
              <span className="uc-card__front">
                <span className="uc-card__label">你的词</span>
                <span className="uc-card__word">{mine?.word || '—'}</span>
                <span className="uc-card__hint">别让别人看到</span>
              </span>
            </button>
          </div>

          <div className="uc-players uc-players--compact">
            {room.players.map(p => (
              <span className="uc-chip" key={p.username}>
                {p.display_name}{p.username === user?.username ? '（你）' : ''}
              </span>
            ))}
          </div>

          {room.status === 'playing' && isHost && (
            <div className="uc-foot">
              <button className="uc-btn" onClick={reveal}>揭晓答案</button>
            </div>
          )}

          {room.status === 'revealed' && room.reveal && (
            <div className="uc-reveal">
              <div><span>平民词</span><strong>{room.reveal.civilian_word}</strong></div>
              <div><span>卧底词</span><strong>{room.reveal.undercover_word}</strong></div>
              <div><span>卧底</span><strong>{room.reveal.undercover}</strong></div>
            </div>
          )}
        </>
      )}

      {toast && <div className="uc-toast">{toast}</div>}
    </div>
  )
}
