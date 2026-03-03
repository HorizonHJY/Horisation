import React, { useState, useEffect, useRef } from 'react'
import { io } from 'socket.io-client'
import { useAuth } from '../../App'

const SIZE = 15

function Avatar({ display, avatar, size = 44, color }) {
  const bg = color === 'black' ? '#1e2a3a' : '#e8e8e8'
  const fg = color === 'black' ? '#fff' : '#333'
  if (avatar) {
    return (
      <img
        src={avatar}
        alt={display}
        style={{ width: size, height: size, borderRadius: '50%', objectFit: 'cover', flexShrink: 0 }}
      />
    )
  }
  return (
    <div style={{
      width: size, height: size, borderRadius: '50%',
      background: bg, color: fg,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontWeight: 700, fontSize: size * 0.4, flexShrink: 0,
    }}>
      {display?.[0]?.toUpperCase() || '?'}
    </div>
  )
}

function StoneIndicator({ color }) {
  return (
    <div style={{
      width: 18, height: 18, borderRadius: '50%', flexShrink: 0,
      background: color === 'black'
        ? 'radial-gradient(circle at 35% 35%, #666, #000)'
        : 'radial-gradient(circle at 35% 35%, #fff, #ccc)',
      boxShadow: '1px 1px 3px rgba(0,0,0,.4)',
      border: color === 'white' ? '1px solid #bbb' : 'none',
    }} />
  )
}

export default function OnlineGomoku() {
  const { user } = useAuth()
  const socketRef = useRef(null)

  const [view, setView]         = useState('lobby')
  const [rooms, setRooms]       = useState([])
  const [room, setRoom]         = useState(null)
  const [creating, setCreating] = useState(false)
  const [roomName, setRoomName] = useState('')
  const [toast, setToast]       = useState(null)

  const flash = (msg, type = 'danger') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3500)
  }

  // ── Socket lifecycle ──────────────────────────────────────────────
  useEffect(() => {
    const socket = io({ withCredentials: true })
    socketRef.current = socket

    socket.on('connect', () => {
      socket.emit('game_join_lobby')
    })
    socket.on('rooms_updated', ({ rooms }) => setRooms(rooms))
    socket.on('game_state',    (r) => { setRoom(r); setView('game') })
    socket.on('game_dissolved', () => {
      setRoom(null)
      setView('lobby')
      flash('The room was closed by the host.', 'warning')
    })
    socket.on('game_error', ({ message }) => flash(message))

    return () => {
      socket.emit('game_leave_lobby')
      socket.disconnect()
    }
  }, [])

  useEffect(() => {
    const onUnload = () => {
      if (room) socketRef.current?.emit('game_leave', { room_id: room.id })
    }
    window.addEventListener('beforeunload', onUnload)
    return () => window.removeEventListener('beforeunload', onUnload)
  }, [room])

  // ── Derived ───────────────────────────────────────────────────────
  const myRole   = !room ? null
    : user.username === room.host    ? 'host'
    : user.username === room.player2 ? 'player2'
    : 'spectator'
  const isMyTurn = room?.status === 'playing' && room?.current_turn === user.username

  // ── Actions ───────────────────────────────────────────────────────
  const createRoom = () => {
    const name = roomName.trim() || `${user.display_name}'s Game`
    socketRef.current.emit('game_create', { name })
    setCreating(false)
    setRoomName('')
  }
  const joinRoom  = (room_id, role) => socketRef.current.emit('game_join', { room_id, role })
  const makeMove  = (index) => {
    if (!isMyTurn || room.board[index] !== null) return
    socketRef.current.emit('game_move', { room_id: room.id, index })
  }
  const kick      = () => socketRef.current.emit('game_kick',    { room_id: room.id })
  const rematch   = () => socketRef.current.emit('game_rematch', { room_id: room.id })
  const leaveRoom = () => {
    socketRef.current.emit('game_leave', { room_id: room.id })
    setRoom(null)
    setView('lobby')
    socketRef.current.emit('game_join_lobby')
  }

  const Toast = toast && (
    <div className={`alert alert-${toast.type} position-fixed top-0 end-0 m-3`}
      style={{ zIndex: 9999, minWidth: 240 }}>
      {toast.msg}
    </div>
  )

  // ══════════════════════════════════════════════════════════════════
  // LOBBY
  // ══════════════════════════════════════════════════════════════════
  if (view === 'lobby') return (
    <div className="container-fluid py-4" style={{ maxWidth: 700 }}>
      {Toast}

      <div className="d-flex align-items-center justify-content-between mb-4">
        <div className="d-flex align-items-center gap-2">
          <i className="fas fa-chess-board fa-lg text-primary" />
          <h4 className="mb-0 fw-bold">Online Gomoku</h4>
        </div>
        <button className="btn btn-primary btn-sm" onClick={() => setCreating(true)}>
          <i className="fas fa-plus me-1" />Create Room
        </button>
      </div>

      {creating && (
        <div className="card p-3 mb-3 shadow-sm">
          <div className="d-flex gap-2">
            <input
              className="form-control form-control-sm"
              placeholder={`${user.display_name}'s Game`}
              value={roomName}
              maxLength={50}
              autoFocus
              onChange={e => setRoomName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && createRoom()}
            />
            <button className="btn btn-primary btn-sm px-3" onClick={createRoom}>Create</button>
            <button className="btn btn-outline-secondary btn-sm" onClick={() => setCreating(false)}>Cancel</button>
          </div>
        </div>
      )}

      {rooms.length === 0 ? (
        <div className="text-center text-muted py-5">
          <i className="fas fa-chess-board fa-3x mb-3 d-block" />
          <p>No rooms yet — create one to start!</p>
        </div>
      ) : (
        <div className="d-flex flex-column gap-2">
          {rooms.map(r => (
            <div key={r.id} className="card shadow-sm px-4 py-3">
              <div className="d-flex align-items-center justify-content-between flex-wrap gap-2">
                <div className="d-flex align-items-center gap-3">
                  <Avatar display={r.host_display} avatar={r.host_avatar} size={38} color="black" />
                  <div>
                    <span className="fw-semibold">{r.name}</span>
                    <div className="text-muted small">
                      {r.host_display}
                      {r.player2_display ? ` vs ${r.player2_display}` : ' — waiting for opponent'}
                    </div>
                  </div>
                </div>
                <div className="d-flex align-items-center gap-2">
                  <span className={`badge ${r.status === 'waiting' ? 'bg-success' : 'bg-warning text-dark'}`}>
                    {r.status === 'waiting' ? 'Waiting' : 'Playing'}
                  </span>
                  {r.status === 'waiting' && r.host !== user.username && (
                    <button className="btn btn-primary btn-sm" onClick={() => joinRoom(r.id, 'player')}>
                      Join
                    </button>
                  )}
                  {r.host === user.username && (
                    <button className="btn btn-outline-primary btn-sm" onClick={() => joinRoom(r.id, 'spectator')}>
                      Re-enter
                    </button>
                  )}
                  <button className="btn btn-outline-secondary btn-sm" onClick={() => joinRoom(r.id, 'spectator')}>
                    <i className="fas fa-eye me-1" />Watch
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )

  // ══════════════════════════════════════════════════════════════════
  // GAME VIEW
  // ══════════════════════════════════════════════════════════════════
  const board    = room?.board || Array(225).fill(null)
  const winSet   = new Set(room?.win_cells || [])
  const finished = room?.status === 'finished'

  const winnerDisplay = room?.winner === room?.host ? room?.host_display : room?.player2_display

  const players = [
    { key: 'host',    username: room?.host,    display: room?.host_display,    avatar: room?.host_avatar,    color: 'black' },
    { key: 'player2', username: room?.player2, display: room?.player2_display, avatar: room?.player2_avatar, color: 'white' },
  ]

  return (
    <div className="container-fluid py-3">
      {Toast}

      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <button className="btn btn-outline-secondary btn-sm" onClick={leaveRoom}>
          <i className="fas fa-arrow-left me-1" />
          {myRole === 'spectator' ? 'Stop watching' : 'Leave room'}
        </button>
        <span className="fw-semibold">{room?.name}</span>
        {myRole === 'spectator' && (
          <span className="badge bg-secondary"><i className="fas fa-eye me-1" />Spectating</span>
        )}
      </div>

      <div className="d-flex flex-wrap gap-4 align-items-start justify-content-center">

        {/* Board */}
        <div
          style={{
            display: 'inline-grid',
            gridTemplateColumns: `repeat(${SIZE}, 36px)`,
            background: '#d4a256',
            border: '3px solid #8b6914',
            borderRadius: 4,
            padding: 12,
            cursor: isMyTurn ? 'pointer' : 'default',
            userSelect: 'none',
          }}
        >
          {board.map((cell, i) => (
            <div key={i} className="gomoku-cell" onClick={() => makeMove(i)}>
              {cell && (
                <div
                  className={`gomoku-stone ${cell}`}
                  style={winSet.has(i) ? { boxShadow: '0 0 0 3px #f59e0b, 2px 2px 4px rgba(0,0,0,.4)' } : undefined}
                />
              )}
            </div>
          ))}
        </div>

        {/* Info panel */}
        <div style={{ minWidth: 220, maxWidth: 260 }}>

          {/* Player cards */}
          <div className="card shadow-sm mb-3 overflow-hidden">
            {players.map(({ key, username, display, avatar, color }, idx) => {
              const isTurn    = room?.current_turn === username && room?.status === 'playing'
              const isMe      = username === user.username
              const isWinner  = finished && room?.winner === username
              return (
                <div key={key}>
                  {idx > 0 && <hr className="my-0" />}
                  <div
                    className={`d-flex align-items-center gap-2 px-3 py-2 ${isTurn ? 'bg-primary bg-opacity-10' : ''}`}
                  >
                    {username ? (
                      <>
                        <Avatar display={display} avatar={avatar} size={40} color={color} />
                        <div className="flex-grow-1 overflow-hidden">
                          <div className="d-flex align-items-center gap-1">
                            <StoneIndicator color={color} />
                            <span className={`text-truncate ${isMe ? 'fw-bold' : ''}`}>
                              {display}{isMe && ' (You)'}
                            </span>
                          </div>
                          {isTurn  && <small className="text-primary fw-semibold">Your turn</small>}
                          {isWinner && <small className="text-success fw-semibold">🎉 Winner!</small>}
                        </div>
                      </>
                    ) : (
                      <>
                        <div style={{
                          width: 40, height: 40, borderRadius: '50%',
                          background: '#f0f0f0', flexShrink: 0,
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                        }}>
                          <i className="fas fa-user text-muted" />
                        </div>
                        <div>
                          <StoneIndicator color={color} />
                          <span className="text-muted fst-italic small ms-1">Waiting…</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Status / controls */}
          {finished && (
            <div className="alert alert-success py-2 mb-3 text-center">
              <strong>
                {room?.winner === user.username ? '🎉 You win!' : `${winnerDisplay} wins!`}
              </strong>
            </div>
          )}

          {room?.status === 'waiting' && (
            <div className="text-muted small mb-3 text-center">
              <i className="fas fa-clock me-1" />Waiting for opponent to join…
            </div>
          )}

          <div className="d-flex flex-column gap-2">
            {myRole === 'host' && room?.player2 && !finished && (
              <button className="btn btn-outline-warning btn-sm" onClick={kick}>
                <i className="fas fa-user-minus me-1" />Kick Player
              </button>
            )}
            {myRole === 'host' && finished && (
              <button className="btn btn-primary btn-sm" onClick={rematch}>
                <i className="fas fa-redo me-1" />Rematch
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
