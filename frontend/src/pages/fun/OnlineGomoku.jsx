import React, { useState, useEffect, useRef } from 'react'
import { io } from 'socket.io-client'
import { useAuth } from '../../App'

const SIZE = 15

const stoneStyle = (color, win = false) => ({
  width: 28, height: 28, borderRadius: '50%',
  background: color === 'black'
    ? 'radial-gradient(circle at 35% 35%, #666, #000)'
    : 'radial-gradient(circle at 35% 35%, #fff, #ccc)',
  boxShadow: win
    ? '0 0 0 3px #f59e0b, 2px 2px 4px rgba(0,0,0,.4)'
    : '2px 2px 4px rgba(0,0,0,.4)',
  flexShrink: 0,
})

export default function OnlineGomoku() {
  const { user } = useAuth()
  const socketRef = useRef(null)

  const [view, setView]           = useState('lobby')   // 'lobby' | 'game'
  const [rooms, setRooms]         = useState([])
  const [room, setRoom]           = useState(null)
  const [creating, setCreating]   = useState(false)
  const [roomName, setRoomName]   = useState('')
  const [toast, setToast]         = useState(null)

  const flash = (msg, type = 'danger') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  // ── Socket lifecycle ────────────────────────────────────────────
  useEffect(() => {
    const socket = io({ withCredentials: true })
    socketRef.current = socket

    socket.on('rooms_updated', ({ rooms }) => setRooms(rooms))
    socket.on('game_state',    (r) => { setRoom(r); setView('game') })
    socket.on('game_dissolved', () => {
      setRoom(null)
      setView('lobby')
      flash('The room was closed by the host.', 'warning')
    })
    socket.on('game_error', ({ message }) => flash(message))

    socket.emit('game_join_lobby')

    return () => {
      socket.emit('game_leave_lobby')
      socket.disconnect()
    }
  }, [])

  // Emit leave on page unload so the room cleans up
  useEffect(() => {
    const onUnload = () => {
      if (room) socketRef.current?.emit('game_leave', { room_id: room.id })
    }
    window.addEventListener('beforeunload', onUnload)
    return () => window.removeEventListener('beforeunload', onUnload)
  }, [room])

  // ── Derived state ───────────────────────────────────────────────
  const myRole = !room ? null
    : user.username === room.host    ? 'host'
    : user.username === room.player2 ? 'player2'
    : 'spectator'

  const isMyTurn = room?.status === 'playing' && room?.current_turn === user.username

  // ── Actions ─────────────────────────────────────────────────────
  function createRoom() {
    const name = roomName.trim() || `${user.display_name}'s Game`
    socketRef.current.emit('game_create', { name })
    setCreating(false)
    setRoomName('')
  }

  function joinRoom(room_id, role) {
    socketRef.current.emit('game_join', { room_id, role })
  }

  function makeMove(index) {
    if (!isMyTurn || room.board[index] !== null) return
    socketRef.current.emit('game_move', { room_id: room.id, index })
  }

  function kick() {
    socketRef.current.emit('game_kick', { room_id: room.id })
  }

  function rematch() {
    socketRef.current.emit('game_rematch', { room_id: room.id })
  }

  function leaveRoom() {
    socketRef.current.emit('game_leave', { room_id: room.id })
    setRoom(null)
    setView('lobby')
    socketRef.current.emit('game_join_lobby')
  }

  // ── Toast ────────────────────────────────────────────────────────
  const Toast = toast && (
    <div
      className={`alert alert-${toast.type} position-fixed top-0 end-0 m-3`}
      style={{ zIndex: 9999, minWidth: 220 }}
    >
      {toast.msg}
    </div>
  )

  // ════════════════════════════════════════════════════════════════
  // LOBBY
  // ════════════════════════════════════════════════════════════════
  if (view === 'lobby') {
    return (
      <div className="container-fluid py-4" style={{ maxWidth: 720 }}>
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
                  <div>
                    <span className="fw-semibold me-2">{r.name}</span>
                    <span className={`badge ${r.status === 'waiting' ? 'bg-success' : 'bg-warning text-dark'}`}>
                      {r.status === 'waiting' ? 'Waiting' : 'Playing'}
                    </span>
                  </div>
                  <div className="d-flex align-items-center gap-3">
                    <span className="text-muted small">
                      <i className="fas fa-user me-1" />
                      {r.host_display}
                      {r.player2_display && ` vs ${r.player2_display}`}
                    </span>
                    <div className="d-flex gap-1">
                      {r.status === 'waiting' && r.host !== user.username && (
                        <button className="btn btn-primary btn-sm" onClick={() => joinRoom(r.id, 'player')}>
                          Join
                        </button>
                      )}
                      {r.host === user.username && (
                        <button className="btn btn-primary btn-sm" onClick={() => joinRoom(r.id, 'spectator')}>
                          Re-enter
                        </button>
                      )}
                      <button className="btn btn-outline-secondary btn-sm" onClick={() => joinRoom(r.id, 'spectator')}>
                        <i className="fas fa-eye me-1" />Watch
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  // ════════════════════════════════════════════════════════════════
  // GAME VIEW
  // ════════════════════════════════════════════════════════════════
  const board    = room?.board    || Array(225).fill(null)
  const winSet   = new Set(room?.win_cells || [])
  const finished = room?.status === 'finished'

  const winnerDisplay = room?.winner === room?.host
    ? room?.host_display
    : room?.player2_display

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
        <span className={`badge ${
          room?.status === 'playing'  ? 'bg-primary' :
          room?.status === 'waiting'  ? 'bg-secondary' : 'bg-success'
        }`}>
          {room?.status}
        </span>
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
              {cell && <div className="gomoku-stone" style={stoneStyle(cell, winSet.has(i))} />}
            </div>
          ))}
        </div>

        {/* Info panel */}
        <div style={{ minWidth: 220 }}>

          {/* Players */}
          <div className="card p-3 mb-3 shadow-sm">
            {[
              { username: room?.host,    display: room?.host_display,    color: 'black' },
              { username: room?.player2, display: room?.player2_display, color: 'white' },
            ].map(({ username, display, color }) => (
              <div key={color} className={`d-flex align-items-center gap-2 ${color === 'black' ? 'mb-2' : ''}`}>
                <div style={stoneStyle(color)} />
                {username ? (
                  <>
                    <span className={username === user.username ? 'fw-bold' : ''}>
                      {display}{username === user.username && ' (You)'}
                    </span>
                    {room?.current_turn === username && room?.status === 'playing' && (
                      <span className="badge bg-primary ms-auto">Turn</span>
                    )}
                  </>
                ) : (
                  <span className="text-muted fst-italic small">Waiting for opponent…</span>
                )}
              </div>
            ))}
          </div>

          {/* Game status messages */}
          {finished && (
            <div className="alert alert-success py-2 mb-3">
              <strong>
                {room?.winner === user.username
                  ? '🎉 You win!'
                  : `${winnerDisplay} wins!`}
              </strong>
            </div>
          )}

          {room?.status === 'waiting' && myRole !== 'host' && (
            <div className="alert alert-secondary py-2 mb-3 small">
              Waiting for the host to start…
            </div>
          )}

          {myRole === 'spectator' && (
            <div className="text-muted small mb-3">
              <i className="fas fa-eye me-1" />Spectating
            </div>
          )}

          {/* Controls */}
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
