import React, { useState, useCallback } from 'react'

const SIZE = 15  // 15×15 board

function createBoard() {
  return Array(SIZE).fill(null).map(() => Array(SIZE).fill(null))
}

function checkWin(board, row, col, player) {
  const DIRS = [[1,0],[0,1],[1,1],[1,-1]]
  for (const [dr, dc] of DIRS) {
    let count = 1
    for (let s of [1, -1]) {
      let r = row + dr*s, c = col + dc*s
      while (r >= 0 && r < SIZE && c >= 0 && c < SIZE && board[r][c] === player) {
        count++; r += dr*s; c += dc*s
      }
    }
    if (count >= 5) return true
  }
  return false
}

function getWinningCells(board, row, col, player) {
  const DIRS = [[1,0],[0,1],[1,1],[1,-1]]
  for (const [dr, dc] of DIRS) {
    const cells = [[row, col]]
    for (let s of [1, -1]) {
      let r = row + dr*s, c = col + dc*s
      while (r >= 0 && r < SIZE && c >= 0 && c < SIZE && board[r][c] === player) {
        cells.push([r, c]); r += dr*s; c += dc*s
      }
    }
    if (cells.length >= 5) return cells
  }
  return []
}

export default function Gomoku() {
  const [board, setBoard]       = useState(createBoard)
  const [current, setCurrent]   = useState('black')
  const [winner, setWinner]     = useState(null)
  const [winCells, setWinCells] = useState([])
  const [history, setHistory]   = useState([])  // move count

  const isWinCell = (r, c) => winCells.some(([wr, wc]) => wr === r && wc === c)

  const handleClick = useCallback((row, col) => {
    if (winner || board[row][col]) return
    const next = board.map(r => [...r])
    next[row][col] = current
    const won = checkWin(next, row, col, current)
    setBoard(next)
    setHistory(h => [...h, { row, col, player: current }])
    if (won) {
      setWinner(current)
      setWinCells(getWinningCells(next, row, col, current))
    } else {
      setCurrent(c => c === 'black' ? 'white' : 'black')
    }
  }, [board, current, winner])

  const reset = () => {
    setBoard(createBoard())
    setCurrent('black')
    setWinner(null)
    setWinCells([])
    setHistory([])
  }

  const undo = () => {
    if (!history.length || winner) return
    const last = history[history.length - 1]
    const next = board.map(r => [...r])
    next[last.row][last.col] = null
    setBoard(next)
    setHistory(h => h.slice(0, -1))
    setCurrent(last.player)
  }

  return (
    <div className="d-flex flex-column align-items-center">
      <h4 className="fw-bold mb-1">Gomoku</h4>
      <p className="text-muted mb-3 text-center">Five in a Row — local 2-player</p>

      {/* Status bar */}
      {winner ? (
        <div className={`alert alert-success d-flex align-items-center gap-3 mb-3 px-4`}>
          <div
            className="gomoku-stone rounded-circle flex-shrink-0"
            style={{ width: 24, height: 24, background: winner === 'black' ? '#000' : '#fff', border: '2px solid #ccc' }}
          />
          <strong>{winner.charAt(0).toUpperCase() + winner.slice(1)} wins! 🎉</strong>
          <button className="btn btn-success btn-sm ms-2" onClick={reset}>Play again</button>
        </div>
      ) : (
        <div className="d-flex align-items-center gap-3 mb-3">
          <span className="text-muted small">Current turn:</span>
          <div className="d-flex align-items-center gap-2">
            <div
              className="rounded-circle"
              style={{
                width: 24, height: 24,
                background: current === 'black'
                  ? 'radial-gradient(circle at 35% 35%, #666, #000)'
                  : 'radial-gradient(circle at 35% 35%, #fff, #ccc)',
                border: '2px solid #999',
                boxShadow: '1px 1px 3px rgba(0,0,0,.3)'
              }}
            />
            <strong>{current.charAt(0).toUpperCase() + current.slice(1)}</strong>
          </div>
          <span className="text-muted small">Move #{history.length + 1}</span>
        </div>
      )}

      {/* Board */}
      <div
        className="gomoku-board mb-3"
        style={{ gridTemplateColumns: `repeat(${SIZE}, 36px)` }}
      >
        {board.map((rowArr, r) =>
          rowArr.map((cell, c) => (
            <div
              key={`${r}-${c}`}
              className="gomoku-cell"
              onClick={() => handleClick(r, c)}
            >
              {cell && (
                <div className={`gomoku-stone ${cell} ${isWinCell(r, c) ? 'win' : ''}`} />
              )}
            </div>
          ))
        )}
      </div>

      {/* Controls */}
      <div className="d-flex gap-2">
        <button className="btn btn-outline-secondary btn-sm" onClick={undo} disabled={!history.length || !!winner}>
          <i className="fas fa-undo me-1" />Undo
        </button>
        <button className="btn btn-outline-danger btn-sm" onClick={reset}>
          <i className="fas fa-redo me-1" />Reset
        </button>
      </div>

      {/* Move log */}
      {history.length > 0 && (
        <div className="mt-3 text-muted small text-center">
          Last move: {history[history.length-1].player} at ({history[history.length-1].row + 1}, {history[history.length-1].col + 1})
        </div>
      )}
    </div>
  )
}
