"""
game_controller.py
Online Gomoku — room management via REST + real-time moves via Socket.IO.
"""

from flask import Blueprint, session
from flask_socketio import emit, join_room, leave_room

from Backend.Controller.socketio_instance import socketio
from Backend.Controller.market_db import (
    get_game_rooms, get_game_room, create_game_room, update_game_room, delete_game_room,
)
from Backend.Controller.user_manager import user_manager

game_bp = Blueprint('game', __name__, url_prefix='/api/game')

SIZE = 15  # 15×15 board


# ── Auth helper ───────────────────────────────────────────────────────────────

def _get_user():
    token = session.get('session_token')
    return user_manager.validate_session(token) if token else None


# ── Enrich room dict with display names ──────────────────────────────────────

def _enrich(room: dict) -> dict:
    if not room:
        return room
    users = user_manager._load_users()

    def display(username):
        if not username:
            return None
        for u in users.values():
            if u.get('username') == username:
                return u.get('display_name', username)
        return username

    room['host_display']    = display(room['host'])
    room['player2_display'] = display(room['player2'])
    return room


# ── Win detection ─────────────────────────────────────────────────────────────

def _check_win(board, index, color):
    row, col = divmod(index, SIZE)
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        count = 1
        for s in [1, -1]:
            r, c = row + dr * s, col + dc * s
            while 0 <= r < SIZE and 0 <= c < SIZE and board[r * SIZE + c] == color:
                count += 1
                r += dr * s
                c += dc * s
        if count >= 5:
            return True
    return False


def _win_cells(board, index, color):
    row, col = divmod(index, SIZE)
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        cells = [index]
        for s in [1, -1]:
            r, c = row + dr * s, col + dc * s
            while 0 <= r < SIZE and 0 <= c < SIZE and board[r * SIZE + c] == color:
                cells.append(r * SIZE + c)
                r += dr * s
                c += dc * s
        if len(cells) >= 5:
            return cells
    return []


# ── Broadcast helpers ─────────────────────────────────────────────────────────

def _broadcast_rooms():
    room_list = [_enrich(r) for r in get_game_rooms()]
    socketio.emit('rooms_updated', {'rooms': room_list}, room='lobby')


def _broadcast_state(room_id):
    room_data = _enrich(get_game_room(room_id))
    socketio.emit('game_state', room_data, room=f'game_{room_id}')


# ── Socket.IO events ──────────────────────────────────────────────────────────

@socketio.on('game_join_lobby')
def on_join_lobby():
    user = _get_user()
    if not user:
        return
    join_room('lobby')
    room_list = [_enrich(r) for r in get_game_rooms()]
    emit('rooms_updated', {'rooms': room_list})


@socketio.on('game_leave_lobby')
def on_leave_lobby():
    leave_room('lobby')


@socketio.on('game_create')
def on_create(data):
    user = _get_user()
    if not user:
        emit('game_error', {'message': 'Not authenticated'})
        return
    name = (data.get('name') or f"{user['display_name']}'s Game")[:50]
    room_id = create_game_room(name, user['username'])
    join_room(f'game_{room_id}')
    emit('game_state', _enrich(get_game_room(room_id)))
    _broadcast_rooms()


@socketio.on('game_join')
def on_join(data):
    user = _get_user()
    if not user:
        emit('game_error', {'message': 'Not authenticated'})
        return
    room_id = data.get('room_id')
    role    = data.get('role', 'spectator')
    room    = get_game_room(room_id)
    if not room:
        emit('game_error', {'message': 'Room not found'})
        return

    username = user['username']

    if role == 'player':
        if room['player2'] is not None:
            emit('game_error', {'message': 'Room is full'})
            return
        if room['host'] == username:
            emit('game_error', {'message': 'You are already in this room as host'})
            return
        update_game_room(room_id, player2=username, status='playing',
                         current_turn=room['host'], board=[None] * 225, win_cells=[])
        join_room(f'game_{room_id}')
        _broadcast_state(room_id)
        _broadcast_rooms()
    else:
        # spectator — just join the socket room and receive current state
        join_room(f'game_{room_id}')
        emit('game_state', _enrich(get_game_room(room_id)))


@socketio.on('game_move')
def on_move(data):
    user = _get_user()
    if not user:
        return
    room_id = data.get('room_id')
    index   = data.get('index')
    if index is None or not (0 <= index < SIZE * SIZE):
        return

    room = get_game_room(room_id)
    if not room or room['status'] != 'playing':
        return

    username = user['username']
    if room['current_turn'] != username:
        emit('game_error', {'message': "Not your turn"})
        return

    board = room['board']
    if board[index] is not None:
        emit('game_error', {'message': 'Cell already occupied'})
        return

    color       = 'black' if username == room['host'] else 'white'
    board[index] = color

    if _check_win(board, index, color):
        cells = _win_cells(board, index, color)
        update_game_room(room_id, board=board, status='finished',
                         winner=username, win_cells=cells)
    else:
        next_turn = room['player2'] if username == room['host'] else room['host']
        update_game_room(room_id, board=board, current_turn=next_turn)

    _broadcast_state(room_id)


@socketio.on('game_kick')
def on_kick(data):
    user = _get_user()
    if not user:
        return
    room_id = data.get('room_id')
    room    = get_game_room(room_id)
    if not room or room['host'] != user['username']:
        return
    update_game_room(room_id, player2=None, status='waiting',
                     current_turn=None, board=[None] * 225, win_cells=[])
    _broadcast_state(room_id)
    _broadcast_rooms()


@socketio.on('game_rematch')
def on_rematch(data):
    user = _get_user()
    if not user:
        return
    room_id = data.get('room_id')
    room    = get_game_room(room_id)
    if not room or room['host'] != user['username'] or room['status'] != 'finished':
        return
    update_game_room(room_id, board=[None] * 225, status='playing',
                     current_turn=room['host'], winner=None, win_cells=[])
    _broadcast_state(room_id)


@socketio.on('game_leave')
def on_leave(data):
    user = _get_user()
    if not user:
        return
    room_id  = data.get('room_id')
    room     = get_game_room(room_id)
    username = user['username']

    leave_room(f'game_{room_id}')

    if not room:
        return

    if username == room['host']:
        delete_game_room(room_id)
        socketio.emit('game_dissolved', {}, room=f'game_{room_id}')
        _broadcast_rooms()
    elif username == room['player2']:
        update_game_room(room_id, player2=None, status='waiting',
                         current_turn=None, board=[None] * 225, win_cells=[])
        _broadcast_state(room_id)
        _broadcast_rooms()
    # spectators just leave silently
