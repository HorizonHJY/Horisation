"""
uc_controller.py
Who is the Undercover — a room, a deal, and a reveal.

  POST /api/undercover/rooms                 create (host joins automatically)
  POST /api/undercover/rooms/<code>/join     join a waiting room
  GET  /api/undercover/rooms/<code>          state; your own word only
  POST /api/undercover/rooms/<code>/start    host only, 3+ players
  POST /api/undercover/rooms/<code>/reveal   host only, ends the round
  POST /api/undercover/rooms/<code>/leave
"""

from flask import Blueprint, jsonify, request

from Backend.Controller.auth_controller import login_required
from Backend.Controller import uc_db

uc_bp = Blueprint('undercover', __name__, url_prefix='/api/undercover')


def _me():
    u = request.current_user
    return u['username'], (u.get('display_name') or u['username'])


@uc_bp.route('/rooms', methods=['POST'])
@login_required
def create_room():
    username, display = _me()
    room = uc_db.create_room(username, display)
    return jsonify({'ok': True, 'room': room}), 201


@uc_bp.route('/rooms/<code>/join', methods=['POST'])
@login_required
def join_room(code):
    username, display = _me()
    room, err = uc_db.join_room(code, username, display)
    if err:
        return jsonify({'ok': False, 'error': err}), 400
    return jsonify({'ok': True, 'room': room})


@uc_bp.route('/rooms/<code>', methods=['GET'])
@login_required
def get_room(code):
    username, _ = _me()
    room = uc_db.state(code, username)
    if not room:
        return jsonify({'ok': False, 'error': 'Room not found.'}), 404
    return jsonify({'ok': True, 'room': room})


@uc_bp.route('/rooms/<code>/start', methods=['POST'])
@login_required
def start(code):
    username, _ = _me()
    room, err = uc_db.start_game(code, username)
    if err:
        return jsonify({'ok': False, 'error': err}), 400
    return jsonify({'ok': True, 'room': room})


@uc_bp.route('/rooms/<code>/reveal', methods=['POST'])
@login_required
def reveal(code):
    username, _ = _me()
    room, err = uc_db.reveal(code, username)
    if err:
        return jsonify({'ok': False, 'error': err}), 400
    return jsonify({'ok': True, 'room': room})


@uc_bp.route('/rooms/<code>/leave', methods=['POST'])
@login_required
def leave(code):
    username, _ = _me()
    ok = uc_db.leave_room(code, username)
    if not ok:
        return jsonify({'ok': False, 'error': 'Cannot leave now.'}), 400
    return jsonify({'ok': True})
