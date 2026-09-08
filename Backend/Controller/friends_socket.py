"""
friends_socket.py
Socket.IO events: personal notification rooms, real-time chat, online presence.
"""

from flask import session, request
from flask_socketio import join_room

from Backend.Controller.socketio_instance import socketio
from Backend.Controller.user_manager import user_manager
from Backend.Controller import market_db

# sid → username  (for quick lookup on incoming events)
_online: dict = {}


@socketio.on('connect')
def _on_connect_friends():
    """Join user_{username} room so the server can push notifications."""
    token = session.get('session_token')
    if token:
        user = user_manager.validate_session(token)
        if user:
            username = user['username']
            _online[request.sid] = username
            join_room(f'user_{username}')


@socketio.on('disconnect')
def _on_disconnect_friends():
    _online.pop(request.sid, None)


@socketio.on('friends_get_online')
def on_get_online():
    """Return list of currently connected usernames to the requester."""
    from flask_socketio import emit
    emit('online_list', {'online': list(set(_online.values()))})


@socketio.on('chat_send')
def on_chat_send(data):
    sender  = _online.get(request.sid)
    if not sender:
        return
    from flask_socketio import emit
    to_user = (data or {}).get('to_user', '').strip()
    content = (data or {}).get('content', '').strip()
    if not content or not to_user:
        return
    if len(content) > 1000:
        emit('chat_error', {'message': 'Message too long (max 1000 chars)'})
        return

    ua, ub   = market_db._friend_pair(sender, to_user)
    room_key = f'{ua}:{ub}'
    msg      = market_db.save_chat_message(room_key, sender, content)

    socketio.emit('chat_message', msg, room=f'user_{sender}')
    socketio.emit('chat_message', msg, room=f'user_{to_user}')


# ── Push helpers called from friends_controller ───────────────────────────────

def notify_friend_request(to_user: str, req_dict: dict) -> None:
    socketio.emit('friend_request_incoming', req_dict, room=f'user_{to_user}')


def notify_contact_request(to_user: str, req_dict: dict) -> None:
    socketio.emit('contact_request_incoming', req_dict, room=f'user_{to_user}')


def notify_contact_resolved(req_id: int, responder: str, requester: str, action: str) -> None:
    """A contact request stopped being pending.

    Both sides are told: the requester so they learn the answer, and the
    responder so their *other* open tabs drop the pending banner instead of
    offering to answer something that is already answered.
    """
    payload = {'id': req_id, 'action': action, 'responder': responder, 'requester': requester}
    socketio.emit('contact_request_resolved', payload, room=f'user_{responder}')
    socketio.emit('contact_request_resolved', payload, room=f'user_{requester}')


def notify_friend_accepted(acceptor: str, req_id: str) -> None:
    with market_db.Session() as s:
        row = s.query(market_db.FriendRequest).filter_by(id=req_id).first()
        if row:
            socketio.emit(
                'friend_accepted',
                {'from_user': acceptor},
                room=f'user_{row.from_user}',
            )


# ── Trade-intent push helpers (意向成单流) ───────────────────────────────────

def notify_intent(event: str, intent: dict) -> None:
    """Push a realtime trade-intent event to the persons who care.

    event: 'trade_intent' (buyer -> seller),
           'trade_intent_accepted' (seller -> buyer),
           'trade_intent_completed' (buyer -> seller),
           'trade_intent_cancelled' (either -> other),
           'trade_intent_declined' (seller -> buyer)
    """
    if not intent:
        return
    buyer  = intent.get('buyer')
    seller = intent.get('seller')
    if event == 'trade_intent':
        socketio.emit(event, intent, room=f'user_{seller}')
    elif event == 'trade_intent_completed':
        socketio.emit(event, intent, room=f'user_{seller}')
    else:  # accepted / declined / cancelled -> notify the buyer
        socketio.emit(event, intent, room=f'user_{buyer}')
