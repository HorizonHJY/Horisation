from flask_socketio import SocketIO

# Shared SocketIO instance — initialised in app.py via init_app()
socketio = SocketIO()

# ── Connected-user registry ───────────────────────────────────────────────────
# sid -> user dict, filled once per connection by the single 'connect' handler
# in friends_socket.py.
#
# This lives here rather than in a feature module because python-socketio stores
# handlers in a plain dict keyed by event name: registering 'connect' in two
# modules does not chain them, the later import silently replaces the earlier
# one. game_controller used to keep its own cache behind its own 'connect'
# handler, which friends_socket then overwrote — so that cache was never
# populated and every game event saw an anonymous user.
#
# The flask session is not reliable inside an eventlet socket context, which is
# why the user is resolved once at connect time and remembered here.
connected_users: dict = {}


def remember_connection(sid: str, user: dict) -> None:
    connected_users[sid] = user


def forget_connection(sid: str) -> None:
    connected_users.pop(sid, None)


def user_for_sid(sid: str):
    return connected_users.get(sid)
