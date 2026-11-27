"""
uc_db.py
Who is the Undercover — rooms, players, and the dealt words.

  uc_rooms    — one row per room: host, status, the pair that was drawn, and
                who got the odd word.
  uc_players  — one row per (room, person). `word` is their private card.

v1 stops after the deal: everyone joins, the host starts, each person flips
their own card, and the host can reveal the pair at the end. Voting and
elimination come later.
"""

import json
import os
import random
import string
from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Integer, String, UniqueConstraint, create_engine, event,
)
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH  = os.path.join(BASE_DIR, '_data', 'market.db')
WORDS_PATH = os.path.join(BASE_DIR, 'Backend', 'data', 'undercover_words.json')

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, connect_args={'check_same_thread': False})
event.listen(engine, 'connect', lambda c, _: c.execute('PRAGMA journal_mode=WAL'))
Session = sessionmaker(bind=engine)
Base = declarative_base()

CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'   # no I/O/0/1
MIN_PLAYERS = 3


class UCRoom(Base):
    __tablename__ = 'uc_rooms'
    code        = Column(String(8), primary_key=True)
    host        = Column(String(100), nullable=False)
    status      = Column(String(12), nullable=False, default='waiting')  # waiting|playing|revealed
    word_a      = Column(String(24), nullable=True)   # civilian word
    word_b      = Column(String(24), nullable=True)   # undercover word
    undercover  = Column(String(100), nullable=True)
    created_at  = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at  = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc),
                         onupdate=lambda: datetime.now(timezone.utc))


class UCPlayer(Base):
    __tablename__ = 'uc_players'
    id           = Column(Integer, primary_key=True, autoincrement=True)
    room_code    = Column(String(8), nullable=False, index=True)
    username     = Column(String(100), nullable=False)
    display_name = Column(String(100), nullable=True)
    role         = Column(String(12), nullable=False, default='civilian')
    word         = Column(String(24), nullable=True)
    joined_at    = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint('room_code', 'username', name='uq_uc_player'),
    )


def init_uc_db():
    Base.metadata.create_all(engine)


# ── Word bank ────────────────────────────────────────────────────────────────

_pairs = None


def load_pairs() -> list[list[str]]:
    global _pairs
    if _pairs is None:
        try:
            with open(WORDS_PATH, encoding='utf-8') as f:
                _pairs = json.load(f).get('pairs') or []
        except (OSError, ValueError):
            _pairs = []
    return _pairs


# ── Rooms ────────────────────────────────────────────────────────────────────

def _new_code(s) -> str:
    for _ in range(20):
        code = ''.join(random.choice(CODE_ALPHABET) for _ in range(6))
        if not s.get(UCRoom, code):
            return code
    raise RuntimeError('Could not allocate a room code')


def create_room(host: str, display_name: str) -> dict:
    with Session() as s:
        code = _new_code(s)
        room = UCRoom(code=code, host=host, status='waiting')
        s.add(room)
        s.add(UCPlayer(room_code=code, username=host, display_name=display_name))
        s.commit()
        return state(code, host)


def join_room(code: str, username: str, display_name: str) -> tuple[dict | None, str | None]:
    with Session() as s:
        room = s.get(UCRoom, code.upper())
        if not room:
            return None, 'Room not found.'
        if room.status != 'waiting':
            existing = s.query(UCPlayer).filter_by(room_code=room.code, username=username).first()
            if not existing:
                return None, 'This game already started.'
            return state(room.code, username), None
        existing = s.query(UCPlayer).filter_by(room_code=room.code, username=username).first()
        if not existing:
            if s.query(UCPlayer).filter_by(room_code=room.code).count() >= 12:
                return None, 'Room is full.'
            s.add(UCPlayer(room_code=room.code, username=username, display_name=display_name))
            s.commit()
        return state(room.code, username), None


def leave_room(code: str, username: str) -> bool:
    with Session() as s:
        room = s.get(UCRoom, code.upper())
        if not room or room.status != 'waiting':
            return False
        p = s.query(UCPlayer).filter_by(room_code=room.code, username=username).first()
        if not p:
            return False
        s.delete(p)
        s.commit()
        # An empty room is removed; a departed host hands over to the next player.
        players = s.query(UCPlayer).filter_by(room_code=room.code).order_by(UCPlayer.joined_at).all()
        if not players:
            s.delete(room)
            s.commit()
        elif room.host == username:
            room.host = players[0].username
            s.commit()
        return True


def start_game(code: str, username: str) -> tuple[dict | None, str | None]:
    with Session() as s:
        room = s.get(UCRoom, code.upper())
        if not room:
            return None, 'Room not found.'
        if room.host != username:
            return None, 'Only the host can start.'
        if room.status != 'waiting':
            return None, 'This game already started.'
        players = s.query(UCPlayer).filter_by(room_code=room.code).all()
        if len(players) < MIN_PLAYERS:
            return None, f'Need at least {MIN_PLAYERS} players.'

        pairs = load_pairs()
        if not pairs:
            return None, 'The word bank is empty.'
        a, b = random.choice(pairs)

        undercover = random.choice(players)
        for p in players:
            p.role = 'undercover' if p.id == undercover.id else 'civilian'
            p.word = b if p.id == undercover.id else a
        room.word_a = a
        room.word_b = b
        room.undercover = undercover.username
        room.status = 'playing'
        s.commit()
        return state(room.code, username), None


def reveal(code: str, username: str) -> tuple[dict | None, str | None]:
    with Session() as s:
        room = s.get(UCRoom, code.upper())
        if not room:
            return None, 'Room not found.'
        if room.host != username:
            return None, 'Only the host can reveal.'
        if room.status == 'waiting':
            return None, 'Start the game first.'
        room.status = 'revealed'
        s.commit()
        return state(room.code, username), None


def state(code: str, viewer: str) -> dict | None:
    with Session() as s:
        room = s.get(UCRoom, code.upper())
        if not room:
            return None
        players = s.query(UCPlayer).filter_by(room_code=room.code).order_by(UCPlayer.joined_at).all()
        me = next((p for p in players if p.username == viewer), None)
        out = {
            'code': room.code,
            'host': room.host,
            'status': room.status,
            'min_players': MIN_PLAYERS,
            'players': [
                {'username': p.username, 'display_name': p.display_name or p.username,
                 'is_host': p.username == room.host}
                for p in players
            ],
            'me': None,
            'reveal': None,
        }
        # A player's own word is private until they look at it — but the state
        # endpoint is the only way to get it, so it ships with their row only.
        if me and room.status in ('playing', 'revealed'):
            out['me'] = {'username': me.username, 'word': me.word}
        if room.status == 'revealed':
            out['reveal'] = {
                'civilian_word': room.word_a,
                'undercover_word': room.word_b,
                'undercover': room.undercover,
            }
        return out
