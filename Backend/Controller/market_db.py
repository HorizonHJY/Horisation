"""
market_db.py
SQLAlchemy models and helper functions for all persistent data.
Database: _data/market.db (SQLite, auto-created on init_db()).
Designed for easy migration to PostgreSQL: swap the engine URL only.
"""

import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from sqlalchemy import (
    create_engine, Column, String, Text, Float, Integer, DateTime, ForeignKey, Boolean, text, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH  = os.path.join(BASE_DIR, '_data', 'market.db')

engine  = create_engine(f'sqlite:///{DB_PATH}', echo=False)
Session = sessionmaker(bind=engine)
Base    = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = 'user'

    id              = Column(Integer, primary_key=True, autoincrement=True)
    username        = Column(String(100), unique=True, nullable=False, index=True)
    password        = Column(String(255), nullable=False)
    role            = Column(String(50),  nullable=False, default='user')
    email           = Column(String(200), default='')
    display_name    = Column(String(100), default='')
    is_active       = Column(Boolean, default=True)
    avatar_url      = Column(String(500), nullable=True)
    contact_info    = Column(Text, nullable=True)   # legacy, kept for migration
    contact_hidden  = Column(Boolean, default=False)
    wechat          = Column(String(200), nullable=True)
    phone           = Column(String(50),  nullable=True)
    created_at      = Column(DateTime, default=lambda: datetime.utcnow())


class UserSession(Base):
    __tablename__ = 'session'

    token      = Column(String(64), primary_key=True)
    username   = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.utcnow())
    expires_at = Column(DateTime, nullable=False)


class Listing(Base):
    __tablename__ = 'listings'

    id               = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    seller_username  = Column(String(100), nullable=False, index=True)
    title            = Column(String(100), nullable=False)
    description      = Column(Text, nullable=False)
    price            = Column(Float, nullable=False)
    category         = Column(String(50), nullable=False, default='other')
    contact          = Column(String(200), nullable=False)
    original_price   = Column(Float, nullable=True)
    status           = Column(String(20), nullable=False, default='active')  # active / sold / removed
    created_at       = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at       = Column(DateTime, nullable=False,
                              default=lambda: datetime.now(timezone.utc),
                              onupdate=lambda: datetime.now(timezone.utc))

    images = relationship('ListingImage', back_populates='listing',
                          cascade='all, delete-orphan', order_by='ListingImage.display_order')


class Message(Base):
    __tablename__ = 'messages'

    id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username     = Column(String(100), nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    content      = Column(Text, nullable=False)
    created_at   = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class GameRoom(Base):
    __tablename__ = 'game_rooms'

    id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name         = Column(String(50),  nullable=False)
    host         = Column(String(100), nullable=False)   # player1, plays black
    player2      = Column(String(100), nullable=True)    # plays white
    status       = Column(String(20),  nullable=False, default='waiting')  # waiting/playing/finished
    board        = Column(Text, nullable=False, default=lambda: json.dumps([None]*225))
    current_turn = Column(String(100), nullable=True)
    winner       = Column(String(100), nullable=True)
    win_cells    = Column(Text, nullable=False, default='[]')
    created_at   = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class ListingImage(Base):
    __tablename__ = 'listing_images'

    id            = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    listing_id    = Column(String(36), ForeignKey('listings.id'), nullable=False, index=True)
    r2_url        = Column(String(500), nullable=False)
    r2_key        = Column(String(500), nullable=False)
    display_order = Column(Integer, nullable=False, default=0)

    listing = relationship('Listing', back_populates='images')


class Memo(Base):
    __tablename__ = 'memos'

    id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username     = Column(String(100), nullable=False, index=True)
    content      = Column(Text, nullable=False)
    type         = Column(String(50), nullable=False, default='general')
    priority     = Column(String(20), nullable=False, default='normal')
    status       = Column(String(20), nullable=False, default='active')
    tags         = Column(Text, nullable=False, default='[]')   # JSON list
    due_date     = Column(String(50), nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at   = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at   = Column(DateTime, nullable=False,
                          default=lambda: datetime.now(timezone.utc),
                          onupdate=lambda: datetime.now(timezone.utc))


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist. Called once at app startup."""
    Base.metadata.create_all(engine)
    _migrate_from_json()
    _migrate_columns()


def _migrate_columns():
    """Idempotently add new columns to existing tables."""
    stmts = [
        "ALTER TABLE listings ADD COLUMN original_price REAL",
        "ALTER TABLE friend_requests ADD COLUMN message TEXT",
        "ALTER TABLE user ADD COLUMN contact_hidden BOOLEAN DEFAULT 0",
        "ALTER TABLE user ADD COLUMN wechat TEXT",
        "ALTER TABLE user ADD COLUMN phone TEXT",
    ]
    with Session() as s:
        for stmt in stmts:
            try:
                s.execute(text(stmt))
                s.commit()
            except Exception:
                pass  # column already exists


def _migrate_from_json():
    """One-time migration: users.json → User table."""
    users_file = os.path.join(BASE_DIR, '_data', 'users.json')
    if not os.path.exists(users_file):
        return
    with Session() as s:
        if s.query(User).count() > 0:
            return  # already migrated
    with open(users_file, encoding='utf-8') as f:
        data = json.load(f)
    with Session() as s:
        for u in data.values():
            row = User(
                username=u.get('username', ''),
                password=u.get('password', ''),
                role=u.get('role', 'user'),
                email=u.get('email', ''),
                display_name=u.get('display_name', u.get('username', '')),
                is_active=u.get('is_active', True),
                avatar_url=u.get('avatar_url'),
                contact_info=u.get('contact_info'),
            )
            s.add(row)
        s.commit()
    os.rename(users_file, users_file + '.migrated')
    print('✅ Migrated users.json → SQLite')


# ── User helpers ───────────────────────────────────────────────────────────────

def _user_to_dict(u: User) -> dict:
    return {
        'username':        u.username,
        'password':        u.password,
        'role':            u.role,
        'email':           u.email or '',
        'display_name':    u.display_name or u.username,
        'is_active':       u.is_active,
        'avatar_url':      u.avatar_url,
        'contact_info':    u.contact_info or '',
        'contact_hidden':  bool(u.contact_hidden),
        'wechat':          u.wechat or '',
        'phone':           u.phone or '',
        'created_at':      u.created_at.isoformat() if u.created_at else '',
    }


def db_create_user(username: str, password: str, role: str,
                   email: str = '', display_name: str = '') -> dict:
    u = User(username=username, password=password, role=role,
             email=email, display_name=display_name or username)
    with Session() as s:
        s.add(u)
        s.commit()
        s.refresh(u)
        return _user_to_dict(u)


def db_get_user(username: str) -> Optional[dict]:
    with Session() as s:
        u = s.query(User).filter_by(username=username).first()
        return _user_to_dict(u) if u else None


def db_list_users() -> List[dict]:
    with Session() as s:
        rows = s.query(User).order_by(User.username).all()
        return [_user_to_dict(u) for u in rows]


def db_search_users(q: str) -> List[dict]:
    pattern = f'%{q}%'
    with Session() as s:
        rows = s.query(User).filter(
            (User.username.ilike(pattern)) |
            (User.display_name.ilike(pattern))
        ).limit(20).all()
        return [_user_to_dict(u) for u in rows]


def db_update_user(username: str, **fields) -> bool:
    allowed = {'password', 'role', 'email', 'display_name',
               'is_active', 'avatar_url', 'contact_info', 'contact_hidden', 'wechat', 'phone'}
    with Session() as s:
        u = s.query(User).filter_by(username=username).first()
        if not u:
            return False
        for key, val in fields.items():
            if key in allowed:
                setattr(u, key, val)
        s.commit()
        return True


def db_delete_user(username: str) -> bool:
    with Session() as s:
        u = s.query(User).filter_by(username=username).first()
        if not u:
            return False
        s.delete(u)
        s.commit()
        return True


# ── Session helpers ────────────────────────────────────────────────────────────

def db_create_session(token: str, username: str, expires_at: datetime) -> None:
    sess = UserSession(token=token, username=username, expires_at=expires_at)
    with Session() as s:
        s.add(sess)
        s.commit()


def db_get_session(token: str) -> Optional[dict]:
    with Session() as s:
        sess = s.query(UserSession).filter_by(token=token).first()
        if not sess:
            return None
        return {
            'token':      sess.token,
            'username':   sess.username,
            'expires_at': sess.expires_at,
        }


def db_delete_session(token: str) -> None:
    with Session() as s:
        sess = s.query(UserSession).filter_by(token=token).first()
        if sess:
            s.delete(sess)
            s.commit()


def db_cleanup_sessions() -> None:
    now = datetime.utcnow()
    with Session() as s:
        s.query(UserSession).filter(UserSession.expires_at < now).delete()
        s.commit()


# ── Serialisers ───────────────────────────────────────────────────────────────

def _listing_to_dict(listing: Listing, seller_user: 'User | None' = None) -> dict:
    return {
        'id':               listing.id,
        'seller_username':  listing.seller_username,
        'seller_display':   (seller_user.display_name or listing.seller_username) if seller_user else listing.seller_username,
        'seller_avatar':    seller_user.avatar_url if seller_user else None,
        'title':            listing.title,
        'description':      listing.description,
        'price':            listing.price,
        'original_price':   listing.original_price,
        'category':         listing.category,
        'contact':          listing.contact,
        'status':           listing.status,
        'created_at':       listing.created_at.isoformat(),
        'updated_at':       listing.updated_at.isoformat(),
        'images':           [{'id': img.id, 'url': img.r2_url, 'order': img.display_order}
                             for img in listing.images],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_messages(limit: int = 100) -> list:
    with Session() as s:
        rows = s.query(Message).order_by(Message.created_at.desc()).limit(limit).all()
        return [
            {
                'id':           r.id,
                'username':     r.username,
                'display_name': r.display_name,
                'content':      r.content,
                'created_at':   r.created_at.isoformat(),
            }
            for r in rows
        ]


def post_message(username: str, display_name: str, content: str) -> dict:
    msg = Message(
        id=str(uuid.uuid4()),
        username=username,
        display_name=display_name,
        content=content,
    )
    with Session() as s:
        s.add(msg)
        s.commit()
        return {
            'id':           msg.id,
            'username':     msg.username,
            'display_name': msg.display_name,
            'content':      msg.content,
            'created_at':   msg.created_at.isoformat(),
        }


def delete_message(message_id: str, username: str, is_admin: bool = False) -> bool:
    """Delete a message. Users can only delete their own; admins can delete any."""
    with Session() as s:
        row = s.query(Message).filter_by(id=message_id).first()
        if not row:
            return False
        if not is_admin and row.username != username:
            return False
        s.delete(row)
        s.commit()
        return True


def _enrich_listings(rows: list, s) -> list[dict]:
    """Batch-fetch seller User rows and attach display info to each listing dict."""
    usernames = list({r.seller_username for r in rows})
    sellers   = {u.username: u for u in s.query(User).filter(User.username.in_(usernames)).all()}
    return [_listing_to_dict(r, sellers.get(r.seller_username)) for r in rows]


def get_all_listings(status: str = 'active') -> list[dict]:
    with Session() as s:
        rows = s.query(Listing).filter_by(status=status)\
                               .order_by(Listing.created_at.desc()).all()
        return _enrich_listings(rows, s)


def get_listing(listing_id: str) -> Optional[dict]:
    with Session() as s:
        row = s.query(Listing).filter_by(id=listing_id).first()
        if not row:
            return None
        seller = s.query(User).filter_by(username=row.seller_username).first()
        return _listing_to_dict(row, seller)


def get_my_listings(username: str) -> list[dict]:
    with Session() as s:
        rows = s.query(Listing).filter_by(seller_username=username)\
                               .order_by(Listing.created_at.desc()).all()
        return _enrich_listings(rows, s)


def get_active_listings_by_user(username: str) -> list[dict]:
    """Active (non-sold) listings for a given seller."""
    with Session() as s:
        rows = s.query(Listing).filter_by(seller_username=username, status='active')\
                               .order_by(Listing.created_at.desc()).all()
        return _enrich_listings(rows, s)


def create_listing(seller: str, title: str, description: str,
                   price: float, category: str, contact: str,
                   original_price: float = None) -> str:
    """Create a listing row and return its id."""
    listing = Listing(
        id=str(uuid.uuid4()),
        seller_username=seller,
        title=title,
        description=description,
        price=price,
        original_price=original_price,
        category=category,
        contact=contact,
    )
    with Session() as s:
        s.add(listing)
        s.commit()
        return listing.id


def add_image(listing_id: str, r2_url: str, r2_key: str, order: int) -> str:
    """Add an image record linked to a listing. Returns image id."""
    img = ListingImage(
        id=str(uuid.uuid4()),
        listing_id=listing_id,
        r2_url=r2_url,
        r2_key=r2_key,
        display_order=order,
    )
    with Session() as s:
        s.add(img)
        s.commit()
        return img.id


def update_listing(listing_id: str, seller: str, **fields) -> bool:
    """Update allowed text fields. Returns False if not found or wrong seller."""
    allowed = {'title', 'description', 'price', 'category', 'contact'}
    with Session() as s:
        row = s.query(Listing).filter_by(id=listing_id, seller_username=seller).first()
        if not row:
            return False
        for key, val in fields.items():
            if key in allowed:
                setattr(row, key, val)
        row.updated_at = datetime.now(timezone.utc)
        s.commit()
        return True


def mark_sold(listing_id: str, seller: str) -> bool:
    """Mark a listing as sold. Returns False if not found or wrong seller."""
    with Session() as s:
        row = s.query(Listing).filter_by(id=listing_id, seller_username=seller).first()
        if not row:
            return False
        row.status     = 'sold'
        row.updated_at = datetime.now(timezone.utc)
        s.commit()
        return True


# ── Game room helpers ─────────────────────────────────────────────────────────

def _room_to_dict(room: GameRoom) -> dict:
    return {
        'id':           room.id,
        'name':         room.name,
        'host':         room.host,
        'player2':      room.player2,
        'status':       room.status,
        'board':        json.loads(room.board),
        'current_turn': room.current_turn,
        'winner':       room.winner,
        'win_cells':    json.loads(room.win_cells),
        'created_at':   room.created_at.isoformat(),
    }


def get_game_rooms() -> list:
    """Return all non-finished rooms."""
    with Session() as s:
        rows = s.query(GameRoom).filter(GameRoom.status != 'finished') \
                                .order_by(GameRoom.created_at.desc()).all()
        return [_room_to_dict(r) for r in rows]


def get_game_room(room_id: str) -> Optional[dict]:
    with Session() as s:
        row = s.query(GameRoom).filter_by(id=room_id).first()
        return _room_to_dict(row) if row else None


def create_game_room(name: str, host: str) -> str:
    room = GameRoom(id=str(uuid.uuid4()), name=name, host=host)
    with Session() as s:
        s.add(room)
        s.commit()
        return room.id


def update_game_room(room_id: str, **fields) -> None:
    allowed = {'player2', 'status', 'board', 'current_turn', 'winner', 'win_cells'}
    with Session() as s:
        row = s.query(GameRoom).filter_by(id=room_id).first()
        if not row:
            return
        for key, val in fields.items():
            if key in allowed:
                if key in ('board', 'win_cells') and not isinstance(val, str):
                    val = json.dumps(val)
                setattr(row, key, val)
        s.commit()


def delete_game_room(room_id: str) -> bool:
    with Session() as s:
        row = s.query(GameRoom).filter_by(id=room_id).first()
        if not row:
            return False
        s.delete(row)
        s.commit()
        return True


def delete_listing(listing_id: str, seller: str) -> Optional[List[str]]:
    """
    Delete a listing and its image records.
    Returns list of r2_keys that must be deleted from R2, or None if not found/wrong seller.
    """
    with Session() as s:
        row = s.query(Listing).filter_by(id=listing_id, seller_username=seller).first()
        if not row:
            return None
        r2_keys = [img.r2_key for img in row.images]
        s.delete(row)
        s.commit()
        return r2_keys


# ── Memo helpers ──────────────────────────────────────────────────────────────

def _memo_to_dict(memo: Memo) -> dict:
    return {
        'id':           memo.id,
        'content':      memo.content,
        'type':         memo.type,
        'priority':     memo.priority,
        'status':       memo.status,
        'tags':         json.loads(memo.tags),
        'due_date':     memo.due_date,
        'completed_at': memo.completed_at.isoformat() if memo.completed_at else None,
        'created_at':   memo.created_at.isoformat(),
        'updated_at':   memo.updated_at.isoformat(),
    }


def get_memos(username: str, status: str = None, memo_type: str = None,
              priority: str = None, limit: int = 50, offset: int = 0) -> tuple:
    """Return (list_of_dicts, total_count) for the user's memos."""
    with Session() as s:
        q = s.query(Memo).filter_by(username=username)
        if status:
            q = q.filter_by(status=status)
        if memo_type:
            q = q.filter_by(type=memo_type)
        if priority:
            q = q.filter_by(priority=priority)
        q = q.order_by(Memo.created_at.desc())
        total = q.count()
        rows  = q.offset(offset).limit(limit).all()
        return [_memo_to_dict(r) for r in rows], total


def create_memo(username: str, content: str, memo_type: str = 'general',
                priority: str = 'normal', tags: list = None, due_date: str = None) -> str:
    """Insert a memo row and return its id."""
    memo = Memo(
        id=str(uuid.uuid4()),
        username=username,
        content=content,
        type=memo_type,
        priority=priority,
        tags=json.dumps(tags or []),
        due_date=due_date,
    )
    with Session() as s:
        s.add(memo)
        s.commit()
        return memo.id


def get_memo_by_id(memo_id: str, username: str) -> Optional[dict]:
    with Session() as s:
        row = s.query(Memo).filter_by(id=memo_id, username=username).first()
        return _memo_to_dict(row) if row else None


def update_memo(memo_id: str, username: str, **fields) -> bool:
    allowed = {'content', 'priority', 'tags', 'due_date', 'status'}
    with Session() as s:
        row = s.query(Memo).filter_by(id=memo_id, username=username).first()
        if not row:
            return False
        for key, val in fields.items():
            if key not in allowed:
                continue
            if key == 'tags':
                val = json.dumps(val) if isinstance(val, list) else val
            setattr(row, key, val)
        if fields.get('status') == 'completed' and row.completed_at is None:
            row.completed_at = datetime.now(timezone.utc)
        row.updated_at = datetime.now(timezone.utc)
        s.commit()
        return True


def complete_memo(memo_id: str, username: str) -> bool:
    with Session() as s:
        row = s.query(Memo).filter_by(id=memo_id, username=username).first()
        if not row:
            return False
        row.status       = 'completed'
        row.completed_at = datetime.now(timezone.utc)
        row.updated_at   = datetime.now(timezone.utc)
        s.commit()
        return True


def delete_memo(memo_id: str, username: str) -> bool:
    with Session() as s:
        row = s.query(Memo).filter_by(id=memo_id, username=username).first()
        if not row:
            return False
        s.delete(row)
        s.commit()
        return True


def get_memo_statistics(username: str) -> dict:
    with Session() as s:
        rows = s.query(Memo).filter_by(username=username).all()
        status_stats   = {}
        priority_stats = {}
        type_stats     = {}
        for r in rows:
            status_stats[r.status]     = status_stats.get(r.status, 0) + 1
            priority_stats[r.priority] = priority_stats.get(r.priority, 0) + 1
            type_stats[r.type]         = type_stats.get(r.type, 0) + 1
        return {
            'total_memos':    len(rows),
            'status_stats':   status_stats,
            'priority_stats': priority_stats,
            'type_stats':     type_stats,
        }


# ── Friend / chat models ───────────────────────────────────────────────────────

class FriendRequest(Base):
    __tablename__ = 'friend_requests'

    id         = Column(String(36),  primary_key=True, default=lambda: str(uuid.uuid4()))
    from_user  = Column(String(100), nullable=False, index=True)
    to_user    = Column(String(100), nullable=False, index=True)
    status     = Column(String(20),  nullable=False, default='pending')  # pending/accepted/rejected
    message    = Column(Text,        nullable=True)
    created_at = Column(DateTime,    nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime,    nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


class Friendship(Base):
    __tablename__ = 'friendships'

    id         = Column(String(36),  primary_key=True, default=lambda: str(uuid.uuid4()))
    user_a     = Column(String(100), nullable=False, index=True)  # lexicographically smaller
    user_b     = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime,    nullable=False, default=lambda: datetime.now(timezone.utc))


class PrivateChatMessage(Base):
    __tablename__ = 'private_chat_messages'

    id         = Column(String(36),   primary_key=True, default=lambda: str(uuid.uuid4()))
    room_key   = Column(String(201),  nullable=False, index=True)  # "{user_a}:{user_b}" sorted
    sender     = Column(String(100),  nullable=False)
    content    = Column(Text,         nullable=False)
    created_at = Column(DateTime,     nullable=False, default=lambda: datetime.now(timezone.utc))


class ContactRequest(Base):
    __tablename__ = 'contact_requests'

    id         = Column(Integer,      primary_key=True, autoincrement=True)
    from_user  = Column(String(100),  nullable=False, index=True)
    to_user    = Column(String(100),  nullable=False, index=True)
    status     = Column(String(20),   nullable=False, default='pending')  # pending/approved/declined
    created_at = Column(DateTime,     nullable=False, default=lambda: datetime.now(timezone.utc))


class ChatRead(Base):
    """Tracks the last time each user read each private chat room."""
    __tablename__ = 'chat_read'

    id         = Column(Integer,     primary_key=True, autoincrement=True)
    username   = Column(String(100), nullable=False, index=True)
    room_key   = Column(String(201), nullable=False, index=True)
    read_at    = Column(DateTime,    nullable=False, default=lambda: datetime.utcnow())


# ── Friend helpers ─────────────────────────────────────────────────────────────

def _friend_pair(a: str, b: str) -> tuple:
    """Return (user_a, user_b) with user_a <= user_b for consistent storage."""
    return (a, b) if a <= b else (b, a)


def _req_to_dict(r: FriendRequest) -> dict:
    return {
        'id':         r.id,
        'from_user':  r.from_user,
        'to_user':    r.to_user,
        'status':     r.status,
        'message':    r.message,
        'created_at': r.created_at.isoformat(),
    }


def send_friend_request(from_user: str, to_user: str, message: str = None) -> Optional[dict]:
    """Create a pending request. Returns None if duplicate pending or already friends."""
    with Session() as s:
        existing = s.query(FriendRequest).filter(
            FriendRequest.status == 'pending',
            ((FriendRequest.from_user == from_user) & (FriendRequest.to_user == to_user)) |
            ((FriendRequest.from_user == to_user)   & (FriendRequest.to_user == from_user))
        ).first()
        if existing:
            return None
        ua, ub = _friend_pair(from_user, to_user)
        if s.query(Friendship).filter_by(user_a=ua, user_b=ub).first():
            return None
        req = FriendRequest(id=str(uuid.uuid4()), from_user=from_user, to_user=to_user, message=message)
        s.add(req)
        s.commit()
        return _req_to_dict(req)


def respond_friend_request(request_id: str, to_user: str, accept: bool) -> bool:
    """Accept or reject. On accept inserts a Friendship row. Returns False if not found."""
    with Session() as s:
        req = s.query(FriendRequest).filter_by(id=request_id, to_user=to_user, status='pending').first()
        if not req:
            return False
        req.status     = 'accepted' if accept else 'rejected'
        req.updated_at = datetime.now(timezone.utc)
        if accept:
            ua, ub = _friend_pair(req.from_user, req.to_user)
            s.add(Friendship(id=str(uuid.uuid4()), user_a=ua, user_b=ub))
        s.commit()
        return True


def get_pending_requests(username: str) -> list:
    with Session() as s:
        rows = s.query(FriendRequest).filter_by(to_user=username, status='pending') \
                                     .order_by(FriendRequest.created_at.desc()).all()
        return [_req_to_dict(r) for r in rows]


def get_sent_requests(username: str) -> list:
    with Session() as s:
        rows = s.query(FriendRequest).filter_by(from_user=username) \
                                     .order_by(FriendRequest.created_at.desc()).all()
        return [_req_to_dict(r) for r in rows]


def get_friends(username: str) -> list:
    with Session() as s:
        rows = s.query(Friendship).filter(
            (Friendship.user_a == username) | (Friendship.user_b == username)
        ).all()
        return [r.user_b if r.user_a == username else r.user_a for r in rows]


def are_friends(a: str, b: str) -> bool:
    ua, ub = _friend_pair(a, b)
    with Session() as s:
        return s.query(Friendship).filter_by(user_a=ua, user_b=ub).first() is not None


def remove_friend(a: str, b: str) -> bool:
    ua, ub = _friend_pair(a, b)
    with Session() as s:
        row = s.query(Friendship).filter_by(user_a=ua, user_b=ub).first()
        if not row:
            return False
        s.delete(row)
        s.commit()
        return True


def get_chat_history(a: str, b: str, limit: int = 100) -> list:
    """Return last N messages between two users, oldest-first."""
    ua, ub   = _friend_pair(a, b)
    room_key = f'{ua}:{ub}'
    with Session() as s:
        rows = s.query(PrivateChatMessage).filter_by(room_key=room_key) \
                                          .order_by(PrivateChatMessage.created_at.desc()) \
                                          .limit(limit).all()
        rows.reverse()
        return [
            {'id': r.id, 'room_key': r.room_key, 'sender': r.sender,
             'content': r.content, 'created_at': r.created_at.isoformat()}
            for r in rows
        ]


def save_chat_message(room_key: str, sender: str, content: str) -> dict:
    msg = PrivateChatMessage(id=str(uuid.uuid4()), room_key=room_key, sender=sender, content=content)
    with Session() as s:
        s.add(msg)
        s.commit()
        return {'id': msg.id, 'room_key': msg.room_key, 'sender': msg.sender,
                'content': msg.content, 'created_at': msg.created_at.isoformat()}


# ── Contact request helpers ────────────────────────────────────────────────────

def _contact_req_to_dict(r: ContactRequest) -> dict:
    return {
        'id':         r.id,
        'from_user':  r.from_user,
        'to_user':    r.to_user,
        'status':     r.status,
        'created_at': r.created_at.isoformat(),
    }


def send_contact_request(from_user: str, to_user: str) -> Optional[dict]:
    """Create a pending contact request. Returns None if duplicate pending/approved."""
    with Session() as s:
        existing = s.query(ContactRequest).filter_by(
            from_user=from_user, to_user=to_user
        ).filter(ContactRequest.status.in_(['pending', 'approved'])).first()
        if existing:
            return None
        req = ContactRequest(from_user=from_user, to_user=to_user)
        s.add(req)
        s.commit()
        s.refresh(req)
        return _contact_req_to_dict(req)


def get_contact_requests_received(username: str) -> list:
    """Pending contact requests where I am the target."""
    with Session() as s:
        rows = s.query(ContactRequest).filter_by(to_user=username, status='pending') \
                                      .order_by(ContactRequest.created_at.desc()).all()
        return [_contact_req_to_dict(r) for r in rows]


def get_contact_requests_sent(username: str) -> list:
    """All contact requests I have sent (any status)."""
    with Session() as s:
        rows = s.query(ContactRequest).filter_by(from_user=username) \
                                      .order_by(ContactRequest.created_at.desc()).all()
        return [_contact_req_to_dict(r) for r in rows]


def respond_contact_request(req_id: int, to_user: str, accept: bool) -> bool:
    with Session() as s:
        req = s.query(ContactRequest).filter_by(id=req_id, to_user=to_user, status='pending').first()
        if not req:
            return False
        req.status = 'approved' if accept else 'declined'
        s.commit()
        return True


def has_contact_access(from_user: str, to_user: str) -> bool:
    """True if from_user's contact request to to_user was approved."""
    with Session() as s:
        return s.query(ContactRequest).filter_by(
            from_user=from_user, to_user=to_user, status='approved'
        ).first() is not None


def get_contact_requests_approved(username: str) -> list:
    """Approved contact requests where I am the target (people I've shared my contact with)."""
    with Session() as s:
        rows = s.query(ContactRequest).filter_by(to_user=username, status='approved') \
                                      .order_by(ContactRequest.created_at.desc()).all()
        return [_contact_req_to_dict(r) for r in rows]


def revoke_contact_access(req_id: int, to_user: str) -> bool:
    """Revoke an approved contact request (set status back to declined)."""
    with Session() as s:
        req = s.query(ContactRequest).filter_by(id=req_id, to_user=to_user, status='approved').first()
        if not req:
            return False
        req.status = 'declined'
        s.commit()
        return True


# ── Chat read / unread helpers ─────────────────────────────────────────────────

def mark_chat_read(username: str, room_key: str) -> None:
    """Upsert the last-read timestamp for this user+room."""
    with Session() as s:
        row = s.query(ChatRead).filter_by(username=username, room_key=room_key).first()
        if row:
            row.read_at = datetime.utcnow()
        else:
            s.add(ChatRead(username=username, room_key=room_key))
        s.commit()


def get_unread_counts(username: str) -> dict:
    """Return {friend_username: unread_count} for all friends with unread messages."""
    with Session() as s:
        friends = get_friends(username)
        result = {}
        for friend in friends:
            ua, ub    = _friend_pair(username, friend)
            room_key  = f'{ua}:{ub}'
            read_row  = s.query(ChatRead).filter_by(username=username, room_key=room_key).first()
            last_read = read_row.read_at if read_row else datetime(1970, 1, 1)
            count = s.query(func.count(PrivateChatMessage.id)).filter(
                PrivateChatMessage.room_key == room_key,
                PrivateChatMessage.sender   == friend,
                PrivateChatMessage.created_at > last_read,
            ).scalar() or 0
            if count > 0:
                result[friend] = count
        return result
