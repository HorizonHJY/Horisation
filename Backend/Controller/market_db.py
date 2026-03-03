"""
market_db.py
SQLAlchemy models and helper functions for the marketplace feature.
Database: _data/market.db (SQLite, auto-created on init_db()).
Designed for easy migration to PostgreSQL: swap the engine URL only.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    create_engine, Column, String, Text, Float, Integer, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH  = os.path.join(BASE_DIR, '_data', 'market.db')

engine  = create_engine(f'sqlite:///{DB_PATH}', echo=False)
Session = sessionmaker(bind=engine)
Base    = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class Listing(Base):
    __tablename__ = 'listings'

    id               = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    seller_username  = Column(String(100), nullable=False, index=True)
    title            = Column(String(100), nullable=False)
    description      = Column(Text, nullable=False)
    price            = Column(Float, nullable=False)
    category         = Column(String(50), nullable=False, default='other')
    contact          = Column(String(200), nullable=False)
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


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist. Called once at app startup."""
    Base.metadata.create_all(engine)


# ── Serialisers ───────────────────────────────────────────────────────────────

def _listing_to_dict(listing: Listing) -> dict:
    return {
        'id':              listing.id,
        'seller_username': listing.seller_username,
        'title':           listing.title,
        'description':     listing.description,
        'price':           listing.price,
        'category':        listing.category,
        'contact':         listing.contact,
        'status':          listing.status,
        'created_at':      listing.created_at.isoformat(),
        'updated_at':      listing.updated_at.isoformat(),
        'images':          [{'id': img.id, 'url': img.r2_url, 'order': img.display_order}
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


def get_all_listings(status: str = 'active') -> list[dict]:
    with Session() as s:
        rows = s.query(Listing).filter_by(status=status)\
                               .order_by(Listing.created_at.desc()).all()
        return [_listing_to_dict(r) for r in rows]


def get_listing(listing_id: str) -> Optional[dict]:
    with Session() as s:
        row = s.query(Listing).filter_by(id=listing_id).first()
        return _listing_to_dict(row) if row else None


def get_my_listings(username: str) -> list[dict]:
    with Session() as s:
        rows = s.query(Listing).filter_by(seller_username=username)\
                               .order_by(Listing.created_at.desc()).all()
        return [_listing_to_dict(r) for r in rows]


def create_listing(seller: str, title: str, description: str,
                   price: float, category: str, contact: str) -> str:
    """Create a listing row and return its id."""
    listing = Listing(
        id=str(uuid.uuid4()),
        seller_username=seller,
        title=title,
        description=description,
        price=price,
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
