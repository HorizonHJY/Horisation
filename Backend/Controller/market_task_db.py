"""
market_task_db.py
SQLAlchemy model + helpers for the Task/Bounty board.
Database lives in the same _data/market.db as the rest of the market.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, String, Text, Float, DateTime, inspect

from Backend.Controller.market_db import Base, engine, Session, User


# ── Task Model ────────────────────────────────────────────────────────────────

class Task(Base):
    __tablename__ = 'tasks'

    id              = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    poster_username = Column(String(100), nullable=False, index=True)
    title           = Column(String(200), nullable=False)
    description     = Column(Text,       nullable=False)
    category        = Column(String(50), nullable=False, default='other')
    bounty          = Column(Float,      nullable=False, default=0)
    location        = Column(String(200), nullable=True)
    due_date        = Column(String(50),  nullable=True)
    status          = Column(String(20), nullable=False, default='open')  # open / in_progress / completed / cancelled
    claimed_by      = Column(String(100), nullable=True)
    created_at      = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at      = Column(DateTime, nullable=False,
                              default=lambda: datetime.now(timezone.utc),
                              onupdate=lambda: datetime.now(timezone.utc))


def init_task_db():
    """Create tasks table if it doesn't exist."""
    inspector = inspect(engine)
    if 'tasks' not in inspector.get_table_names():
        Base.metadata.create_all(engine, tables=[Task.__table__])
        print('✅ Created tasks table')
    else:
        _migrate_task_columns()


def _migrate_task_columns():
    """Idempotently add missing columns."""
    stmts = [
        "ALTER TABLE tasks ADD COLUMN location TEXT",
        "ALTER TABLE tasks ADD COLUMN claimed_by TEXT",
    ]
    with Session() as s:
        for stmt in stmts:
            try:
                s.execute(stmt)
                s.commit()
            except Exception:
                pass  # column already exists


# ── Task helpers ──────────────────────────────────────────────────────────────

def _task_to_dict(task: Task, poster: Optional[User] = None) -> dict:
    return {
        'id':              task.id,
        'poster_username': task.poster_username,
        'poster_display':  (poster.display_name or task.poster_username) if poster else task.poster_username,
        'poster_avatar':   poster.avatar_url if poster else None,
        'title':           task.title,
        'description':     task.description,
        'category':        task.category,
        'bounty':          task.bounty,
        'location':        task.location or '',
        'due_date':        task.due_date or '',
        'status':          task.status,
        'claimed_by':      task.claimed_by,
        'created_at':      task.created_at.isoformat(),
        'updated_at':      task.updated_at.isoformat(),
    }


def _enrich_tasks(tasks: list) -> list[dict]:
    """Batch-fetch poster User rows and attach display info."""
    if not tasks:
        return []
    usernames = list({t.poster_username for t in tasks})
    with Session() as s:
        posters = {u.username: u for u in s.query(User).filter(User.username.in_(usernames)).all()}
    return [_task_to_dict(t, posters.get(t.poster_username)) for t in tasks]


def get_all_tasks(category: str = '', search: str = '') -> list[dict]:
    """Return all open tasks, optionally filtered by category and/or search."""
    with Session() as s:
        q = s.query(Task).filter(Task.status == 'open')
        if category:
            q = q.filter(Task.category == category)
        if search:
            pattern = f'%{search}%'
            q = q.filter(
                Task.title.ilike(pattern) |
                Task.description.ilike(pattern)
            )
        rows = q.order_by(Task.created_at.desc()).all()
        return _enrich_tasks(rows)


def get_task(task_id: str) -> Optional[dict]:
    with Session() as s:
        row = s.query(Task).filter_by(id=task_id).first()
        if not row:
            return None
        poster = s.query(User).filter_by(username=row.poster_username).first()
        return _task_to_dict(row, poster)


def get_my_tasks(username: str) -> list[dict]:
    """All tasks by the current user (any status)."""
    with Session() as s:
        rows = s.query(Task).filter_by(poster_username=username)\
                            .order_by(Task.created_at.desc()).all()
        return _enrich_tasks(rows)


def get_tasks_by_category(category: str) -> list[dict]:
    """Open tasks in a specific category."""
    with Session() as s:
        rows = s.query(Task).filter_by(category=category, status='open')\
                            .order_by(Task.created_at.desc()).all()
        return _enrich_tasks(rows)


def get_tasks_by_user(username: str) -> list[dict]:
    """Open tasks by a specific poster."""
    with Session() as s:
        rows = s.query(Task).filter_by(poster_username=username, status='open')\
                            .order_by(Task.created_at.desc()).all()
        return _enrich_tasks(rows)


def create_task(poster_username: str, title: str, description: str,
                category: str, bounty: float,
                location: Optional[str] = None,
                due_date: Optional[str] = None) -> dict:
    task = Task(
        id=str(uuid.uuid4()),
        poster_username=poster_username,
        title=title,
        description=description,
        category=category,
        bounty=bounty,
        location=location,
        due_date=due_date,
    )
    with Session() as s:
        s.add(task)
        s.commit()
        s.refresh(task)
        poster = s.query(User).filter_by(username=poster_username).first()
        return _task_to_dict(task, poster)


def update_task(task_id: str, username: str, **fields) -> Optional[dict]:
    """Update allowed fields. Only the poster can update. Returns updated task or None."""
    allowed = {'title', 'description', 'category', 'bounty', 'location', 'due_date'}
    with Session() as s:
        row = s.query(Task).filter_by(id=task_id, poster_username=username).first()
        if not row:
            return None
        for key, val in fields.items():
            if key in allowed:
                setattr(row, key, val)
        row.updated_at = datetime.now(timezone.utc)
        s.commit()
        poster = s.query(User).filter_by(username=username).first()
        return _task_to_dict(row, poster)


def delete_task_db(task_id: str, username: str) -> bool:
    """Delete a task. Only the poster can delete. Returns True on success."""
    with Session() as s:
        row = s.query(Task).filter_by(id=task_id, poster_username=username).first()
        if not row:
            return False
        s.delete(row)
        s.commit()
        return True


def mark_task_in_progress(task_id: str, username: str) -> bool:
    """Set status to 'in_progress'. Only the poster can do this."""
    with Session() as s:
        row = s.query(Task).filter_by(id=task_id, poster_username=username, status='open').first()
        if not row:
            return False
        row.status     = 'in_progress'
        row.updated_at = datetime.now(timezone.utc)
        s.commit()
        return True


def mark_task_completed(task_id: str, username: str) -> bool:
    """Set status to 'completed'. Only the poster can do this."""
    with Session() as s:
        row = s.query(Task).filter_by(id=task_id, poster_username=username).first()
        if not row:
            return False
        if row.status not in ('open', 'in_progress'):
            return False
        row.status     = 'completed'
        row.updated_at = datetime.now(timezone.utc)
        s.commit()
        return True


def get_all_categories() -> list:
    """Return hardcoded task categories (matching the controller)."""
    return [
        {'slug': 'grocery',      'label': '逛超市',   'icon': 'fa-shopping-basket'},
        {'slug': 'airport',      'label': '接送机',   'icon': 'fa-plane-departure'},
        {'slug': 'delivery',     'label': '跑腿代取',  'icon': 'fa-box'},
        {'slug': 'pet',          'label': '遛狗/宠物',  'icon': 'fa-paw'},
        {'slug': 'moving',       'label': '搬家搬运',  'icon': 'fa-truck-moving'},
        {'slug': 'tech_support', 'label': '技术支援',  'icon': 'fa-wrench'},
        {'slug': 'tutoring',     'label': '辅导/教学',  'icon': 'fa-chalkboard-teacher'},
        {'slug': 'other',        'label': '其他',      'icon': 'fa-ellipsis-h'},
    ]


def get_task_statistics(username: str) -> dict:
    """Return various stats about the user's tasks."""
    with Session() as s:
        rows = s.query(Task).filter_by(poster_username=username).all()
        total   = len(rows)
        open_count    = sum(1 for r in rows if r.status == 'open')
        in_progress   = sum(1 for r in rows if r.status == 'in_progress')
        completed     = sum(1 for r in rows if r.status == 'completed')
        cancelled     = sum(1 for r in rows if r.status == 'cancelled')
        total_bounty  = sum(r.bounty for r in rows if r.status == 'completed')
        return {
            'total':         total,
            'open':          open_count,
            'in_progress':   in_progress,
            'completed':     completed,
            'cancelled':     cancelled,
            'total_paid':    round(total_bounty, 2),
        }
