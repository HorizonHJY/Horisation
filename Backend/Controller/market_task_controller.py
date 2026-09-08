"""
market_task_controller.py
Flask Blueprint for the Task/Bounty board — a place to post errands,
grocery runs, airport pickups, and other tasks with a bounty (悬赏金额).
"""

from flask import Blueprint, request, jsonify

from Backend.Controller.auth_controller import login_required
from Backend.Controller.market_task_db import (
    get_all_tasks, get_task, get_my_tasks,
    get_tasks_by_category, get_tasks_by_user,
    create_task, update_task, delete_task_db,
    mark_task_in_progress, mark_task_completed,
    get_all_categories, get_task_statistics,
)

task_bp = Blueprint('market_tasks', __name__, url_prefix='/api/market/tasks')

TASK_CATEGORIES = [
    {'slug': 'grocery',      'label': '逛超市',   'icon': 'fa-shopping-basket',  'color': '#27ae60'},
    {'slug': 'airport',      'label': '接送机',   'icon': 'fa-plane-departure',  'color': '#2d7dd2'},
    {'slug': 'delivery',     'label': '跑腿代取',  'icon': 'fa-box',              'color': '#e67e22'},
    {'slug': 'pet',          'label': '遛狗/宠物',  'icon': 'fa-paw',             'color': '#8e44ad'},
    {'slug': 'moving',       'label': '搬家搬运',  'icon': 'fa-truck-moving',     'color': '#c0392b'},
    {'slug': 'tech_support', 'label': '技术支援',  'icon': 'fa-wrench',           'color': '#2c3e50'},
    {'slug': 'tutoring',     'label': '辅导/教学',  'icon': 'fa-chalkboard-teacher','color': '#16a085'},
    {'slug': 'other',        'label': '其他',      'icon': 'fa-ellipsis-h',       'color': '#7f8c8d'},
]


# ── Categories ───────────────────────────────────────────────────────────────

@task_bp.route('/categories', methods=['GET'])
@login_required
def list_categories():
    """Return available task categories."""
    return jsonify({'ok': True, 'categories': TASK_CATEGORIES})


# ── List / Browse ────────────────────────────────────────────────────────────

@task_bp.route('/', methods=['GET'])
@login_required
def list_tasks():
    """Return open tasks, optionally filtered by category."""
    category = request.args.get('category', '').strip()
    search   = request.args.get('search', '').strip()
    tasks = get_all_tasks(category=category, search=search)
    return jsonify({'ok': True, 'tasks': tasks})


@task_bp.route('/<task_id>', methods=['GET'])
@login_required
def get_task_route(task_id):
    task = get_task(task_id)
    if not task:
        return jsonify({'ok': False, 'error': 'Task not found.'}), 404
    return jsonify({'ok': True, 'task': task})


@task_bp.route('/my', methods=['GET'])
@login_required
def my_tasks():
    """Tasks posted by the current user."""
    username = request.current_user['username']
    tasks = get_my_tasks(username)
    return jsonify({'ok': True, 'tasks': tasks})


@task_bp.route('/user/<username>', methods=['GET'])
@login_required
def user_tasks(username):
    """Open tasks by a specific user."""
    tasks = get_tasks_by_user(username)
    return jsonify({'ok': True, 'tasks': tasks})


# ── Create ──────────────────────────────────────────────────────────────────

@task_bp.route('/', methods=['POST'])
@login_required
def create_task_route():
    username = request.current_user['username']

    data = request.get_json() or {}

    title       = str(data.get('title', '')).strip()
    description = str(data.get('description', '')).strip()
    category    = str(data.get('category', 'other')).strip()
    location    = str(data.get('location', '')).strip()
    due_date    = str(data.get('due_date', '')).strip()

    try:
        bounty = float(data.get('bounty', 0))
    except (ValueError, TypeError):
        return jsonify({'ok': False, 'error': 'Bounty must be a number.'}), 400

    if not title:
        return jsonify({'ok': False, 'error': 'Title is required.'}), 400
    if not description:
        return jsonify({'ok': False, 'error': 'Description is required.'}), 400
    if bounty < 0:
        return jsonify({'ok': False, 'error': 'Bounty cannot be negative.'}), 400

    valid_slugs = {c['slug'] for c in TASK_CATEGORIES}
    if category not in valid_slugs:
        return jsonify({'ok': False, 'error': f'Invalid category. Must be one of: {", ".join(sorted(valid_slugs))}'}), 400

    task = create_task(
        poster_username=username,
        title=title,
        description=description,
        category=category,
        bounty=bounty,
        location=location or None,
        due_date=due_date or None,
    )
    return jsonify({'ok': True, 'task': task}), 201


# ── Update ──────────────────────────────────────────────────────────────────

@task_bp.route('/<task_id>', methods=['PUT'])
@login_required
def update_task_route(task_id):
    username = request.current_user['username']
    data     = request.get_json() or {}

    fields = {}
    for key in ('title', 'description', 'category', 'location', 'due_date'):
        if key in data:
            val = str(data[key]).strip()
            if val:
                fields[key] = val

    if 'bounty' in data:
        try:
            fields['bounty'] = float(data['bounty'])
        except (ValueError, TypeError):
            return jsonify({'ok': False, 'error': 'Bounty must be a number.'}), 400

    task = update_task(task_id, username, **fields)
    if not task:
        return jsonify({'ok': False, 'error': 'Task not found or permission denied.'}), 404
    return jsonify({'ok': True, 'task': task})


# ── Status changes ──────────────────────────────────────────────────────────

@task_bp.route('/<task_id>/in-progress', methods=['POST'])
@login_required
def mark_in_progress(task_id):
    """Mark a task as in-progress (assigned/taken). Poster sets this when someone takes the task."""
    username = request.current_user['username']
    ok = mark_task_in_progress(task_id, username)
    if not ok:
        return jsonify({'ok': False, 'error': 'Task not found or permission denied.'}), 404
    return jsonify({'ok': True})


@task_bp.route('/<task_id>/complete', methods=['POST'])
@login_required
def mark_completed(task_id):
    """Mark a task as completed (poster confirms)."""
    username = request.current_user['username']
    ok = mark_task_completed(task_id, username)
    if not ok:
        return jsonify({'ok': False, 'error': 'Task not found or permission denied.'}), 404
    return jsonify({'ok': True})


# ── Delete ──────────────────────────────────────────────────────────────────

@task_bp.route('/<task_id>', methods=['DELETE'])
@login_required
def delete_task(task_id):
    username = request.current_user['username']
    ok = delete_task_db(task_id, username)
    if not ok:
        return jsonify({'ok': False, 'error': 'Task not found or permission denied.'}), 404
    return jsonify({'ok': True})


# ── Statistics ──────────────────────────────────────────────────────────────

@task_bp.route('/stats', methods=['GET'])
@login_required
def task_stats():
    username = request.current_user['username']
    stats = get_task_statistics(username)
    return jsonify({'ok': True, 'stats': stats})
