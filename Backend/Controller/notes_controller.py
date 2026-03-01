# Backend/Controller/notes_controller.py
"""
笔记控制器
处理笔记和日记相关的API接口
"""

from flask import Blueprint, request, jsonify
from .auth_controller import login_required
from .notes_manager import notes_manager

# 创建笔记蓝图
notes_bp = Blueprint('notes', __name__, url_prefix='/api/notes')

@notes_bp.route('/', methods=['GET'])
@login_required
def list_notes():
    """列出用户笔记"""
    try:
        username = request.current_user['username']

        # 获取查询参数
        category = request.args.get('category')
        search_query = request.args.get('search')
        tags = request.args.getlist('tags')
        is_favorite = request.args.get('is_favorite')
        is_archived = request.args.get('is_archived', 'false').lower() == 'true'
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        # 转换is_favorite参数
        if is_favorite is not None:
            is_favorite = is_favorite.lower() == 'true'

        # 获取笔记列表
        result = notes_manager.list_notes(
            username=username,
            category=category,
            search_query=search_query,
            tags=tags,
            is_favorite=is_favorite,
            is_archived=is_archived,
            limit=limit,
            offset=offset
        )

        return jsonify({
            'ok': True,
            **result
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to list notes: {str(e)}'}), 500

@notes_bp.route('/', methods=['POST'])
@login_required
def create_note():
    """创建新笔记"""
    try:
        username = request.current_user['username']
        data = request.get_json()

        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        title = data.get('title', '').strip()
        content = data.get('content', '')
        category = data.get('category', '日记')
        tags = data.get('tags', [])

        if not title:
            return jsonify({'ok': False, 'error': 'Title is required'}), 400

        success, message, note_id = notes_manager.create_note(
            username=username,
            title=title,
            content=content,
            category=category,
            tags=tags
        )

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({
            'ok': True,
            'message': message,
            'note_id': note_id
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to create note: {str(e)}'}), 500

@notes_bp.route('/<note_id>', methods=['GET'])
@login_required
def get_note(note_id):
    """获取单个笔记"""
    try:
        username = request.current_user['username']
        note = notes_manager.get_note(username, note_id)

        if not note:
            return jsonify({'ok': False, 'error': 'Note not found'}), 404

        return jsonify({
            'ok': True,
            'note': note
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to get note: {str(e)}'}), 500

@notes_bp.route('/<note_id>', methods=['PUT'])
@login_required
def update_note(note_id):
    """更新笔记"""
    try:
        username = request.current_user['username']
        data = request.get_json()

        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        title = data.get('title')
        content = data.get('content')
        category = data.get('category')
        tags = data.get('tags')

        success, message = notes_manager.update_note(
            username=username,
            note_id=note_id,
            title=title,
            content=content,
            category=category,
            tags=tags
        )

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({
            'ok': True,
            'message': message
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to update note: {str(e)}'}), 500

@notes_bp.route('/<note_id>', methods=['DELETE'])
@login_required
def delete_note(note_id):
    """删除笔记"""
    try:
        username = request.current_user['username']

        success, message = notes_manager.delete_note(username, note_id)

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({
            'ok': True,
            'message': message
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to delete note: {str(e)}'}), 500

@notes_bp.route('/<note_id>/favorite', methods=['POST'])
@login_required
def toggle_favorite(note_id):
    """切换笔记收藏状态"""
    try:
        username = request.current_user['username']

        success, message, is_favorite = notes_manager.toggle_favorite(username, note_id)

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({
            'ok': True,
            'message': message,
            'is_favorite': is_favorite
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to toggle favorite: {str(e)}'}), 500

@notes_bp.route('/<note_id>/archive', methods=['POST'])
@login_required
def toggle_archive(note_id):
    """切换笔记归档状态"""
    try:
        username = request.current_user['username']

        success, message, is_archived = notes_manager.toggle_archive(username, note_id)

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({
            'ok': True,
            'message': message,
            'is_archived': is_archived
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to toggle archive: {str(e)}'}), 500

@notes_bp.route('/categories', methods=['GET'])
@login_required
def get_categories():
    """获取用户分类列表"""
    try:
        username = request.current_user['username']
        result = notes_manager.list_notes(username, limit=0)  # 只获取分类信息

        return jsonify({
            'ok': True,
            'categories': result['categories']
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to get categories: {str(e)}'}), 500

@notes_bp.route('/categories', methods=['POST'])
@login_required
def add_category():
    """添加新分类"""
    try:
        username = request.current_user['username']
        data = request.get_json()

        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        category = data.get('category', '').strip()

        if not category:
            return jsonify({'ok': False, 'error': 'Category name is required'}), 400

        success, message = notes_manager.add_category(username, category)

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({
            'ok': True,
            'message': message
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to add category: {str(e)}'}), 500

@notes_bp.route('/categories/<category>', methods=['DELETE'])
@login_required
def remove_category(category):
    """删除分类"""
    try:
        username = request.current_user['username']

        success, message = notes_manager.remove_category(username, category)

        if not success:
            return jsonify({'ok': False, 'error': message}), 400

        return jsonify({
            'ok': True,
            'message': message
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to remove category: {str(e)}'}), 500

@notes_bp.route('/statistics', methods=['GET'])
@login_required
def get_statistics():
    """获取笔记统计信息"""
    try:
        username = request.current_user['username']
        stats = notes_manager.get_statistics(username)

        return jsonify({
            'ok': True,
            'statistics': stats
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to get statistics: {str(e)}'}), 500