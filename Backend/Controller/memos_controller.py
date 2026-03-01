# Backend/Controller/memos_controller.py
"""
备忘录控制器 - 更新的Hormemo功能
支持按用户隔离的备忘录管理
"""

from flask import Blueprint, request, jsonify
from .auth_controller import login_required
from .user_manager import user_manager
from datetime import datetime
import uuid

# 创建备忘录蓝图
memos_bp = Blueprint('memos', __name__, url_prefix='/api/memos')

@memos_bp.route('/', methods=['GET'])
@login_required
def list_memos():
    """列出用户备忘录"""
    try:
        username = request.current_user['username']

        # 获取查询参数
        status = request.args.get('status')
        memo_type = request.args.get('type')
        priority = request.args.get('priority')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        # 获取用户备忘录
        users = user_manager._load_users()
        if username not in users:
            return jsonify({'ok': False, 'error': 'User not found'}), 404

        user = users[username]
        memos = user.get('memos', [])

        # 过滤条件
        if status:
            memos = [memo for memo in memos if memo.get('status') == status]
        if memo_type:
            memos = [memo for memo in memos if memo.get('type') == memo_type]
        if priority:
            memos = [memo for memo in memos if memo.get('priority') == priority]

        # 按创建时间倒序排序
        memos.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # 分页
        total_count = len(memos)
        memos = memos[offset:offset + limit]

        return jsonify({
            'ok': True,
            'memos': memos,
            'total_count': total_count
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to list memos: {str(e)}'}), 500

@memos_bp.route('/', methods=['POST'])
@login_required
def create_memo():
    """创建备忘录"""
    try:
        username = request.current_user['username']
        data = request.get_json()

        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        content = data.get('content', '').strip()
        if not content:
            return jsonify({'ok': False, 'error': 'Content is required'}), 400

        memo_type = data.get('type', 'general')
        tags = data.get('tags', [])
        priority = data.get('priority', 'normal')

        # 获取用户数据
        users = user_manager._load_users()
        if username not in users:
            return jsonify({'ok': False, 'error': 'User not found'}), 404

        user = users[username]
        if 'memos' not in user:
            user['memos'] = []

        # 生成备忘录ID
        memo_id = str(uuid.uuid4())

        # 创建备忘录
        memo = {
            'id': memo_id,
            'content': content,
            'type': memo_type,
            'tags': tags,
            'priority': priority,
            'status': 'active',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'due_date': data.get('due_date'),
            'completed_at': None
        }

        user['memos'].append(memo)
        user_manager._save_users(users)

        return jsonify({
            'ok': True,
            'message': 'Memo created successfully',
            'memo_id': memo_id
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to create memo: {str(e)}'}), 500

@memos_bp.route('/<memo_id>', methods=['GET'])
@login_required
def get_memo(memo_id):
    """获取指定备忘录"""
    try:
        username = request.current_user['username']

        users = user_manager._load_users()
        if username not in users:
            return jsonify({'ok': False, 'error': 'User not found'}), 404

        user = users[username]
        memos = user.get('memos', [])

        memo = next((memo for memo in memos if memo['id'] == memo_id), None)
        if not memo:
            return jsonify({'ok': False, 'error': 'Memo not found'}), 404

        return jsonify({
            'ok': True,
            'memo': memo
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to get memo: {str(e)}'}), 500

@memos_bp.route('/<memo_id>', methods=['PUT'])
@login_required
def update_memo(memo_id):
    """更新备忘录"""
    try:
        username = request.current_user['username']
        data = request.get_json()

        if not data:
            return jsonify({'ok': False, 'error': 'Invalid JSON data'}), 400

        users = user_manager._load_users()
        if username not in users:
            return jsonify({'ok': False, 'error': 'User not found'}), 404

        user = users[username]
        memos = user.get('memos', [])

        memo = next((memo for memo in memos if memo['id'] == memo_id), None)
        if not memo:
            return jsonify({'ok': False, 'error': 'Memo not found'}), 404

        # 更新字段
        if 'content' in data:
            memo['content'] = data['content']
        if 'status' in data:
            memo['status'] = data['status']
            if data['status'] == 'completed' and memo.get('completed_at') is None:
                memo['completed_at'] = datetime.now().isoformat()
        if 'priority' in data:
            memo['priority'] = data['priority']
        if 'tags' in data:
            memo['tags'] = data['tags']
        if 'due_date' in data:
            memo['due_date'] = data['due_date']

        memo['updated_at'] = datetime.now().isoformat()

        user_manager._save_users(users)

        return jsonify({
            'ok': True,
            'message': 'Memo updated successfully'
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to update memo: {str(e)}'}), 500

@memos_bp.route('/<memo_id>', methods=['DELETE'])
@login_required
def delete_memo(memo_id):
    """删除备忘录"""
    try:
        username = request.current_user['username']

        users = user_manager._load_users()
        if username not in users:
            return jsonify({'ok': False, 'error': 'User not found'}), 404

        user = users[username]
        memos = user.get('memos', [])

        memo_index = next((i for i, memo in enumerate(memos) if memo['id'] == memo_id), None)
        if memo_index is None:
            return jsonify({'ok': False, 'error': 'Memo not found'}), 404

        del memos[memo_index]
        user_manager._save_users(users)

        return jsonify({
            'ok': True,
            'message': 'Memo deleted successfully'
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to delete memo: {str(e)}'}), 500

@memos_bp.route('/statistics', methods=['GET'])
@login_required
def get_memo_statistics():
    """获取备忘录统计信息"""
    try:
        username = request.current_user['username']

        users = user_manager._load_users()
        if username not in users:
            return jsonify({'ok': False, 'error': 'User not found'}), 404

        user = users[username]
        memos = user.get('memos', [])

        # 统计信息
        total_memos = len(memos)
        status_stats = {}
        priority_stats = {}
        type_stats = {}

        for memo in memos:
            # 按状态统计
            status = memo.get('status', 'active')
            status_stats[status] = status_stats.get(status, 0) + 1

            # 按优先级统计
            priority = memo.get('priority', 'normal')
            priority_stats[priority] = priority_stats.get(priority, 0) + 1

            # 按类型统计
            memo_type = memo.get('type', 'general')
            type_stats[memo_type] = type_stats.get(memo_type, 0) + 1

        return jsonify({
            'ok': True,
            'statistics': {
                'total_memos': total_memos,
                'status_stats': status_stats,
                'priority_stats': priority_stats,
                'type_stats': type_stats
            }
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to get statistics: {str(e)}'}), 500

@memos_bp.route('/<memo_id>/complete', methods=['POST'])
@login_required
def complete_memo(memo_id):
    """标记备忘录为完成"""
    try:
        username = request.current_user['username']

        users = user_manager._load_users()
        if username not in users:
            return jsonify({'ok': False, 'error': 'User not found'}), 404

        user = users[username]
        memos = user.get('memos', [])

        memo = next((memo for memo in memos if memo['id'] == memo_id), None)
        if not memo:
            return jsonify({'ok': False, 'error': 'Memo not found'}), 404

        memo['status'] = 'completed'
        memo['completed_at'] = datetime.now().isoformat()
        memo['updated_at'] = datetime.now().isoformat()

        user_manager._save_users(users)

        return jsonify({
            'ok': True,
            'message': 'Memo marked as completed'
        })

    except Exception as e:
        return jsonify({'ok': False, 'error': f'Failed to complete memo: {str(e)}'}), 500