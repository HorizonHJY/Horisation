# Backend/Controller/notes_manager.py
"""
私人笔记和日记管理系统
每个用户拥有独立的笔记空间，完全私密
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

class NotesManager:
    """笔记管理类"""

    def __init__(self, data_dir: str = "_data"):
        """初始化笔记管理器"""
        self.data_dir = data_dir
        self.notes_dir = os.path.join(data_dir, "notes")

        # 确保笔记目录存在
        os.makedirs(self.notes_dir, exist_ok=True)

    def _get_user_notes_file(self, username: str) -> str:
        """获取用户笔记文件路径"""
        return os.path.join(self.notes_dir, f"{username}_notes.json")

    def _load_user_notes(self, username: str) -> Dict:
        """加载用户笔记数据"""
        notes_file = self._get_user_notes_file(username)
        try:
            with open(notes_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                'notes': {},
                'categories': ['日记', '工作笔记', '想法记录', '待办事项'],
                'settings': {
                    'default_category': '日记',
                    'auto_save': True,
                    'created_at': datetime.now().isoformat()
                }
            }

    def _save_user_notes(self, username: str, notes_data: Dict):
        """保存用户笔记数据"""
        notes_file = self._get_user_notes_file(username)
        with open(notes_file, 'w', encoding='utf-8') as f:
            json.dump(notes_data, f, ensure_ascii=False, indent=2)

    def create_note(self, username: str, title: str, content: str = "",
                   category: str = "日记", tags: List[str] = None) -> Tuple[bool, str, Optional[str]]:
        """创建新笔记"""
        try:
            notes_data = self._load_user_notes(username)

            # 生成唯一ID
            note_id = str(uuid.uuid4())

            # 创建笔记对象
            note = {
                'id': note_id,
                'title': title.strip(),
                'content': content,
                'category': category,
                'tags': tags or [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'word_count': len(content),
                'is_favorite': False,
                'is_archived': False
            }

            # 保存笔记
            notes_data['notes'][note_id] = note

            # 如果分类不存在，添加到分类列表
            if category not in notes_data['categories']:
                notes_data['categories'].append(category)

            self._save_user_notes(username, notes_data)

            return True, "Note created successfully", note_id

        except Exception as e:
            return False, f"Failed to create note: {str(e)}", None

    def update_note(self, username: str, note_id: str, title: str = None,
                   content: str = None, category: str = None,
                   tags: List[str] = None) -> Tuple[bool, str]:
        """更新笔记"""
        try:
            notes_data = self._load_user_notes(username)

            if note_id not in notes_data['notes']:
                return False, "Note not found"

            note = notes_data['notes'][note_id]

            # 更新字段
            if title is not None:
                note['title'] = title.strip()
            if content is not None:
                note['content'] = content
                note['word_count'] = len(content)
            if category is not None:
                note['category'] = category
                # 如果分类不存在，添加到分类列表
                if category not in notes_data['categories']:
                    notes_data['categories'].append(category)
            if tags is not None:
                note['tags'] = tags

            note['updated_at'] = datetime.now().isoformat()

            self._save_user_notes(username, notes_data)

            return True, "Note updated successfully"

        except Exception as e:
            return False, f"Failed to update note: {str(e)}"

    def delete_note(self, username: str, note_id: str) -> Tuple[bool, str]:
        """删除笔记"""
        try:
            notes_data = self._load_user_notes(username)

            if note_id not in notes_data['notes']:
                return False, "Note not found"

            del notes_data['notes'][note_id]
            self._save_user_notes(username, notes_data)

            return True, "Note deleted successfully"

        except Exception as e:
            return False, f"Failed to delete note: {str(e)}"

    def get_note(self, username: str, note_id: str) -> Optional[Dict]:
        """获取单个笔记"""
        try:
            notes_data = self._load_user_notes(username)
            return notes_data['notes'].get(note_id)
        except Exception:
            return None

    def list_notes(self, username: str, category: str = None,
                  search_query: str = None, tags: List[str] = None,
                  is_favorite: bool = None, is_archived: bool = False,
                  limit: int = 50, offset: int = 0) -> Dict:
        """列出用户笔记"""
        try:
            notes_data = self._load_user_notes(username)
            notes = list(notes_data['notes'].values())

            # 过滤条件
            if category:
                notes = [note for note in notes if note['category'] == category]

            if search_query:
                query = search_query.lower()
                notes = [note for note in notes
                        if query in note['title'].lower() or query in note['content'].lower()]

            if tags:
                notes = [note for note in notes
                        if any(tag in note['tags'] for tag in tags)]

            if is_favorite is not None:
                notes = [note for note in notes if note.get('is_favorite', False) == is_favorite]

            notes = [note for note in notes if note.get('is_archived', False) == is_archived]

            # 按更新时间排序（最新的在前）
            notes.sort(key=lambda x: x['updated_at'], reverse=True)

            # 分页
            total_count = len(notes)
            notes = notes[offset:offset + limit]

            return {
                'notes': notes,
                'total_count': total_count,
                'categories': notes_data['categories'],
                'settings': notes_data['settings']
            }

        except Exception as e:
            return {
                'notes': [],
                'total_count': 0,
                'categories': ['日记'],
                'settings': {},
                'error': str(e)
            }

    def toggle_favorite(self, username: str, note_id: str) -> Tuple[bool, str, bool]:
        """切换笔记收藏状态"""
        try:
            notes_data = self._load_user_notes(username)

            if note_id not in notes_data['notes']:
                return False, "Note not found", False

            note = notes_data['notes'][note_id]
            note['is_favorite'] = not note.get('is_favorite', False)
            note['updated_at'] = datetime.now().isoformat()

            self._save_user_notes(username, notes_data)

            return True, "Favorite status updated", note['is_favorite']

        except Exception as e:
            return False, f"Failed to update favorite: {str(e)}", False

    def toggle_archive(self, username: str, note_id: str) -> Tuple[bool, str, bool]:
        """切换笔记归档状态"""
        try:
            notes_data = self._load_user_notes(username)

            if note_id not in notes_data['notes']:
                return False, "Note not found", False

            note = notes_data['notes'][note_id]
            note['is_archived'] = not note.get('is_archived', False)
            note['updated_at'] = datetime.now().isoformat()

            self._save_user_notes(username, notes_data)

            return True, "Archive status updated", note['is_archived']

        except Exception as e:
            return False, f"Failed to update archive: {str(e)}", False

    def add_category(self, username: str, category: str) -> Tuple[bool, str]:
        """添加新分类"""
        try:
            notes_data = self._load_user_notes(username)

            category = category.strip()
            if not category:
                return False, "Category name cannot be empty"

            if category in notes_data['categories']:
                return False, "Category already exists"

            notes_data['categories'].append(category)
            self._save_user_notes(username, notes_data)

            return True, f"Category '{category}' added successfully"

        except Exception as e:
            return False, f"Failed to add category: {str(e)}"

    def remove_category(self, username: str, category: str) -> Tuple[bool, str]:
        """删除分类"""
        try:
            notes_data = self._load_user_notes(username)

            if category not in notes_data['categories']:
                return False, "Category not found"

            # 检查是否有笔记使用这个分类
            notes_using_category = [note for note in notes_data['notes'].values()
                                  if note['category'] == category]

            if notes_using_category:
                return False, f"Cannot delete category: {len(notes_using_category)} notes are using it"

            notes_data['categories'].remove(category)
            self._save_user_notes(username, notes_data)

            return True, f"Category '{category}' removed successfully"

        except Exception as e:
            return False, f"Failed to remove category: {str(e)}"

    def get_statistics(self, username: str) -> Dict:
        """获取用户笔记统计信息"""
        try:
            notes_data = self._load_user_notes(username)
            notes = list(notes_data['notes'].values())

            # 基础统计
            total_notes = len(notes)
            total_words = sum(note['word_count'] for note in notes)
            favorites_count = len([note for note in notes if note.get('is_favorite', False)])
            archived_count = len([note for note in notes if note.get('is_archived', False)])

            # 按分类统计
            category_stats = {}
            for note in notes:
                category = note['category']
                if category not in category_stats:
                    category_stats[category] = 0
                category_stats[category] += 1

            # 最近活动（最近7天）
            from datetime import datetime, timedelta
            week_ago = datetime.now() - timedelta(days=7)
            recent_notes = [note for note in notes
                          if datetime.fromisoformat(note['updated_at']) >= week_ago]

            return {
                'total_notes': total_notes,
                'total_words': total_words,
                'favorites_count': favorites_count,
                'archived_count': archived_count,
                'active_notes': total_notes - archived_count,
                'category_stats': category_stats,
                'recent_activity': len(recent_notes),
                'average_words_per_note': total_words // total_notes if total_notes > 0 else 0
            }

        except Exception as e:
            return {
                'error': str(e),
                'total_notes': 0,
                'total_words': 0
            }

# 创建全局笔记管理器实例
notes_manager = NotesManager()