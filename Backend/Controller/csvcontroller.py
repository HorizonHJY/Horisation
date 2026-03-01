# Backend/Controller/csvcontroller.py
"""
CSV API 控制器 - 只负责路由和请求处理
业务逻辑已移至 Backend/Functions 模块
"""

from flask import Blueprint, request, jsonify
import os
from typing import Optional, Tuple

# 导入核心功能模块
from Backend.Functions.csv_processor import csv_processor
from Backend.Functions.csv_cleaner import csv_cleaner

bp = Blueprint("csv_api", __name__)

# 配置常量
MAX_BYTES = 100 * 1024 * 1024  # 100MB
ALLOWED_EXT = {'.csv', '.xls', '.xlsx'}


def _get_file_and_bytes() -> Tuple[Optional[str], Optional[bytes], Optional[Tuple[str, int]]]:
    """
    从请求中提取文件并验证

    Returns:
        Tuple: (filename, binary_data, error)
            - filename: 文件名
            - binary_data: 文件的二进制数据
            - error: 错误信息元组 (message, status_code)，无错误时为 None
    """
    if 'file' not in request.files:
        return None, None, ('no file field', 400)

    f = request.files['file']
    if not f or not f.filename or f.filename.strip() == '':
        return None, None, ('empty filename', 400)

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return None, None, ('only .csv/.xls/.xlsx allowed', 400)

    # 检查文件大小
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    if size > MAX_BYTES:
        return None, None, ('file too large', 413)

    data = f.read()
    return f.filename, data, None


# ==================== API 路由 ====================

@bp.post("/api/csv/preview")
def api_preview():
    """
    预览CSV/Excel文件前N行

    Query Parameters:
        n (int): 预览行数，默认5，最大2000
        sep (str): CSV分隔符，可选
        encoding (str): CSV编码，可选

    Returns:
        JSON: {
            'ok': bool,
            'filename': str,
            'columns': list,
            'rows': list[dict]
        }
    """
    filename, data, err = _get_file_and_bytes()
    if err:
        msg, code = err
        return jsonify({'ok': False, 'error': msg}), code

    # 解析参数
    try:
        n = int(request.args.get('n', '5'))
        n = max(1, min(n, 2000))
    except Exception:
        n = 5

    sep = request.args.get('sep')
    encoding = request.args.get('encoding')

    # 调用核心功能
    try:
        payload = csv_processor.get_preview(
            data,
            n=n,
            sep=sep,
            filename=filename,
            encoding=encoding
        )
        return jsonify({'ok': True, 'filename': filename, **payload})
    except ImportError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'ok': False, 'error': f'parse failed: {e}'}), 400


@bp.post("/api/csv/summary")
def api_summary():
    """
    获取CSV/Excel文件概要信息

    Query Parameters:
        sep (str): CSV分隔符，可选
        encoding (str): CSV编码，可选

    Returns:
        JSON: {
            'ok': bool,
            'filename': str,
            'summary': {
                'rows': int,
                'cols': int,
                'columns': list,
                'dtypes': dict,
                'na_count': dict,
                'na_ratio': dict
            }
        }
    """
    filename, data, err = _get_file_and_bytes()
    if err:
        msg, code = err
        return jsonify({'ok': False, 'error': msg}), code

    sep = request.args.get('sep')
    encoding = request.args.get('encoding')

    try:
        summary = csv_processor.get_summary(
            data,
            sep=sep,
            filename=filename,
            encoding=encoding
        )
        return jsonify({'ok': True, 'filename': filename, 'summary': summary})
    except ImportError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'ok': False, 'error': f'parse failed: {e}'}), 400


@bp.post("/api/csv/clean")
def api_clean():
    """
    清洗CSV数据

    Query Parameters:
        case (str): 列名大小写 ('upper', 'lower', 'title')
        strip_special (bool): 去除特殊字符
        remove_duplicates (bool): 去除重复行

    Returns:
        JSON: {
            'ok': bool,
            'filename': str,
            'cleaned_rows': int,
            'removed_duplicates': int
        }
    """
    filename, data, err = _get_file_and_bytes()
    if err:
        msg, code = err
        return jsonify({'ok': False, 'error': msg}), code

    # 解析参数
    case = request.args.get('case', 'upper')
    strip_special = request.args.get('strip_special', 'true').lower() == 'true'
    remove_dups = request.args.get('remove_duplicates', 'false').lower() == 'true'

    try:
        # 读取数据
        from Backend.Functions.csv_processor import csv_processor
        df = csv_processor.read_file_to_dataframe(data, filename=filename)
        original_rows = len(df)

        # 清洗列名
        df = csv_cleaner.clean_column_names(df, case=case, strip_special=strip_special)

        # 清洗单元格
        df = csv_cleaner.clean_cell_values(df)

        # 去重
        if remove_dups:
            df = csv_cleaner.remove_duplicates(df)

        removed = original_rows - len(df)

        return jsonify({
            'ok': True,
            'filename': filename,
            'cleaned_rows': len(df),
            'removed_duplicates': removed,
            'columns': list(df.columns)
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': f'clean failed: {e}'}), 400
