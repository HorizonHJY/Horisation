# Backend/Controller/csvcontroller.py
"""
CSV API 控制器 - 只负责路由和请求处理
业务逻辑已移至 Backend/Functions 模块
"""

from flask import Blueprint, request, jsonify, current_app, send_from_directory, url_for
import os
from typing import Optional, Tuple
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from werkzeug.utils import secure_filename

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

def _parse_mapping(default_prefix: str, required: bool = False):
    raw = request.form.get('mapping') or request.args.get('mapping')

    if raw is None:
        json_payload = request.get_json(silent=True) or {}
        raw = json_payload.get('mapping')

    if raw in (None, ""):
        if required:
            return None, ('mapping is required', 400)
        return [], None

    if isinstance(raw, str):
        try:
            mapping = json.loads(raw)
        except json.JSONDecodeError:
            return None, ('invalid mapping json', 400)
    else:
        mapping = raw

    if not isinstance(mapping, list):
        return None, ('mapping must be a list', 400)

    final = []
    for idx, item in enumerate(mapping):
        if not isinstance(item, dict):
            return None, ('mapping items must be objects', 400)
        entry = dict(item)
        entry.setdefault('out_file', f"{default_prefix}_{idx + 1}.xlsx")
        final.append(entry)

    return final, None


def _prepare_output_mapping(mapping: list[dict], default_prefix: str):
    """Resolve output filenames to the uploads directory.

    Ensures every mapping entry writes to the configured upload folder so the
    generated Excel files can be downloaded through the shared download
    endpoint. Returns a tuple of (prepared_mapping, downloadable_names).
    """

    upload_dir = current_app.config.get('UPLOAD_FOLDER') or os.path.join(current_app.root_path, '_uploads')
    os.makedirs(upload_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prepared: list[dict] = []
    download_names: list[str] = []

    for idx, item in enumerate(mapping, start=1):
        entry = dict(item)
        requested_name = entry.get('out_file') or f"{default_prefix}_{idx}.xlsx"
        safe_name = secure_filename(requested_name) or f"{default_prefix}_{idx}.xlsx"

        stem, ext = os.path.splitext(safe_name)
        if not ext:
            ext = '.xlsx'

        final_name = f"{stem}_{timestamp}{ext}"
        entry['out_file'] = os.path.join(upload_dir, final_name)

        prepared.append(entry)
        download_names.append(final_name)

    return prepared, download_names

def _read_diff_frame(data: bytes, filename: str, sep: Optional[str] = None, encoding: Optional[str] = None):
    df = csv_processor.read_file_to_dataframe(
        data,
        filename=filename,
        sep=sep,
        encoding=encoding
    )
    return df.apply(pd.to_numeric, errors='ignore')


# ==================== API 路由 ====================

@bp.post("/api/csv/diff-metadata")
def api_diff_metadata():
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    if not file1 or not file1.filename:
        return jsonify({'ok': False, 'error': 'file1 is required'}), 400
    if not file2 or not file2.filename:
        return jsonify({'ok': False, 'error': 'file2 is required'}), 400

    sep1 = request.args.get('sep1') or request.form.get('sep1')
    sep2 = request.args.get('sep2') or request.form.get('sep2')
    encoding1 = request.args.get('encoding1') or request.form.get('encoding1')
    encoding2 = request.args.get('encoding2') or request.form.get('encoding2')

    try:
        data1 = file1.read()
        data2 = file2.read()

        df1 = _read_diff_frame(data1, file1.filename, sep=sep1, encoding=encoding1)
        df2 = _read_diff_frame(data2, file2.filename, sep=sep2, encoding=encoding2)

        dtypes1 = csv_processor._infer_column_types(df1)
        dtypes2 = csv_processor._infer_column_types(df2)

        numeric1 = [col for col, dtype in dtypes1.items() if dtype == 'numeric']
        numeric2 = [col for col, dtype in dtypes2.items() if dtype == 'numeric']

        numeric1_set = set(numeric1)
        numeric2_set = set(numeric2)

        shared_numeric = [
            col for col in df1.columns
            if col in numeric1_set and col in numeric2_set
        ]

        return jsonify({
            'ok': True,
            'columns1': list(df1.columns),
            'columns2': list(df2.columns),
            'numeric_columns1': numeric1,
            'numeric_columns2': numeric2,
            'shared_numeric_columns': shared_numeric
        })
    except ImportError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'ok': False, 'error': f'diff metadata failed: {e}'}), 400

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
        clean_columns (bool): 是否清洗列名
        strip_special (bool): 是否移除列名中的特殊字符
        clean_cells (bool): 是否清洗单元格取值
        normalize_strings (bool): 是否对字符串字段进行标准化
        round_decimals (bool): 是否对小数进行四舍五入
        decimal_places (int): 四舍五入保留的小数位
        scale_numeric (bool): 是否执行数值缩放/平移
        scale_factor (float): 缩放因子（与 scale_numeric 搭配使用）
        scale_offset (float): 平移偏移量（与 scale_numeric 搭配使用）
        format_percentages (bool): 是否格式化百分比
        percent_decimals (int): 百分比保留的小数位
        format_dates (bool): 是否格式化日期
        date_format (str): 日期格式，例如 %Y-%m-%d
        fill_missing (bool): 是否自动填充缺失值
        missing_numeric_strategy (str): 数值列填充策略（mean/median/zero）
        missing_categorical_strategy (str): 分类列填充策略（mode/constant）
        missing_constant (str): 常量填充值（当策略为 constant 时）
        handle_outliers (bool): 是否处理异常值
        outlier_method (str): 异常值检测方法（zscore/iqr）
        outlier_threshold (float): 异常值阈值
        outlier_strategy (str): 异常值替换策略（median/mean/clip）
        remove_duplicates (bool): 是否去除重复行

    Returns:
        JSON: {
            'ok': bool,
            'filename': str,
            'cleaned_rows': int,
            'removed_duplicates': int,
            'applied_steps': list[str]
        }
    """
    filename, data, err = _get_file_and_bytes()
    if err:
        msg, code = err
        return jsonify({'ok': False, 'error': msg}), code

    # 解析参数
    case = request.args.get('case', 'upper')
    clean_columns = request.args.get('clean_columns', 'true').lower() == 'true'
    strip_special = request.args.get('strip_special', 'true').lower() == 'true'
    clean_cells = request.args.get('clean_cells', 'true').lower() == 'true'
    normalize_strings = request.args.get('normalize_strings', 'false').lower() == 'true'
    round_decimals = request.args.get('round_decimals', 'false').lower() == 'true'
    scale_numeric = request.args.get('scale_numeric', 'false').lower() == 'true'
    format_percentages = request.args.get('format_percentages', 'false').lower() == 'true'
    format_dates = request.args.get('format_dates', 'false').lower() == 'true'
    fill_missing = request.args.get('fill_missing', 'false').lower() == 'true'
    handle_outliers = request.args.get('handle_outliers', 'false').lower() == 'true'
    remove_dups = request.args.get('remove_duplicates', 'false').lower() == 'true'

    def _parse_int(value: Optional[str], default: int) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _parse_float(value: Optional[str]) -> Optional[float]:
        try:
            if value is None or value == '':
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    decimal_places = _parse_int(request.args.get('decimal_places'), 2)
    percent_decimals = _parse_int(request.args.get('percent_decimals'), 2)
    scale_factor = _parse_float(request.args.get('scale_factor'))
    scale_offset = _parse_float(request.args.get('scale_offset'))
    date_format = request.args.get('date_format') or '%Y-%m-%d'
    missing_numeric_strategy = request.args.get('missing_numeric_strategy', 'mean')
    missing_categorical_strategy = request.args.get('missing_categorical_strategy', 'mode')
    missing_constant = request.args.get('missing_constant')
    outlier_method = request.args.get('outlier_method', 'zscore')
    outlier_threshold = _parse_float(request.args.get('outlier_threshold')) or 3.0
    outlier_strategy = request.args.get('outlier_strategy', 'median')

    try:
        # 读取数据
        from Backend.Functions.csv_processor import csv_processor
        df = csv_processor.read_file_to_dataframe(data, filename=filename)
        original_rows = len(df)

        applied_steps = []

        # 清洗列名
        if clean_columns:
            df = csv_cleaner.clean_column_name(df, case=case, strip_special=strip_special)
            applied_steps.append('列名标准化')
            if strip_special:
                applied_steps.append('移除特殊字符')

        # 清洗单元格
        if clean_cells:
            df = csv_cleaner.clean_cell_values(df)
            applied_steps.append('单元格清洗')

        formatting_jobs: list[dict] = []

        def add_job(trans_type: str, columns: list[str], **params) -> bool:
            valid_cols = [col for col in columns if col in df.columns]
            if not valid_cols:
                return False
            job = {'Column': valid_cols, 'trans_type': trans_type}
            for key, value in params.items():
                if value is not None:
                    job[key] = value
            formatting_jobs.append(job)
            return True

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        string_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        if normalize_strings:
            if add_job('str', string_cols):
                applied_steps.append('字符串标准化')

        if round_decimals:
            if add_job('float', numeric_cols, decimals=decimal_places):
                applied_steps.append(f'小数四舍五入(保留{decimal_places}位)')

        if scale_numeric:
            scale_jobs_added = False
            if scale_factor is not None:
                scale_jobs_added |= add_job('scale', numeric_cols, operation='mul', factor=scale_factor)
            if scale_offset is not None:
                scale_jobs_added |= add_job('scale', numeric_cols, operation='add', factor=scale_offset)
            if scale_factor is None and scale_offset is None:
                for col in numeric_cols:
                    series = pd.to_numeric(df[col], errors='coerce')
                    valid = series.dropna()
                    if valid.empty:
                        continue
                    span = valid.max() - valid.min()
                    if span == 0:
                        continue
                    formatting_jobs.append(
                        {'Column': [col], 'trans_type': 'scale', 'operation': 'sub', 'factor': float(valid.min())})
                    formatting_jobs.append(
                        {'Column': [col], 'trans_type': 'scale', 'operation': 'div', 'factor': float(span)})
                    scale_jobs_added = True
            if scale_jobs_added:
                applied_steps.append('数值缩放/平移')

        if format_percentages:
            percent_cols = []
            for col in numeric_cols:
                series = pd.to_numeric(df[col], errors='coerce')
                valid = series.dropna()
                if valid.empty:
                    continue
                if ((valid >= 0) & (valid <= 1)).mean() >= 0.6:
                    percent_cols.append(col)
            if add_job('percent', percent_cols, decimals=percent_decimals):
                applied_steps.append('百分比格式化')

        if format_dates:
            def _normalize_date_format(fmt: str) -> str:
                mapping = {
                    '%Y-%m-%d': 'YYYY-MM-DD',
                    '%Y/%m/%d': 'YYYY-MM-DD',
                    '%d-%m-%y': 'DD_MM_YY',
                    '%d/%m/%y': 'DD_MM_YY',
                    '%m-%y': 'MM-YY',
                    '%m/%y': 'MM-YY'
                }
                return mapping.get(fmt, 'YYYY-MM-DD')

            date_cols: list[str] = []
            for col in df.columns:
                series = df[col]
                if isinstance(series, pd.DataFrame):
                    if series.shape[1] == 0:
                        continue
                    series = series.iloc[:, 0]
                if pd.api.types.is_datetime64_any_dtype(series):
                    date_cols.append(col)
                    continue
                if series.dtype == 'O' or pd.api.types.is_string_dtype(series):
                    sample = series.dropna().astype(str).head(20)
                    if sample.empty:
                        continue
                    parsed = pd.to_datetime(sample, errors='coerce', utc=False)
                    if parsed.notna().mean() >= 0.6:
                        date_cols.append(col)
            if add_job('date', date_cols, format=_normalize_date_format(date_format)):
                applied_steps.append('日期格式化')

        if fill_missing:
            fill_jobs_added = False
            numeric_strategy = (missing_numeric_strategy or 'mean').lower()
            categorical_strategy = (missing_categorical_strategy or 'mode').lower()

            if numeric_cols:
                strategy = 'mean'
                fill_value = None
                if numeric_strategy == 'median':
                    strategy = 'median'
                elif numeric_strategy in {'zero', 'constant'}:
                    strategy = 'constant'
                    fill_value = 0
                elif numeric_strategy == 'nan':
                    strategy = 'nan'
                fill_jobs_added |= add_job('missing', numeric_cols, strategy=strategy, fill_value=fill_value)

            categorical_cols = [col for col in df.columns if col not in numeric_cols]
            if categorical_cols:
                strategy = 'mode'
                fill_value = None
                if categorical_strategy == 'constant':
                    strategy = 'constant'
                    fill_value = missing_constant if missing_constant is not None else ''
                elif categorical_strategy == 'nan':
                    strategy = 'nan'
                elif categorical_strategy == 'mode':
                    strategy = 'mode'
                fill_jobs_added |= add_job('missing', categorical_cols, strategy=strategy, fill_value=fill_value)

            if fill_jobs_added:
                applied_steps.append('缺失值填充')

        if handle_outliers:
            replace_mapping = {
                'median': 'median',
                'mean': 'mean',
                'clip': 'clip'
            }
            replace_value = replace_mapping.get((outlier_strategy or 'median').lower(), 'median')
            if add_job('outlier', numeric_cols, method=outlier_method, threshold=outlier_threshold,
                       replace=replace_value):
                applied_steps.append('异常值处理')

        if formatting_jobs:
            df = csv_cleaner.formatting(df, formatting_jobs)
            df = df.apply(pd.to_numeric, errors='ignore')

        removed = 0

        # 去重
        if remove_dups:
            df = csv_cleaner.remove_duplicates(df)
            removed = original_rows - len(df)
            applied_steps.append('重复行去重')

        # 生成导出文件
        upload_dir = current_app.config.get('UPLOAD_FOLDER') or os.path.join(current_app.root_path, '_uploads')
        os.makedirs(upload_dir, exist_ok=True)

        base_name = Path(filename).stem if filename else 'cleaned'
        safe_stem = secure_filename(base_name) or 'cleaned'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_suffix = Path(filename).suffix.lower() if filename else ''
        export_ext = '.xlsx' if original_suffix in {'.xls', '.xlsx'} else '.csv'
        output_filename = f"{safe_stem}_cleaned_{timestamp}{export_ext}"
        output_path = os.path.join(upload_dir, output_filename)

        try:
            if export_ext == '.xlsx':
                df.to_excel(output_path, index=False)
            else:
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
        except Exception as export_err:
            return jsonify({'ok': False, 'error': f'clean succeeded but export failed: {export_err}'}), 500

        download_url = url_for('csv_api.download_cleaned_file', filename=output_filename)



        return jsonify({
            'ok': True,
            'filename': filename,
            'cleaned_rows': len(df),
            'removed_duplicates': removed,
            'columns': list(df.columns),
            'applied_steps': applied_steps,
            'output_filename': output_filename,
            'download_url': download_url
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': f'clean failed: {e}'}), 400

@bp.get("/api/csv/download/<path:filename>")
def download_cleaned_file(filename: str):
    upload_dir = current_app.config.get('UPLOAD_FOLDER') or os.path.join(current_app.root_path, '_uploads')
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({'ok': False, 'error': 'invalid filename'}), 400
    file_path = os.path.join(upload_dir, safe_name)
    if not os.path.isfile(file_path):
        return jsonify({'ok': False, 'error': 'file not found'}), 404
    return send_from_directory(upload_dir, safe_name, as_attachment=True)


@bp.post("/api/csv/diff_highlight")
def api_diff_highlight():
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    if not file1 or not file1.filename:
        return jsonify({'ok': False, 'error': 'file1 is required'}), 400
    if not file2 or not file2.filename:
        return jsonify({'ok': False, 'error': 'file2 is required'}), 400

    mapping, err = _parse_mapping('diff_highlight', required=True)
    if err:
        msg, code = err
        return jsonify({'ok': False, 'error': msg}), code

    sep1 = request.args.get('sep1') or request.form.get('sep1')
    sep2 = request.args.get('sep2') or request.form.get('sep2')
    encoding1 = request.args.get('encoding1') or request.form.get('encoding1')
    encoding2 = request.args.get('encoding2') or request.form.get('encoding2')

    try:
        data1 = file1.read()
        data2 = file2.read()

        df1 = _read_diff_frame(data1, file1.filename, sep=sep1, encoding=encoding1)
        df2 = _read_diff_frame(data2, file2.filename, sep=sep2, encoding=encoding2)

        if df1.shape != df2.shape:
            return jsonify({'ok': False, 'error': 'files must have the same shape'}), 400
        if list(df1.columns) != list(df2.columns):
            return jsonify({'ok': False, 'error': 'files must share the same columns'}), 400

        prepared_mapping, download_names = _prepare_output_mapping(mapping, 'diff_highlight')

        csv_processor.diff_highlight(df1, df2, prepared_mapping)

        download_urls = [
            url_for('csv_api.download_cleaned_file', filename=name)
            for name in download_names
        ]

        return jsonify({'ok': True, 'created_files': download_names, 'download_urls': download_urls})
    except ImportError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'ok': False, 'error': f'diff highlight failed: {e}'}), 400


@bp.post("/api/csv/diff_report")
def api_diff_report():
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    if not file1 or not file1.filename:
        return jsonify({'ok': False, 'error': 'file1 is required'}), 400
    if not file2 or not file2.filename:
        return jsonify({'ok': False, 'error': 'file2 is required'}), 400

    mapping, err = _parse_mapping('diff_report', required=True)
    if err:
        msg, code = err
        return jsonify({'ok': False, 'error': msg}), code

    sep1 = request.args.get('sep1') or request.form.get('sep1')
    sep2 = request.args.get('sep2') or request.form.get('sep2')
    encoding1 = request.args.get('encoding1') or request.form.get('encoding1')
    encoding2 = request.args.get('encoding2') or request.form.get('encoding2')

    try:
        data1 = file1.read()
        data2 = file2.read()

        df1 = _read_diff_frame(data1, file1.filename, sep=sep1, encoding=encoding1)
        df2 = _read_diff_frame(data2, file2.filename, sep=sep2, encoding=encoding2)

        if df1.shape != df2.shape:
            return jsonify({'ok': False, 'error': 'files must have the same shape'}), 400
        if list(df1.columns) != list(df2.columns):
            return jsonify({'ok': False, 'error': 'files must share the same columns'}), 400

        prepared_mapping, download_names = _prepare_output_mapping(mapping, 'diff_report')

        csv_processor.write_diff_report(df1, df2, prepared_mapping)

        download_urls = [
            url_for('csv_api.download_cleaned_file', filename=name)
            for name in download_names
        ]

        return jsonify({'ok': True, 'created_files': download_names, 'download_urls': download_urls})
    except ImportError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'ok': False, 'error': f'diff report failed: {e}'}), 400