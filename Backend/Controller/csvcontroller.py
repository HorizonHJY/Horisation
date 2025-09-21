# Backend/Controller/csvcontroller.py
from flask import Blueprint, request, jsonify
from io import BytesIO
import pandas as pd
import numpy as np
import os,openpyxl
import re

bp = Blueprint("csv_api", __name__)

MAX_BYTES = 100 * 1024 * 1024
ALLOWED_EXT = {'.csv', '.xls', '.xlsx'}

# ---- CSV 编码回退读取 ----
_USE_PYARROW = False
try:
    import pyarrow  # noqa: F401

    _USE_PYARROW = True
except Exception:
    _USE_PYARROW = False

# 检查 Excel 引擎依赖
_HAS_OPENPYXL = False
_HAS_XLRD = False
try:
    import openpyxl  # noqa: F401

    _HAS_OPENPYXL = True
except ImportError:
    pass

try:
    import xlrd  # noqa: F401

    _HAS_XLRD = True
except ImportError:
    pass


def _read_csv_with_fallback(binary: bytes, nrows: int | None = None,
                            sep: str | None = None, encoding: str | None = None) -> pd.DataFrame:
    """
    优先尝试 UTF-8（及 UTF-8-SIG），失败则回退至常见本地编码。
    可限制 nrows；可指定分隔符 sep；可手动传入 encoding 则直接使用该编码。
    """
    base_kwargs: dict = {}
    if nrows is not None:
        base_kwargs["nrows"] = nrows
    if sep is not None:
        base_kwargs["sep"] = sep

    if encoding:
        bio = BytesIO(binary)
        return pd.read_csv(bio, encoding=encoding, **base_kwargs)

    utf8_tries = [("utf-8", _USE_PYARROW), ("utf-8-sig", _USE_PYARROW)]
    local_fallbacks = ["gbk", "gb2312", "big5", "shift_jis", "cp1252", "latin1"]

    for enc, use_pa in utf8_tries:
        try:
            bio = BytesIO(binary)
            if use_pa:
                return pd.read_csv(bio, encoding=enc, engine="pyarrow", **base_kwargs)
            else:
                return pd.read_csv(bio, encoding=enc, **base_kwargs)
        except Exception:
            pass

    for enc in local_fallbacks:
        try:
            bio = BytesIO(binary)
            return pd.read_csv(bio, encoding=enc, **base_kwargs)
        except Exception:
            continue

    # 3) 最后兜底：latin1
    bio = BytesIO(binary)
    return pd.read_csv(bio, encoding="latin1", **base_kwargs)


# ---- 统一入口：CSV/Excel → DataFrame ----
def _to_df(binary: bytes, nrows: int | None = None, sep: str | None = None,
           filename: str | None = None, encoding: str | None = None) -> pd.DataFrame:
    """
    将二进制数据读成 DataFrame。
    - 若 filename 指向 .xls/.xlsx，使用 read_excel
    - 否则按 CSV 处理并做编码回退
    - Excel 文件如果存在多级表头，会自动展平成单级列名
    """
    name = (filename or "").lower()

    # Excel 文件处理
    if name.endswith((".xls", ".xlsx")):
        bio = BytesIO(binary)

        # 检查 Excel 依赖
        if name.endswith(".xlsx") and not _HAS_OPENPYXL:
            raise ImportError("处理 .xlsx 文件需要 openpyxl 库，请安装: pip install openpyxl")
        elif name.endswith(".xls") and not _HAS_XLRD:
            raise ImportError("处理 .xls 文件需要 xlrd 库，请安装: pip install xlrd")

        try:
            # 优先使用 openpyxl 处理 xlsx，xlrd 处理 xls
            if name.endswith(".xlsx"):
                df = pd.read_excel(bio, engine="openpyxl")
            else:
                df = pd.read_excel(bio, engine="xlrd")
        except Exception as e:
            # 兼容性兜底
            try:
                df = pd.read_excel(bio)
            except Exception:
                raise e
        # Excel 展平多级表头
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns.values]

    # CSV 文件处理
    else:
        df = _read_csv_with_fallback(binary, nrows=nrows, sep=sep, encoding=encoding)

    # 确保所有列都是字符串类型，避免类型比较错误
    for col in df.columns:
        df[col] = df[col].astype(str)

    return df


# ---- 预览和概要函数 ----
def read_csv_preview(binary: bytes, n: int = 5, sep: str | None = None,
                     filename: str | None = None, encoding: str | None = None):
    """返回前 n 行的预览记录与列名；支持 CSV/Excel（通过 filename 判断）"""
    df = _to_df(binary, nrows=n, sep=sep, filename=filename, encoding=encoding)
    head = df.head(n)
    return {
        'columns': list(head.columns),
        'rows': head.to_dict(orient='records')
    }


def summarize_csv(binary: bytes, sep: str | None = None,
                  filename: str | None = None, encoding: str | None = None):
    """返回整体概要信息：行/列、字段类型、缺失统计等；支持 CSV/Excel"""
    df = _to_df(binary, sep=sep, filename=filename, encoding=encoding)

    # 计算缺失值（空字符串视为缺失）
    na_count = {}
    for col in df.columns:
        # 计算空字符串和 NaN 的数量
        na_count[col] = int((df[col] == '') | (df[col] == 'nan') | (df[col].isna()) | (df[col] == 'None')).sum()

    total_rows = len(df)
    na_ratio = {k: (v / total_rows if total_rows else 0.0) for k, v in na_count.items()}

    # 推断数据类型（基于非空值）
    dtypes = {}
    for col in df.columns:
        # 获取非空值
        non_empty = df[col][(df[col] != '') & (df[col] != 'nan') & (~df[col].isna()) & (df[col] != 'None')]

        if len(non_empty) == 0:
            dtypes[col] = 'unknown'
            continue

        # 尝试推断类型
        sample = non_empty.iloc[0] if len(non_empty) > 0 else ''

        # 检查是否为数字
        if re.match(r'^-?\d+\.?\d*$', str(sample)):
            dtypes[col] = 'numeric'
        # 检查是否为日期
        elif re.match(r'^\d{4}-\d{2}-\d{2}', str(sample)):
            dtypes[col] = 'date'
        else:
            dtypes[col] = 'text'

    return {
        'rows': int(total_rows),
        'cols': int(df.shape[1]),
        'columns': list(df.columns),
        'dtypes': dtypes,
        'na_count': na_count,
        'na_ratio': na_ratio
    }


# ---- 文件处理函数 ----
def _get_file_and_bytes():
    if 'file' not in request.files:
        return None, None, ('no file field', 400)
    f = request.files['file']
    if not f or f.filename.strip() == '':
        return None, None, ('empty filename', 400)

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return None, None, ('only .csv/.xls/.xlsx allowed', 400)

    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    if size > MAX_BYTES:
        return None, None, ('file too large', 413)

    data = f.read()
    return f.filename, data, None


# ---- API 路由 ----
@bp.post("/api/csv/preview")
def api_preview():
    filename, data, err = _get_file_and_bytes()
    if err:
        msg, code = err
        return jsonify({'ok': False, 'error': msg}), code

    try:
        n = int(request.args.get('n', '5'))
        n = max(1, min(n, 2000))
    except Exception:
        n = 5

    # 获取编码和分隔符参数
    sep = request.args.get('sep')
    encoding = request.args.get('encoding')

    try:
        payload = read_csv_preview(data, n=n, sep=sep, filename=filename, encoding=encoding)
        return jsonify({'ok': True, 'filename': filename, **payload})
    except ImportError as e:
        # 处理缺少依赖的情况
        return jsonify({'ok': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'ok': False, 'error': f'parse failed: {e}'}), 400


@bp.post("/api/csv/summary")
def api_summary():
    filename, data, err = _get_file_and_bytes()
    if err:
        msg, code = err
        return jsonify({'ok': False, 'error': msg}), code

    sep = request.args.get('sep')
    encoding = request.args.get('encoding')

    try:
        summary = summarize_csv(data, sep=sep, filename=filename, encoding=encoding)
        return jsonify({'ok': True, 'filename': filename, 'summary': summary})
    except ImportError as e:
        # 处理缺少依赖的情况
        return jsonify({'ok': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'ok': False, 'error': f'parse failed: {e}'}), 400