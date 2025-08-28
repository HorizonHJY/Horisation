# Backend/Controller/csv_handling.py
from io import BytesIO
import pandas as pd

def _to_df(binary: bytes) -> pd.DataFrame:
    """
    将二进制CSV读成DataFrame。
    - 自动推断编码（常见UTF-8/GBK情况pandas会处理；若有特殊编码可扩展）
    """
    bio = BytesIO(binary)
    # 你也可以在 read_csv 里加参数，如 sep=';', encoding='utf-8', dtype=str 等
    df = pd.read_csv(bio)
    return df

def read_csv_preview(binary: bytes, n: int = 5):
    """返回前 n 行的预览记录（list[dict]）与列名"""
    df = _to_df(binary)
    head = df.head(n)
    return {
        'columns': list(head.columns),
        'rows': head.to_dict(orient='records')
    }

def summarize_csv(binary: bytes):
    """返回整体概要信息：行/列、字段类型、缺失统计等"""
    df = _to_df(binary)

    # 字段类型（pandas dtype -> 简单字符串）
    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}

    # 缺失值计数与占比
    na_count = df.isna().sum().to_dict()
    total_rows = len(df)
    na_ratio = {k: (v / total_rows if total_rows else 0.0) for k, v in na_count.items()}

    summary = {
        'rows': int(total_rows),
        'cols': int(df.shape[1]),
        'columns': list(df.columns),
        'dtypes': dtypes,
        'na_count': na_count,
        'na_ratio': na_ratio
    }
    return summary
