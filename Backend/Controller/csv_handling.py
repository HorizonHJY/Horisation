from io import BytesIO
import pandas as pd
import numpy as np

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

#Data cleaning
def clean_csv(binary: bytes,
              subset: list[str] | None = None,
              keep: str = 'first',
              strip_cell_space: bool = True,
              dedupe_columns: bool = True
              ) -> pd.DataFrame :
    df = _to_df(binary)

    #列名大写
    new_cols = []
    seen = {}
    for c in df.columns:
        name = str(c).strip().upper()  # 转字符串 + 去空格 + 大写
        if dedupe_columns:
            k = seen.get(name, 0)
            if k > 0:
                name = f"{name}_{k}"
            seen[name] = k + 1
        new_cols.append(name)
    df.columns = new_cols

    #去除首尾空格
    if strip_cell_space:
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    #去重
    df = df.drop_duplicates(subset = subset, keep = keep).reset_index(drop=True)

    return df

#DataFrame merge
def concat_dfs(dfs: list[pd.DataFrame],
                uppercase_cols: bool = True,
                alias: dict[str, str] | None = None) -> pd.DataFrame:
    """
    纵向合并多个 DataFrame：列取并集，缺失列自动补 NaN

    Args:
        dfs: 要合并的 DataFrame 列表
        uppercase_cols: 是否把列名统一成大写（避免 id vs ID）
        alias: 列名同义映射，如 {'user_id':'ID', 'uid':'ID'}
    """
    normed = []
    for df in dfs:
        df = df.copy()
        if alias:
            df.rename(columns = alias, inplace = True)
        if uppercase_cols:
            df.columns = [str(c).strip().upper() for c in df.columns]
        normed.append(df)

    result = pd.concat(normed, axis = 0, ignore_index = True, sort = False)
    return result


#DataFrame Exporting
#export as csv
def export_csv(df: pd.DataFrame, filename: str = 'output.csv') -> None:
    return df.to_csv(filename, index = False, encoding = 'utf-8')

def export_excel(df: pd.DataFrame, filename: str = 'output.xlsx') -> None:
    return df.to_excel(filename, index = False, engine = 'openpyxl')

def export_json(df: pd.DataFrame, filename: str = 'output.json') -> None:
    return df.to_json(filename, orient = 'records', force_ascii = False)


