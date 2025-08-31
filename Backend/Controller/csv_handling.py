# Backend/Controller/csv_handling.py
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
def dataframe_merge(df_list: list[pd.DataFrame],
                   how: str = 'inner',
                   on: str | list[str] | None = None,
                   left_on: str | list[str] | None = None,
                   right_on: str | list[str] | None = None,
                   suffixes: tuple = ('_x', '_y')) -> pd.DataFrame:
    """
    合并多个DataFrame（简化版）

    Args:
        df_list: 要合并的DataFrame列表（假定至少有一个）
        how: 合并方式 ('inner'交集, 'outer'并集, 'left', 'right')
        on: 用于合并的列名（当所有DataFrame都有相同列名时使用）
        left_on: 左侧DataFrame的合并列
        right_on: 右侧DataFrame的合并列
        suffixes: 重复列名的后缀

    Returns:
        合并后的DataFrame
    """
    result_df = df_list[0]

    for i, next_df in enumerate(df_list[1:], 1):
        if on is not None and (left_on is not None or right_on is not None):
            raise ValueError("不能同时指定'on'和'left_on/right_on'参数")

        if on is not None:
            result_df = result_df.merge(next_df, how = how, on = on, suffixes = suffixes)
        elif left_on is not None and right_on is not None:
            result_df = result_df.merge(next_df, how = how, left_on = left_on,
                                        right_on = right_on, suffixes = suffixes)
        else:
            # 自动推断合并列（公共列交集）
            common_cols = list(set(result_df.columns) & set(next_df.columns))
            if not common_cols:
                raise ValueError(f"DataFrame {i} 和 {i+1} 没有共同的列名用于合并")

            result_df = result_df.merge(next_df, how = how, on = common_cols, suffixes = suffixes)

    return result_df


#DataFrame Exporting
#export as csv
def export_csv(df: pd.DataFrame, filename: str = 'output.csv') -> None:
    return df.to_csv(filename, index = False, encoding = 'utf-8')

def export_excel(df: pd.DataFrame, filename: str = 'output.xlsx') -> None:
    return df.to_excel(filename, index = False)

def export_json(df: pd.DataFrame, filename: str = 'output.json') -> None:
    return df.to_json(filename, orient = 'records', force_ascii = False)



#随机生成csv数据文件
def generate_demo_csv(filename: str = "demo.csv", rows: int = 20):
    # 随机数据
    data = {
        "id": range(1, rows + 1),
        "name": [f"User{i}" for i in range(1, rows + 1)],
        "age": np.random.randint(18, 60, size=rows),   # 18~59岁
        "city": np.random.choice(
            ["New York", "Los Angeles", "Chicago", "Houston", "San Francisco"],
            size=rows
        )
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"✅ 已生成 {filename}，包含 {rows} 行数据")

#生成demo数据
generate_demo_csv("demo2.csv", rows=30)

#测试clean
#转换为二进制
#with open("demo.csv", "rb") as f:
#    binary = f.read()
#df_clean = clean_csv(binary)
#export_csv(df_clean, "demo_clean.csv")


