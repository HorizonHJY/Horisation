from io import BytesIO
import pandas as pd
from functools import reduce
import numpy as np
import csv
import re
from typing import Any


def detect_separator(binary: bytes, encodings=('utf-8', 'utf-8-sig', 'latin1', 'gbk')):
    """
    自动检测 CSV 文件的分隔符和编码
    - 默认尝试多种常见编码
    - 支持常见分隔符：逗号、分号、制表符、竖线
    """
    for enc in encodings:
        try:
            text = binary.decode(enc)
            sample = text[:10000]  # 取前一段文本作为样本
            # 用 Sniffer 猜分隔符
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
            print(f"[DEBUG] 检测到分隔符: {repr(dialect.delimiter)}, 编码: {enc}")
            return dialect.delimiter, enc
        except Exception:
            continue

    # 如果都失败，默认逗号 + utf-8
    print("[DEBUG] 自动检测失败，使用默认分隔符 ',' 和编码 'utf-8'")
    return ',', 'utf-8'


def _to_df(binary: bytes) -> pd.DataFrame:
    """
    将二进制CSV读成DataFrame。
    - 自动推断编码（常见UTF-8/GBK情况pandas会处理；若有特殊编码可扩展）
    """
    # 你也可以在 read_csv 里加参数，如 sep=';', encoding='utf-8', dtype=str 等
    sep, encoding = detect_separator(binary)
    bio = BytesIO(binary)
    df = pd.read_csv(bio, sep=sep, encoding=encoding)
    df.attrs["sep"] = sep
    df.attrs["encoding"] = encoding
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

def change_separator(df: pd.DataFrame, sep: str = ';') -> pd.DataFrame:
    """修改 DataFrame 的分隔符设定（只修改属性，不导出）"""
    df.attrs["sep"] = sep
    return df

def clean_csv(binary: bytes,
              subset: list[str] | None = None,
              keep: str = 'first',
              strip_cell_space: bool = True,
              dedupe_columns: bool = True,
              case: str = "upper",
              prefix: str | None = None,
              strip_special: bool = True,
              flatten_header: bool = True,
              drop_duplicates: bool = False
              ) -> pd.DataFrame:

    """清洗CSV/Excel文件数据并标准化表头
    可选将多级表头 (MultiIndex) 展平为单级。
    对列名进行标准化处理，包括大小写统一，加前缀，去除特殊字符（仅保留字母、数字、下划线、中文），避免重复列名（自动添加_1, _2后缀）
    对字符串单元格去除首尾空格。
    删除重复行"""
    df = _to_df(binary)

    # 如果是 MultiIndex 表头，展平
    if flatten_header and isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x]) for col in df.columns.values]

    # 列名处理
    new_cols = []
    seen = {}
    for c in df.columns:
        name = str(c).strip()

        # 统一大小写， {'upper', 'lower', 'title', None}, 默认upper
        if case == "upper":
            name = name.upper()
        elif case == "lower":
            name = name.lower()
        elif case == "title":
            name = name.title()

        # 去除特殊符号
        if strip_special:
            import re
            name = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "_", name).strip("_")

        # 加前缀
        if prefix:
            name = f"{prefix}{name}"

        # 列名去重
        if dedupe_columns:
            k = seen.get(name, 0)
            if k > 0:
                name = f"{name}_{k}"
            seen[name] = k + 1
        new_cols.append(name)
    df.columns = new_cols

    # 去除单元格首尾空格
    if strip_cell_space:
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # 数据行去重
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    return df

def normalize_dates(df, date_cols=None, date_format="%Y-%m-%d"):
    """
    日期标准化
    - 将指定列（或所有列尝试）转为统一格式
    - 默认格式为 "%Y-%m-%d"
    """
    df = df.copy()

    for col in (date_cols if date_cols else df.columns):
        def convert(x):
            if pd.isna(x):
                return x
            try:
                return pd.to_datetime(x, errors="raise").strftime(date_format)
            except Exception:
                return x  # 保留原值

        df[col] = df[col].apply(convert)

    return df


def normalize_units(df:pd.DataFrame, unit_map:dict[str:int], base_units):
    """
    单位换算
    - 根据列名中的单位（如 金额(万元)）换算成统一单位（如 元）
    - unit_map: 定义支持的单位换算关系
    - base_units: 定义目标标准单位
    """

    # 默认支持的单位换算表
    unit_map = unit_map or {
        "金额": {"元": 1, "千元": 1000, "万元": 10000},
        "人口": {"人": 1, "千人": 1000, "万人": 10000}
    }
    # 默认目标标准单位
    base_units = base_units or {"金额": "元", "人口": "人"}

    rename_dict = {}
    for col in df.columns:
        # 正则匹配列名中的单位部分，例如 "金额(万元)"
        match = re.match(r"^(.*?)[(_]?([^\d\W]+)?\)?$", str(col))
        if not match:
            continue
        base_name, unit = match.groups()
        base_name = base_name.strip("_ (")
        unit = unit if unit else None

        if base_name in unit_map:
            # 获取目标统一单位
            target_unit = base_units.get(base_name, list(unit_map[base_name].keys())[0])
            if unit and unit in unit_map[base_name]:
                # 按比例换算
                factor = unit_map[base_name][unit]
                target_factor = unit_map[base_name][target_unit]
                df[col] = pd.to_numeric(df[col], errors="coerce") * (factor / target_factor)

            # 修改列名为统一单位
            rename_dict[col] = f"{base_name}({target_unit})"

    df.rename(columns=rename_dict, inplace=True)
    return df


def normalize_percent(df, percent_cols=None):
    """
    百分比标准化
    - 将百分比字符串（"50%"）或大于1的数值（50）转为小数（0.5）
    - 默认自动识别列名中包含 ["率", "比", "%"] 的列
    """
    if percent_cols is None:
        percent_cols = [c for c in df.columns if any(k in str(c) for k in ["率", "比", "%"])]

    def convert(x):
        if pd.isna(x):
            return x
        if isinstance(x, str) and x.endswith("%"):
            try:
                return float(x.strip("%")) / 100
            except:
                return None
        try:
            val = float(x)
            return val / 100 if val > 1 else val
        except:
            return None

    for col in percent_cols:
        df[col] = df[col].apply(convert)
        # 列名加后缀，避免混淆
        df.rename(columns={col: f"{col}(小数)"}, inplace=True)

    return df

def normalize_values(df, value_map=None):
    """
    分类字段标准化
    - 根据 value_map 将同一字段的不同取值映射为统一值
    例如：性别（男/女/Male/Female/M/F）统一为 M/F
         状态（是/否/yes/no/Y/N）统一为 1/0
    """
    if not value_map:
        value_map = {
            "性别": {"男": "M", "女": "F", "male": "M", "female": "F", "M": "M", "F": "F"},
            "状态": {"是": 1, "否": 0, "yes": 1, "no": 0, "Y": 1, "N": 0}
        }

    for col, mapping in value_map.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    return df

class Cleaning:

    def normalize_strings(slef,df, normalize_case="upper", strip_whitespace=True, strip_invisible=True):
        """
        字符串清理
        - 去除前后空格
        - 删除不可见字符（换行、制表符、零宽空格）
        - 大小写统一（upper/lower/title）
        """
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            def clean(x):
                if not isinstance(x, str):
                    return x

                # 去掉前后空格
                if strip_whitespace:
                    x = x.strip()

                # 删除不可见字符（零宽空格，换行，制表符）
                if strip_invisible:
                    x = re.sub(r"[\u200b\r\n\t]", " ", x)

                # 大小写转换
                if normalize_case == "upper":
                    x = x.upper()
                elif normalize_case == "lower":
                    x = x.lower()
                elif normalize_case == "title":
                    x = x.title()

                # 压缩多余空格
                return re.sub(" +", " ", x)

            df[col] = df[col].apply(clean)
        return df




def normalize_missing(df, missing_values=None):
    """
    缺失值统一
    - 将各种形式的缺失值标记（"NA", "null", "-", "N/A"...）替换为 np.nan
    """
    missing_values = missing_values or ["", "NA", "N/A", "-", "null", "None"]
    df.replace(missing_values, np.nan, inplace=True)
    return df

def handle_missing_outliers(df: pd.DataFrame,
                            mark_missing: bool = True,
                            fill_strategy: str | None = None,  # {"mean", "median", "max", "min", "ffill", "bfill", None}
                            fill_value: any = None,
                            custom_fill: dict[str, str | Any] | None = None,
                            missing_values: list[str] = ["", "NA", "N/A", "na", "-", "null", "None"],


                            mark_outliers: bool = True,
                            outlier_method: str = "zscore",  # {"zscore", "iqr"}
                            z_threshold: float = 3.0,
                            iqr_factor: float = 1.5,
                            replace_outliers: bool = False,
                            outlier_fill: str = "median"  # {"mean", "median", "nan"}
                            ) -> pd.DataFrame:
    """
    缺失值和异常值处理函数
    - 缺失值标记 + 填充
    - 异常值标记 + 填充
    """
    df = df.copy()

    #统一缺失值标记
    df.replace(missing_values, np.nan, inplace=True)

    # 缺失值标记
    if mark_missing:
        for col in df.columns:
            df[f"{col}_isna"] = df[col].isna().astype(int)

    # 缺失值填充
    for col in df.columns:
        if df[col].isna().any():
            strategy = None
            if custom_fill and col in custom_fill:
                strategy = custom_fill[col]
            elif fill_strategy:
                strategy = fill_strategy

            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "max":
                df[col] = df[col].fillna(df[col].max())
            elif strategy == "min":
                df[col] = df[col].fillna(df[col].min())
            elif strategy == "ffill":
                df[col] = df[col].fillna(method="ffill")
            elif strategy == "bfill":
                df[col] = df[col].fillna(method="bfill")
            elif strategy is None and fill_value is not None:
                df[col] = df[col].fillna(fill_value)

    # 异常值检测
    if mark_outliers:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if outlier_method == "zscore":
                mean, std = df[col].mean(), df[col].std()
                df[f"{col}_outlier"] = ((df[col] - mean).abs() > z_threshold * std).astype(int)
            elif outlier_method == "iqr":
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
                df[f"{col}_outlier"] = ((df[col] < lower) | (df[col] > upper)).astype(int)

            # 异常值替换
            if replace_outliers:
                if outlier_fill == "mean":
                    df.loc[df[f"{col}_outlier"] == 1, col] = df[col].mean()
                elif outlier_fill == "median":
                    df.loc[df[f"{col}_outlier"] == 1, col] = df[col].median()
                elif outlier_fill == "nan":
                    df.loc[df[f"{col}_outlier"] == 1, col] = np.nan

    return df

def validate_ranges(df, validate_ranges=None):
    """
    数值范围校验
    - 对指定字段做范围限制
    - 超出范围的值替换为 NaN
    - 例如：{"年龄": (0, 120)}，超过 120 或小于 0 的设为 NaN
    """
    if validate_ranges:
        for col, (low, high) in validate_ranges.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[(df[col] < low) | (df[col] > high), col] = np.nan
    return df

def normalize_codes(df, code_length=None):
    """
    编码字段补零
    - 用于像邮编、ID 这样的字段，统一长度
    - 例如：邮编 123 -> 000123（长度 6）
    """
    if code_length:
        for col, length in code_length.items():
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: str(x).zfill(length) if pd.notna(x) else x
                )
    return df

def normalize_booleans(df, boolean_map=None):
    """
    布尔字段标准化
    - 将 "是/否"、"yes/no"、"Y/N"、1/0 等统一转为 True/False
    """
    if boolean_map:
        for col, mapping in boolean_map.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)
    return df

def combine_dfs(dfs: list[pd.DataFrame],
                method: str = "merge",   # "merge" / "concat"
                # concat 相关参数
                uppercase_cols: bool = True,
                alias: dict[str, str] | None = None,
                # merge 相关参数
                how: str = "outer",
                on: str | list[str] | None = None,
                left_on: str | list[str] | None = None,
                right_on: str | list[str] | None = None,
                suffixes: tuple[str, str] = ("_left", "_right")) -> pd.DataFrame:
    """
    合并多个 DataFrame：支持纵向 concat 和横向 merge，默认使用 merge

    Args:
        dfs: 要合并的 DataFrame 列表
        method: "merge"(横向，SQL join) 或 "concat"(纵向，append)
        uppercase_cols: concat 模式下是否把列名统一成大写
        alias: concat 模式下的列名同义映射
        how: merge 的连接方式，默认 "outer"
        on: merge 的公共列
        left_on/right_on: merge 的左右连接键
        suffixes: merge 后相同列名的后缀
    """
    if method == "concat":
        normed = []
        for df in dfs:
            df = df.copy()
            if alias:
                df.rename(columns=alias, inplace=True)
            if uppercase_cols:
                df.columns = [str(c).strip().upper() for c in df.columns]
            normed.append(df)
        result = pd.concat(normed, axis=0, ignore_index=True, sort=False)
        return result

    elif method == "merge":
        if on is not None and (left_on is not None or right_on is not None):
            raise ValueError("不能同时指定 'on' 和 'left_on/right_on'")

        def _merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
            if on is not None:
                return a.merge(b, how=how, on=on, suffixes=suffixes)
            elif left_on is not None and right_on is not None:
                return a.merge(b, how=how, left_on=left_on, right_on=right_on, suffixes=suffixes)
            else:
                common = list(set(a.columns) & set(b.columns))
                if not common:
                    raise ValueError("未指定键，且没有公共列可用于合并")
                return a.merge(b, how=how, on=common, suffixes=suffixes)

        return reduce(_merge, dfs)

    else:
        raise ValueError("method 必须是 'merge' 或 'concat'")

def export_data(df: pd.DataFrame, filename: str = 'output.csv', file_format: str = None) -> None:
    """
    通用导出函数：支持 CSV, Excel, JSON
    :param df: 需要导出的 DataFrame
    :param filename: 文件名，默认 output.csv
    :param file_format: 文件格式，可选 'csv', 'excel', 'json'，默认根据文件扩展名判断
    """
    # 如果没有指定格式，则从文件名扩展名推断
    if file_format is None:
        if filename.lower().endswith('.csv'):
            file_format = 'csv'
        elif filename.lower().endswith(('.xls', '.xlsx')):
            file_format = 'excel'
        elif filename.lower().endswith('.json'):
            file_format = 'json'
        else:
            raise ValueError("无法识别文件格式，请指定 file_format 参数 ('csv', 'excel', 'json')")

    # 根据不同格式导出
    if file_format == 'csv':
        df.to_csv(filename, index=False, encoding='utf-8')
    elif file_format == 'excel':
        df.to_excel(filename, index=False, engine='openpyxl')
    elif file_format == 'json':
        df.to_json(filename, orient='records', force_ascii=False)
    else:
        raise ValueError("不支持的文件格式: {}".format(file_format))


def main():
    csv_text = """姓名,年龄,金额(万元),比率,状态
    张三,25,1.5,50%,是
    李四,130,2.0,0.8,否
    王五,NA,3.5,120,yes
    赵六,40,4.0,abc,no
    """
    binary = csv_text.encode("utf-8")

    print("=== detect_separator ===")
    sep, enc = detect_separator(binary)
    print("sep:", sep, "encoding:", enc)

    print("\n=== read_csv_preview ===")
    preview = read_csv_preview(binary, n=3)
    print(preview)

    print("\n=== summarize_csv ===")
    summary = summarize_csv(binary)
    print(summary)

    print("\n=== clean_csv ===")
    df_clean = clean_csv(binary)
    print(df_clean.head())

    print("\n=== normalize_dates ===")
    df_dates = pd.DataFrame({"日期": ["2024-01-01", "2024/02/01", "20240301"]})
    print(normalize_dates(df_dates))

    print("\n=== normalize_units ===")
    df_units = pd.DataFrame({"金额(万元)": [1.5, 2, 3]})
    print(normalize_units(df_units, None, None))

    print("\n=== normalize_percent ===")
    df_percent = pd.DataFrame({"比率": ["50%", "0.8", "120"]})
    print(normalize_percent(df_percent))

    print("\n=== normalize_values ===")
    df_val = pd.DataFrame({"性别": ["男", "F", "female"]})
    print(normalize_values(df_val))

    print("\n=== normalize_strings ===")
    df_str = pd.DataFrame({"名字": ["  Zhang ", "li\nsi", "WANG\tWU "]})
    print(normalize_strings(df_str))



    print("\n=== normalize_missing ===")
    df_missing = pd.DataFrame({"数据": ["NA", "N/A", "-", "123"]})
    print(normalize_missing(df_missing))

    print("\n=== handle_missing_outliers ===")
    df_outlier = pd.DataFrame({"数值": [1, 2, 1000, None]})
    print(handle_missing_outliers(df_outlier))

    print("\n=== validate_ranges ===")
    df_range = pd.DataFrame({"年龄": [25, -1, 200]})
    print(validate_ranges(df_range, {"年龄": (0, 120)}))

    print("\n=== normalize_codes ===")
    df_code = pd.DataFrame({"邮编": [123, "45", None]})
    print(normalize_codes(df_code, {"邮编": 6}))

    print("\n=== normalize_booleans ===")
    df_bool = pd.DataFrame({"状态": ["是", "否", "yes", "no"]})
    mapping = {"状态": {"是": True, "否": False, "yes": True, "no": False}}
    print(normalize_booleans(df_bool, mapping))

    print("\n=== combine_dfs (concat) ===")
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
    print(combine_dfs([df1, df2], method="concat"))

    print("\n=== combine_dfs (merge) ===")
    df3 = pd.DataFrame({"ID": [1, 2], "VAL": [10, 20]})
    df4 = pd.DataFrame({"ID": [1, 3], "NAME": ["张", "李"]})
    print(combine_dfs([df3, df4], method="merge", on="ID"))

    print("\n=== export_data ===")
    export_data(df1, "test_output.csv")  # 默认 CSV
    export_data(df1, "test_output.xlsx")
    export_data(df1, "test_output.json")
    print("数据已导出到 test_output.*")

if __name__ == "__main__":
    main()