"""
CSV数据清洗和转换功能模块
包含列名标准化、数据去重、类型转换等功能
"""

import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, List, Any


class CSVCleaner:
    """CSV数据清洗器"""

    @staticmethod
    def clean_column_names(
        df: pd.DataFrame,
        case: str = "upper",
        prefix: Optional[str] = None,
        strip_special: bool = True,
        dedupe: bool = True
    ) -> pd.DataFrame:
        """
        清洗列名

        Args:
            df: 原始DataFrame
            case: 大小写转换 ('upper', 'lower', 'title', None)
            prefix: 添加前缀
            strip_special: 去除特殊字符（仅保留字母数字下划线中文）
            dedupe: 列名去重（添加_1, _2后缀）

        Returns:
            pd.DataFrame: 列名清洗后的数据
        """
        df = df.copy()
        new_cols = []
        seen = {}

        for col in df.columns:
            name = str(col).strip()

            # 大小写转换
            if case == "upper":
                name = name.upper()
            elif case == "lower":
                name = name.lower()
            elif case == "title":
                name = name.title()

            # 去除特殊字符
            if strip_special:
                name = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "_", name).strip("_")

            # 添加前缀
            if prefix:
                name = f"{prefix}{name}"

            # 去重
            if dedupe:
                count = seen.get(name, 0)
                if count > 0:
                    name = f"{name}_{count}"
                seen[name] = count + 1

            new_cols.append(name)

        df.columns = new_cols
        return df

    @staticmethod
    def clean_cell_values(
        df: pd.DataFrame,
        strip_whitespace: bool = True,
        normalize_missing: bool = True,
        missing_values: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        清洗单元格数据

        Args:
            df: 原始DataFrame
            strip_whitespace: 去除首尾空格
            normalize_missing: 统一缺失值标记为NaN
            missing_values: 缺失值标识列表

        Returns:
            pd.DataFrame: 数据清洗后的DataFrame
        """
        df = df.copy()

        # 去除空格
        if strip_whitespace:
            obj_cols = df.select_dtypes(include=["object"]).columns
            for col in obj_cols:
                df[col] = df[col].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )

        # 统一缺失值
        if normalize_missing:
            if missing_values is None:
                missing_values = ["", "NA", "N/A", "na", "-", "null", "None", "nan"]
            df.replace(missing_values, np.nan, inplace=True)

        return df

    @staticmethod
    def remove_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        去除重复行

        Args:
            df: 原始DataFrame
            subset: 用于判断重复的列（None表示全部列）
            keep: 保留策略 ('first', 'last', False)

        Returns:
            pd.DataFrame: 去重后的数据
        """
        return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    @staticmethod
    def normalize_percent(
        df: pd.DataFrame,
        percent_cols: Optional[List[str]] = None,
        auto_detect: bool = True
    ) -> pd.DataFrame:
        """
        百分比标准化（转为小数）

        Args:
            df: 原始DataFrame
            percent_cols: 指定的百分比列
            auto_detect: 是否自动检测包含"率"、"比"、"%"的列

        Returns:
            pd.DataFrame: 处理后的数据
        """
        df = df.copy()

        if auto_detect and percent_cols is None:
            percent_cols = [
                c for c in df.columns
                if any(k in str(c) for k in ["率", "比", "%"])
            ]

        if not percent_cols:
            return df

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
            if col in df.columns:
                df[col] = df[col].apply(convert)

        return df

    @staticmethod
    def normalize_dates(
        df: pd.DataFrame,
        date_cols: Optional[List[str]] = None,
        date_format: str = "%Y-%m-%d"
    ) -> pd.DataFrame:
        """
        日期标准化

        Args:
            df: 原始DataFrame
            date_cols: 日期列列表（None表示尝试所有列）
            date_format: 目标日期格式

        Returns:
            pd.DataFrame: 处理后的数据
        """
        df = df.copy()
        cols_to_process = date_cols if date_cols else df.columns

        for col in cols_to_process:
            if col not in df.columns:
                continue

            def convert(x):
                if pd.isna(x):
                    return x
                try:
                    return pd.to_datetime(x, errors="raise").strftime(date_format)
                except Exception:
                    return x

            df[col] = df[col].apply(convert)

        return df

    @staticmethod
    def handle_outliers(
        df: pd.DataFrame,
        method: str = "zscore",
        threshold: float = 3.0,
        replace_with: str = "median",
        mark_only: bool = False
    ) -> pd.DataFrame:
        """
        异常值处理

        Args:
            df: 原始DataFrame
            method: 检测方法 ('zscore' 或 'iqr')
            threshold: Z-score阈值（默认3.0）或IQR倍数（默认1.5）
            replace_with: 替换策略 ('mean', 'median', 'nan')
            mark_only: 仅标记不替换（添加_outlier列）

        Returns:
            pd.DataFrame: 处理后的数据
        """
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            # 检测异常值
            if method == "zscore":
                mean, std = df[col].mean(), df[col].std()
                is_outlier = (df[col] - mean).abs() > threshold * std
            elif method == "iqr":
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
                is_outlier = (df[col] < lower) | (df[col] > upper)
            else:
                continue

            # 标记异常值
            if mark_only:
                df[f"{col}_outlier"] = is_outlier.astype(int)
            else:
                # 替换异常值
                if replace_with == "mean":
                    df.loc[is_outlier, col] = df[col].mean()
                elif replace_with == "median":
                    df.loc[is_outlier, col] = df[col].median()
                elif replace_with == "nan":
                    df.loc[is_outlier, col] = np.nan

        return df


# 全局清洗器实例
csv_cleaner = CSVCleaner()
