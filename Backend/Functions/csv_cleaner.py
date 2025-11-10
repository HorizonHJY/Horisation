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
            cols: list[str] | None = None,
            *,
            case: str | None = "upper",
            strip_special: bool = True
    ) -> pd.DataFrame:
        """标准化列名.

        Args:
            df: 待处理的数据框。
            cols: 需要处理的列名列表, ``None`` 表示全部列。
            case: 大小写控制, 支持 ``upper``/``lower``/``title``/``None``。
            strip_special: 是否移除除 ``_`` 以外的特殊字符。

        Returns:
            ``pd.DataFrame``: 列名规范化后的数据框, 返回副本避免原地修改。
        """

        df = df.copy()
        target_cols = set(df.columns if cols is None else cols)
        normalised = []

        for original in df.columns:
            if original not in target_cols:
                normalised.append(original)
                continue

            if not isinstance(original, str):
                normalised.append(original)
                continue

            col = original.strip()
            col = re.sub(r"\s+", "_", col)
            if strip_special:
                col = re.sub(r"[^0-9a-zA-Z_]+", "", col)

            case_value = (case or "").lower()
            if case_value == "upper":
                col = col.upper()
            elif case_value == "lower":
                col = col.lower()
            elif case_value == "title":
                col = "_".join(part.capitalize() for part in col.split("_"))

            normalised.append(col)

        df.columns = normalised
        return df

    @staticmethod
    def clean_column_name(
            df: pd.DataFrame,
            cols: list[str] | None = None,
            **kwargs: Any
    ) -> pd.DataFrame:
        """保持兼容的旧接口, 默认转为大写并替换空格。

        旧的调用方可能会额外传入 ``case``、``strip_special`` 等关键字参数,
        因此这里透传给 ``clean_column_names`` 以避免 ``TypeError``。
        """

        if "case" not in kwargs:
            kwargs["case"] = "upper"

        return CSVCleaner.clean_column_names(df, cols=cols, **kwargs)

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
    def formatting(df: pd.DataFrame, mapping: list[dict]) -> pd.DataFrame:
        df = df.copy().astype(object)

        def _column_indices(column: str) -> list[int]:
            try:
                loc = df.columns.get_loc(column)
            except KeyError:
                return []

            if isinstance(loc, slice):
                return list(range(loc.start, loc.stop))
            if isinstance(loc, (list, tuple)):
                return [int(i) for i in loc]
            if isinstance(loc, np.ndarray):
                if loc.dtype == bool:
                    return np.flatnonzero(loc).tolist()
                return [int(i) for i in loc.tolist()]
            return [int(loc)]

        def _update_column(column: str, transform):
            indices = _column_indices(column)
            for idx in indices:
                original = df.iloc[:, idx]
                transformed = transform(original)
                if isinstance(transformed, pd.Series):
                    transformed = transformed.reindex(df.index)
                    if transformed.dtype == object:
                        df.iloc[:, idx] = transformed.tolist()
                    else:
                        df.iloc[:, idx] = transformed
                else:
                    df.iloc[:, idx] = transformed

        for m in mapping:
            cols = m.get('Column', [])
            trans_type = m.get('trans_type', None)

            if trans_type == 'str':
                for col in cols:
                    if col in df.columns:
                        def _str_transform(series: pd.Series) -> pd.Series:
                            series = series.astype(str)
                            series = series.str.upper()
                            series = series.str.strip()
                            return series.str.replace(" ", "_")

                        _update_column(col, _str_transform)

            elif trans_type == 'int':
                for col in cols:
                    if col in df.columns:
                        def _int_transform(series: pd.Series) -> pd.Series:
                            return pd.to_numeric(series, errors="coerce").astype("Int64")

                        _update_column(col, _int_transform)

            elif trans_type == 'float':
                decimals = int(m.get('decimals', 4)) if m.get('decimals') is not None else 4
                for col in cols:
                    if col in df.columns:
                        def _float_transform(series: pd.Series) -> pd.Series:
                            return pd.to_numeric(series, errors="coerce").round(decimals)

                        _update_column(col, _float_transform)

            elif trans_type == 'bool':
                for col in cols:
                    if col in df.columns:
                        _update_column(col, lambda series: series.astype("boolean"))

            elif trans_type == 'percent':
                decimals = int(m.get('decimals', 2)) if m.get('decimals') is not None else 2
                for col in cols:
                    if col in df.columns:
                        def _percent_transform(series: pd.Series) -> pd.Series:
                            series = pd.to_numeric(series, errors="coerce") * 100
                            series = series.round(decimals)
                            return series.apply(lambda x: f"{x}%" if pd.notna(x) else pd.NA)

                        _update_column(col, _percent_transform)

            elif trans_type == 'date':
                date_format = m.get("format", "YYYY-MM-DD")

                from dateutil import parser
                def parse_date(val):
                    try:
                        return parser.parse(str(val), dayfirst=False, yearfirst=False)
                    except Exception:
                        return pd.NaT

                for col in cols:
                    if col in df.columns:
                        def _date_transform(series: pd.Series) -> pd.Series:
                            series = series.apply(parse_date)
                            series = pd.to_datetime(series, errors="coerce")
                            if date_format == "YYYY-MM-DD":
                                return series.dt.strftime("%Y-%m-%d")
                            if date_format == "DD_MM_YY":
                                return series.dt.strftime("%d-%m-%y")
                            if date_format == "MM-YY":
                                return series.dt.strftime("%m-%y")
                            return series

                        _update_column(col, _date_transform)

            elif trans_type == "scale":
                factor = m.get("factor", 1)
                operation = m.get("operation", "mul")
                for col in cols:
                    if col in df.columns:
                        def _scale_transform(series: pd.Series) -> pd.Series:
                            series = pd.to_numeric(series, errors="coerce")
                            if operation == "mul":
                                return series * factor
                            if operation == "div":
                                return series / factor
                            if operation == "add":
                                return series + factor
                            if operation == "sub":
                                return series - factor
                            return series

                        _update_column(col, _scale_transform)


            elif trans_type == 'missing':
                strategy = m.get("strategy", None)
                fill_value = m.get("fill_value")
                for col in cols:
                    if col in df.columns:
                        def _missing_transform(series: pd.Series) -> pd.Series:
                            if strategy == "mean":
                                numeric = pd.to_numeric(series, errors="coerce")
                                return numeric.fillna(numeric.mean())
                            if strategy == "median":
                                numeric = pd.to_numeric(series, errors="coerce")
                                return numeric.fillna(numeric.median())
                            if strategy == "mode":
                                mode_series = series.mode(dropna=True)
                                if not mode_series.empty:
                                    return series.fillna(mode_series.iloc[0])
                                return series
                            if strategy == "constant":
                                return series.fillna(fill_value)
                            if strategy == "nan":
                                return series.fillna(pd.NA)
                            return series

                        _update_column(col, _missing_transform)

            elif trans_type == "outlier":
                method = m.get("method", "zscore")
                replace = m.get("replace", "nan")
                threshold = m.get("threshold", 3)
                for col in cols:
                    if col in df.columns:
                        def _outlier_transform(series: pd.Series) -> pd.Series:
                            numeric = pd.to_numeric(series, errors="coerce")
                            if numeric.isna().all():
                                return numeric

                            mask = pd.Series(False, index=series.index)
                            lower_bound = None
                            upper_bound = None

                            if method == "zscore":
                                mean = numeric.mean()
                                std = numeric.std()
                                if pd.isna(std) or std == 0:
                                    return numeric
                                mask = (numeric - mean).abs() > threshold * std
                                lower_bound = mean - threshold * std
                                upper_bound = mean + threshold * std
                            elif method == "iqr":
                                q1 = numeric.quantile(0.25)
                                q3 = numeric.quantile(0.75)
                                if pd.isna(q1) or pd.isna(q3):
                                    return numeric
                                iqr = q3 - q1
                                lower_bound = q1 - threshold * iqr
                                upper_bound = q3 + threshold * iqr
                                mask = (numeric < lower_bound) | (numeric > upper_bound)

                            if not mask.any():
                                return numeric

                            if replace == "mean":
                                return numeric.mask(mask, numeric.mean())
                            if replace == "median":
                                return numeric.mask(mask, numeric.median())
                            if replace == "clip":
                                if lower_bound is None or upper_bound is None:
                                    return numeric
                                clipped = numeric.clip(lower_bound, upper_bound)
                                return numeric.mask(mask, clipped)
                            if replace == "nan":
                                return numeric.mask(mask, pd.NA)
                            return numeric

                        _update_column(col, _outlier_transform)

        df = df.astype(object).where(df.notna(), float("nan"))
        return df

# 全局清洗器实例
csv_cleaner = CSVCleaner()
