"""
CSV/Excel 文件处理核心功能模块
将文件读取、预览、摘要等功能从controller中分离
"""

from io import BytesIO
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
import re

# 依赖检查
_USE_PYARROW = False
_HAS_OPENPYXL = False
_HAS_XLRD = False

try:
    import pyarrow
    _USE_PYARROW = True
except ImportError:
    pass

try:
    import openpyxl
    _HAS_OPENPYXL = True
except ImportError:
    pass

try:
    import xlrd
    _HAS_XLRD = True
except ImportError:
    pass

# 常量配置
MISSING_VALUES = {'', 'nan', 'None', 'NaN', 'null', 'NULL', 'NA', 'N/A'}
NUMERIC_PATTERN = re.compile(r'^-?\d+(\.\d+)?$')
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}')


class CSVProcessor:
    """CSV/Excel 文件处理器"""

    def __init__(self):
        self.use_pyarrow = _USE_PYARROW
        self.has_openpyxl = _HAS_OPENPYXL
        self.has_xlrd = _HAS_XLRD

    def read_with_encoding_fallback(
        self,
        binary: bytes,
        nrows: Optional[int] = None,
        sep: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> pd.DataFrame:
        """
        使用编码回退机制读取CSV

        Args:
            binary: CSV二进制数据
            nrows: 限制读取行数
            sep: 分隔符
            encoding: 指定编码（如提供则直接使用）

        Returns:
            pd.DataFrame: 解析后的数据
        """
        base_kwargs = {}
        if nrows is not None:
            base_kwargs["nrows"] = nrows
        if sep is not None:
            base_kwargs["sep"] = sep

        # 指定编码直接使用
        if encoding:
            bio = BytesIO(binary)
            return pd.read_csv(bio, encoding=encoding, **base_kwargs)

        # UTF-8 优先（使用PyArrow加速）
        utf8_tries = [
            ("utf-8", self.use_pyarrow),
            ("utf-8-sig", self.use_pyarrow)
        ]

        for enc, use_pa in utf8_tries:
            try:
                bio = BytesIO(binary)
                engine = "pyarrow" if use_pa else "c"
                return pd.read_csv(bio, encoding=enc, engine=engine, **base_kwargs)
            except Exception:
                continue

        # 本地编码回退
        local_encodings = ["gbk", "gb2312", "big5", "shift_jis", "cp1252"]
        for enc in local_encodings:
            try:
                bio = BytesIO(binary)
                return pd.read_csv(bio, encoding=enc, **base_kwargs)
            except Exception:
                continue

        # 最后兜底：latin1
        bio = BytesIO(binary)
        return pd.read_csv(bio, encoding="latin1", **base_kwargs)

    def read_file_to_dataframe(
        self,
        binary: bytes,
        filename: Optional[str] = None,
        nrows: Optional[int] = None,
        sep: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> pd.DataFrame:
        """
        统一入口：将CSV/Excel读取为DataFrame

        Args:
            binary: 文件二进制数据
            filename: 文件名（用于判断文件类型）
            nrows: 限制读取行数
            sep: CSV分隔符
            encoding: CSV编码

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        name = (filename or "").lower()

        # Excel文件处理
        if name.endswith((".xls", ".xlsx")):
            return self._read_excel(binary, name)

        # CSV文件处理
        df = self.read_with_encoding_fallback(binary, nrows=nrows, sep=sep, encoding=encoding)

        # 统一转换为字符串，避免类型问题
        df = df.astype(str)

        return df

    def _read_excel(self, binary: bytes, filename: str) -> pd.DataFrame:
        """读取Excel文件"""
        bio = BytesIO(binary)

        # 检查依赖
        if filename.endswith(".xlsx") and not self.has_openpyxl:
            raise ImportError("处理 .xlsx 文件需要 openpyxl 库，请安装: pip install openpyxl")
        elif filename.endswith(".xls") and not self.has_xlrd:
            raise ImportError("处理 .xls 文件需要 xlrd 库，请安装: pip install xlrd")

        # 读取Excel
        try:
            engine = "openpyxl" if filename.endswith(".xlsx") else "xlrd"
            df = pd.read_excel(bio, engine=engine)
        except Exception as e:
            # 兜底尝试自动检测
            try:
                df = pd.read_excel(bio)
            except Exception:
                raise e

        # 展平多级表头
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(x) for x in col if x])
                for col in df.columns.values
            ]

        # 转换为字符串
        df = df.astype(str)

        return df

    def get_preview(
        self,
        binary: bytes,
        n: int = 5,
        sep: Optional[str] = None,
        filename: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> Dict:
        """
        获取文件预览（前N行）

        Args:
            binary: 文件二进制数据
            n: 预览行数
            sep: CSV分隔符
            filename: 文件名
            encoding: CSV编码

        Returns:
            dict: {'columns': [...], 'rows': [{...}, ...]}
        """
        df = self.read_file_to_dataframe(
            binary,
            filename=filename,
            nrows=n,
            sep=sep,
            encoding=encoding
        )

        head = df.head(n)

        return {
            'columns': list(head.columns),
            'rows': head.to_dict(orient='records')
        }

    def get_summary(
        self,
        binary: bytes,
        sep: Optional[str] = None,
        filename: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> Dict:
        """
        获取文件概要统计信息

        Args:
            binary: 文件二进制数据
            sep: CSV分隔符
            filename: 文件名
            encoding: CSV编码

        Returns:
            dict: 包含行数、列数、数据类型、缺失值统计等
        """
        df = self.read_file_to_dataframe(
            binary,
            filename=filename,
            sep=sep,
            encoding=encoding
        )

        # 计算缺失值
        na_count = {}
        for col in df.columns:
            is_missing = df[col].isin(MISSING_VALUES) | df[col].isna()
            na_count[col] = int(is_missing.sum())

        total_rows = len(df)
        na_ratio = {
            k: round(v / total_rows, 4) if total_rows else 0.0
            for k, v in na_count.items()
        }

        # 推断数据类型
        dtypes = self._infer_column_types(df)

        return {
            'rows': int(total_rows),
            'cols': int(df.shape[1]),
            'columns': list(df.columns),
            'dtypes': dtypes,
            'na_count': na_count,
            'na_ratio': na_ratio
        }

    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """推断列数据类型"""
        dtypes = {}

        for col in df.columns:
            # 获取非空值
            non_empty = df[col][~(df[col].isin(MISSING_VALUES) | df[col].isna())]

            if len(non_empty) == 0:
                dtypes[col] = 'unknown'
                continue

            # 取样本值
            sample = str(non_empty.iloc[0])

            # 判断类型
            if NUMERIC_PATTERN.match(sample):
                dtypes[col] = 'numeric'
            elif DATE_PATTERN.match(sample):
                dtypes[col] = 'date'
            else:
                dtypes[col] = 'text'

        return dtypes


# 全局处理器实例
csv_processor = CSVProcessor()
