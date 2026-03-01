# Backend Functions 模块

## 📂 目录结构

```
Backend/Functions/
├── __init__.py           # 模块初始化
├── csv_processor.py      # CSV/Excel文件处理核心
├── csv_cleaner.py        # 数据清洗和转换
└── README.md            # 本文档
```

## 🎯 设计理念

将业务逻辑从 Controller 中分离，实现：
- **关注点分离**：Controller只负责路由，Functions负责业务逻辑
- **代码复用**：核心功能可被多个controller调用
- **易于测试**：纯函数逻辑，方便单元测试
- **可维护性**：业务逻辑集中管理，修改更方便

## 📦 模块说明

### 1. csv_processor.py - 文件处理器

**CSVProcessor 类**

核心功能：
- ✅ 多编码自动回退（UTF-8 → GBK → GB2312 → Big5...）
- ✅ Excel 文件支持（.xls / .xlsx）
- ✅ 多级表头自动展平
- ✅ PyArrow 加速（可选）
- ✅ 文件预览和概要统计

**主要方法：**

```python
from Backend.Functions import csv_processor

# 读取文件为 DataFrame
df = csv_processor.read_file_to_dataframe(
    binary=file_bytes,
    filename="data.csv",
    nrows=100,              # 限制行数（可选）
    sep=",",                # 分隔符（可选）
    encoding="utf-8"        # 编码（可选，自动检测）
)

# 获取预览（前N行）
preview = csv_processor.get_preview(
    binary=file_bytes,
    n=10,
    filename="data.xlsx"
)
# 返回: {'columns': [...], 'rows': [{...}, ...]}

# 获取概要统计
summary = csv_processor.get_summary(
    binary=file_bytes,
    filename="data.csv"
)
# 返回: {'rows': 1000, 'cols': 10, 'dtypes': {...}, 'na_count': {...}, ...}
```

**编码回退顺序：**
1. UTF-8（PyArrow引擎）
2. UTF-8-SIG（BOM处理）
3. GBK
4. GB2312
5. Big5
6. Shift_JIS
7. CP1252
8. Latin1（兜底）

### 2. csv_cleaner.py - 数据清洗器

**CSVCleaner 类**

核心功能：
- ✅ 列名标准化（大小写、去特殊字符、去重）
- ✅ 单元格数据清洗（去空格、统一缺失值）
- ✅ 重复行去除
- ✅ 百分比标准化
- ✅ 日期标准化
- ✅ 异常值检测和处理

**主要方法：**

```python
from Backend.Functions import csv_cleaner
import pandas as pd

# 1. 清洗列名
df = csv_cleaner.clean_column_names(
    df,
    case="upper",           # 'upper', 'lower', 'title'
    prefix="COL_",          # 添加前缀
    strip_special=True,     # 去除特殊字符
    dedupe=True            # 列名去重
)

# 2. 清洗单元格数据
df = csv_cleaner.clean_cell_values(
    df,
    strip_whitespace=True,     # 去除首尾空格
    normalize_missing=True     # 统一缺失值为NaN
)

# 3. 去除重复行
df = csv_cleaner.remove_duplicates(
    df,
    subset=['ID', 'Name'],    # 基于哪些列判断重复
    keep='first'              # 保留第一个
)

# 4. 百分比标准化（转为小数）
df = csv_cleaner.normalize_percent(
    df,
    percent_cols=['增长率', '收益率'],
    auto_detect=True          # 自动检测包含"率"、"比"的列
)

# 5. 日期标准化
df = csv_cleaner.normalize_dates(
    df,
    date_cols=['日期', '时间'],
    date_format="%Y-%m-%d"    # 目标格式
)

# 6. 异常值处理
df = csv_cleaner.handle_outliers(
    df,
    method="zscore",          # 'zscore' 或 'iqr'
    threshold=3.0,            # Z-score阈值
    replace_with="median",    # 'mean', 'median', 'nan'
    mark_only=False          # True=仅标记不替换
)
```

## 🔌 Controller 集成示例

**在 csvcontroller.py 中的用法：**

```python
from Backend.Functions.csv_processor import csv_processor
from Backend.Functions.csv_cleaner import csv_cleaner

@bp.post("/api/csv/preview")
def api_preview():
    filename, data, err = _get_file_and_bytes()
    if err:
        return jsonify({'ok': False, 'error': err[0]}), err[1]

    # 调用 csv_processor
    payload = csv_processor.get_preview(
        data,
        n=10,
        filename=filename
    )

    return jsonify({'ok': True, **payload})
```

## 🧪 测试示例

```python
import pytest
from Backend.Functions import csv_processor, csv_cleaner

def test_csv_processor():
    # 准备测试数据
    csv_content = b"Name,Age\nAlice,25\nBob,30"

    # 测试预览
    preview = csv_processor.get_preview(csv_content, n=5)
    assert len(preview['columns']) == 2
    assert preview['columns'] == ['Name', 'Age']

def test_csv_cleaner():
    import pandas as pd

    # 准备测试数据
    df = pd.DataFrame({
        'name ': ['  Alice  ', 'Bob'],
        '增长率': ['50%', '0.8']
    })

    # 测试列名清洗
    df = csv_cleaner.clean_column_names(df, case='upper')
    assert 'NAME' in df.columns

    # 测试百分比标准化
    df = csv_cleaner.normalize_percent(df)
    assert df['增长率'].iloc[0] == 0.5
```

## 📊 数据流图

```
用户上传文件
    ↓
csvcontroller.py (路由层)
    ├─ _get_file_and_bytes()     [验证文件]
    ├─ csv_processor.get_preview()   [读取+预览]
    ├─ csv_processor.get_summary()   [统计信息]
    └─ csv_cleaner.clean_*()        [数据清洗]
    ↓
返回 JSON 结果给前端
```

## 🚀 性能优化

1. **PyArrow 加速**：
   - 安装：`pip install pyarrow`
   - 自动检测并使用PyArrow引擎读取CSV
   - 性能提升约30-50%

2. **分块读取**：
   ```python
   # 大文件只读前1000行预览
   df = csv_processor.read_file_to_dataframe(
       binary=data,
       nrows=1000
   )
   ```

3. **类型优化**：
   - 字符串列自动转为category（节省内存）
   - 数值列推断并转换类型

## 🛠️ 扩展开发指南

### 添加新的清洗功能

1. 在 `csv_cleaner.py` 中添加静态方法：

```python
@staticmethod
def normalize_currency(df: pd.DataFrame, currency_cols: List[str]):
    """货币标准化：将 $1,234.56 转为 1234.56"""
    for col in currency_cols:
        df[col] = df[col].str.replace('[$,]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
```

2. 在 controller 中调用：

```python
@bp.post("/api/csv/normalize_currency")
def api_normalize_currency():
    # ... 获取数据 ...
    df = csv_processor.read_file_to_dataframe(data, filename=filename)
    df = csv_cleaner.normalize_currency(df, ['价格', '金额'])
    # ... 返回结果 ...
```

## 📝 注意事项

1. **编码问题**：如果自动检测失败，用户可通过API参数指定编码
2. **Excel依赖**：处理Excel需要 `openpyxl` 或 `xlrd`
3. **内存限制**：大文件（>100MB）建议分块处理或使用Dask
4. **类型转换**：所有数据统一转为字符串，避免类型错误

## 🔗 相关文档

- [Flask Blueprint 文档](https://flask.palletsprojects.com/en/2.3.x/blueprints/)
- [Pandas 性能优化](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [PyArrow 引擎说明](https://arrow.apache.org/docs/python/)
