# 🚀 Horisation Quick Start Guide

## 📋 前置要求

- Python 3.8+
- pip

## 🔧 安装步骤

### 1. 克隆项目（如果尚未克隆）

```bash
git clone <your-repo-url>
cd Horisation
```

### 2. 创建虚拟环境（推荐）

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**可选性能优化（推荐）：**
```bash
# 安装 PyArrow 以获得 30-50% 的 CSV 读取性能提升
pip install pyarrow
```

### 4. 验证安装

```bash
python -c "from Backend.Functions import csv_processor; print('Installation successful!')"
```

## 🎮 运行应用

```bash
python app.py
```

你将看到：
```
============================================================
🚀 Horisation Application Starting...
============================================================
📂 Template Directory: .../Template (exists: True)
📂 Static Directory: .../Static (exists: True)
📂 Upload Directory: .../_uploads (exists: True)
============================================================
✅ Registered Blueprint: csv_api

============================================================
🌐 Server running at: http://localhost:5000
📊 CSV Workspace: http://localhost:5000/csv
============================================================
```

## 📱 访问应用

打开浏览器访问：

- 🏠 **主页**: http://localhost:5000
- 📊 **CSV工作区**: http://localhost:5000/csv
- 📝 **备忘录**: http://localhost:5000/hormemo
- 📈 **限额跟踪**: http://localhost:5000/limit

## 🧪 测试 API

### 预览 CSV 文件

```bash
curl -X POST \
  -F "file=@your_data.csv" \
  "http://localhost:5000/api/csv/preview?n=10"
```

### 获取文件概要

```bash
curl -X POST \
  -F "file=@your_data.csv" \
  "http://localhost:5000/api/csv/summary"
```

### 清洗数据

```bash
curl -X POST \
  -F "file=@your_data.csv" \
  "http://localhost:5000/api/csv/clean?case=upper&remove_duplicates=true"
```

## 📊 使用示例

### 1. 上传 CSV 文件

1. 访问 http://localhost:5000/csv
2. 拖放 CSV/Excel 文件到上传区域，或点击"选择文件"
3. 选择编码、分隔符等选项
4. 点击"预览数据"或"数据概要"

### 2. 查看数据预览

- 查看前 N 行数据
- 列名带颜色标签：
  - 🔵 文本类型
  - 🟢 数值类型
  - 🟡 日期类型

### 3. 查看统计信息

切换到"统计信息"Tab：
- 总行数、列数
- 数据类型分布
- 缺失值统计
- 缺失率

### 4. 数据清洗

点击"清洗数据"按钮：
- 列名标准化（大小写、去特殊字符）
- 去除重复行
- 单元格数据清洗

## 🔍 功能模块说明

### Backend/Functions/

**csv_processor.py** - 文件处理器
- 多编码自动检测（UTF-8 → GBK → GB2312...）
- Excel 文件支持
- 数据类型推断

**csv_cleaner.py** - 数据清洗器
- 列名标准化
- 数据去重
- 百分比/日期标准化
- 异常值检测

## ⚙️ 配置说明

### 修改最大文件大小

编辑 `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 改为 200MB
```

### 修改编码检测顺序

编辑 `Backend/Functions/csv_processor.py`:
```python
local_encodings = ["gbk", "gb2312", "big5", "shift_jis", "cp1252"]
# 调整顺序或添加新编码
```

## 🐛 常见问题

### Q1: 找不到 pandas 模块

```bash
pip install pandas numpy
```

### Q2: Excel 文件无法读取

```bash
# 安装 Excel 依赖
pip install openpyxl xlrd
```

### Q3: 性能较慢

```bash
# 安装 PyArrow 加速
pip install pyarrow
```

### Q4: 端口被占用

修改 `app.py` 最后一行：
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # 改为 5001
```

## 📚 进阶使用

### 代码示例：使用 Functions 模块

```python
from Backend.Functions import csv_processor, csv_cleaner
import pandas as pd

# 读取 CSV
with open('data.csv', 'rb') as f:
    binary_data = f.read()

# 获取预览
preview = csv_processor.get_preview(binary_data, n=10)
print(preview['columns'])  # 列名列表
print(preview['rows'])     # 前10行数据

# 获取概要
summary = csv_processor.get_summary(binary_data)
print(f"总行数: {summary['rows']}")
print(f"总列数: {summary['cols']}")

# 数据清洗
df = csv_processor.read_file_to_dataframe(binary_data)
df = csv_cleaner.clean_column_names(df, case='upper')
df = csv_cleaner.remove_duplicates(df)
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_csv_processor.py

# 查看覆盖率
pytest --cov=Backend/Functions
```

## 🔗 相关资源

- 📖 [完整文档](./OPTIMIZATION_SUMMARY.md)
- 🛠️ [Functions 模块文档](./Backend/Functions/README.md)
- 📊 [Pandas 文档](https://pandas.pydata.org/docs/)
- 🌐 [Flask 文档](https://flask.palletsprojects.com/)

## 🆘 获取帮助

遇到问题？
1. 查看 [OPTIMIZATION_SUMMARY.md](./OPTIMIZATION_SUMMARY.md)
2. 查看 [Backend/Functions/README.md](./Backend/Functions/README.md)
3. 提交 GitHub Issue

---

**Happy Coding! 🎉**
