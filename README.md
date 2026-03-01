# 🚀 Horisation

> 一个基于 Flask 的 CSV/Excel 数据分析与金融建模 Web 应用

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-orange.svg)](https://pandas.pydata.org/)

## ✨ 特性

### 📊 CSV/Excel 处理
- ✅ **智能编码检测**：自动识别 UTF-8、GBK、GB2312、Big5 等编码
- ✅ **多格式支持**：CSV / Excel (.xls / .xlsx)
- ✅ **数据预览**：快速查看文件前 N 行，无需加载全部数据
- ✅ **概要统计**：行数、列数、数据类型、缺失值分析
- ✅ **数据清洗**：列名标准化、去重、类型转换
- ✅ **性能优化**：PyArrow 引擎加速（提升 30-50%）

### 🎨 现代化 UI
- 🎯 卡片式布局，分区清晰
- 📱 响应式设计，适配多种屏幕
- 🏷️ 彩色数据类型标签（文本/数值/日期）
- 🔄 平滑的 Tab 切换动画
- 📤 拖拽上传，操作便捷

### 🛠️ 架构设计
- 🏗️ **模块化设计**：业务逻辑与控制器分离
- 🧪 **易于测试**：纯函数逻辑，方便单元测试
- 📦 **代码复用**：核心功能可被多处调用
- 🔌 **可扩展性**：轻松添加新功能

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd Horisation

# 安装依赖
pip install -r requirements.txt

# 可选：安装性能优化包
pip install pyarrow  # 提升 30-50% 性能
```

### 2. 运行应用

```bash
python app.py
```

访问：http://localhost:5000

### 3. 使用 CSV 工作区

1. 访问 http://localhost:5000/csv
2. 上传 CSV/Excel 文件（拖拽或点击选择）
3. 选择编码、分隔符等选项
4. 点击"预览数据"或"数据概要"

## 📂 项目结构

```
Horisation/
├── app.py                      # Flask 应用入口
├── requirements.txt            # 项目依赖
├── QUICKSTART.md              # 快速开始指南
├── OPTIMIZATION_SUMMARY.md    # 优化总结
│
├── Backend/
│   ├── Controller/
│   │   └── csvcontroller.py   # CSV API 控制器
│   │
│   ├── Functions/             # 🆕 核心业务逻辑模块
│   │   ├── csv_processor.py   # 文件处理器
│   │   ├── csv_cleaner.py     # 数据清洗器
│   │   └── README.md          # 模块文档
│   │
│   └── Horfunc/
│       └── finpkg.py          # 金融建模函数
│
├── Template/                  # Jinja2 模板
│   ├── CSV.html              # CSV 工作区（优化版）
│   ├── Home.html             # 主页
│   └── ...
│
└── Static/                   # 静态资源
    ├── js/
    │   └── horcsv.js
    └── css/
```

## 🔌 API 端点

### 预览文件
```bash
POST /api/csv/preview
参数: ?n=10&encoding=utf-8&sep=,

返回:
{
  "ok": true,
  "filename": "data.csv",
  "columns": ["col1", "col2"],
  "rows": [{...}, {...}]
}
```

### 获取概要
```bash
POST /api/csv/summary
参数: ?encoding=utf-8

返回:
{
  "ok": true,
  "summary": {
    "rows": 1000,
    "cols": 10,
    "dtypes": {...},
    "na_count": {...},
    "na_ratio": {...}
  }
}
```

### 清洗数据
```bash
POST /api/csv/clean
参数: ?case=upper&remove_duplicates=true

返回:
{
  "ok": true,
  "cleaned_rows": 950,
  "removed_duplicates": 50,
  "columns": [...]
}
```

## 💻 代码示例

### 使用 Functions 模块

```python
from Backend.Functions import csv_processor, csv_cleaner

# 读取文件
with open('data.csv', 'rb') as f:
    binary_data = f.read()

# 获取预览
preview = csv_processor.get_preview(binary_data, n=10)
print(preview['columns'])

# 获取概要
summary = csv_processor.get_summary(binary_data)
print(f"总行数: {summary['rows']}")

# 数据清洗
df = csv_processor.read_file_to_dataframe(binary_data)
df = csv_cleaner.clean_column_names(df, case='upper')
df = csv_cleaner.remove_duplicates(df)
```

## 🎯 核心功能

### CSV Processor（文件处理器）
- 📥 **智能读取**：自动检测编码和分隔符
- 📊 **数据预览**：前 N 行快速预览
- 📈 **统计分析**：行数、列数、数据类型、缺失值
- 🚀 **性能优化**：PyArrow 引擎加速

### CSV Cleaner（数据清洗器）
- 🏷️ **列名标准化**：大小写转换、去特殊字符、去重
- 🧹 **单元格清洗**：去空格、统一缺失值
- 🔄 **数据转换**：百分比/日期标准化
- 📉 **异常值处理**：Z-score / IQR 检测

## 🧪 测试

```bash
# 运行所有测试
pytest

# 查看覆盖率
pytest --cov=Backend/Functions

# 运行特定测试
pytest tests/test_csv_processor.py
```

## 📚 文档

- 📖 [快速开始指南](./QUICKSTART.md)
- 🔧 [优化总结](./OPTIMIZATION_SUMMARY.md)
- 🛠️ [Functions 模块文档](./Backend/Functions/README.md)
- 📝 [CLAUDE.md](./CLAUDE.md) - AI 开发指南

## 🌟 最近更新

### v2.0 (2025-10-11)
- ✨ 重构架构：业务逻辑与 Controller 分离
- 🎨 优化前端：现代化卡片式布局 + Tab 切换
- 🚀 性能提升：PyArrow 引擎，编码检测优化
- 📦 新增模块：Backend/Functions（csv_processor + csv_cleaner）
- 🔌 新增 API：数据清洗端点
- 📚 完善文档：QUICKSTART、OPTIMIZATION_SUMMARY、Functions README

## 🛠️ 技术栈

**后端：**
- Flask 2.3+
- Pandas 2.0+
- NumPy
- PyArrow（可选）
- openpyxl / xlrd

**前端：**
- HTML5 + CSS3
- Vanilla JavaScript
- Font Awesome

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📞 联系方式

- GitHub: [@Horiz](https://github.com/Horiz)
- Email: your-email@example.com

---

<div align="center">

**Made with ❤️ by Horiz**

[⭐ Star this project](https://github.com/Horiz/Horisation) if you find it helpful!

</div>
