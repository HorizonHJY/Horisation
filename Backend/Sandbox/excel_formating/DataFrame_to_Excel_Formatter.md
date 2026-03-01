# 🧩 DataFrame-to-Excel Formatting Function — Concept & Design Document  
**Author:** Horizon Yuan  
**Version:** v0.1 (Concept Stage)  
**Last Updated:** 2025-10-12  

---

## 📘 1. Project Overview

This project aims to design a **Python-based universal formatting function** that transforms any given **pandas DataFrame** into a **stylized Excel output**.  

Unlike standard `.to_excel()` exports, this system will provide a flexible **formatting configuration layer** — allowing developers to control **how each column, row, and cell** is displayed inside Excel, with a focus on **clarity, modularity, and customization**.

The project will be the foundation for a larger formatting framework, where **style configuration and logic are decoupled** — meaning data logic stays clean, while Excel appearance can be governed externally through dictionaries or JSON templates.

---

## 🧠 2. Core Design Philosophy

### 🎯 Purpose
To build a **functional yet extensible** system that can:
- Reformat **DataFrame column data types** into well-defined Excel column formats.
- Apply **custom styling** at the **cell level** — controlling font, size, color, and background.
- Optionally define **sheet-wide properties** like borders.

### 🔍 Key Principle
> “Data defines structure, configuration defines style.”

The goal is **not** to handle complex visualization (like freezing panes or conditional formatting), but to establish a **robust and reusable pipeline** that defines how raw DataFrames should appear in Excel, column by column and cell by cell.

---

## 🧩 3. Functional Architecture Overview

### (1) Inputs
| Component | Description |
|------------|--------------|
| `DataFrame` | The source data table. |
| `style_dict` | A dictionary (or JSON file) that defines formatting rules at the column, cell, and sheet level. |
| `output_path` | Target Excel file name or path. |

### (2) Processing Flow
1. **Read** style definitions (from dict or JSON).  
2. **Map** each column’s type to an Excel display format.  
3. **Apply** font and color settings to specific cells.  
4. **Render** the styled DataFrame to Excel using openpyxl/xlsxwriter.  

---

## 🧱 4. Formatting Logic Structure

### 🧩 (a) Column-Level Formatting
Each column will be reformatted based on its **data type** and target Excel format.  

Example:
| Data Type | Excel Format | Example Output |
|------------|--------------|----------------|
| Float | `0.00` | `1234.56` |
| Integer | `0` | `1234` |
| Date | `yyyy-mm-dd` | `2025-10-12` |
| String | Text, left-aligned | `Sample Text` |

---

### 🧩 (b) Cell-Level Formatting
At the cell level, each cell can have fully customizable styles:
- Font size  
- Bold or normal  
- Font color  
- Background color  
- Border (on or off)

Example definition:
```python
"cells": {
    (0, 0): {"font_size": 14, "bold": True, "bg_color": "#D6EAF8"},
    (1, 2): {"font_color": "#FF0000", "font_size": 11},
    (2, 3): {"bg_color": "#F9EBEA", "bold": False}
}
```

---

### 🧩 (c) Sheet-Level Settings
- Global border control  
- Sheet title or name customization  
- Future extension: gridline visibility, cell alignment, etc.

Example:
```python
"sheet": {
    "border": True,
    "title": "Portfolio Risk Report"
}
```

---

### 📋 Full Example of a Style Dictionary
```python
style_dict = {
    "columns": {
        "A": {"type": "float", "format": "0.00"},
        "B": {"type": "date", "format": "yyyy-mm-dd"},
        "C": {"type": "string"}
    },
    "cells": {
        (0, 0): {"font_size": 14, "bold": True, "bg_color": "#AED6F1"},
        (1, 1): {"font_color": "#FF0000"},
        (2, 3): {"bg_color": "#FADBD8", "border": True}
    },
    "sheet": {
        "border": True
    }
}
```

---

## ⚙️ 5. Implementation Plan

### (a) Library Selection
| Library | Reason |
|----------|--------|
| **openpyxl** | Best suited for post-processing and per-cell style editing. |
| **pandas** | DataFrame handling and initial Excel write-out. |
| **xlsxwriter** | Optional alternative for performance when handling large files. |

---

### (b) Core Class Design

```python
class ExcelFormatter:
    """
    A class to convert DataFrame into styled Excel outputs.
    """

    def __init__(self, style_config: dict):
        self.style_config = style_config

    def apply_column_format(self, worksheet):
        """
        Apply formatting rules at the column level.
        """
        pass

    def apply_cell_format(self, worksheet):
        """
        Apply styling (font, color, background) at the cell level.
        """
        pass

    def export(self, df, output_path: str):
        """
        Combine formatting and export to Excel file.
        """
        pass
```

---

## 📦 6. Reference Projects for Design Inspiration

| Project | Description | Key Takeaways |
|----------|--------------|----------------|
| **StyleFrame** ([GitHub](https://github.com/DeepSpace2/StyleFrame)) | Combines pandas and openpyxl for cell-level styling. | Learn structure for style abstraction and column mapping. |
| **XlsxPandasFormatter** ([GitHub](https://github.com/webermarcolivier/xlsxpandasformatter)) | Provides API for hierarchical formatting. | Study how column/row/cell styles are prioritized and applied. |
| **Pandabook** ([GitHub](https://github.com/pretoriusdre/pandabook)) | Enables multi-sheet exports with predefined style templates. | Learn reusable style template management. |
| **copy_xlsx_styles** ([GitHub](https://github.com/Sydney-Informatics-Hub/copy_xlsx_styles)) | Copies Excel style from one sheet to another. | Useful for template-based formatting extensions. |

---

## 🎯 7. Current Stage Goals (Milestone v0.1)

| Level | Functionality | Status |
|--------|----------------|---------|
| **Column-level** | Reformat DataFrame types → Excel display formats | 🟢 Planned |
| **Cell-level** | Font size, bold, font color, background color | 🟢 Planned |
| **Sheet-level** | Border toggle | 🟢 Planned |
| **Export** | Output single DataFrame → Excel sheet | 🟢 Planned |
| **Testing** | Verify via openpyxl style attributes | 🔜 Next step |

---

## 🔮 8. Future Extensions (Milestone v0.2+)
- Conditional formatting rules  
- JSON-based reusable templates  
- Column auto-width calculation  
- Support for multiple DataFrames → multiple sheets  
- Merge-cell / title-row features  
- Integration with front-end UI (Flask or Vue dashboard)

---

## 💬 9. General Project Prompt (for Docs or AI-assisted Continuation)

> **Prompt Summary:**  
>  
> Design a Python module that converts a pandas DataFrame into an Excel file with customizable formatting.  
>  
> The project should use a dictionary (or JSON) configuration to define styling rules at three levels:
> - **Column level** → map data types to Excel number formats.  
> - **Cell level** → control font size, boldness, font color, and background color.  
> - **Sheet level** → optionally apply border or sheet-wide rules.  
>  
> This phase focuses on building a lightweight, reusable, and extensible pipeline for formatting.  
>  
> Future versions may introduce conditional formatting, reusable templates, and style inheritance.

---

## 📈 10. Summary of Progress

- ✅ Idea & structure defined  
- ✅ Core dictionary schema designed  
- ✅ Technical direction (openpyxl-based) chosen  
- ✅ Reference libraries identified  
- 🔜 Implementation: class skeleton & function logic  
- 🔜 Testing: sample DataFrame export with formatting applied  

---

**📅 Next milestone:**  
Implement and test `ExcelFormatter.export()` on small mock DataFrames (5×5) to confirm per-cell and per-column style application.
