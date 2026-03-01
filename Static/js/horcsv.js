// Static/js/horcsv.js
(() => {
  const $ = (id) => document.getElementById(id);

  // DOM 元素
  const fileInput = $('fileInput');
  const btnChoose = $('btnChoose');
  const btnPreview = $('btnPreview');
  const btnSummary = $('btnSummary');
  const btnClean = $('btnClean');
  const dropzone = $('dropzone');
  const rowsN = $('rowsN');
  const encodingSelect = $('encodingSelect');
  const separatorInput = $('separatorInput');
  const caseSelect = $('caseSelect');
  const columnsArea = $('columnsArea');
  const previewTable = $('previewTable');
  const statusEl = $('statusMessage');  // 修正：从 'status' 改为 'statusMessage'
  const fileNameEl = $('fileName');
  const resultSection = $('resultSection');
  const summaryCards = $('summaryCards');
  const summaryDetails = $('summaryDetails');

  let currentFile = null;

  // 工具函数
  function setStatus(msg, type = 'info') {
    if (!statusEl) return;
    statusEl.innerHTML = '';
    if (!msg) return;

    const badge = document.createElement('div');
    badge.className = `status-badge status-${type}`;
    badge.textContent = msg;
    statusEl.appendChild(badge);
  }

  function clearTable() {
    if (previewTable) previewTable.innerHTML = '';
  }

  function clearColumns() {
    if (columnsArea) columnsArea.innerHTML = '';
  }

  function showResultSection() {
    if (resultSection) {
      resultSection.style.display = 'block';
    }
  }

  // 渲染列标签（带类型颜色）
  function renderColumns(cols, dtypes = {}) {
    clearColumns();
    (cols || []).forEach(c => {
      const span = document.createElement('span');
      span.className = 'chip';

      // 根据数据类型添加样式
      const dtype = dtypes[c] || 'text';
      if (dtype === 'numeric') {
        span.classList.add('numeric');
      } else if (dtype === 'date') {
        span.classList.add('date');
      }

      span.textContent = c;
      columnsArea.appendChild(span);
    });
  }

  // 渲染表格
  function renderTable(cols, rows) {
    clearTable();
    if (!cols?.length) return;

    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    cols.forEach(c => {
      const th = document.createElement('th');
      th.textContent = c;
      trh.appendChild(th);
    });
    thead.appendChild(trh);

    const tbody = document.createElement('tbody');
    (rows || []).forEach(r => {
      const tr = document.createElement('tr');
      cols.forEach(c => {
        const td = document.createElement('td');
        let v = r[c];
        if (v === null || v === undefined) v = '';
        td.textContent = String(v);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    previewTable.appendChild(thead);
    previewTable.appendChild(tbody);
  }

  // 渲染概要卡片
  function renderSummaryCards(summary) {
    if (!summaryCards) return;
    summaryCards.innerHTML = '';

    const cards = [
      { label: '总行数', value: summary.rows || 0 },
      { label: '总列数', value: summary.cols || 0 },
      { label: '缺失值', value: Object.values(summary.na_count || {}).reduce((a, b) => a + b, 0) }
    ];

    cards.forEach(card => {
      const div = document.createElement('div');
      div.className = 'summary-card';
      div.innerHTML = `
        <div class="summary-value">${card.value}</div>
        <div class="summary-label">${card.label}</div>
      `;
      summaryCards.appendChild(div);
    });
  }

  // 渲染概要详情
  function renderSummaryDetails(summary) {
    if (!summaryDetails) return;

    const lines = [];

    // 列名
    if (summary.columns) {
      lines.push(`<strong>列名：</strong>${summary.columns.join(', ')}`);
    }

    // 数据类型
    if (summary.dtypes) {
      const dtypesStr = Object.entries(summary.dtypes)
        .map(([col, type]) => `${col}: ${type}`)
        .join(', ');
      lines.push(`<strong>数据类型：</strong>${dtypesStr}`);
    }

    // 缺失值统计
    if (summary.na_count) {
      const naStr = Object.entries(summary.na_count)
        .filter(([_, count]) => count > 0)
        .map(([col, count]) => `${col}: ${count}`)
        .join(', ');
      if (naStr) {
        lines.push(`<strong>缺失值统计：</strong>${naStr}`);
      }
    }

    summaryDetails.innerHTML = lines.join('<br>');
  }

  // 选择文件按钮
  btnChoose?.addEventListener('click', () => fileInput?.click());

  // 文件选择事件
  fileInput?.addEventListener('change', () => {
    currentFile = fileInput.files?.[0] || null;
    if (currentFile) {
      fileNameEl.textContent = `📄 ${currentFile.name}`;
      fileNameEl.style.display = 'inline-flex';
    } else {
      fileNameEl.textContent = '';
      fileNameEl.style.display = 'none';
    }
    setStatus('');
    clearTable();
    clearColumns();
  });

  // 全局阻止拖拽默认行为
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
    window.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  // 拖拽上传
  if (dropzone) {
    let dragCounter = 0;

    dropzone.addEventListener('dragenter', () => {
      dragCounter++;
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragover', (e) => {
      e.dataTransfer.dropEffect = 'copy';
    });

    dropzone.addEventListener('dragleave', () => {
      dragCounter = Math.max(0, dragCounter - 1);
      if (dragCounter === 0) {
        dropzone.classList.remove('dragover');
      }
    });

    dropzone.addEventListener('drop', (e) => {
      dragCounter = 0;
      dropzone.classList.remove('dragover');

      const files = e.dataTransfer?.files;
      if (!files || !files.length) return;

      const file = files[0];
      currentFile = file;

      try {
        // 同步到 file input
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
      } catch (err) {
        console.warn('无法同步到 file input:', err);
      }

      fileNameEl.textContent = `📄 ${file.name}`;
      fileNameEl.style.display = 'inline-flex';
      setStatus('');
      clearTable();
      clearColumns();
    });
  }

  // 构建查询参数
  function buildQueryParams() {
    const params = new URLSearchParams();

    // 预览行数
    const n = Math.max(1, Math.min(2000, parseInt(rowsN?.value || '10', 10)));
    params.append('n', n);

    // 编码
    const encoding = encodingSelect?.value;
    if (encoding) {
      params.append('encoding', encoding);
    }

    // 分隔符
    const separator = separatorInput?.value?.trim();
    if (separator && separator !== ',') {
      params.append('sep', separator);
    }

    return params.toString();
  }

  // 预览按钮
  btnPreview?.addEventListener('click', async () => {
    try {
      if (!currentFile) {
        setStatus('请先选择文件', 'error');
        return;
      }

      const queryParams = buildQueryParams();
      const fd = new FormData();
      fd.append('file', currentFile, currentFile.name);

      setStatus('上传并解析中...', 'info');
      btnPreview.disabled = true;

      const resp = await fetch(`/api/csv/preview?${queryParams}`, {
        method: 'POST',
        body: fd
      });

      const data = await resp.json();

      if (!resp.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      // 渲染结果
      renderColumns(data.columns || []);
      renderTable(data.columns || [], data.rows || []);
      showResultSection();

      setStatus(`预览完成：${(data.columns || []).length} 列，${(data.rows || []).length} 行`, 'success');
    } catch (e) {
      console.error(e);
      setStatus(`解析失败：${e.message}`, 'error');
    } finally {
      btnPreview.disabled = false;
    }
  });

  // 概要按钮
  btnSummary?.addEventListener('click', async () => {
    try {
      if (!currentFile) {
        setStatus('请先选择文件', 'error');
        return;
      }

      const queryParams = buildQueryParams();
      const fd = new FormData();
      fd.append('file', currentFile, currentFile.name);

      setStatus('计算概要统计中...', 'info');
      btnSummary.disabled = true;

      const resp = await fetch(`/api/csv/summary?${queryParams}`, {
        method: 'POST',
        body: fd
      });

      const data = await resp.json();

      if (!resp.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      const summary = data.summary || {};

      // 渲染概要信息
      renderSummaryCards(summary);
      renderSummaryDetails(summary);
      renderColumns(summary.columns || [], summary.dtypes || {});
      showResultSection();

      // 切换到概要Tab
      document.querySelector('.tab[data-tab="summary"]')?.click();

      setStatus(`概要已生成`, 'success');
    } catch (e) {
      console.error(e);
      setStatus(`概要失败：${e.message}`, 'error');
    } finally {
      btnSummary.disabled = false;
    }
  });

  // 清洗按钮
  btnClean?.addEventListener('click', async () => {
    try {
      if (!currentFile) {
        setStatus('请先选择文件', 'error');
        return;
      }

      const fd = new FormData();
      fd.append('file', currentFile, currentFile.name);

      // 构建清洗参数
      const params = new URLSearchParams();
      const caseValue = caseSelect?.value || 'upper';
      params.append('case', caseValue);
      params.append('strip_special', 'true');
      params.append('remove_duplicates', 'true');

      setStatus('清洗数据中...', 'info');
      btnClean.disabled = true;

      const resp = await fetch(`/api/csv/clean?${params.toString()}`, {
        method: 'POST',
        body: fd
      });

      const data = await resp.json();

      if (!resp.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      // 显示清洗结果
      const msg = `清洗完成：${data.cleaned_rows} 行，移除重复 ${data.removed_duplicates} 行`;
      setStatus(msg, 'success');

      // 显示清洗后的列名
      if (data.columns) {
        renderColumns(data.columns);
        showResultSection();
      }
    } catch (e) {
      console.error(e);
      setStatus(`清洗失败：${e.message}`, 'error');
    } finally {
      btnClean.disabled = false;
    }
  });

  console.log('✅ CSV 工作区已初始化');
})();
