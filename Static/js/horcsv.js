// Static/js/horcsv.js
(() => {
  const $ = (id) => document.getElementById(id);
  const fileInput = $('fileInput');
  const btnChoose = $('btnChoose');
  const btnPreview = $('btnPreview');
  const btnSummary = $('btnSummary');
  const dropzone = $('dropzone');
  const rowsN = $('rowsN');
  const encodingSelect = $('encodingSelect');
  const separatorInput = $('separatorInput');
  const columnsArea = $('columnsArea');
  const previewTable = $('previewTable');
  const statusEl = $('status');
  const fileNameEl = $('fileName');
  const summaryBox = $('summaryBox');

  let currentFile = null;

  function setStatus(msg) { if (statusEl) statusEl.textContent = msg || ''; }
  function clearTable() { if (previewTable) previewTable.innerHTML = ''; }
  function clearColumns() { if (columnsArea) columnsArea.innerHTML = ''; }

  function renderColumns(cols) {
    clearColumns();
    (cols || []).forEach(c => {
      const span = document.createElement('span');
      span.className = 'chip';
      span.textContent = c;
      columnsArea.appendChild(span);
    });
  }

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

  // 选择文件
  btnChoose?.addEventListener('click', () => fileInput?.click());
  fileInput?.addEventListener('change', () => {
    currentFile = fileInput.files?.[0] || null;
    fileNameEl.textContent = currentFile ? `选中文件: ${currentFile.name}` : '';
    setStatus('');
    clearTable();
    clearColumns();
    if (summaryBox) summaryBox.textContent = '';
  });

  // 全局阻止默认
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
    window.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  // dropzone 管理
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
      if (dragCounter === 0) dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
      dragCounter = 0;
      dropzone.classList.remove('dragover');
      const files = e.dataTransfer?.files;
      if (!files || !files.length) return;
      const file = files[0];
      currentFile = file;
      try {
        if (fileInput) fileInput.files = files;
      } catch { }
      fileNameEl.textContent = `选中文件: ${file.name}`;
      setStatus('');
      clearTable();
      clearColumns();
      if (summaryBox) summaryBox.textContent = '';
    });
  }

  // 构建查询参数
  function buildQueryParams() {
    const params = new URLSearchParams();
    const n = Math.max(1, Math.min(2000, parseInt(rowsN?.value || '5', 10)));
    params.append('n', n);

    const encoding = encodingSelect.value;
    if (encoding) {
      params.append('encoding', encoding);
    }

    const separator = separatorInput.value.trim();
    if (separator && separator !== ',') {
      params.append('sep', encodeURIComponent(separator));
    }

    return params.toString();
  }

  // 预览
  btnPreview?.addEventListener('click', async () => {
    try {
      if (!currentFile) {
        setStatus('请先选择文件 / Please choose a file.');
        return;
      }

      const queryParams = buildQueryParams();
      const fd = new FormData();
      fd.append('file', currentFile, currentFile.name);
      setStatus('上传并解析中… Uploading & parsing…');

      const resp = await fetch(`/api/csv/preview?${queryParams}`, {
        method: 'POST',
        body: fd
      });

      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      renderColumns(data.columns || []);
      renderTable(data.columns || [], data.rows || []);

      if (summaryBox) summaryBox.textContent = '';
      setStatus(`预览完成：${(data.columns || []).length} 列 / ${(data.rows || []).length} 行`);
    } catch (e) {
      console.error(e);
      setStatus(`解析失败 / Failed: ${e.message || e}`);
    }
  });

  // 概要
  btnSummary?.addEventListener('click', async () => {
    try {
      if (!currentFile) {
        setStatus('请先选择文件 / Please choose a file.');
        return;
      }

      const queryParams = buildQueryParams();
      const fd = new FormData();
      fd.append('file', currentFile, currentFile.name);
      setStatus('计算概要中… Summarizing…');

      const resp = await fetch(`/api/csv/summary?${queryParams}`, {
        method: 'POST',
        body: fd
      });

      const data = await resp.json();
      if (!resp.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      const s = data.summary || {};
      const lines = [
        `Rows: ${s.rows}  Cols: ${s.cols}`,
        `Columns: ${Array.isArray(s.columns) ? s.columns.join(', ') : ''}`,
        `Dtypes: ${s.dtypes ? JSON.stringify(s.dtypes) : ''}`,
        `NA Count: ${s.na_count ? JSON.stringify(s.na_count) : ''}`
      ];

      if (summaryBox) summaryBox.textContent = lines.join('  |  ');
      setStatus('概要已生成 / Summary ready.');
    } catch (e) {
      console.error(e);
      setStatus(`概要失败 / Failed: ${e.message || e}`);
    }
  });
})();