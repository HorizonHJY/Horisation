// Static/js/horcsv.js
(() => {
  const $ = (id) => document.getElementById(id);

  // DOM 元素
  const fileInput = $('fileInput');
  const btnChoose = $('btnChoose');
  const btnClean = $('btnClean');
  const btnPreview = $('btnPreview');
  const btnSummary = $('btnSummary');
  const uploadDropzone = $('uploadDropzone');
  const encodingSelect = $('encodingSelect');
  const separatorInput = $('separatorInput');
  const previewRowsInput = $('previewRowsInput');
  const caseSelect = $('caseSelect');
  const cleanColumnsCheckbox = $('cleanColumnsCheckbox');
  const stripSpecialCheckbox = $('stripSpecialCheckbox');
  const cleanCellsCheckbox = $('cleanCellsCheckbox');
  const removeDuplicatesCheckbox = $('removeDuplicatesCheckbox');
  const normalizeStringsCheckbox = $('normalizeStringsCheckbox');
  const roundDecimalsCheckbox = $('roundDecimalsCheckbox');
  const scaleNumericCheckbox = $('scaleNumericCheckbox');
  const formatPercentCheckbox = $('formatPercentCheckbox');
  const formatDatesCheckbox = $('formatDatesCheckbox');
  const fillMissingCheckbox = $('fillMissingCheckbox');
  const handleOutliersCheckbox = $('handleOutliersCheckbox');
  const diffFile1Input = $('diffFile1Input');
  const diffFile2Input = $('diffFile2Input');
  const diffFile1Button = $('diffFile1Button');
  const diffFile2Button = $('diffFile2Button');
  const diffFile1Dropzone = $('diffFile1Dropzone');
  const diffFile2Dropzone = $('diffFile2Dropzone');
  const diffFile1Name = $('diffFile1Name');
  const diffFile2Name = $('diffFile2Name');
  const diffEncoding1 = $('diffEncoding1');
  const diffEncoding2 = $('diffEncoding2');
  const diffSeparator1 = $('diffSeparator1');
  const diffSeparator2 = $('diffSeparator2');
  const diffMappingContainer = $('diffMappingContainer');
  const diffSelectAllColumns = $('diffSelectAllColumns');
  const diffClearColumns = $('diffClearColumns');
  const diffPrimaryKey = $('diffPrimaryKey');
  const btnDiffHighlight = $('btnDiffHighlight');
  const btnDiffReport = $('btnDiffReport');
  const statusEl = $('statusMessage');
  const fileNameEl = $('fileName');
  const fileNameText = fileNameEl?.querySelector('span');
  const diffStatusEl = $('diffStatus');
  const diffCreatedFiles = $('diffCreatedFiles');
  const cleanDownloadWrap = $('cleanDownload');
  const cleanDownloadLink = $('cleanDownloadLink');
  const resultSection = $('resultSection');
  const tabPreview = $('tabPreview');
  const tabSummary = $('tabSummary');
  const columnsArea = $('columnsArea');
  const previewTable = $('previewTable');
  const summaryCards = $('summaryCards');
  const summaryDetails = $('summaryDetails');
  const previewEmpty = $('previewEmpty');
  const summaryEmpty = $('summaryEmpty');
  const btnResetOptions = $('btnResetOptions');
  const resultTabs = Array.from(document.querySelectorAll('.result-tab'));

  const DEFAULT_OPTIONS = {
    encoding: 'utf-8',
    separator: ',',
    previewRows: 10,
    case: 'upper',
    cleanColumns: true,
    stripSpecial: true,
    cleanCells: true,
    removeDuplicates: true,
    normalizeStrings: true,
    roundDecimals: true,
    scaleNumeric: false,
    formatPercentages: false,
    formatDates: true,
    fillMissing: true,
    handleOutliers: false
  };

  let currentFile = null;
  const diffSelectedFiles = new Map();
  let diffCheckboxGroups = new Map();
  let diffMetadataTimer = null;
  let diffMetadataAbortController = null;
  let diffMetadataLoading = false;

  function setCurrentFile(file) {
    currentFile = file || null;

    if (fileInput) {
      if (currentFile) {
        try {
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(currentFile);
          fileInput.files = dataTransfer.files;
        } catch (err) {
          // Older browsers may not support programmatic assignment; ignore.
        }
      } else {
        fileInput.value = '';
      }
    }

    if (currentFile) {
      if (fileNameText) {
        fileNameText.textContent = currentFile.name;
      }
      if (fileNameEl) {
        fileNameEl.style.display = 'inline-flex';
      }
    } else {
      if (fileNameText) {
        fileNameText.textContent = '';
      }
      if (fileNameEl) {
        fileNameEl.style.display = 'none';
      }
    }
  }

  function clearTable() {
    if (previewTable) {
      previewTable.innerHTML = '';
    }
    updatePreviewEmptyState(false);
  }

  function clearColumns() {
    if (columnsArea) {
      columnsArea.innerHTML = '';
    }
  }

  function clearSummary() {
    if (summaryCards) {
      summaryCards.innerHTML = '';
    }
    if (summaryDetails) {
      summaryDetails.innerHTML = '';
    }
    updateSummaryEmptyState(false);
  }

  function hideResultSection() {
    if (resultSection) {
      resultSection.style.display = 'none';
    }
    activateResultTab('preview');
  }

  function showResultSection(tab = 'preview') {
    if (resultSection) {
      resultSection.style.display = 'flex';
    }
    activateResultTab(tab);
  }

  function activateResultTab(tabName) {
    if (!resultTabs.length) return;

    resultTabs.forEach((tab) => {
      const isActive = tab.dataset.tab === tabName;
      tab.classList.toggle('active', isActive);
      tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    tabPreview?.classList.toggle('active', tabName === 'preview');
    tabSummary?.classList.toggle('active', tabName === 'summary');
  }

  function updatePreviewEmptyState(hasData) {
    if (!previewEmpty) return;
    previewEmpty.style.display = hasData ? 'none' : 'block';
  }

  function updateSummaryEmptyState(hasData) {
    if (!summaryEmpty) return;
    summaryEmpty.style.display = hasData ? 'none' : 'block';
  }

  function renderColumns(cols, dtypes = {}) {
    clearColumns();
    if (!columnsArea) return;

    (cols || []).forEach((col) => {
      const chip = document.createElement('span');
      chip.className = 'chip';

      const dtype = dtypes[col] || '';
      if (dtype === 'numeric') {
        chip.classList.add('numeric');
      } else if (dtype === 'date') {
        chip.classList.add('date');
      }

      chip.textContent = col;
      columnsArea.appendChild(chip);
    });
  }

  function renderTable(cols, rows) {
    clearTable();
    if (!previewTable || !Array.isArray(cols) || cols.length === 0) {
      updatePreviewEmptyState(false);
      return;
    }

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    cols.forEach((col) => {
      const th = document.createElement('th');
      th.textContent = col;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);

    const tbody = document.createElement('tbody');
    (rows || []).forEach((row) => {
      const tr = document.createElement('tr');
      cols.forEach((col) => {
        const td = document.createElement('td');
        let value = row[col];
        if (value === null || value === undefined) {
          value = '';
        }
        td.textContent = String(value);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    previewTable.appendChild(thead);
    previewTable.appendChild(tbody);

    updatePreviewEmptyState((rows || []).length > 0);
  }

  function renderSummaryCards(summary = {}) {
    if (!summaryCards) return;
    summaryCards.innerHTML = '';

    const totalRows = summary.rows ?? 0;
    const totalCols = summary.cols ?? 0;
    const missingTotal = Object.values(summary.na_count || {}).reduce((acc, value) => acc + Number(value || 0), 0);

    const cards = [
      { label: '总行数', value: totalRows },
      { label: '总列数', value: totalCols },
      { label: '缺失值总数', value: missingTotal }
    ];

    cards.forEach(({ label, value }) => {
      const card = document.createElement('div');
      card.className = 'summary-card';
      card.innerHTML = `
        <div class="summary-value">${value}</div>
        <div class="summary-label">${label}</div>
      `;
      summaryCards.appendChild(card);
    });
  }

  function renderSummaryDetails(summary = {}) {
    if (!summaryDetails) return;

    summaryDetails.innerHTML = '';

    const fragments = [];

    if (Array.isArray(summary.columns) && summary.columns.length) {
      fragments.push({ label: '列名：', value: summary.columns.join(', ') });
    }

    if (summary.dtypes) {
      const dtypePairs = Object.entries(summary.dtypes)
        .map(([col, dtype]) => `${col}: ${dtype}`)
        .join(', ');
      if (dtypePairs) {
        fragments.push({ label: '数据类型：', value: dtypePairs });
      }
    }

    if (summary.na_count) {
      const naPairs = Object.entries(summary.na_count)
        .filter(([, count]) => Number(count) > 0)
        .map(([col, count]) => `${col}: ${count}`)
        .join(', ');
      if (naPairs) {
        fragments.push({ label: '缺失值统计：', value: naPairs });
      }
    }

    if (!fragments.length) {
      const emptyParagraph = document.createElement('p');
      emptyParagraph.textContent = '暂无更多概要信息。';
      summaryDetails.appendChild(emptyParagraph);
      return;
    }

    fragments.forEach(({ label, value }) => {
      const paragraph = document.createElement('p');
      const strong = document.createElement('strong');
      strong.textContent = label;
      paragraph.appendChild(strong);
      paragraph.appendChild(document.createTextNode(value));
      summaryDetails.appendChild(paragraph);
    });
  }

  function buildPreviewQueryParams() {
    const params = new URLSearchParams();

    const previewRows = Math.max(1, Math.min(2000, parseInt(previewRowsInput?.value || '10', 10)));
    params.append('n', previewRows);

    const encoding = encodingSelect?.value;
    if (encoding) {
      params.append('encoding', encoding);
    }

    const separator = separatorInput?.value?.trim();
    if (separator && separator !== ',') {
      params.append('sep', separator);
    }

    return params.toString();
  }

  function clearAllOutputs() {
    clearTable();
    clearColumns();
    clearSummary();
    hideResultSection();
  }

  function syncColumnControls() {
    const enabled = !!cleanColumnsCheckbox?.checked;
    if (caseSelect) {
      caseSelect.disabled = !enabled;
    }
    if (stripSpecialCheckbox) {
      stripSpecialCheckbox.disabled = !enabled;
    }
  }

  if (cleanColumnsCheckbox) {
    cleanColumnsCheckbox.addEventListener('change', syncColumnControls);
    syncColumnControls();
  }
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

  function setDiffStatus(msg, type = 'info') {
    if (!diffStatusEl) return;
    diffStatusEl.innerHTML = '';
    if (!msg) return;

    const badge = document.createElement('div');
    badge.className = `status-badge status-${type}`;
    badge.textContent = msg;
    diffStatusEl.appendChild(badge);
  }

  function renderCreatedFiles(files, urls = []) {
    if (!diffCreatedFiles) return;
    diffCreatedFiles.innerHTML = '';
    (files || []).forEach((file, idx) => {
      const li = document.createElement('li');
      const url = urls[idx];

      if (url) {
        const link = document.createElement('a');
        link.href = url;
        link.textContent = `📁 ${file}`;
        link.setAttribute('download', file || 'diff.xlsx');
        link.target = '_blank';
        link.rel = 'noopener';
        li.appendChild(link);
      } else {
        li.textContent = `📁 ${file}`;
      }
      diffCreatedFiles.appendChild(li);
    });
  }


  function resetCleanDownload() {
    if (cleanDownloadWrap) {
      cleanDownloadWrap.style.display = 'none';
    }
    if (cleanDownloadLink) {
      cleanDownloadLink.removeAttribute('href');
      cleanDownloadLink.removeAttribute('download');
      cleanDownloadLink.textContent = '下载清洗后的文件';
    }
  }

  function showCleanDownload(url, filename) {
    if (!cleanDownloadWrap || !cleanDownloadLink || !url) return;
    cleanDownloadLink.href = url;
    const displayName = filename || 'cleaned.csv';
    cleanDownloadLink.textContent = `下载清洗后的文件 (${displayName})`;
    cleanDownloadLink.setAttribute('download', displayName);
    cleanDownloadWrap.style.display = 'inline-flex';
  }


  function updateFileBadge(badgeEl, file) {
    if (!badgeEl) return;
    const span = badgeEl.querySelector('span');
    if (file) {
      if (span) span.textContent = file.name;
      badgeEl.style.display = 'inline-flex';
    } else {
      if (span) span.textContent = '';
      badgeEl.style.display = 'none';
    }
  }

  function appendIfValue(fd, key, value) {
    if (value) {
      fd.append(key, value);
    }
  }

  function updateMappingActionButtons() {
    const hasColumns = diffCheckboxGroups.size > 0;
    if (diffSelectAllColumns) {
      diffSelectAllColumns.disabled = !hasColumns;
    }
    if (diffClearColumns) {
      diffClearColumns.disabled = !hasColumns;
    }
  }

  function renderDiffMappingPlaceholder(text, variant = 'subtle') {
    if (!diffMappingContainer) return;
    diffMappingContainer.innerHTML = '';
    const message = document.createElement('div');
    const classes = ['mapping-placeholder'];
    if (variant) {
      classes.push(variant);
    }
    message.className = classes.join(' ');
    message.textContent = text;
    diffMappingContainer.appendChild(message);
    diffCheckboxGroups = new Map();
    updateMappingActionButtons();
  }

  function captureDiffSelection() {
    const state = new Map();
    diffCheckboxGroups.forEach((entries, column) => {
      const isChecked = entries.some(({ input }) => input.checked);
      state.set(column, isChecked);
    });
    return state;
  }

  function handleDiffCheckboxChange(column, checked, sourceInput = null) {
    const peers = diffCheckboxGroups.get(column) || [];
    peers.forEach(({ input, wrapper }) => {
      if (input !== sourceInput) {
        input.checked = checked;
      }
      wrapper.classList.toggle('checked', checked);
    });
  }

  function getSelectedDiffColumns() {
    const selected = [];
    diffCheckboxGroups.forEach((entries, column) => {
      if (entries.some(({ input }) => input.checked)) {
        selected.push(column);
      }
    });
    return selected;
  }

  function buildDiffMappingPayload({ required = false } = {}) {
    if (!diffMappingContainer) {
      if (required) {
        throw new Error('缺少映射配置区域');
      }
      return '[]';
    }

    if (diffCheckboxGroups.size === 0) {
      if (required) {
        if (diffMetadataLoading) {
          throw new Error('数值列正在加载，请稍候...');
        }
        throw new Error('未检测到可比较的数值列，请检查文件或解析设置');
      }
      return '[]';
    }

    const selected = getSelectedDiffColumns();
    if (!selected.length) {
      if (required) {
        throw new Error('请至少选择一个需要比较的数值列');
      }
      return '[]';
    }

    return JSON.stringify([{ columns: selected }]);
  }

  function buildMappingColumn(title, numericColumns, sharedSet, fileKey, previousState) {
    const columnEl = document.createElement('div');
    columnEl.className = 'mapping-column';

    const header = document.createElement('div');
    header.className = 'mapping-column-header';

    const titleEl = document.createElement('div');
    titleEl.className = 'mapping-column-title';
    titleEl.textContent = title;
    header.appendChild(titleEl);

    const countEl = document.createElement('div');
    countEl.className = 'mapping-column-count';
    countEl.textContent = `${numericColumns.length} 列`;
    header.appendChild(countEl);

    columnEl.appendChild(header);

    const listEl = document.createElement('div');
    listEl.className = 'mapping-checkbox-list';

    if (!numericColumns.length) {
      const placeholder = document.createElement('div');
      placeholder.className = 'mapping-placeholder subtle';
      placeholder.textContent = '未检测到数值列';
      listEl.appendChild(placeholder);
    }

    numericColumns.forEach((col, idx) => {
      const isShared = sharedSet.has(col);
      const labelEl = document.createElement('label');
      labelEl.className = 'mapping-checkbox';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.value = col;
      checkbox.id = `diff-${fileKey}-${idx}`;

      if (!isShared) {
        checkbox.disabled = true;
        checkbox.checked = false;
        labelEl.classList.add('disabled');
      } else {
        const hasPrev = previousState.has(col);
        const shouldCheck = hasPrev ? previousState.get(col) : true;
        checkbox.checked = shouldCheck;
      }

      const span = document.createElement('span');
      span.textContent = isShared ? col : `${col}（仅此文件）`;

      labelEl.appendChild(checkbox);
      labelEl.appendChild(span);
      listEl.appendChild(labelEl);

      if (isShared) {
        let group = diffCheckboxGroups.get(col);
        if (!group) {
          group = [];
          diffCheckboxGroups.set(col, group);
        }
        group.push({ input: checkbox, wrapper: labelEl });
        checkbox.addEventListener('change', () => handleDiffCheckboxChange(col, checkbox.checked, checkbox));
        handleDiffCheckboxChange(col, checkbox.checked, checkbox);
      }
    });

    columnEl.appendChild(listEl);
    return columnEl;
  }

  function renderDiffMappingOptions(metadata, previousState = new Map()) {
    if (!diffMappingContainer) return;

    diffMappingContainer.innerHTML = '';
    diffCheckboxGroups = new Map();

    const sharedSet = new Set(metadata?.shared_numeric_columns || []);
    const numeric1 = metadata?.numeric_columns1 || [];
    const numeric2 = metadata?.numeric_columns2 || [];

    if (!sharedSet.size) {
      const warning = document.createElement('div');
      warning.className = 'mapping-placeholder warning';
      warning.textContent = '未检测到两个文件共有的数值列，请检查编码、分隔符或数据内容。';
      diffMappingContainer.appendChild(warning);
    }
    const columnsWrapper = document.createElement('div');
    columnsWrapper.className = 'mapping-columns';
    columnsWrapper.appendChild(buildMappingColumn('文件一数值列', numeric1, sharedSet, 'file1', previousState));
    columnsWrapper.appendChild(buildMappingColumn('文件二数值列', numeric2, sharedSet, 'file2', previousState));

    diffMappingContainer.appendChild(columnsWrapper);
    updateMappingActionButtons();
  }

  async function loadDiffMetadata() {
    if (!diffMappingContainer) return;

    const file1 = diffFile1Input?.files?.[0] || diffSelectedFiles.get('file1');
    const file2 = diffFile2Input?.files?.[0] || diffSelectedFiles.get('file2');

    if (!file1 || !file2) {
      renderDiffMappingPlaceholder('请先上传两个文件以加载数值列。', 'subtle');
      diffMetadataLoading = false;
      return;
    }

    const formData = new FormData();
    formData.append('file1', file1, file1.name);
    formData.append('file2', file2, file2.name);
    appendIfValue(formData, 'encoding1', diffEncoding1?.value || '');
    appendIfValue(formData, 'encoding2', diffEncoding2?.value || '');
    appendIfValue(formData, 'sep1', diffSeparator1?.value?.trim() || '');
    appendIfValue(formData, 'sep2', diffSeparator2?.value?.trim() || '');

    const previousState = captureDiffSelection();

    if (diffMetadataAbortController) {
      diffMetadataAbortController.abort();
    }

    diffMetadataAbortController = new AbortController();
    diffMetadataLoading = true;
    renderDiffMappingPlaceholder('正在检测数值列...', 'subtle');

    try {
      const resp = await fetch('/api/csv/diff-metadata', {
        method: 'POST',
        body: formData,
        signal: diffMetadataAbortController.signal
      });
      const data = await parseJsonResponse(resp);

      if (!resp.ok || !data?.ok) {
        throw new Error(data?.error || `HTTP ${resp.status}`);
      }

      renderDiffMappingOptions(data, previousState);
    } catch (err) {
      if (err.name === 'AbortError') {
        return;
      }
      renderDiffMappingPlaceholder(`加载数值列失败：${err.message || err}`, 'error');
    } finally {
      diffMetadataAbortController = null;
      diffMetadataLoading = false;
      updateMappingActionButtons();
      }
    }
    function scheduleDiffMetadataRefresh(delay = 300) {
      if (!diffMappingContainer) return;
      if (diffMetadataTimer) {
        clearTimeout(diffMetadataTimer);
      }
      diffMetadataTimer = setTimeout(() => {
        diffMetadataTimer = null;
        loadDiffMetadata();
      }, delay);
    }
  updateMappingActionButtons();

  diffSelectAllColumns?.addEventListener('click', () => {
    diffCheckboxGroups.forEach((_, column) => {
      handleDiffCheckboxChange(column, true);
    });
  });

  diffClearColumns?.addEventListener('click', () => {
    diffCheckboxGroups.forEach((_, column) => {
      handleDiffCheckboxChange(column, false);
    });
  });

  [diffEncoding1, diffEncoding2].forEach((el) => {
    el?.addEventListener('change', () => scheduleDiffMetadataRefresh());
  });

  [diffSeparator1, diffSeparator2].forEach((el) => {
    if (!el) return;
    el.addEventListener('change', () => scheduleDiffMetadataRefresh());
    el.addEventListener('input', () => scheduleDiffMetadataRefresh(500));
  });

  async function parseJsonResponse(resp) {
    const contentType = resp.headers?.get?.('content-type') || '';

    if (contentType.includes('application/json')) {
      return resp.json();
    }

    const text = await resp.text();
    const trimmed = (text || '').trim();
    const looksHtml = /^<!doctype html/i.test(trimmed) || /^<html/i.test(trimmed);

    if (looksHtml) {
      const hint = resp.status === 404
        ? '请确认接口地址是否正确'
        : '请确认已登录并检查服务器日志';
      return {
        ok: false,
        error: `服务器返回 HTML 响应 (HTTP ${resp.status})：${hint}`
      };
    }

    if (!trimmed) {
      return {
        ok: resp.ok,
        error: `HTTP ${resp.status}`
      };
    }

    try {
      return JSON.parse(trimmed);
    } catch (err) {
      return {
        ok: false,
        error: trimmed || `HTTP ${resp.status}`
      };
    }
  }

  // 选择文件按钮
  btnChoose?.addEventListener('click', () => fileInput?.click());

  uploadDropzone?.addEventListener('click', () => fileInput?.click());

  if (uploadDropzone) {
    const activateDropzone = (event) => {
      event.preventDefault();
      event.stopPropagation();
      uploadDropzone.classList.add('active');
    };

    const deactivateDropzone = (event) => {
      event.preventDefault();
      event.stopPropagation();
      uploadDropzone.classList.remove('active');
    };

      ['dragenter', 'dragover'].forEach((evtName) => {
      uploadDropzone.addEventListener(evtName, activateDropzone);
    });

    ['dragleave', 'dragend'].forEach((evtName) => {
      uploadDropzone.addEventListener(evtName, deactivateDropzone);
    });

    uploadDropzone.addEventListener('drop', (event) => {
      event.preventDefault();
      event.stopPropagation();
      uploadDropzone.classList.remove('active');

      const file = event.dataTransfer?.files?.[0];
      if (file) {
        setCurrentFile(file);
        setStatus('');
        clearAllOutputs();
        resetCleanDownload();
      }
    });
  }

  // 文件选择事件
  fileInput?.addEventListener('change', () => {
    setCurrentFile(fileInput.files?.[0] || null);
    setStatus('');
    clearAllOutputs();
    resetCleanDownload();
  });

  // diff 文件选择
  const assignFileToInput = (inputEl, file, key) => {
    if (!inputEl) return;

    if (file) {
      if (typeof DataTransfer !== 'undefined') {
        try {
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(file);
          inputEl.files = dataTransfer.files;
        } catch (err) {
          console.warn('DataTransfer not supported, falling back to stored reference');
        }
      }

      if (key) {
        diffSelectedFiles.set(key, file);
      }
    } else {
      inputEl.value = '';
      if (key) {
        diffSelectedFiles.delete(key);
      }
    }
  };

  const setupDiffUploader = ({ input, button, dropzone, badge, key }) => {
    if (!input) return;

    button?.addEventListener('click', () => input.click());

    if (dropzone) {
      const activate = (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropzone.classList.add('active');
      };

      const deactivate = (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropzone.classList.remove('active');
      };

      dropzone.addEventListener('click', (event) => {
        event.preventDefault();
        input.click();
      });

      ['dragenter', 'dragover'].forEach((name) => dropzone.addEventListener(name, activate));
      ['dragleave', 'dragend'].forEach((name) => dropzone.addEventListener(name, deactivate));

      dropzone.addEventListener('drop', (event) => {
        event.preventDefault();
        event.stopPropagation();
        dropzone.classList.remove('active');

        const file = event.dataTransfer?.files?.[0];
        if (file) {
          assignFileToInput(input, file, key);
          updateFileBadge(badge, file);
          scheduleDiffMetadataRefresh();
        }
      });
    }

    input.addEventListener('change', () => {
      const file = input.files?.[0] || null;
      if (key) {
        if (file) {
          diffSelectedFiles.set(key, file);
        } else {
          diffSelectedFiles.delete(key);
        }
      }
      updateFileBadge(badge, file);
      scheduleDiffMetadataRefresh();
    });
  };

  setupDiffUploader({
    input: diffFile1Input,
    button: diffFile1Button,
    dropzone: diffFile1Dropzone,
    badge: diffFile1Name,
    key: 'file1'
  });

  setupDiffUploader({
    input: diffFile2Input,
    button: diffFile2Button,
    dropzone: diffFile2Dropzone,
    badge: diffFile2Name,
    key: 'file2'
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
      const cleanColumns = cleanColumnsCheckbox?.checked ?? true;
      const stripSpecial = cleanColumns && (stripSpecialCheckbox?.checked ?? true);
      const cleanCells = cleanCellsCheckbox?.checked ?? true;
      const removeDuplicates = removeDuplicatesCheckbox?.checked ?? true;
      const normalizeStrings = normalizeStringsCheckbox?.checked ?? true;
      const roundDecimals = roundDecimalsCheckbox?.checked ?? true;
      const scaleNumeric = scaleNumericCheckbox?.checked ?? false;
      const formatPercentages = formatPercentCheckbox?.checked ?? false;
      const formatDates = formatDatesCheckbox?.checked ?? true;
      const fillMissing = fillMissingCheckbox?.checked ?? true;
      const handleOutliers = handleOutliersCheckbox?.checked ?? false;

      params.append('case', caseValue);
      params.append('clean_columns', cleanColumns ? 'true' : 'false');
      params.append('strip_special', stripSpecial ? 'true' : 'false');
      params.append('clean_cells', cleanCells ? 'true' : 'false');
      params.append('normalize_strings', normalizeStrings ? 'true' : 'false');
      params.append('round_decimals', roundDecimals ? 'true' : 'false');
      params.append('scale_numeric', scaleNumeric ? 'true' : 'false');
      params.append('format_percentages', formatPercentages ? 'true' : 'false');
      params.append('format_dates', formatDates ? 'true' : 'false');
      params.append('fill_missing', fillMissing ? 'true' : 'false');
      params.append('handle_outliers', handleOutliers ? 'true' : 'false');
      params.append('remove_duplicates', removeDuplicates ? 'true' : 'false');

      const encoding = encodingSelect?.value;
      if (encoding) {
        params.append('encoding', encoding);
      }

      const separator = separatorInput?.value?.trim();
      if (separator) {
        params.append('sep', separator);
      }

      setStatus('清洗数据中...', 'info');
      resetCleanDownload();
      btnClean.disabled = true;

      const resp = await fetch(`/api/csv/clean?${params.toString()}`, {
        method: 'POST',
        body: fd
      });

      const data = await resp.json();

      if (!resp.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      const cleanedRows = data.cleaned_rows ?? 0;
      const removedDuplicates = data.removed_duplicates ?? 0;
      const steps = Array.isArray(data.applied_steps) ? data.applied_steps : [];

      if (steps.length) {
        const parts = [`清洗完成：${cleanedRows} 行`];
        if (steps.includes('重复行去重')) {
          parts.push(`移除重复 ${removedDuplicates} 行`);
        }
        parts.push(`执行步骤：${steps.join('、')}`);
        setStatus(parts.join('，'), 'success');
      } else {
        setStatus(`未执行任何清洗步骤，${cleanedRows} 行数据保持不变`, 'info');
      }

      if (Array.isArray(data.columns)) {
        renderColumns(data.columns);
        updatePreviewEmptyState(false);
        showResultSection('preview');
      }

      if (data.download_url) {
        showCleanDownload(data.download_url, data.output_filename);
      }
    } catch (e) {
      console.error(e);
      setStatus(`清洗失败：${e.message}`, 'error');
      resetCleanDownload();
    } finally {
      if (btnClean) {
        btnClean.disabled = false;
      }
    }
  });

  btnPreview?.addEventListener('click', async () => {
    try {
      if (!currentFile) {
        setStatus('请先选择文件', 'error');
        return;
      }

      const queryParams = buildPreviewQueryParams();
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

      renderColumns(data.columns || []);
      renderTable(data.columns || [], data.rows || []);
      showResultSection('preview');
      setStatus(`预览完成：${(data.columns || []).length} 列，${(data.rows || []).length} 行`, 'success');
    } catch (e) {
      console.error(e);
      clearTable();
      setStatus(`解析失败：${e.message}`, 'error');
    } finally {
      if (btnPreview) {
        btnPreview.disabled = false;
      }
    }
  });

  btnSummary?.addEventListener('click', async () => {
    try {
      if (!currentFile) {
        setStatus('请先选择文件', 'error');
        return;
      }

      const queryParams = buildPreviewQueryParams();
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
      renderSummaryCards(summary);
      renderSummaryDetails(summary);
      renderColumns(summary.columns || [], summary.dtypes || {});
      const hasSummaryContent = Object.keys(summary || {}).length > 0;
      updateSummaryEmptyState(hasSummaryContent);
      showResultSection('summary');
      setStatus('概要已生成', 'success');
    } catch (e) {
      console.error(e);
      clearSummary();
      setStatus(`概要失败：${e.message}`, 'error');
    } finally {
      if (btnSummary) {
        btnSummary.disabled = false;
      }
    }
  });

  btnResetOptions?.addEventListener('click', () => {
    if (encodingSelect) {
      encodingSelect.value = DEFAULT_OPTIONS.encoding;
    }
    if (separatorInput) {
      separatorInput.value = DEFAULT_OPTIONS.separator;
    }
    if (previewRowsInput) {
      previewRowsInput.value = DEFAULT_OPTIONS.previewRows;
    }
    if (caseSelect) {
      caseSelect.value = DEFAULT_OPTIONS.case;
    }
    if (cleanColumnsCheckbox) {
      cleanColumnsCheckbox.checked = DEFAULT_OPTIONS.cleanColumns;
    }
    if (stripSpecialCheckbox) {
      stripSpecialCheckbox.checked = DEFAULT_OPTIONS.stripSpecial;
    }
    if (cleanCellsCheckbox) {
      cleanCellsCheckbox.checked = DEFAULT_OPTIONS.cleanCells;
    }
    if (removeDuplicatesCheckbox) {
      removeDuplicatesCheckbox.checked = DEFAULT_OPTIONS.removeDuplicates;
    }
    if (normalizeStringsCheckbox) {
      normalizeStringsCheckbox.checked = DEFAULT_OPTIONS.normalizeStrings;
    }
    if (roundDecimalsCheckbox) {
      roundDecimalsCheckbox.checked = DEFAULT_OPTIONS.roundDecimals;
    }
    if (scaleNumericCheckbox) {
      scaleNumericCheckbox.checked = DEFAULT_OPTIONS.scaleNumeric;
    }
    if (formatPercentCheckbox) {
      formatPercentCheckbox.checked = DEFAULT_OPTIONS.formatPercentages;
    }
    if (formatDatesCheckbox) {
      formatDatesCheckbox.checked = DEFAULT_OPTIONS.formatDates;
    }
    if (fillMissingCheckbox) {
      fillMissingCheckbox.checked = DEFAULT_OPTIONS.fillMissing;
    }
    if (handleOutliersCheckbox) {
      handleOutliersCheckbox.checked = DEFAULT_OPTIONS.handleOutliers;
    }

    syncColumnControls();
    setStatus('选项已恢复默认值', 'info');
  });

  if (resultTabs.length) {
    resultTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
        if (!tabName) return;
        activateResultTab(tabName);
      });
    });
  }

  async function handleDiffRequest(endpoint) {
    try {
      setDiffStatus('', 'info');
      renderCreatedFiles([], []);

      const file1 = diffFile1Input?.files?.[0] || diffSelectedFiles.get('file1');
      const file2 = diffFile2Input?.files?.[0] || diffSelectedFiles.get('file2');

      if (!file1 || !file2) {
        setDiffStatus('请同时选择 File 1 与 File 2', 'error');
        return;
      }

      let mappingPayload = '';
      try {
        mappingPayload = buildDiffMappingPayload({ required: true });
      } catch (err) {
        setDiffStatus(err.message || '请选择需要比较的数值列', 'error');
        return;
      }

      const fd = new FormData();
      fd.append('file1', file1, file1.name);
      fd.append('file2', file2, file2.name);
      fd.append('mapping', mappingPayload);

      appendIfValue(fd, 'encoding1', diffEncoding1?.value || '');
      appendIfValue(fd, 'encoding2', diffEncoding2?.value || '');

      const primaryKey = diffPrimaryKey?.value?.trim();
      if (primaryKey) {
        fd.append('primary_key', primaryKey);
      }

      const sep1 = diffSeparator1?.value?.trim();
      if (sep1) {
        fd.append('sep1', sep1);
      }

      const sep2 = diffSeparator2?.value?.trim();
      if (sep2) {
        fd.append('sep2', sep2);
      }

      const payloadEntries = Array.from(fd.entries());
      const sendRequest = async (url) => {
        const body = new FormData();
        payloadEntries.forEach(([key, value]) => body.append(key, value));
        const resp = await fetch(url, {
          method: 'POST',
          body
        });
        const data = await parseJsonResponse(resp);
        return { resp, data };
      };

      setDiffStatus('生成差异中，请稍候...', 'info');
      if (btnDiffHighlight) btnDiffHighlight.disabled = true;
      if (btnDiffReport) btnDiffReport.disabled = true;

      let { resp, data } = await sendRequest(endpoint);

      const fallbackEndpoint = endpoint.includes('-')
        ? endpoint.replace(/-/g, '_')
        : endpoint;

      const needsFallback =
        resp.status === 404 &&
        fallbackEndpoint !== endpoint &&
        (!data?.ok) &&
        (!data?.error || /not\s+found/i.test(data.error));

      if (needsFallback) {
        console.warn(`API route ${endpoint} missing, retrying as ${fallbackEndpoint}`);
        ({ resp, data } = await sendRequest(fallbackEndpoint));
      }

      if (!resp.ok || !data?.ok) {
        const message = data?.error || `HTTP ${resp.status}`;
        throw new Error(message);
      }

      const files = data.created_files || [];
      const urls = data.download_urls || [];
      setDiffStatus(`生成成功：${files.length} 个输出文件`, 'success');
      renderCreatedFiles(files, urls);
    } catch (e) {
      console.error(e);
      setDiffStatus(`生成失败：${e.message}`, 'error');
    } finally {
      if (btnDiffHighlight) btnDiffHighlight.disabled = false;
      if (btnDiffReport) btnDiffReport.disabled = false;
    }
  }

  btnDiffHighlight?.addEventListener('click', () => handleDiffRequest('/api/csv/diff-highlight'));
  btnDiffReport?.addEventListener('click', () => handleDiffRequest('/api/csv/diff-report'));

  console.log('✅ CSV 工作区已初始化');
})();
