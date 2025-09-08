document.addEventListener('DOMContentLoaded', function () {
  const toast = document.getElementById('toast');
  const show = (msg, type = 'success') => {
    toast.querySelector('span').textContent = msg;
    toast.className = `notification ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 2500);
  };

  const statTotal = document.getElementById('stat-total');
  const statDone = document.getElementById('stat-done');
  const statPend = document.getElementById('stat-pending');

  function recalcStats() {
    const items = [...document.querySelectorAll('#task-list .task-item')];
    const done = items.filter(i => i.classList.contains('completed')).length;
    const total = items.length;
    statTotal.textContent = total;
    statDone.textContent = done;
    statPend.textContent = Math.max(total - done, 0);
    document.getElementById('empty-state').style.display = total ? 'none' : 'block';
  }

  function addTaskFromInput() {
    const input = document.getElementById('new-task');
    const text = (input.value || '').trim();
    if (!text) { show('请输入任务内容', 'error'); return; }

    const li = document.createElement('li');
    li.className = 'task-item';
    li.innerHTML = `
      <div class="task-content">
        <div class="task-checkbox"></div>
        <div class="task-text"></div>
        <span class="task-priority priority-low">低优先级</span>
      </div>
      <div class="task-actions">
        <button class="btn-action btn-edit"><i class="fas fa-edit"></i></button>
        <button class="btn-action btn-delete"><i class="fas fa-trash"></i></button>
      </div>`;
    li.querySelector('.task-text').textContent = text;
    document.getElementById('task-list').appendChild(li);
    input.value = '';
    show('任务已成功添加！');
    recalcStats();
  }

  const addBtn = document.getElementById('btn-add');
  if (addBtn) addBtn.addEventListener('click', addTaskFromInput);

  const addFirst = document.getElementById('btn-add-first');
  if (addFirst) addFirst.addEventListener('click', addTaskFromInput);

  // 事件委托：复选、删除、编辑、过滤
  document.addEventListener('click', function (e) {
    const checkbox = e.target.closest('.task-checkbox');
    if (checkbox) {
      const item = checkbox.closest('.task-item');
      checkbox.classList.toggle('checked');
      const textEl = item.querySelector('.task-text');
      textEl.classList.toggle('completed');
      item.classList.toggle('completed');
      show(item.classList.contains('completed') ? '任务标记为已完成！' : '任务标记为未完成！');
      recalcStats();
    }

    const delBtn = e.target.closest('.btn-delete');
    if (delBtn) {
      const item = delBtn.closest('.task-item');
      item.style.opacity = '0';
      item.style.transform = 'translateX(100px)';
      setTimeout(() => { item.remove(); recalcStats(); show('任务已删除！'); }, 250);
    }

    const editBtn = e.target.closest('.btn-edit');
    if (editBtn) {
      const item = editBtn.closest('.task-item');
      const textEl = item.querySelector('.task-text');
      const newText = prompt('编辑任务内容：', textEl.textContent.trim());
      if (newText !== null) { textEl.textContent = newText.trim(); show('任务已更新'); }
    }

    const filterBtn = e.target.closest('.filter-btn');
    if (filterBtn) {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      filterBtn.classList.add('active');
      const mode = filterBtn.dataset.filter;
      document.querySelectorAll('#task-list .task-item').forEach(li => {
        const isDone = li.classList.contains('completed');
        li.style.display =
          mode === 'all' ? '' :
          mode === 'active' ? (isDone ? 'none' : '') :
          mode === 'done' ? (isDone ? '' : 'none') : '';
      });
      show(`已切换到 "${filterBtn.textContent}" 筛选模式`);
    }
  });

  recalcStats();
});
