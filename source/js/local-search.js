/* Search Fluid's generated index; render results as text, including literal symbols. */
(() => {
  const input = document.querySelector('#local-search-input');
  const results = document.querySelector('#local-search-result');
  if (!input || !results) return;
  let entries = null;
  let loading = null;

  function readableContent(html) {
    const doc = new DOMParser().parseFromString(html, 'text/html');
    doc.querySelectorAll('script, style').forEach(node => node.remove());
    doc.querySelectorAll('.katex').forEach(node => {
      node.replaceWith(doc.createTextNode(node.querySelector('.katex-html')?.textContent || ''));
    });
    doc.querySelectorAll('p, h2, h3, li, tr, td, th').forEach(node => node.append(doc.createTextNode(' ')));
    return doc.body.textContent.replace(/\s+/g, ' ').trim();
  }

  function render() {
    const query = input.value.trim().toLocaleLowerCase();
    results.replaceChildren();
    if (!query) return;
    if (!entries) { results.textContent = '正在加载文章索引…'; return; }
    const words = query.split(/\s+/);
    let count = 0;
    for (const entry of entries) {
      if (!words.every(word => (entry.title + ' ' + entry.content).toLocaleLowerCase().includes(word))) continue;
      const link = document.createElement('a');
      link.href = entry.url;
      link.className = 'list-group-item list-group-item-action font-weight-bolder search-list-title';
      link.textContent = entry.title;
      const snippet = document.createElement('p');
      snippet.className = 'search-list-content';
      const index = entry.content.toLocaleLowerCase().indexOf(words[0]);
      const start = Math.max(0, index - 24);
      snippet.textContent = (start ? '…' : '') + entry.content.slice(start, start + 130) + '…';
      results.append(link, snippet);
      count++;
    }
    if (!count) results.textContent = '没有找到相关内容，试试其他关键词。';
  }

  async function load() {
    if (entries) return;
    if (!loading) {
      loading = fetch(CONFIG.search_path || '/local-search.xml')
        .then(response => { if (!response.ok) throw new Error('Search index unavailable'); return response.text(); })
        .then(xml => {
          const doc = new DOMParser().parseFromString(xml, 'application/xml');
          if (doc.querySelector('parsererror')) throw new Error('Invalid search index');
          entries = Array.from(doc.querySelectorAll('entry')).map(entry => ({
            title: entry.querySelector('title')?.textContent || '',
            url: entry.querySelector('url')?.textContent || '/',
            content: readableContent(entry.querySelector('content')?.textContent || '')
          }));
        });
    }
    try { await loading; render(); }
    catch { loading = null; results.textContent = '文章索引暂时无法加载，请关闭后重新打开搜索。'; }
  }
  input.addEventListener('input', render);
  jQuery('#modalSearch').on('show.bs.modal', load)
    .on('shown.bs.modal', () => input.focus())
    .on('hidden.bs.modal', () => { input.value = ''; results.replaceChildren(); });
})();
