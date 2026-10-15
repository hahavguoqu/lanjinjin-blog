(() => {
  const root = document.documentElement;
  const title = document.querySelector('.typed-title');
  if (title && root.classList.contains('home-entering')) {
    const completeTitle = title.textContent;
    const letters = Array.from(completeTitle);
    const hero = document.querySelector('.hero-picture img');
    const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
    title.textContent = '';
    async function revealHome() {
      try {
        // Begin after the chosen responsive image is ready, with a bounded fallback.
        await Promise.race([hero.decode().catch(() => {}), pause(2000)]);
        await pause(350);
        root.classList.add('intro-backdrop');
        await pause(850);
        root.classList.add('intro-ui');
        await pause(250);
        for (const letter of letters) {
          title.textContent += letter;
          await pause(280);
        }
        await pause(450);
      } finally {
        title.textContent = completeTitle;
        root.classList.remove('home-entering', 'intro-backdrop', 'intro-ui');
      }
    }
    revealHome();
  }
  const article = document.querySelector('.post-content .markdown-body');
  if (!article) return;
  const headings = Array.from(article.querySelectorAll('h2[id]'));
  if (headings.length < 2) return;
  const details = document.createElement('details');
  details.className = 'mobile-toc';
  const summary = document.createElement('summary');
  summary.textContent = '文章目录';
  const list = document.createElement('ul');
  for (const heading of headings) {
    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = '#' + encodeURIComponent(heading.id);
    link.textContent = heading.textContent;
    link.addEventListener('click', () => { details.open = false; });
    item.append(link);
    list.append(item);
  }
  details.append(summary, list);
  article.prepend(details);
})();
