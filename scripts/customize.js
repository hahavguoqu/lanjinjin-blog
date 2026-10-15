/* Keep the published Fluid package intact; customize through its extension points. */
hexo.extend.filter.register('theme_inject', injects => {
  injects.header.file('default', 'layout/header.ejs');
  injects.head.raw('entrance', `<script>
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      document.documentElement.classList.add('home-entering');
      window.setTimeout(function () {
        document.documentElement.classList.remove('home-entering');
      }, 6000);
    }
  </script>`);
});

hexo.extend.filter.register('after_render:html', html => {
  return html.replace('<body>', '<body class="lanjinjin-site">');
});

// Hexo's archive generator skips the route until the first article is published.
hexo.extend.generator.register('empty-archive', locals => {
  if (locals.posts.length) return;
  return { path: 'archives/index.html', layout: ['page'], data: { title: '归档', content: '', comments: false } };
});

hexo.extend.generator.register('empty-home', locals => {
  if (locals.posts.length) return;
  return { path: 'index.html', layout: ['index'], data: { __index: true, posts: locals.posts, current: 1, total: 1 } };
});
