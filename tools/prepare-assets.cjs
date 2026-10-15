const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const assets = [
  ['jquery/dist/jquery.min.js', 'jquery/jquery.min.js'],
  ['bootstrap/dist/css/bootstrap.min.css', 'bootstrap/css/bootstrap.min.css'],
  ['bootstrap/dist/js/bootstrap.min.js', 'bootstrap/js/bootstrap.min.js'],
  ['tocbot/dist/tocbot.min.js', 'tocbot/tocbot.min.js'],
  ['clipboard/dist/clipboard.min.js', 'clipboard/clipboard.min.js'],
  ['github-markdown-css/github-markdown.css', 'github-markdown/github-markdown.min.css'],
  ['hint.css/hint.min.css', 'hint/hint.min.css'],
  ['katex/dist/katex.min.css', 'katex/katex.min.css'],
  ['katex/dist/fonts', 'katex/fonts']
];
for (const [source, destination] of assets) {
  const target = path.join(root, 'source/vendor', destination);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.cpSync(path.join(root, 'node_modules', source), target, { recursive: true });
}
// Keep upstream license texts next to redistributed browser assets.
for (const name of ['jquery', 'bootstrap', 'tocbot', 'clipboard', 'github-markdown-css', 'hint.css', 'katex']) {
  const packageDir = path.join(root, 'node_modules', name);
  for (const file of fs.readdirSync(packageDir).filter(file => /^licen[sc]e/i.test(file))) {
    fs.copyFileSync(path.join(packageDir, file), path.join(root, 'source/vendor', `${name}-LICENSE`));
  }
}
