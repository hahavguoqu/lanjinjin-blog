$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$blogNode = (Get-Command node -ErrorAction SilentlyContinue).Source
if ($blogNode) {
    $blogMajor = [int]((& $blogNode --version).TrimStart('v').Split('.')[0])
} else { $blogMajor = 0 }
if ($blogMajor -lt 20) {
    $bundledBlogNode = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
    if (Test-Path -LiteralPath $bundledBlogNode) { $blogNode = $bundledBlogNode }
    else { throw '请安装 Node.js 24 后重新运行。' }
}
if (!(Test-Path -LiteralPath 'node_modules\hexo\bin\hexo')) { throw '请先使用 Node.js 20.19+ 执行 npm ci。' }
& $blogNode 'tools\prepare-assets.cjs'
if ($LASTEXITCODE -ne 0) { throw '静态依赖准备失败。' }
& $blogNode 'node_modules\hexo\bin\hexo' server --ip 127.0.0.1 --port 4000
