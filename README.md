# 兰斤斤博客

Hexo 8 + Fluid 1.9。正式域名：`https://blog.lanjinjin.site`。

## 本地预览

使用 Node.js 24：

```sh
npm ci
npm run dev
```

访问 `http://127.0.0.1:4000/`。Windows 也可运行 `./preview.ps1`，自动选择已安装的新版 Node.js。修改站点配置后需要重启预览服务。

## 内容

目前仅保留首页、笔记、归档、关于的页面结构，正文留空。

- `source/_posts/`：以后发布的文章。
- `source/notes/index.md`：笔记目录。
- `source/about/index.md`：关于页。
- `_config.yml`：正式域名及站点配置。
- `_config.fluid.yml`：导航、主题和页脚配置。
- `layout/header.ejs`：封面与各页面标题。
- `source/css/custom.css`、`source/js/custom.js`：布局和动画。

本地 `.local-drafts/` 与 `preview/` 不进入 Git，也不发布。没有文章时仍生成首页与空归档页；发布第一篇文章后归档由 Hexo 自动生成。

## 发布

推送到 `main` 后，GitHub Actions 使用 Node.js 24 安装依赖、构建并部署 `public/`。Pages 的发布来源设置为 **GitHub Actions**，自定义域名设置为 **blog.lanjinjin.site**。

腾讯云 DNSPod 中，在 `lanjinjin.site` 下配置 `blog` 的 CNAME，记录值为 `hahavguoqu.github.io`，线路选择默认。不要包含协议或仓库路径。DNS 生效且 GitHub 签发证书后开启 Enforce HTTPS。

`npm run build` 会先从依赖中复制前端资源到 `source/vendor/`，再生成静态页面。部署只上传 `public/`，评论暂未启用。
