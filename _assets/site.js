/*! 学习库站点增强 v1.0
 *  功能：已读进度标记 + Gist 云同步 + 随手记 + 学习进度面板
 *        全站搜索(Ctrl+K) + Obsidian 双链转换 + Mermaid + 暗色模式 + Callout 样式
 *  依赖：docsify@4，脚本需在 docsify.min.js 之前加载
 */
(function () {
  'use strict';

  /* ---------------- 配置与状态 ---------------- */

  var LS = {
    progress: 'lr_progress_v1',
    token: 'lr_gist_token',
    gistId: 'lr_gist_id',
    theme: 'lr_theme',
    lastSync: 'lr_last_sync_at'
  };
  var GIST_FILE = 'learning-progress.json';
  var API = 'https://api.github.com/gists/';
  var MODULE_ORDER = ['顶层文档', '算法与AI', '搜推广与生成式推荐', '基础', '工程与职业', '面试', '科目四'];

  var D = (window.$docsify = {
    homepage: 'README.md',
    loadSidebar: true,
    loadNavbar: true,
    relativePath: true,
    subMaxLevel: 3,
    maxLevel: 3,
    auto2top: true,
    sidebarDisplayLevel: 1,
    externalLinkTarget: '_blank',
    gistId: '4cd7838a7fa749aa41483d1eec7b4f81',
    plugins: [lrPlugin]
  });

  function currentGistId() {
    return localStorage.getItem(LS.gistId) || D.gistId || '';
  }
  function hasConfig() {
    return !!(localStorage.getItem(LS.token) && currentGistId());
  }

  function loadJSON(key, fallback) {
    try {
      var v = localStorage.getItem(key);
      return v ? JSON.parse(v) : fallback;
    } catch (e) { return fallback; }
  }

  var state = loadJSON(LS.progress, null) || { version: 1, read: {}, scratch: [], scratchDel: {}, lastRead: null, updatedAt: '' };
  if (!state.scratch) state.scratch = [];
  if (!state.scratchDel) state.scratchDel = {};

  var searchIndex = null;
  var searchIndexPromise = null;
  var wikiMap = null;
  var wikiMapPromise = null;
  var pendingWikiRoute = null;
  var pushTimer = null;
  var syncBusy = false;
  var mermaidReady = false;

  function saveLocal(quiet) {
    try { localStorage.setItem(LS.progress, JSON.stringify(state)); } catch (e) {}
    if (!quiet) schedulePush();
  }
  function schedulePush() {
    if (!hasConfig()) return;
    clearTimeout(pushTimer);
    pushTimer = setTimeout(function () { syncNow('auto'); }, 4000);
  }

  /* ---------------- Gist 云同步 ---------------- */

  function apiHeaders(extra) {
    var h = {
      Authorization: 'Bearer ' + (localStorage.getItem(LS.token) || ''),
      Accept: 'application/vnd.github+json'
    };
    if (extra) for (var k in extra) h[k] = extra[k];
    return h;
  }

  function gistFetch() {
    var id = currentGistId();
    if (!id) return Promise.reject(new Error('未配置 Gist ID'));
    return fetch(API + id, { headers: apiHeaders() }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status + (r.status === 401 ? '（token 无效）' : r.status === 404 ? '（Gist 不存在或 token 无 gist 权限）' : ''));
      return r.json();
    });
  }

  function gistWrite(data) {
    var id = currentGistId();
    if (!id) return Promise.reject(new Error('未配置 Gist ID'));
    var body = { files: {} };
    data.updatedAt = new Date().toISOString();
    body.files[GIST_FILE] = { content: JSON.stringify(data) };
    return fetch(API + id, {
      method: 'PATCH',
      headers: apiHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body)
    }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      try { localStorage.setItem(LS.lastSync, new Date().toISOString()); } catch (e) {}
      return true;
    });
  }

  /* 按“每个条目新者胜”合并远端进度（含删除墓碑） */
  function mergeRemote(remote) {
    if (!remote) return false;
    var changed = false;
    var rr = remote.read || {};
    Object.keys(rr).forEach(function (k) {
      var mine = state.read[k];
      if (!mine || (rr[k].at || 0) >= (mine.at || 0)) { state.read[k] = rr[k]; changed = true; }
    });
    (remote.scratch || []).forEach(function (s) {
      var exists = state.scratch.some(function (x) { return x.id === s.id; });
      var tomb = state.scratchDel[s.id];
      if (!exists && !(tomb && (tomb.at || 0) >= (s.at || 0))) { state.scratch.push(s); changed = true; }
    });
    var rd = remote.scratchDel || {};
    Object.keys(rd).forEach(function (k) {
      var mine = state.scratchDel[k];
      if (!mine || (rd[k].at || 0) >= (mine.at || 0)) { state.scratchDel[k] = rd[k]; changed = true; }
    });
    if (remote.lastRead && (!state.lastRead || (remote.lastRead.at || 0) >= (state.lastRead.at || 0))) {
      state.lastRead = remote.lastRead; changed = true;
    }
    return changed;
  }

  function syncNow(mode) {
    if (syncBusy || !hasConfig()) return Promise.resolve(false);
    syncBusy = true;
    return gistFetch()
      .then(function (j) {
        var f = j.files && j.files[GIST_FILE];
        var remote = null;
        try { remote = f && f.content ? JSON.parse(f.content) : null; } catch (e) {}
        return mergeRemote(remote);
      }, function () { return false; })
      .then(function () { return gistWrite(state); })
      .then(function () {
        saveLocal(true);
        if (route() === 'progress.md') renderDashboard();
        if (route() === 'scratch.md') renderScratch();
        if (mode !== 'silent') toast('☁️ 已同步');
      })
      .catch(function (e) {
        if (mode !== 'silent') toast('⚠️ 同步失败：' + e.message);
      })
      .then(function () { syncBusy = false; });
  }

  /* ---------------- 工具函数 ---------------- */

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function route() {
    var h = location.hash.replace(/^#\/?/, '');
    var p = h.split('?')[0];
    try { p = decodeURIComponent(p); } catch (e) {}
    p = (p || 'README.md').replace(/\/+$/, '');
    if (!p) p = 'README.md';
    // docsify 渲染的导航/侧边栏链接会省略 .md（如 #/progress），这里统一补全
    if (!/\.md$/.test(p)) p += '.md';
    return p;
  }

  function routeDir() {
    var p = route();
    var i = p.lastIndexOf('/');
    return i < 0 ? '' : p.slice(0, i);
  }

  function isSpecial(p) { return p === 'progress.md' || p === 'scratch.md'; }

  function encPath(p) {
    return p.split('/').map(encodeURIComponent).join('/');
  }

  function go(rootPath) {
    location.hash = '#/' + encPath(rootPath);
  }

  /* 把“相对仓库根”的路径转为相对当前页面的路径（relativePath: true 模式下必须） */
  function relPath(rootPath) {
    var from = routeDir() ? routeDir().split('/') : [];
    var to = rootPath.split('/');
    var i = 0;
    while (i < from.length && i < to.length - 1 && from[i] === to[i]) i++;
    var out = from.slice(i).map(function () { return '..'; }).concat(to.slice(i));
    return out.join('/') || to[to.length - 1];
  }

  function withTimeout(promise, ms) {
    return Promise.race([
      promise,
      new Promise(function (res) { setTimeout(function () { res(null); }, ms); })
    ]);
  }

  function ensureWikiMap() {
    if (wikiMap) return Promise.resolve(wikiMap);
    if (!wikiMapPromise) {
      wikiMapPromise = fetch('_assets/wikimap.json', { cache: 'no-cache' })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (j) {
          wikiMap = j;
          // 若首屏渲染时索引还没就绪（双链未转换），整页刷新一次补上（每个会话最多一次）
          if (j && pendingWikiRoute && pendingWikiRoute === route() && !sessionStorage.getItem('lr_wr')) {
            try { sessionStorage.setItem('lr_wr', '1'); } catch (e) {}
            setTimeout(function () { try { sessionStorage.removeItem('lr_wr'); } catch (e) {} }, 8000);
            location.reload();
          }
          return j;
        })
        .catch(function () { return null; });
    }
    return wikiMapPromise;
  }

  function getSearchIndex() {
    if (searchIndex) return Promise.resolve(searchIndex);
    if (!searchIndexPromise) {
      searchIndexPromise = fetch('_assets/search-index.json', { cache: 'no-cache' })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (j) { searchIndex = j; return j; })
        .catch(function () { return null; });
    }
    return searchIndexPromise;
  }

  function resolveWiki(target) {
    if (!wikiMap) return null;
    var t = target.trim().replace(/^\.?\//, '').replace(/\.md$/, '');
    if (t.indexOf('/') >= 0 && wikiMap.byPath[t]) return wikiMap.byPath[t];
    if (wikiMap.byName[t]) return wikiMap.byName[t];
    return null;
  }

  /* ---------------- markdown 变换 ---------------- */

  var FM_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;
  var CODE_SPLIT_RE = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`|\$\$[\s\S]*?\$\$|\$[^$\n]*\$)/g;
  var WL_RE = /\[\[([^\[\]|]*?)(?:#([^\[\]|]*?))?(?:\|([^\[\]]*?))?\]\]/g;

  function extractMeta(content) {
    var m = content.match(FM_RE);
    if (!m) return null;
    var meta = {};
    var g = function (re) { var x = m[1].match(re); return x ? x[1].replace(/^["'\s]+|["'\s]+$/g, '') : ''; };
    meta.status = g(/^status:\s*(.+)$/m);
    meta.updated = g(/^last_updated:\s*(.+)$/m);
    return meta.status || meta.updated ? meta : null;
  }

  function transformSegment(seg) {
    if (!wikiMap) return seg;
    return seg.replace(WL_RE, function (m, target, head, alias) {
      if (!target && !head) return '';
      var label = (alias || head || target || '').trim();
      if (!target) return '[' + label + '](?id=' + encodeURIComponent(head.trim()) + ')';
      var p = resolveWiki(target);
      if (!p) return '<span class="lr-wl-missing" title="未找到对应笔记">' + esc(label) + ' ⚠</span>';
      var href = relPath(p) + (head ? '?id=' + encodeURIComponent(head.trim()) : '');
      return '[' + label + '](' + href + ')';
    });
  }

  function transformAll(content) {
    if (!wikiMap) return content;
    var parts = content.split(CODE_SPLIT_RE);
    for (var i = 0; i < parts.length; i += 2) parts[i] = transformSegment(parts[i]);
    return parts.join('');
  }

  var CALLOUT_RE = /<blockquote>\s*<p>\s*\[!(\w+)\]\s*([^<]*?)<\/p>/i;
  var CALLOUT_ICONS = { note: '📝', abstract: '📋', info: 'ℹ️', todo: '☑️', tip: '💡', success: '✅', question: '❓', warning: '⚠️', caution: '⚠️', failure: '❌', danger: '🚨', bug: '🐞', example: '🔍', quote: '💬', important: '❗' };

  function transformCallouts(html) {
    return html.replace(CALLOUT_RE, function (m, type, rest) {
      var t = type.toLowerCase();
      var icon = CALLOUT_ICONS[t] || '📌';
      var title = rest.trim();
      return '<blockquote class="lr-callout lr-callout-' + t + '"><p class="lr-callout-title">' +
        icon + (title ? ' ' + esc(title) : '') + '</p>';
    });
  }

  function metaChips(meta) {
    var chips = '<div class="lr-meta">';
    if (meta.status) chips += '<span class="lr-chip lr-chip-status">' + esc(meta.status) + '</span>';
    if (meta.updated) chips += '<span class="lr-chip">更新: ' + esc(meta.updated) + '</span>';
    return chips + '</div>';
  }

  /* ---------------- 主题 ---------------- */

  function applyTheme(t) {
    document.documentElement.classList.toggle('lr-dark', t === 'dark');
    var b = document.getElementById('lr-theme-btn');
    if (b) b.textContent = t === 'dark' ? '☀️' : '🌙';
  }
  function initTheme() {
    var t = localStorage.getItem(LS.theme);
    if (!t) t = window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    applyTheme(t);
  }
  function toggleTheme() {
    var t = document.documentElement.classList.contains('lr-dark') ? 'light' : 'dark';
    localStorage.setItem(LS.theme, t);
    applyTheme(t);
  }

  /* ---------------- UI 构建 ---------------- */

  function el(id) { return document.getElementById(id); }

  function buildUI() {
    if (el('lr-fab')) return;

    var fab = document.createElement('div');
    fab.id = 'lr-fab';
    fab.innerHTML =
      '<div class="lr-fab-col">' +
      '<button class="lr-fab-btn" data-lr-act="open-search" title="搜索 (Ctrl+K)">🔍</button>' +
      '<button class="lr-fab-btn" id="lr-theme-btn" data-lr-act="toggle-theme" title="暗色 / 亮色">🌙</button>' +
      '<button class="lr-fab-btn" data-lr-act="open-settings" title="同步设置">⚙️</button>' +
      '</div>' +
      '<button id="lr-read-btn" class="lr-fab-pill" data-lr-act="toggle-read" style="display:none">✓ 标记已读</button>';
    document.body.appendChild(fab);

    var search = document.createElement('div');
    search.id = 'lr-search';
    search.className = 'lr-overlay hidden';
    search.innerHTML =
      '<div class="lr-search-box">' +
      '<input id="lr-search-input" placeholder="搜索标题 / 章节 / 摘要…  （Ctrl+K）" autocomplete="off">' +
      '<div id="lr-search-hint" class="lr-search-hint"></div>' +
      '<div id="lr-search-results"></div>' +
      '</div>';
    document.body.appendChild(search);

    var settings = document.createElement('div');
    settings.id = 'lr-settings';
    settings.className = 'lr-overlay hidden';
    settings.innerHTML =
      '<div class="lr-modal">' +
      '<h3>⚙️ 同步设置</h3>' +
      '<label>GitHub Token（只需 gist 权限）</label>' +
      '<input type="password" id="lr-token-input" placeholder="ghp_…" autocomplete="off">' +
      '<div class="lr-hint">还没有？<a href="https://github.com/settings/tokens/new?scopes=gist&amp;description=learning-progress-sync" target="_blank">点此一键创建</a>（页面里只勾 gist，其他不用勾，生成后粘贴到上面）</div>' +
      '<label>Gist ID</label>' +
      '<input id="lr-gist-input" autocomplete="off">' +
      '<div class="lr-actions">' +
      '<button data-lr-act="save-config" class="lr-btn-primary">保存</button>' +
      '<button data-lr-act="test">测试连接</button>' +
      '<button data-lr-act="sync-now">立即同步</button>' +
      '<button data-lr-act="export">导出备份</button>' +
      '<button data-lr-act="import">导入备份</button>' +
      '<button data-lr-act="reset-local" class="lr-danger">清空本地进度</button>' +
      '<button data-lr-act="close-settings">关闭</button>' +
      '</div>' +
      '<div id="lr-settings-status" class="lr-status"></div>' +
      '<input type="file" id="lr-import-file" accept=".json" style="display:none">' +
      '</div>';
    document.body.appendChild(settings);

    var toast = document.createElement('div');
    toast.id = 'lr-toast';
    document.body.appendChild(toast);
  }

  var toastTimer = null;
  function toast(msg) {
    var t = el('lr-toast');
    if (!t) return;
    t.textContent = msg;
    t.className = 'show';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.className = ''; }, 2600);
  }

  function openOverlay(id) {
    var ov = el(id);
    if (!ov) return;
    ov.classList.remove('hidden');
    if (id === 'lr-search') {
      var input = el('lr-search-input');
      input.value = '';
      el('lr-search-results').innerHTML = '';
      el('lr-search-hint').textContent = '';
      setTimeout(function () { input.focus(); }, 50);
      doSearch('');
    } else if (id === 'lr-settings') {
      el('lr-token-input').value = localStorage.getItem(LS.token) || '';
      el('lr-gist-input').value = currentGistId();
      el('lr-settings-status').textContent = '';
    }
  }
  function closeOverlays() {
    ['lr-search', 'lr-settings'].forEach(function (id) { var o = el(id); if (o) o.classList.add('hidden'); });
  }

  /* ---------------- 已读按钮 ---------------- */

  function refreshReadBtn() {
    var btn = el('lr-read-btn');
    if (!btn) return;
    var p = route();
    var show = /\.md$/.test(p) && !isSpecial(p) && p !== 'README.md';
    btn.style.display = show ? 'block' : 'none';
    if (!show) return;
    var entry = state.read[p];
    var done = entry && !entry.d;
    btn.textContent = done ? '✅ 已读（点击取消）' : '✓ 标记已读';
    btn.classList.toggle('lr-done', !!done);
  }

  function toggleRead() {
    var p = route();
    if (isSpecial(p) || p === 'README.md') return;
    var entry = state.read[p];
    if (entry && !entry.d) state.read[p] = { at: Date.now(), d: 1 };
    else state.read[p] = { at: Date.now() };
    saveLocal();
    refreshReadBtn();
    toast(state.read[p].d ? '已取消已读标记' : '✅ 已标记已读');
  }

  /* ---------------- 学习进度面板 ---------------- */

  function moduleOf(p) { return p.indexOf('/') > 0 ? p.split('/')[0] : '其他'; }

  function pct(a, b) { return b ? Math.round((a / b) * 100) : 0; }

  function fmtTime(ts) {
    if (!ts) return '';
    var d = new Date(ts);
    var pad = function (n) { return (n < 10 ? '0' : '') + n; };
    return (d.getMonth() + 1) + '月' + d.getDate() + '日 ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
  }

  function renderDashboard() {
    var box = el('lr-dashboard');
    if (!box) return Promise.resolve();
    return getSearchIndex().then(function (idx) {
      var box2 = el('lr-dashboard');
      if (!box2) return;
      var files = idx ? idx.files : [];
      var titleMap = {};
      var mods = {};
      var order = MODULE_ORDER.slice();

      files.forEach(function (f) {
        if (f.p === 'README.md') return;
        titleMap[f.p] = f.t;
        var m = f.m || '其他';
        if (order.indexOf(m) < 0) order.push(m);
        mods[m] = mods[m] || { done: 0, total: 0 };
        mods[m].total++;
        var e = state.read[f.p];
        if (e && !e.d) mods[m].done++;
      });
      Object.keys(state.read).forEach(function (p) {
        var e = state.read[p];
        if (e && !e.d && !titleMap[p]) {
          var m = moduleOf(p);
          if (order.indexOf(m) < 0) order.push(m);
          mods[m] = mods[m] || { done: 0, total: 0 };
          mods[m].total++; mods[m].done++;
          titleMap[p] = p;
        }
      });

      var done = 0, total = 0;
      order.forEach(function (m) { if (mods[m]) { done += mods[m].done; total += mods[m].total; } });

      var html = '<h1 id="学习进度面板">📈 学习进度面板</h1>';
      html += '<div class="lr-dash-overall"><div class="lr-dash-big">' + done + ' <span>/ ' + total + ' 篇</span></div>' +
        '<div class="lr-dash-pct">总体 ' + pct(done, total) + '%</div></div>' +
        '<div class="lr-bar lr-bar-big"><i style="width:' + pct(done, total) + '%"></i></div>';

      if (state.lastRead) {
        html += '<div class="lr-dash-continue"><button class="lr-btn-primary" data-lr-go="' + esc(state.lastRead.path) + '">↩️ 继续上次：' + esc(state.lastRead.title || state.lastRead.path) + '</button></div>';
      }

      var syncLine = hasConfig()
        ? '☁️ 云同步已开启 · 上次同步 ' + (localStorage.getItem(LS.lastSync) ? fmtTime(new Date(localStorage.getItem(LS.lastSync)).getTime()) : '从未') +
          ' · <a href="javascript:void 0" data-lr-act="sync-now">立即同步</a> · <a href="javascript:void 0" data-lr-act="open-settings">设置</a>'
        : '⚠️ 尚未配置云同步（进度只存在本机）· <a href="javascript:void 0" data-lr-act="open-settings">去设置 →</a>';
      html += '<div class="lr-dash-sync">' + syncLine + '</div>';

      html += '<h2 id="分模块进度">分模块进度</h2><div class="lr-mods">';
      order.forEach(function (m) {
        if (!mods[m]) return;
        var p = pct(mods[m].done, mods[m].total);
        html += '<div class="lr-mod"><div class="lr-mod-head"><span class="lr-mod-name">' + esc(m) +
          '</span><span class="lr-mod-stat">' + mods[m].done + ' / ' + mods[m].total + ' · ' + p + '%</span></div>' +
          '<div class="lr-bar"><i style="width:' + p + '%"></i></div></div>';
      });
      html += '</div>';

      var recent = Object.keys(state.read)
        .filter(function (p) { var e = state.read[p]; return e && !e.d; })
        .sort(function (a, b) { return (state.read[b].at || 0) - (state.read[a].at || 0); })
        .slice(0, 12);
      if (recent.length) {
        html += '<h2 id="最近已读">最近已读</h2><ul class="lr-recent">';
        recent.forEach(function (p) {
          html += '<li><a href="#/' + encPath(p) + '">' + esc(titleMap[p] || p) + '</a><span class="lr-recent-time">' + fmtTime(state.read[p].at) + '</span></li>';
        });
        html += '</ul>';
      }

      if (!idx) html += '<div class="lr-hint">（索引加载失败，统计可能不完整）</div>';
      box2.innerHTML = html;
    });
  }

  /* ---------------- 随手记 ---------------- */

  function renderScratch() {
    var list = el('lr-scratch-list');
    if (!list) return;
    var del = state.scratchDel || {};
    var items = state.scratch
      .filter(function (s) { return !(del[s.id] && (del[s.id].at || 0) >= (s.at || 0)); })
      .sort(function (a, b) { return (b.at || 0) - (a.at || 0); });
    if (!items.length) {
      list.innerHTML = '<div class="lr-hint">还没有记录。随手记会同步到你的私有 Gist，所有设备可见。</div>';
      return;
    }
    list.innerHTML = items.map(function (s) {
      return '<div class="lr-scratch-item"><div class="lr-scratch-head"><span>' + fmtTime(s.at) +
        '</span><a href="javascript:void 0" data-lr-del="' + esc(s.id) + '">删除</a></div><div class="lr-scratch-text">' + esc(s.text) + '</div></div>';
    }).join('');
  }

  function addScratch() {
    var input = el('lr-scratch-input');
    var text = (input.value || '').trim();
    if (!text) return;
    state.scratch.push({ id: Date.now() + '-' + Math.random().toString(36).slice(2, 8), at: Date.now(), text: text });
    input.value = '';
    saveLocal();
    renderScratch();
    toast(hasConfig() ? '已记录并同步中…' : '已记录（未配置云同步，仅本机）');
  }

  /* ---------------- 搜索 ---------------- */

  var searchTimer = null;

  function doSearch(q) {
    q = (q || '').trim().toLowerCase();
    var hint = el('lr-search-hint');
    var box = el('lr-search-results');
    if (!box) return;
    if (!q) {
      box.innerHTML = '<div class="lr-search-empty">输入关键词开始搜索<br><span class="lr-hint">搜索范围：全部 684 篇笔记的标题、章节标题与开头摘要；进入笔记后可用 Ctrl+F 搜全文</span></div>';
      hint.textContent = '';
      return;
    }
    if (!searchIndex) {
      box.innerHTML = '<div class="lr-search-empty">⏳ 索引加载中…</div>';
      getSearchIndex().then(function () { if (el('lr-search-input').value.trim()) doSearch(el('lr-search-input').value); });
      return;
    }
    var tokens = q.split(/\s+/);
    var results = [];
    var files = searchIndex.files || [];
    for (var i = 0; i < files.length && results.length < 200; i++) {
      var f = files[i];
      var hay = (f.t + ' ' + (f.h || []).join(' ') + ' ' + (f.x || '')).toLowerCase();
      var score = 0, ok = true;
      for (var j = 0; j < tokens.length; j++) {
        var tk = tokens[j];
        var it = f.t.toLowerCase().indexOf(tk);
        var ih = (f.h || []).join(' ').toLowerCase().indexOf(tk);
        var ix = (f.x || '').toLowerCase().indexOf(tk);
        if (it < 0 && ih < 0 && ix < 0) { ok = false; break; }
        score += (it >= 0 ? 100 : 0) + (ih >= 0 ? 10 : 0) + (ix >= 0 ? 1 : 0);
      }
      if (ok) results.push({ f: f, s: score });
    }
    results.sort(function (a, b) { return b.s - a.s; });
    hint.textContent = '共 ' + results.length + ' 条结果';
    var top = results.slice(0, 40);
    box.innerHTML = top.length ? top.map(function (r) {
      var f = r.f;
      var snippet = '';
      for (var k = 0; k < tokens.length; k++) {
        var hit = (f.h || []).filter(function (h) { return h.toLowerCase().indexOf(tokens[k]) >= 0; })[0];
        if (hit) { snippet = '§ ' + hit; break; }
      }
      if (!snippet && f.x) {
        var lx = f.x.toLowerCase().indexOf(tokens[0]);
        if (lx >= 0) snippet = f.x.slice(Math.max(0, lx - 20), lx + 60);
      }
      var dir = f.p.split('/');
      dir.pop();
      return '<div class="lr-search-item" data-lr-go="' + esc(f.p) + '">' +
        '<div class="lr-si-title">' + esc(f.t) + '</div>' +
        '<div class="lr-si-path">' + esc((f.m ? f.m + ' / ' : '') + dir.join(' / ')) + '</div>' +
        (snippet ? '<div class="lr-si-snip">' + esc(snippet) + '</div>' : '') +
        '</div>';
    }).join('') : '<div class="lr-search-empty">没有找到「' + esc(q) + '」相关笔记</div>';
  }

  /* ---------------- Docsify 插件 ---------------- */

  function lrPlugin(hook) {
    hook.init(function () {
      initTheme();
      ensureWikiMap();
    });

    hook.beforeEach(function (content) {
      if (typeof content !== 'string') return content;
      var p = route();
      if (p === 'progress.md') return '<div id="lr-dashboard" class="lr-loading">⏳ 正在加载学习进度…</div>';
      if (p === 'scratch.md') {
        return (
          '<h1 id="随手记">📝 随手记</h1>' +
          '<div class="lr-scratch">' +
          '<div class="lr-hint">在这里随手记录灵感 / 待办 / 公式片段。保存到你的私有 Gist，<b>所有设备同步</b>；之后可在 Obsidian 里查看 <code>learning-progress.json</code>。</div>' +
          '<textarea id="lr-scratch-input" placeholder="记点什么…（Ctrl+Enter 快速添加）"></textarea>' +
          '<div class="lr-scratch-actions"><button class="lr-btn-primary" data-lr-act="add-scratch">添加</button>' +
          '<span id="lr-scratch-status"></span></div>' +
          '<div id="lr-scratch-list"></div>' +
          '</div>');
      }
      var meta = extractMeta(content);
      var stripped = content.replace(FM_RE, '');
      var body = transformAll(stripped);
      if (!wikiMap && new RegExp(WL_RE.source).test(stripped)) pendingWikiRoute = p;
      return (meta ? metaChips(meta) : '') + body;
    });

    hook.afterEach(function (html, next) {
      var out = transformCallouts(html);
      if (route() === 'README.md') {
        var readCount = 0;
        Object.keys(state.read).forEach(function (k) { if (!state.read[k].d) readCount++; });
        var total = searchIndex ? searchIndex.count : 684;
        var bar = '<div class="lr-homebar">' +
          (state.lastRead ? '<button class="lr-btn-primary" data-lr-go="' + esc(state.lastRead.path) + '">↩️ 继续上次：' + esc(state.lastRead.title || state.lastRead.path) + '</button>' : '') +
          '<span class="lr-homebar-stat">📊 已读 ' + readCount + ' / ' + total + ' · ' + pct(readCount, total) + '%　' +
          '<a href="#/progress.md">进度详情 →</a></span></div>';
        out = bar + out;
      }
      next(out);
    });

    hook.doneEach(function () {
      var p = route();
      refreshReadBtn();
      if (p === 'progress.md') renderDashboard();
      if (p === 'scratch.md') renderScratch();
      if (/\.md$/.test(p) && !isSpecial(p) && p !== 'README.md') {
        var h1 = document.querySelector('.markdown-section h1');
        state.lastRead = { path: p, title: h1 ? h1.textContent.trim() : p, at: Date.now() };
        saveLocal();
      }
      renderMermaid();
    });

    hook.mounted(function () {
      buildUI();
      initTheme();
      getSearchIndex();
      if (hasConfig()) syncNow('silent');
      window.addEventListener('focus', function () {
        if (hasConfig()) {
          var last = localStorage.getItem(LS.lastSync);
          if (!last || Date.now() - new Date(last).getTime() > 10 * 60 * 1000) syncNow('silent');
        }
      });
    });
  }

  function renderMermaid() {
    if (!window.mermaid) return;
    if (!mermaidReady) {
      try { window.mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose' }); } catch (e) { return; }
      mermaidReady = true;
    }
    var nodes = document.querySelectorAll('pre code.lang-mermaid, pre code.language-mermaid');
    if (!nodes.length) return;
    var blocks = [];
    Array.prototype.forEach.call(nodes, function (code) {
      var pre = code.closest('pre');
      if (!pre || pre.dataset.lrMermaid) return;
      var div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = code.textContent;
      pre.replaceWith(div);
      div.dataset.lrMermaid = '1';
      blocks.push(div);
    });
    if (blocks.length && window.mermaid.run) {
      window.mermaid.run({ nodes: blocks }).catch(function () {});
    }
  }

  /* ---------------- 事件绑定 ---------------- */

  function bindEvents() {
    document.addEventListener('click', function (e) {
      if (e.target.classList && e.target.classList.contains('lr-overlay')) { closeOverlays(); return; }
      var t = e.target.closest('[data-lr-act],[data-lr-go],[data-lr-del]');
      if (!t) return;

      if (t.dataset.lrGo) { closeOverlays(); go(t.dataset.lrGo); return; }

      if (t.dataset.lrDel) {
        var id = t.dataset.lrDel;
        state.scratchDel[id] = { at: Date.now() };
        saveLocal();
        renderScratch();
        return;
      }

      switch (t.dataset.lrAct) {
        case 'toggle-read': toggleRead(); break;
        case 'toggle-theme': toggleTheme(); break;
        case 'open-search': openOverlay('lr-search'); break;
        case 'open-settings': openOverlay('lr-settings'); break;
        case 'close-settings': closeOverlays(); break;
        case 'add-scratch': addScratch(); break;
        case 'sync-now':
          if (!hasConfig()) { el('lr-settings-status').textContent = '请先填写并保存 Token'; openOverlay('lr-settings'); break; }
          t.textContent = '同步中…';
          syncNow().then(function () { t.textContent = '立即同步'; });
          break;
        case 'save-config': {
          var token = el('lr-token-input').value.trim();
          var gid = el('lr-gist-input').value.trim();
          if (token) localStorage.setItem(LS.token, token); else localStorage.removeItem(LS.token);
          if (gid) localStorage.setItem(LS.gistId, gid); else localStorage.removeItem(LS.gistId);
          el('lr-settings-status').textContent = hasConfig() ? '✓ 已保存，正在同步…' : '已保存（未完整配置，云同步未开启）';
          if (hasConfig()) syncNow();
          break;
        }
        case 'test': {
          var st = el('lr-settings-status');
          st.textContent = '测试中…';
          var tk = el('lr-token-input').value.trim();
          var saved = localStorage.getItem(LS.token);
          if (tk && tk !== saved) localStorage.setItem(LS.token, tk);
          gistFetch().then(function (j) {
            st.textContent = '✓ 连接成功 · Gist 更新于 ' + fmtTime(new Date(j.updated_at).getTime());
          }).catch(function (err) { st.textContent = '✗ 连接失败：' + err.message; });
          break;
        }
        case 'export': {
          var blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
          var a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = 'lr-progress-backup-' + new Date().toISOString().slice(0, 10) + '.json';
          a.click();
          URL.revokeObjectURL(a.href);
          break;
        }
        case 'import': el('lr-import-file').click(); break;
        case 'reset-local':
          if (confirm('确定清空本机已读进度？（云端数据在下次同步前不受影响）')) {
            state.read = {};
            saveLocal(true);
            refreshReadBtn();
            toast('已清空本地进度');
          }
          break;
      }
    });

    document.addEventListener('change', function (e) {
      if (e.target && e.target.id === 'lr-import-file') {
        var file = e.target.files[0];
        if (!file) return;
        var reader = new FileReader();
        reader.onload = function () {
          try {
            var data = JSON.parse(reader.result);
            if (!data.read) throw new Error('格式不对');
            state.read = data.read || {};
            state.scratch = data.scratch || [];
            state.scratchDel = data.scratchDel || {};
            state.lastRead = data.lastRead || null;
            saveLocal();
            refreshReadBtn();
            el('lr-settings-status').textContent = '✓ 导入成功';
            if (route() === 'progress.md') renderDashboard();
          } catch (err) {
            el('lr-settings-status').textContent = '✗ 导入失败：' + err.message;
          }
        };
        reader.readAsText(file);
        e.target.value = '';
      }
    });

    document.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (el('lr-search').classList.contains('hidden')) openOverlay('lr-search'); else closeOverlays();
      }
      if (e.key === 'Escape') closeOverlays();
    });

    document.addEventListener('keydown', function (e) {
      if (e.target && e.target.id === 'lr-scratch-input' && (e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        addScratch();
      }
      if (e.target && e.target.id === 'lr-search-input' && e.key === 'Enter') {
        var first = document.querySelector('#lr-search-results .lr-search-item');
        if (first) go(first.dataset.lrGo);
      }
    });

    document.addEventListener('input', function (e) {
      if (e.target && e.target.id === 'lr-search-input') {
        clearTimeout(searchTimer);
        var v = e.target.value;
        searchTimer = setTimeout(function () { doSearch(v); }, 120);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindEvents);
  } else {
    bindEvents();
  }
})();
