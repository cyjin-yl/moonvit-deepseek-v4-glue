/* ============================================================
   jsonviz.js — JSON 浏览器(零依赖)
   - 按需 fetch:点击才加载,显示大小与下载进度
   - 三种视图:图表 / 可折叠树 / 原始文本
   - 自动检测数值数组 → SVG 折线/柱状图
   - 对象数组的平行数值字段 → 多序列折线图
   用法: JsonViz.mount(container, { url, size, title })
   ============================================================ */
(function (global) {
  "use strict";

  // 图表配色(与设计系统一致)
  var PALETTE = ["#111111", "#FF6B6B", "#6B8CFF", "#7FBC8C", "#A388EE", "#FF9EC6", "#b45400"];
  var MAX_CHARTS = 20;     // 最多自动生成的图表数
  var MAX_POINTS = 500;    // 单序列采样上限
  var MAX_TREE_NODES = 4000; // 树节点上限,防大卡死
  var RAW_MAX = 200 * 1024;  // 原始文本展示上限 200KB

  function humanSize(n) {
    if (n == null) return "未知大小";
    var units = ["B", "KB", "MB", "GB"], u = 0;
    while (n >= 1024 && u < units.length - 1) { n /= 1024; u++; }
    return n.toFixed(u ? 1 : 0) + units[u];
  }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function isFiniteNum(x) { return typeof x === "number" && isFinite(x); }

  // 等距采样到 max 个点
  function sample(arr, max) {
    if (arr.length <= max) return arr;
    var out = [], step = arr.length / max;
    for (var i = 0; i < max; i++) out.push(arr[Math.floor(i * step)]);
    return out;
  }

  // 数值数组判定:长度≥3 且 ≥80% 是有限数字
  function isNumericArray(a) {
    if (!Array.isArray(a) || a.length < 3) return false;
    if (a.length > 100000) return false;
    var n = 0;
    for (var i = 0; i < a.length; i++) if (isFiniteNum(a[i])) n++;
    return n / a.length >= 0.8;
  }

  // 对象数组的平行数值字段:{ numericKeys, rows }
  function parallelSeries(a) {
    if (!Array.isArray(a) || a.length < 3 || a.length > 100000) return null;
    var objCount = 0;
    for (var i = 0; i < a.length; i++) {
      if (a[i] && typeof a[i] === "object" && !Array.isArray(a[i])) objCount++;
    }
    if (objCount / a.length < 0.8) return null;
    // 统计每个字段为有限数字的比例
    var keyStat = {}, total = a.length;
    for (var j = 0; j < a.length; j++) {
      var o = a[j];
      if (!o || typeof o !== "object") continue;
      for (var k in o) {
        if (isFiniteNum(o[k])) keyStat[k] = (keyStat[k] || 0) + 1;
      }
    }
    var keys = Object.keys(keyStat).filter(function (k) {
      return keyStat[k] / total >= 0.8;
    });
    if (keys.length === 0) return null;
    return { numericKeys: keys.slice(0, 6), rows: a };
  }

  // 遍历 JSON,收集可图表化的数组(限制访问节点数)
  function collectCharts(root) {
    var charts = [];
    var visited = 0;
    function walk(node, path) {
      if (charts.length >= MAX_CHARTS || visited > 50000) return;
      visited++;
      if (Array.isArray(node)) {
        if (isNumericArray(node)) {
          charts.push({ path: path, type: "numeric", data: node.filter(isFiniteNum) });
          return;
        }
        var ps = parallelSeries(node);
        if (ps) {
          charts.push({ path: path, type: "series", data: ps });
          return;
        }
        for (var i = 0; i < node.length && i < 200; i++) walk(node[i], path + "[" + i + "]");
      } else if (node && typeof node === "object") {
        for (var k in node) walk(node[k], path ? path + "." + k : k);
      }
    }
    walk(root, "");
    return charts;
  }

  // 画一条/多条折线(或柱状)SVG
  function drawChart(chart) {
    var W = 720, H = 260, PAD = 40;
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    var seriesList; // [{name, values, color}]
    if (chart.type === "numeric") {
      seriesList = [{ name: chart.path || "value", values: sample(chart.data, MAX_POINTS), color: PALETTE[0] }];
    } else {
      var rows = sample(chart.data.rows, MAX_POINTS);
      seriesList = chart.data.numericKeys.map(function (k, idx) {
        return {
          name: k,
          values: rows.map(function (r) { return isFiniteNum(r && r[k]) ? r[k] : null; }),
          color: PALETTE[idx % PALETTE.length]
        };
      });
    }

    // 全局范围
    var vmin = Infinity, vmax = -Infinity, n = 1;
    seriesList.forEach(function (s) {
      n = Math.max(n, s.values.length);
      s.values.forEach(function (v) {
        if (v == null) return;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      });
    });
    if (!isFinite(vmin)) { vmin = 0; vmax = 1; }
    if (vmin === vmax) { vmin -= 1; vmax += 1; }

    function X(i) { return PAD + (i / Math.max(1, n - 1)) * (W - 2 * PAD); }
    function Y(v) { return H - PAD - ((v - vmin) / (vmax - vmin)) * (H - 2 * PAD); }

    // 坐标轴 + 刻度文字
    var axis = document.createElementNS("http://www.w3.org/2000/svg", "g");
    axis.innerHTML =
      '<line x1="' + PAD + '" y1="' + (H - PAD) + '" x2="' + (W - PAD) + '" y2="' + (H - PAD) + '" stroke="#111" stroke-width="2"/>' +
      '<line x1="' + PAD + '" y1="' + PAD + '" x2="' + PAD + '" y2="' + (H - PAD) + '" stroke="#111" stroke-width="2"/>' +
      '<text x="' + (PAD - 6) + '" y="' + (PAD + 4) + '" text-anchor="end" font-size="11" font-family="monospace">' + fmtNum(vmax) + "</text>" +
      '<text x="' + (PAD - 6) + '" y="' + (H - PAD) + '" text-anchor="end" font-size="11" font-family="monospace">' + fmtNum(vmin) + "</text>" +
      '<text x="' + PAD + '" y="' + (H - PAD + 16) + '" font-size="11" font-family="monospace">0</text>' +
      '<text x="' + (W - PAD) + '" y="' + (H - PAD + 16) + '" text-anchor="end" font-size="11" font-family="monospace">' + (n - 1) + "</text>";
    svg.appendChild(axis);

    if (chart.type === "numeric" && seriesList[0].values.length <= 60) {
      // 点数少 → 柱状图
      var vals = seriesList[0].values;
      var bw = (W - 2 * PAD) / vals.length;
      var base = Y(Math.max(0, vmin));
      vals.forEach(function (v, i) {
        var r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        r.setAttribute("x", X(i) - bw * 0.4 + bw / 2);
        r.setAttribute("y", Math.min(Y(v), base));
        r.setAttribute("width", Math.max(1, bw * 0.8));
        r.setAttribute("height", Math.abs(Y(v) - base));
        r.setAttribute("fill", "#FFDC58");
        r.setAttribute("stroke", "#111");
        r.setAttribute("stroke-width", "1.5");
        svg.appendChild(r);
      });
    } else {
      seriesList.forEach(function (s) {
        var d = "", started = false;
        s.values.forEach(function (v, i) {
          if (v == null) { started = false; return; }
          d += (started ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1);
          started = true;
        });
        var p = document.createElementNS("http://www.w3.org/2000/svg", "path");
        p.setAttribute("d", d);
        p.setAttribute("fill", "none");
        p.setAttribute("stroke", s.color);
        p.setAttribute("stroke-width", "2.5");
        svg.appendChild(p);
      });
    }
    return { svg: svg, seriesList: seriesList };
  }

  function fmtNum(v) {
    if (Math.abs(v) >= 1000) return v.toExponential(2);
    return Math.abs(v) < 1 && v !== 0 ? v.toFixed(4) : v.toFixed(2);
  }

  // 可折叠树(原生 details/summary)
  function buildTree(root) {
    var box = el("div", "jv-tree");
    var count = 0;
    function leaf(parent, key, value) {
      var row = el("div", "jv-leaf");
      var k = el("span", "jv-key", key + ": ");
      row.appendChild(k);
      var v;
      if (typeof value === "number") v = el("span", "jv-num", String(value));
      else if (typeof value === "boolean") v = el("span", "jv-bool", String(value));
      else if (value === null) v = el("span", "jv-bool", "null");
      else {
        var s = String(value);
        v = el("span", "jv-str", JSON.stringify(s.length > 120 ? s.slice(0, 120) + "…" : s));
      }
      row.appendChild(v);
      parent.appendChild(row);
    }
    function node(parent, key, value) {
      if (++count > MAX_TREE_NODES) {
        if (count === MAX_TREE_NODES + 1) {
          parent.appendChild(el("div", "jv-leaf", "… 节点过多,已截断,请用原始文本视图 …"));
        }
        return;
      }
      if (Array.isArray(value)) {
        var d1 = el("details");
        var s1 = el("summary", null, key + "  [数组 × " + value.length + "]");
        d1.appendChild(s1);
        var lim = Math.min(value.length, 200);
        for (var i = 0; i < lim; i++) node(d1, "[" + i + "]", value[i]);
        if (value.length > lim) d1.appendChild(el("div", "jv-leaf", "… 还有 " + (value.length - lim) + " 项 …"));
        parent.appendChild(d1);
      } else if (value && typeof value === "object") {
        var d2 = el("details");
        var keys = Object.keys(value);
        var s2 = el("summary", null, key + "  {对象 × " + keys.length + "}");
        d2.appendChild(s2);
        keys.slice(0, 300).forEach(function (k) { node(d2, k, value[k]); });
        if (keys.length > 300) d2.appendChild(el("div", "jv-leaf", "… 还有 " + (keys.length - 300) + " 个字段 …"));
        parent.appendChild(d2);
      } else {
        leaf(parent, key, value);
      }
    }
    node(box, "root", root);
    return box;
  }

  // 带进度地 fetch JSON
  function fetchWithProgress(url, onProgress) {
    return fetch(url).then(function (resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      var total = Number(resp.headers.get("Content-Length")) || 0;
      if (!resp.body || !resp.body.getReader) return resp.text();
      var reader = resp.body.getReader();
      var chunks = [], received = 0;
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) {
            var all = new Uint8Array(received), off = 0;
            chunks.forEach(function (c) { all.set(c, off); off += c.length; });
            return new TextDecoder("utf-8").decode(all);
          }
          chunks.push(r.value);
          received += r.value.length;
          onProgress(received, total);
          return pump();
        });
      }
      return pump();
    });
  }

  function mount(container, opts) {
    container.innerHTML = "";

    var info = el("div", "notice",
      "JSON 文件" + (opts.size ? ",大小 " + humanSize(opts.size) : "") + "。点击按钮按需加载,不会一整坨糊在页面上。");
    var loadBtn = el("button", "btn btn-primary", "加载 JSON");
    info.appendChild(document.createTextNode(" "));
    info.appendChild(loadBtn);
    container.appendChild(info);

    var progress = el("div", "progress");
    var bar = el("div", "bar");
    progress.appendChild(bar);
    container.appendChild(progress);

    var body = el("div");
    container.appendChild(body);

    loadBtn.addEventListener("click", function () {
      loadBtn.disabled = true;
      loadBtn.textContent = "加载中…";
      progress.style.display = "block";
      fetchWithProgress(opts.url, function (received, total) {
        var denom = total || opts.size || 0;
        bar.style.width = denom ? Math.min(100, (received / denom) * 100).toFixed(1) + "%" : "100%";
      }).then(function (text) {
        var data = JSON.parse(text);
        info.style.display = "none";
        progress.style.display = "none";
        renderViews(body, data, text);
      }).catch(function (err) {
        info.textContent = "加载失败:" + err.message;
        info.classList.add("notice");
      });
    });
  }

  function renderViews(body, data, rawText) {
    body.innerHTML = "";
    var toolbar = el("div", "jv-toolbar");
    var panel = el("div");
    var btns = {};

    ["图表", "树", "原始文本"].forEach(function (name) {
      var b = el("button", "btn", name);
      btns[name] = b;
      b.addEventListener("click", function () { show(name); });
      toolbar.appendChild(b);
    });

    function show(name) {
      Object.keys(btns).forEach(function (k) { btns[k].classList.toggle("active", k === name); });
      panel.innerHTML = "";
      if (name === "图表") showCharts();
      else if (name === "树") panel.appendChild(buildTree(data));
      else {
        var pre = el("pre", "code-view");
        var pretty = rawText.length > RAW_MAX
          ? rawText.slice(0, RAW_MAX) + "\n… 原始文本过大,已截断 …"
          : rawText;
        pre.textContent = pretty;
        panel.appendChild(pre);
      }
    }

    function showCharts() {
      var charts = collectCharts(data);
      if (charts.length === 0) {
        panel.appendChild(el("div", "notice", "没检测到大段数值序列,试试树视图。"));
        return;
      }
      var tip = el("div", "notice", "自动检测到 " + charts.length + " 组数值序列,已生成图表(长序列已采样到 " + MAX_POINTS + " 点以内)。");
      panel.appendChild(tip);
      charts.forEach(function (c) {
        var box = el("div", "jv-chart");
        box.appendChild(el("div", "jv-chart-title", c.path || "(根数组)"));
        var drawn = drawChart(c);
        box.appendChild(drawn.svg);
        if (drawn.seriesList.length > 1) {
          var legend = el("div", "jv-legend");
          drawn.seriesList.forEach(function (s) {
            var item = el("span");
            var sw = el("span", "swatch");
            sw.style.background = s.color;
            item.appendChild(sw);
            item.appendChild(document.createTextNode(s.name));
            legend.appendChild(item);
          });
          box.appendChild(legend);
        }
        panel.appendChild(box);
      });
    }

    body.appendChild(toolbar);
    body.appendChild(panel);
    show("图表");
  }

  global.JsonViz = { mount: mount };
})(window);
