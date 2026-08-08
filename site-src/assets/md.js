/* ============================================================
   md.js — 小型手写 Markdown 渲染器(零依赖)
   支持:标题 / 粗体 / 斜体 / 行内代码 / 代码块 / 无序与有序列表
        表格 / 链接 / 图片 / 引用 / 分隔线
   用法: MD.render(markdownText) -> HTML 字符串
   ============================================================ */
(function (global) {
  "use strict";

  // HTML 转义,防注入
  function esc(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // 行内语法:图片、链接、行内代码、粗体、斜体
  function inline(s) {
    s = esc(s);
    // 图片 ![alt](src)
    s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g,
      '<img src="$2" alt="$1" loading="lazy">');
    // 链接 [text](url)
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // 行内代码 `code`
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    // 粗体 **text** / __text__
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    // 斜体 *text* / _text_(避免误伤单词内下划线)
    s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    s = s.replace(/(^|[^_\w])_([^_\n]+)_(?=[^_\w]|$)/g, "$1<em>$2</em>");
    return s;
  }

  // 判断表格分隔行 | --- | :-: |
  function isTableSep(line) {
    return /^\|?[\s:|-]+\|[\s:|-]*$/.test(line) && line.indexOf("-") !== -1;
  }

  function splitRow(line) {
    var t = line.trim();
    if (t.charAt(0) === "|") t = t.slice(1);
    if (t.charAt(t.length - 1) === "|") t = t.slice(0, -1);
    return t.split("|").map(function (c) { return c.trim(); });
  }

  function render(src) {
    var lines = String(src).replace(/\r\n?/g, "\n").split("\n");
    var html = [];
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];

      // 空行
      if (/^\s*$/.test(line)) { i++; continue; }

      // 代码块 ```
      var fence = line.match(/^```(\w*)\s*$/);
      if (fence) {
        var buf = [];
        i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i])) {
          buf.push(lines[i]); i++;
        }
        i++; // 跳过结束 fence
        html.push("<pre><code" + (fence[1] ? ' class="lang-' + fence[1] + '"' : "") +
          ">" + esc(buf.join("\n")) + "</code></pre>");
        continue;
      }

      // 标题 # ~ ######
      var h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        var lv = h[1].length;
        html.push("<h" + lv + ">" + inline(h[2]) + "</h" + lv + ">");
        i++; continue;
      }

      // 分隔线
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        html.push("<hr>"); i++; continue;
      }

      // 引用块(连续 > 行)
      if (/^\s*>/.test(line)) {
        var qbuf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) {
          qbuf.push(lines[i].replace(/^\s*>\s?/, "")); i++;
        }
        html.push("<blockquote>" + render(qbuf.join("\n")) + "</blockquote>");
        continue;
      }

      // 表格:当前行含 | 且下一行是分隔行
      if (line.indexOf("|") !== -1 && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        var heads = splitRow(line);
        i += 2;
        var rows = [];
        while (i < lines.length && lines[i].indexOf("|") !== -1 && !/^\s*$/.test(lines[i])) {
          rows.push(splitRow(lines[i])); i++;
        }
        var t = ["<table><thead><tr>"];
        heads.forEach(function (c) { t.push("<th>" + inline(c) + "</th>"); });
        t.push("</tr></thead><tbody>");
        rows.forEach(function (r) {
          t.push("<tr>");
          r.forEach(function (c) { t.push("<td>" + inline(c) + "</td>"); });
          t.push("</tr>");
        });
        t.push("</tbody></table>");
        html.push(t.join(""));
        continue;
      }

      // 无序列表
      if (/^\s*[-*+]\s+/.test(line)) {
        var ul = ["<ul>"];
        while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
          ul.push("<li>" + inline(lines[i].replace(/^\s*[-*+]\s+/, "")) + "</li>");
          i++;
        }
        ul.push("</ul>");
        html.push(ul.join(""));
        continue;
      }

      // 有序列表
      if (/^\s*\d+[.)]\s+/.test(line)) {
        var ol = ["<ol>"];
        while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
          ol.push("<li>" + inline(lines[i].replace(/^\s*\d+[.)]\s+/, "")) + "</li>");
          i++;
        }
        ol.push("</ol>");
        html.push(ol.join(""));
        continue;
      }

      // 普通段落:合并连续非空行
      var pbuf = [line];
      i++;
      while (i < lines.length && !/^\s*$/.test(lines[i]) &&
             !/^(#{1,6}\s|```|\s*[-*+]\s|\s*\d+[.)]\s|\s*>)/.test(lines[i]) &&
             !(lines[i].indexOf("|") !== -1 && i + 1 < lines.length && isTableSep(lines[i + 1]))) {
        pbuf.push(lines[i]); i++;
      }
      html.push("<p>" + inline(pbuf.join(" ")) + "</p>");
    }

    return html.join("\n");
  }

  global.MD = { render: render };
})(window);
