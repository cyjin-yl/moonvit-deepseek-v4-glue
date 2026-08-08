/* ============================================================
   app.js — hash 路由 + 视图渲染(零依赖)
   数据源: data/manifest.json(由 tools/build_site.py 生成)
   ============================================================ */
(function () {
  "use strict";

  var view = document.getElementById("view");
  var manifest = null;
  var fileByRepoPath = {};

  /* ---------- 工具 ---------- */

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function humanSize(n) {
    if (n == null) return "?";
    var units = ["B", "KB", "MB", "GB"], u = 0;
    while (n >= 1024 && u < units.length - 1) { n /= 1024; u++; }
    return n.toFixed(u ? 1 : 0) + units[u];
  }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function itemLink(repoPath) {
    return "#/item?path=" + encodeURIComponent(repoPath);
  }

  function summaryOf(repoPath) {
    return (manifest.summaries && manifest.summaries[repoPath]) || null;
  }

  function baseName(p) {
    return p.split("/").pop();
  }

  // 附加字段(story_arc 等)在 manifest 顶层按来源存:{文件名: 值},取第一份
  function extra(name) {
    var m = manifest[name];
    if (!m) return null;
    if (typeof m === "object" && !Array.isArray(m)) {
      var keys = Object.keys(m);
      if (keys.length && m[keys[0]] !== undefined &&
          (keys[0].endsWith(".json") || typeof m[keys[0]] === "object" || Array.isArray(m[keys[0]]))) {
        // 按来源存的结构:合并数组或取第一个值
        var merged = [];
        var sawArray = false;
        keys.forEach(function (k) {
          if (Array.isArray(m[k])) { sawArray = true; merged = merged.concat(m[k]); }
        });
        if (sawArray) return merged;
        return m[keys[0]];
      }
    }
    return m;
  }

  // 自然排序(page-2 < page-10)
  function naturalSort(a, b) {
    return a.localeCompare(b, "zh-Hans-CN", { numeric: true });
  }

  /* ---------- 灯箱 ---------- */

  function openLightbox(src) {
    var lb = document.getElementById("lightbox");
    document.getElementById("lightbox-img").src = src;
    lb.hidden = false;
  }
  document.getElementById("lightbox-close").addEventListener("click", function () {
    document.getElementById("lightbox").hidden = true;
  });
  document.getElementById("lightbox").addEventListener("click", function (e) {
    if (e.target === this) this.hidden = true;
  });

  /* ---------- 摘要头(详情页固定显示) ---------- */

  function summaryHead(repoPath, fallbackTitle) {
    var s = summaryOf(repoPath);
    var box = el("div", "summary-head");
    var h = el("h2", null, (s && s.title) || fallbackTitle || repoPath);
    box.appendChild(h);
    var p = el("p", "meta", repoPath);
    p.style.fontFamily = "var(--mono)";
    p.style.fontSize = "12px";
    box.appendChild(p);
    if (s && s.summary) box.appendChild(el("p", null, s.summary));
    if (!s) {
      box.appendChild(el("p", null, "(该条目暂无中文摘要数据,等待 summaries 接入)"));
    }
    if (s && Array.isArray(s.conclusions) && s.conclusions.length) {
      box.appendChild(el("div", "block-label", "结论 × " + s.conclusions.length));
      var ul = el("ul", "conclusions");
      s.conclusions.forEach(function (c) { ul.appendChild(el("li", null, c)); });
      box.appendChild(ul);
    }
    if (s && Array.isArray(s.lessons) && s.lessons.length) {
      box.appendChild(el("div", "block-label", "经验教训 × " + s.lessons.length));
      var ul2 = el("ul", "lessons");
      s.lessons.forEach(function (c) { ul2.appendChild(el("li", null, c)); });
      box.appendChild(ul2);
    }
    if (s && Array.isArray(s.tags) && s.tags.length) {
      var tagBox = el("div");
      var colors = ["t-yellow", "t-blue", "t-green", "t-pink", "t-purple", "t-red"];
      s.tags.forEach(function (t, i) {
        tagBox.appendChild(el("span", "tag " + colors[i % colors.length], t));
      });
      box.appendChild(tagBox);
    }
    return box;
  }

  /* ---------- 卡片 ---------- */

  function entryCard(repoPath, accentCls) {
    var s = summaryOf(repoPath);
    var f = fileByRepoPath[repoPath];
    var a = el("a", "card clickable reveal" + (accentCls ? " " + accentCls : ""));
    a.href = itemLink(repoPath);
    a.setAttribute("data-tilt", "");
    a.appendChild(el("h3", null, (s && s.title) || baseName(repoPath)));
    a.appendChild(el("div", "meta", repoPath + (f ? " · " + humanSize(f.size) : "")));
    if (s && s.summary) a.appendChild(el("p", null, s.summary));
    var bits = [];
    if (s && s.conclusions) bits.push("结论 × " + s.conclusions.length);
    if (s && s.lessons) bits.push("经验 × " + s.lessons.length);
    if (bits.length) {
      var meta = el("p", "meta", bits.join(" · "));
      meta.style.fontWeight = "700";
      a.appendChild(meta);
    }
    return a;
  }

  /* ================= 视图:首页 ================= */

  function viewHome() {
    view.innerHTML = "";

    var proj = manifest.project || {};

    // Hero:3D 流水线
    var hero = el("section", "hero");
    var canvas = document.createElement("canvas");
    hero.appendChild(canvas);
    hero.appendChild(el("div", "hero-tag", "Gate D: NO-GO"));
    hero.appendChild(el("div", "hero-caption",
      proj.tagline || "MoonViT 视觉塔 → Projector → DeepSeek-V4:一次诚实的技术尽调"));
    view.appendChild(hero);
    FX.hero(canvas);

    // 海报标题 + 一句话
    var poster = el("h1", "poster");
    poster.innerHTML = esc(proj.title || "租卡训练之前,") + "<br><span class='stroke'>" +
      esc(proj.title_stroke || "先看清几何坍缩。") + "</span>";
    view.appendChild(poster);
    if (proj.one_liner || proj.project_one_liner) {
      var one = el("p", null, proj.one_liner || proj.project_one_liner);
      one.style.fontSize = "18px";
      one.style.fontWeight = "700";
      one.style.maxWidth = "720px";
      view.appendChild(one);
    }

    // 导读入口:全站最显眼的 CTA
    var cta = el("a", "guide-cta");
    cta.href = "#/guide";
    cta.appendChild(el("div", "guide-cta-title", "第一次来?从《十课导读》开始 →"));
    cta.appendChild(el("div", "guide-cta-sub",
      "约 25 分钟,零背景要求。每个数字都能点开看解释,每章一句话总结,读完再看下面的数字你全都能看懂。"));
    view.appendChild(cta);

    // 最终结论
    if (proj.final_verdict) {
      view.appendChild(el("div", "notice", "最终结论:" + proj.final_verdict));
    }

    // 关键数字
    var nums = computeKeyNumbers(proj);
    var numHead = el("h2", "section-title", "关键数字");
    view.appendChild(numHead);
    var numTip = el("p", "meta", "看不懂某个数字?去 <a href='#/guide'>导读</a>,每个都有通俗解释。");
    numTip.innerHTML = "看不懂某个数字?去 <a href='#/guide'>导读</a>,每个都有通俗解释。";
    view.appendChild(numTip);
    var grid = el("div", "grid cols-3");
    var colors = ["c-yellow", "c-red", "c-blue", "c-green", "c-pink", "c-purple"];
    nums.forEach(function (n, i) {
      var c = el("div", "num-card reveal " + colors[i % colors.length]);
      c.setAttribute("data-tilt", "");
      c.appendChild(el("div", "num", String(n.value)));
      c.appendChild(el("div", "label", n.label));
      grid.appendChild(c);
    });
    view.appendChild(grid);

    // 故事时间线(兼容:overview 内嵌 或 附加字段;条目兼容字符串或对象)
    var arc = proj.story_arc || extra("story_arc");
    if (arc && arc.length) {
      view.appendChild(el("h2", "section-title blue", "故事时间线"));
      var tl = el("div", "timeline");
      arc.forEach(function (step) {
        var item = el("div", "timeline-item reveal");
        if (typeof step === "string") {
          item.appendChild(el("div", "t-desc", step));
        } else {
          if (step.phase) item.appendChild(el("div", "t-phase", step.phase));
          item.appendChild(el("div", "t-title", step.title || ""));
          if (step.desc) item.appendChild(el("div", "t-desc", step.desc));
        }
        tl.appendChild(item);
      });
      view.appendChild(tl);
    }

    // 六个分类入口
    view.appendChild(el("h2", "section-title green", "档案分类"));
    var cats = [
      ["#/guide", "十课导读", "本科生也能读懂的完整导览,数字逐个解释", "c-red"],
      ["#/reports", "报告包", "20 个 typst 渲染报告包 + 完整报告 PDF", "c-yellow"],
      ["#/docs", "设计文档", "架构、合同、门禁 runbook 等 markdown", "c-blue"],
      ["#/configs", "实验配置", "全部训练/评测配置 JSON", "c-green"],
      ["#/experiments", "实验数据", "V100 与 Qwen3B 两轮实验的结果文件", "c-pink"],
      ["#/browser", "全仓浏览", "按目录树 + 搜索浏览所有已发布文件", "c-purple"],
      ["#/lessons", "经验教训", "花钱买来的全部教训,聚合去重", "c-red"]
    ];
    var cg = el("div", "grid cols-3");
    cats.forEach(function (c) {
      var a = el("a", "cat-card reveal " + c[3]);
      a.href = c[0];
      a.setAttribute("data-tilt", "");
      a.appendChild(el("div", "cat-name", c[1]));
      a.appendChild(el("div", "cat-desc", c[2]));
      cg.appendChild(a);
    });
    view.appendChild(cg);

    // 全项目经验教训精选(兼容字符串或 {text, source})
    var top = proj.top_lessons || extra("top_lessons");
    if (top && top.length) {
      view.appendChild(el("h2", "section-title pink", "花钱买来的教训"));
      top.forEach(function (t) {
        var card = el("div", "lesson-card reveal");
        card.appendChild(el("div", "lesson-text", typeof t === "string" ? t : (t.text || "")));
        if (t && t.source) {
          var src = el("div", "lesson-src");
          var link = el("a", null, "来源:" + t.source);
          link.href = itemLink(t.source);
          src.appendChild(link);
          card.appendChild(src);
        }
        view.appendChild(card);
      });
    }

    // 名词表(overview.glossary,可选)
    if (Array.isArray(proj.glossary) && proj.glossary.length) {
      view.appendChild(el("h2", "section-title purple", "名词表"));
      var gg = el("div", "grid cols-2");
      proj.glossary.forEach(function (g) {
        var c = el("div", "card reveal");
        c.appendChild(el("h3", null, g.term || ""));
        c.appendChild(el("p", null, g.meaning || ""));
        gg.appendChild(c);
      });
      view.appendChild(gg);
    }

    // 跑马灯
    FX.marquee(nums.map(function (n) { return n.label + " " + n.value; })
      .concat(["Gate D NO-GO", "projector 训练第 1-2 步几何坍缩", "不进入付费训练"]));

    FX.tiltAll(view);
    FX.reveals(view);
  }

  function computeKeyNumbers(proj) {
    if (Array.isArray(proj.key_numbers) && proj.key_numbers.length) return proj.key_numbers;
    var files = manifest.files || [];
    var reports = new Set();
    files.forEach(function (f) {
      var m = f.content_path.match(/^content\/report\/(package[^/]+)\//);
      if (m) reports.add(m[1]);
    });
    function count(prefix) {
      return files.filter(function (f) { return f.repo_path.indexOf(prefix) === 0; }).length;
    }
    return [
      { value: reports.size, label: "报告包" },
      { value: count("docs/"), label: "设计文档" },
      { value: count("configs/"), label: "实验配置" },
      { value: count("experiments/"), label: "实验结果文件" },
      { value: files.length, label: "已发布文件" },
      { value: (manifest.skipped || []).length, label: "未发布大文件" }
    ];
  }

  /* ================= 视图:报告列表 ================= */

  function reportPackages() {
    var pkgs = {};
    (manifest.files || []).forEach(function (f) {
      var m = f.content_path.match(/^content\/report\/([^/]+)\/(.+)$/);
      if (!m) return;
      if (!pkgs[m[1]]) pkgs[m[1]] = { name: m[1], files: [] };
      pkgs[m[1]].files.push(f);
    });
    var list = Object.keys(pkgs).map(function (k) { return pkgs[k]; });
    list.sort(function (a, b) { return naturalSort(a.name, b.name); });
    return list;
  }

  function viewReports() {
    view.innerHTML = "";
    var h = el("h1", "poster", "报告包");
    view.appendChild(h);

    // 完整报告卡
    var mainPdf = (manifest.files || []).find(function (f) {
      return f.repo_path === "report/main.pdf";
    });
    if (mainPdf) {
      var full = el("a", "cat-card c-red reveal");
      full.href = mainPdf.content_path;
      full.target = "_blank";
      full.setAttribute("data-tilt", "");
      full.appendChild(el("div", "cat-name", "完整报告 main.pdf"));
      full.appendChild(el("div", "cat-desc",
        "一次看完整个项目的主报告(" + humanSize(mainPdf.size) + "),新窗口打开"));
      view.appendChild(full);
    }

    view.appendChild(el("h2", "section-title yellow", "全部报告包"));
    var grid = el("div", "grid cols-3");
    reportPackages().forEach(function (p) {
      var pngs = p.files.filter(function (f) { return f.kind === "png"; }).length;
      var s = summaryOf("report/" + p.name);
      var a = el("a", "card clickable reveal");
      a.href = "#/report/" + encodeURIComponent(p.name);
      a.setAttribute("data-tilt", "");
      a.appendChild(el("h3", null, (s && s.title) || p.name));
      var meta = el("div", "meta", "PNG × " + pngs +
        (p.files.some(function (f) { return f.kind === "pdf"; }) ? " · 含 PDF" : "") +
        (p.files.some(function (f) { return f.kind === "typ"; }) ? " · 含 typ" : ""));
      a.appendChild(meta);
      if (s && s.summary) a.appendChild(el("p", null, s.summary));
      grid.appendChild(a);
    });
    view.appendChild(grid);
    FX.tiltAll(view);
    FX.reveals(view);
    FX.marquee(["报告包 × " + reportPackages().length, "typst 渲染", "Gate D NO-GO"]);
  }

  /* ================= 视图:报告详情(画廊) ================= */

  function viewReportDetail(pkg) {
    view.innerHTML = "";
    var files = (manifest.files || []).filter(function (f) {
      return f.content_path.indexOf("content/report/" + pkg + "/") === 0;
    });
    if (!files.length) {
      view.appendChild(el("div", "notice", "没找到报告包:" + pkg));
      return;
    }
    view.appendChild(summaryHead("report/" + pkg, pkg));

    var back = el("a", "btn", "← 返回报告列表");
    back.href = "#/reports";
    view.appendChild(back);

    // PDF / typ 入口
    files.filter(function (f) { return f.kind === "pdf"; }).forEach(function (f) {
      var a = el("a", "btn btn-primary", "打开 PDF(新窗口," + humanSize(f.size) + ")");
      a.href = f.content_path;
      a.target = "_blank";
      a.style.marginLeft = "10px";
      view.appendChild(a);
    });
    files.filter(function (f) { return f.kind === "typ"; }).forEach(function (f) {
      var box = el("div");
      box.appendChild(el("h2", "section-title purple", "typst 源码 " + baseName(f.repo_path)));
      var pre = el("pre", "code-view", "加载中…");
      box.appendChild(pre);
      view.appendChild(box);
      fetch(f.content_path).then(function (r) { return r.text(); }).then(function (t) {
        pre.textContent = t.length > 100 * 1024 ? t.slice(0, 100 * 1024) + "\n… 过长已截断 …" : t;
      }).catch(function () { pre.textContent = "加载失败"; });
    });

    // PNG 画廊(懒加载 + 灯箱)
    var pngs = files.filter(function (f) { return f.kind === "png"; })
      .sort(function (a, b) { return naturalSort(a.repo_path, b.repo_path); });
    if (pngs.length) {
      view.appendChild(el("h2", "section-title", "页面画廊 × " + pngs.length));
      var gal = el("div", "gallery");
      pngs.forEach(function (f) {
        var fig = document.createElement("figure");
        fig.className = "reveal";
        var img = document.createElement("img");
        img.src = f.content_path;
        img.loading = "lazy";
        img.alt = baseName(f.repo_path);
        fig.appendChild(img);
        fig.appendChild(el("figcaption", null, baseName(f.repo_path) + " · " + humanSize(f.size)));
        fig.addEventListener("click", function () { openLightbox(f.content_path); });
        gal.appendChild(fig);
      });
      view.appendChild(gal);
    }
    FX.reveals(view);
    FX.marquee([pkg, "PNG × " + pngs.length, "点击放大"]);
  }

  /* ================= 视图:文档/配置/实验列表 ================= */

  function viewList(kind) {
    var prefix = { docs: "docs/", configs: "configs/", experiments: "experiments/" }[kind];
    var titles = { docs: "设计文档", configs: "实验配置", experiments: "实验数据" };
    var colors = { docs: "section-title blue", configs: "section-title green", experiments: "section-title pink" };
    view.innerHTML = "";

    var h = el("h1", "poster");
    h.textContent = titles[kind];
    view.appendChild(h);

    var files = (manifest.files || []).filter(function (f) {
      return f.repo_path.indexOf(prefix) === 0;
    });
    var withSummary = files.filter(function (f) { return summaryOf(f.repo_path); });
    var without = files.filter(function (f) { return !summaryOf(f.repo_path); });

    if (withSummary.length) {
      view.appendChild(el("h2", colors[kind], "带中文摘要 × " + withSummary.length));
      var g1 = el("div", "grid cols-2");
      withSummary.forEach(function (f) { g1.appendChild(entryCard(f.repo_path)); });
      view.appendChild(g1);
    }
    if (without.length) {
      view.appendChild(el("h2", colors[kind], "全部文件 × " + without.length));
      var g2 = el("div", "grid cols-3");
      without.forEach(function (f) { g2.appendChild(entryCard(f.repo_path)); });
      view.appendChild(g2);
    }
    if (!files.length) view.appendChild(el("div", "notice", "该分类下没有已发布文件。"));
    FX.tiltAll(view);
    FX.reveals(view);
    FX.marquee([titles[kind] + " × " + files.length]);
  }

  /* ================= 视图:实验数据(两棵实验树,分区块 + 表格) ================= */

  // 两棵实验树的元信息;tree_note 的键是 summaries 来源文件名
  var EXP_TREES = [
    {
      prefix: "experiments/v100_perception_20260804",
      name: "第一轮:V100 感知实验链",
      date: "2026-08-04 · 本地 V100 · Package 1–14 · 合成数据 + 0.5B 小模型",
      question: "这一轮回答的问题:视觉塔看到的信息,语言模型到底用没用上?",
      noteKey: "experiments-v100.json",
      cls: "tree-yellow",
      titleCls: "section-title"
    },
    {
      prefix: "experiments/qwen3b_community_eval_20260805",
      name: "第二轮:Qwen2.5-3B 租前归因",
      date: "2026-08-05 起 · 本地 V100 · 真实数据 + 3B 纯文本代理模型",
      question: "这一轮回答的问题:在足够大的代理模型上,projector 能不能学会'看图'?不能的话,死在哪一步?",
      noteKey: "experiments-qwen3b.json",
      cls: "tree-pink",
      titleCls: "section-title pink"
    }
  ];

  function expFileStat(prefix) {
    var n = 0, sz = 0;
    (manifest.files || []).forEach(function (f) {
      if (f.repo_path.indexOf(prefix + "/") === 0) { n++; sz += f.size || 0; }
    });
    return { n: n, sz: sz };
  }

  // heads: [名称列, 描述列, 结论/教训列, 文件列];第三列取 conclusions 还是 lessons 由 listField 决定
  function expTable(keys, heads, listField) {
    var wrap = el("div", "btable-wrap reveal");
    var table = el("table", "btable");
    var thead = document.createElement("thead");
    var hr = document.createElement("tr");
    heads.forEach(function (t) { hr.appendChild(el("th", null, t)); });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = document.createElement("tbody");
    keys.forEach(function (k) {
      var s = summaryOf(k) || {};
      var tr = document.createElement("tr");
      tr.className = "clickable-row";
      tr.title = "点击查看该实验的全部原始文件";
      tr.addEventListener("click", function () { location.hash = itemLink(k); });

      var td1 = document.createElement("td");
      td1.className = "col-name";
      td1.appendChild(el("strong", null, s.title || baseName(k)));
      td1.appendChild(el("div", "meta", k.replace(/^experiments\//, "")));
      tr.appendChild(td1);

      var td2 = document.createElement("td");
      td2.className = "col-what";
      td2.textContent = s.summary || "—";
      tr.appendChild(td2);

      var td3 = document.createElement("td");
      td3.className = "col-result";
      var list = s[listField] || [];
      if (list.length) {
        var ul = el("ul", "mini-list");
        list.slice(0, 2).forEach(function (c) { ul.appendChild(el("li", null, c)); });
        td3.appendChild(ul);
        if (list.length > 2) td3.appendChild(el("div", "meta", "还有 " + (list.length - 2) + " 条,点进详情看全部 →"));
      } else {
        td3.textContent = "—";
      }
      tr.appendChild(td3);

      var st = expFileStat(k);
      var td4 = el("td", "col-num", st.n ? st.n + " 个\n" + humanSize(st.sz) : "—");
      tr.appendChild(td4);

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function viewExperiments() {
    view.innerHTML = "";
    var h = el("h1", "poster");
    h.textContent = "实验数据";
    view.appendChild(h);
    var lede = el("p", "lede");
    lede.textContent = "这个项目一共跑过两轮真实硬件实验,全部在本地 V100 上完成,没花一分钱租卡。"
      + "每一轮先看'总说'理解整轮在干什么,再看表格——每行是一个实验,写清了它干了什么、结论是什么。"
      + "点任意一行可以进到那个实验的原始文件(JSON 结果、日志、图表)。";
    view.appendChild(lede);

    var expCount = 0;
    EXP_TREES.forEach(function (tree) {
      var entries = Object.keys(manifest.summaries || {}).filter(function (k) {
        return k.indexOf(tree.prefix + "/") === 0;
      });
      var top = entries.filter(function (k) { return k.split("/").length === 3; }).sort(naturalSort);
      var deep = entries.filter(function (k) { return k.split("/").length > 3; }).sort(naturalSort);
      expCount += top.length;

      var head = el("div", "tree-head " + tree.cls);
      head.appendChild(el("h2", null, tree.name));
      head.appendChild(el("div", "meta", tree.date + " · 实验 × " + top.length +
        (deep.length ? " · 失败/事故留档 × " + deep.length : "")));
      head.appendChild(el("p", "tree-question", tree.question));
      var note = (manifest.tree_note || {})[tree.noteKey];
      if (note) {
        var noteP = el("p", "tree-note");
        noteP.textContent = "总说:" + note;
        head.appendChild(noteP);
      }
      view.appendChild(head);

      view.appendChild(expTable(top, ["实验", "干了什么 / 怎么做的", "结论(关键数字)", "原始文件"], "conclusions"));

      if (deep.length) {
        var failLabel = el("h3", "block-label", "失败与事故留档 × " + deep.length + " —— 这些'翻车记录'也是财富,别跳过");
        view.appendChild(failLabel);
        view.appendChild(expTable(deep, ["失败记录", "发生了什么", "教训", "原始文件"], "lessons"));
      }
    });

    var expFiles = (manifest.files || []).filter(function (f) {
      return f.repo_path.indexOf("experiments/") === 0;
    });
    var foot = el("div", "notice");
    foot.innerHTML = "两棵树共发布原始结果文件 <strong>" + expFiles.length + "</strong> 个(JSON / 日志 / SVG 图 / CSV),"
      + "全部可以在 <a href='#/browser'>全仓浏览</a> 里按目录翻阅;权重等二进制大文件未发布,清单和原因也在浏览器里。";
    view.appendChild(foot);

    FX.reveals(view);
    FX.marquee(["实验树 × 2", "实验 × " + expCount, "结果文件 × " + expFiles.length, "全程零付费资源", "Gate D NO-GO"]);
  }

  /* ================= 视图:导读(本科生友好,逐章讲解) ================= */

  function firstSentence(s) {
    if (!s) return "";
    var m = s.match(/^[^。!?\n]*[。!?]/);
    return m ? m[0] : s.slice(0, 80) + "…";
  }

  function viewGuide() {
    view.innerHTML = "";
    var guide = manifest.guide || {};
    var chapters = guide.chapters || [];

    var h = el("h1", "poster");
    h.textContent = "导读:从零读懂这个项目";
    view.appendChild(h);
    var lede = el("p", "lede");
    lede.textContent = "这一页写给所有人,不需要任何 AI 背景。十章,大约 25 分钟,讲清楚这个项目想干什么、发现了什么、为什么停下来。"
      + "每个黄色数字徽章都可以点开——里面是这个数字的通俗解释;每章结尾都有'记住这个'一句话总结,和值得继续读的原文链接。";
    view.appendChild(lede);

    // 章节目录
    var toc = el("div", "guide-toc");
    chapters.forEach(function (c, i) {
      var a = el("a", "guide-toc-chip", (i + 1) + ". " + c.title.split(":")[0]);
      a.href = "#/guide";
      a.addEventListener("click", function (e) {
        e.preventDefault();
        var t = document.getElementById("ch-" + c.id);
        if (t) t.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      toc.appendChild(a);
    });
    view.appendChild(toc);

    var chapterColors = ["ch-yellow", "ch-blue", "ch-green", "ch-pink", "ch-purple"];
    chapters.forEach(function (c, i) {
      var sec = el("section", "chapter reveal " + chapterColors[i % chapterColors.length]);
      sec.id = "ch-" + c.id;

      var head = el("div", "chapter-head");
      head.appendChild(el("div", "chapter-no", "第 " + (i + 1) + " 课"));
      var ht = el("h2", null, c.title);
      head.appendChild(ht);
      if (c.subtitle) head.appendChild(el("div", "chapter-sub", c.subtitle));
      sec.appendChild(head);

      (c.body || []).forEach(function (p) { sec.appendChild(el("p", "chapter-p", p)); });

      if (c.analogy && c.analogy.text) {
        var ab = el("div", "analogy-box");
        ab.appendChild(el("div", "block-label", c.analogy.title || "打个比方"));
        ab.appendChild(el("p", null, c.analogy.text));
        sec.appendChild(ab);
      }

      if (c.numbers && c.numbers.length) {
        sec.appendChild(el("div", "block-label", "本章数字 × " + c.numbers.length + "(点开看解释)"));
        var chips = el("div", "num-chips");
        var explain = el("div", "num-explain");
        explain.hidden = true;
        c.numbers.forEach(function (n) {
          var chip = el("button", "num-chip", n.value);
          chip.type = "button";
          chip.addEventListener("click", function () {
            chips.querySelectorAll(".num-chip").forEach(function (x) { x.classList.remove("on"); });
            chip.classList.add("on");
            explain.innerHTML = "";
            explain.appendChild(el("strong", null, n.value + " —— " + n.label));
            explain.appendChild(el("p", null, n.explain));
            explain.hidden = false;
          });
          chips.appendChild(chip);
        });
        sec.appendChild(chips);
        sec.appendChild(explain);
      }

      if (c.reflection && c.reflection.points && c.reflection.points.length) {
        var rb = el("div", "reflection-box");
        rb.appendChild(el("div", "block-label", c.reflection.title || "我们的反思"));
        var rl = el("ol", "reflection-list");
        c.reflection.points.forEach(function (p) { rl.appendChild(el("li", null, p)); });
        rb.appendChild(rl);
        sec.appendChild(rb);
      }

      if (c.links && c.links.length) {
        sec.appendChild(el("div", "block-label", "想深入?读这些 × " + c.links.length));
        var lb = el("div", "guide-links");
        c.links.forEach(function (lk) {
          var s = summaryOf(lk.path);
          var a = el("a", "guide-link");
          a.href = itemLink(lk.path);
          a.appendChild(el("strong", null, (s && s.title) || baseName(lk.path)));
          a.appendChild(el("div", "meta", lk.path));
          a.appendChild(el("p", null, "为什么读它:" + lk.why));
          lb.appendChild(a);
        });
        sec.appendChild(lb);
      }

      if (c.takeaway) {
        var tb = el("div", "takeaway-box");
        tb.appendChild(el("span", "takeaway-label", "记住这个 →"));
        tb.appendChild(el("span", null, c.takeaway));
        sec.appendChild(tb);
      }
      view.appendChild(sec);
    });

    // 结尾:自动生成的阅读地图(全部文档一句话索引,确保一个不漏)
    view.appendChild(el("h2", "section-title purple", "毕业后:全部档案的一句话索引"));
    var mapNote = el("p", "lede");
    mapNote.textContent = "读完十课,你就是这个项目最懂行的外人了。下面是仓库里每一份文档、每一个实验的一句话说明,点哪个看哪个,没人会迷路。";
    view.appendChild(mapNote);

    var mapGroups = [
      ["文档(设计/合同/状态)", "docs/", "#/docs"],
      ["配置(实验的全部参数)", "configs/", "#/configs"],
      ["实验(两轮实测)", "experiments/", "#/experiments"],
      ["报告(75 页正式叙事)", "report/", "#/reports"]
    ];
    mapGroups.forEach(function (g) {
      var keys = Object.keys(manifest.summaries || {}).filter(function (k) {
        return k.indexOf(g[1]) === 0;
      }).sort(naturalSort);
      if (!keys.length) return;
      var gh = el("h3", "block-label");
      var ga = el("a", null, g[0] + " × " + keys.length + " →");
      ga.href = g[2];
      gh.appendChild(ga);
      view.appendChild(gh);
      var ul = el("div", "readmap");
      keys.forEach(function (k) {
        var s = summaryOf(k) || {};
        var a = el("a", "readmap-row");
        a.href = itemLink(k);
        a.appendChild(el("strong", null, s.title || baseName(k)));
        a.appendChild(el("span", null, firstSentence(s.summary) || k));
        ul.appendChild(a);
      });
      view.appendChild(ul);
    });

    FX.reveals(view);
    FX.marquee(["导读 × " + chapters.length + " 课", "约 25 分钟", "每个数字都有解释", "读完不迷路"]);
  }

  /* ================= 视图:条目详情 ================= */
  function viewItem(repoPath) {
    view.innerHTML = "";
    var f = fileByRepoPath[repoPath];
    view.appendChild(summaryHead(repoPath, f ? baseName(repoPath) : repoPath));

    if (!f) {
      // 可能是目录级摘要(如 report/package4-render 或实验子目录)
      var children = (manifest.files || []).filter(function (x) {
        return x.repo_path.indexOf(repoPath + "/") === 0;
      });
      if (children.length) {
        view.appendChild(el("h2", "section-title", "目录内容 × " + children.length));
        var g = el("div", "grid cols-3");
        children.forEach(function (c) { g.appendChild(entryCard(c.repo_path)); });
        view.appendChild(g);
      } else {
        view.appendChild(el("div", "notice", "该路径未发布(可能是被跳过的大文件),见仓库原始目录。"));
      }
      FX.reveals(view);
      return;
    }

    var kindBox = el("div");
    var meta = el("p", "meta", "类型 " + f.kind + " · 大小 " + humanSize(f.size));
    meta.style.fontFamily = "var(--mono)";
    kindBox.appendChild(meta);

    if (f.kind === "md") {
      var mdBox = el("div", "md-body card");
      mdBox.innerHTML = "<p>加载中…</p>";
      kindBox.appendChild(mdBox);
      fetch(f.content_path).then(function (r) { return r.text(); }).then(function (t) {
        mdBox.innerHTML = MD.render(t);
      }).catch(function (e) { mdBox.textContent = "加载失败:" + e.message; });
    } else if (f.kind === "json") {
      var jvBox = el("div");
      JsonViz.mount(jvBox, { url: f.content_path, size: f.size });
      kindBox.appendChild(jvBox);
    } else if (f.kind === "png" || f.kind === "svg") {
      var fig = document.createElement("figure");
      fig.style.cursor = "zoom-in";
      var img = document.createElement("img");
      img.src = f.content_path;
      img.style.cssText = "max-width:100%;border:3px solid #111;box-shadow:6px 6px 0 #111;";
      img.alt = baseName(repoPath);
      fig.appendChild(img);
      fig.addEventListener("click", function () { openLightbox(f.content_path); });
      kindBox.appendChild(fig);
    } else if (f.kind === "pdf") {
      var a = el("a", "btn btn-primary", "打开 PDF(新窗口)");
      a.href = f.content_path;
      a.target = "_blank";
      kindBox.appendChild(a);
      var em = document.createElement("embed");
      em.src = f.content_path;
      em.type = "application/pdf";
      em.style.cssText = "width:100%;height:70vh;border:3px solid #111;box-shadow:6px 6px 0 #111;margin-top:14px;";
      kindBox.appendChild(em);
    } else {
      // log / csv / typ / jsonl / 其他文本 → 等宽代码视图
      var pre = el("pre", "code-view", "加载中…");
      kindBox.appendChild(pre);
      fetch(f.content_path).then(function (r) { return r.text(); }).then(function (t) {
        pre.textContent = t.length > 200 * 1024 ? t.slice(0, 200 * 1024) + "\n… 过长已截断 …" : t;
      }).catch(function (e) { pre.textContent = "加载失败:" + e.message; });
    }
    view.appendChild(kindBox);
    FX.reveals(view);
    FX.marquee([baseName(repoPath), f.kind, humanSize(f.size)]);
  }

  /* ================= 视图:全仓浏览器 ================= */

  var browserState = { node: "", q: "" };

  function viewBrowser() {
    view.innerHTML = "";
    var h = el("h1", "poster");
    h.textContent = "全仓浏览";
    view.appendChild(h);

    var search = document.createElement("input");
    search.className = "search-box";
    search.placeholder = "搜索:标题 / 摘要 / 路径…";
    search.value = browserState.q;
    view.appendChild(search);

    var wrap = el("div", "browser");
    var treeBox = el("div", "browser-tree");
    var listBox = el("div");
    wrap.appendChild(treeBox);
    wrap.appendChild(listBox);
    view.appendChild(wrap);

    // 构建两级树
    var tree = buildTreeIndex();

    function renderTree() {
      treeBox.innerHTML = "";
      var root = el("div", "bt-node" + (browserState.node === "" ? " active" : ""),
        "全部文件 (" + manifest.files.length + ")");
      root.addEventListener("click", function () { browserState.node = ""; renderAll(); });
      treeBox.appendChild(root);
      Object.keys(tree).forEach(function (top) {
        var node = el("div", "bt-node" + (browserState.node === top ? " active" : ""),
          top + " (" + tree[top].count + ")");
        node.addEventListener("click", function () { browserState.node = top; renderAll(); });
        treeBox.appendChild(node);
        var childWrap = el("div", "bt-child");
        Object.keys(tree[top].subs).forEach(function (sub) {
          var key = top + "/" + sub;
          var c = el("div", "bt-node" + (browserState.node === key ? " active" : ""),
            sub + " (" + tree[top].subs[sub] + ")");
          c.addEventListener("click", function (e) {
            e.stopPropagation();
            browserState.node = key;
            renderAll();
          });
          childWrap.appendChild(c);
        });
        treeBox.appendChild(childWrap);
      });
    }

    function matchFile(f) {
      if (browserState.node && f.repo_path !== browserState.node &&
          f.repo_path.indexOf(browserState.node + "/") !== 0) return false;
      var q = browserState.q.trim().toLowerCase();
      if (!q) return true;
      var s = summaryOf(f.repo_path);
      var hay = (f.repo_path + " " + ((s && s.title) || "") + " " + ((s && s.summary) || "")).toLowerCase();
      return hay.indexOf(q) !== -1;
    }

    function renderList() {
      listBox.innerHTML = "";
      var files = manifest.files.filter(matchFile);
      if (!files.length) listBox.appendChild(el("div", "notice", "没有匹配的文件。"));
      files.slice(0, 300).forEach(function (f) {
        var s = summaryOf(f.repo_path);
        var a = el("a", "file-row");
        a.href = itemLink(f.repo_path);
        a.appendChild(el("div", "fr-title", (s && s.title) || baseName(f.repo_path)));
        a.appendChild(el("div", "fr-path", f.repo_path));
        if (s && s.summary) a.appendChild(el("div", "fr-summary", s.summary));
        a.appendChild(el("div", "fr-meta", f.kind + " · " + humanSize(f.size)));
        listBox.appendChild(a);
      });
      if (files.length > 300) {
        listBox.appendChild(el("div", "notice", "结果过多,只显示前 300 条,请用搜索缩小范围。"));
      }

      // 未发布的大文件也要列出
      var skipped = (manifest.skipped || []).filter(function (sk) {
        if (browserState.node && sk.repo_path !== browserState.node &&
            sk.repo_path.indexOf(browserState.node + "/") !== 0) return false;
        var q = browserState.q.trim().toLowerCase();
        return !q || sk.repo_path.toLowerCase().indexOf(q) !== -1;
      });
      if (skipped.length) {
        listBox.appendChild(el("h2", "section-title pink", "未发布 × " + skipped.length));
        skipped.slice(0, 200).forEach(function (sk) {
          var row = el("div", "file-row skipped");
          row.appendChild(el("div", "fr-title", baseName(sk.repo_path)));
          row.appendChild(el("div", "fr-path", sk.repo_path));
          row.appendChild(el("div", "fr-meta",
            humanSize(sk.size) + " · 未发布(" + sk.reason + "),见仓库"));
          listBox.appendChild(row);
        });
        if (skipped.length > 200) {
          listBox.appendChild(el("div", "notice", "未发布条目过多,只显示前 200 条。"));
        }
      }
    }

    var debounce = null;
    search.addEventListener("input", function () {
      clearTimeout(debounce);
      debounce = setTimeout(function () {
        browserState.q = search.value;
        renderList();
      }, 150);
    });

    function renderAll() { renderTree(); renderList(); }
    renderAll();
    FX.marquee(["全仓浏览", "已发布 × " + manifest.files.length, "未发布 × " + (manifest.skipped || []).length]);
  }

  function buildTreeIndex() {
    var tree = {};
    (manifest.files || []).forEach(function (f) {
      var parts = f.repo_path.split("/");
      var top = parts[0];
      var sub = parts.length > 2 ? parts[1] : "(根)";
      if (!tree[top]) tree[top] = { count: 0, subs: {} };
      tree[top].count++;
      tree[top].subs[sub] = (tree[top].subs[sub] || 0) + 1;
    });
    return tree;
  }

  /* ================= 视图:经验教训总集 ================= */

  function viewLessons() {
    view.innerHTML = "";
    var h = el("h1", "poster");
    h.textContent = "经验教训总集";
    view.appendChild(h);
    view.appendChild(el("p", null, "聚合所有摘要数据里的 lessons,按文本去重,每条带来源链接。"));

    var seen = {};
    var items = [];
    Object.keys(manifest.summaries || {}).forEach(function (path) {
      var s = manifest.summaries[path];
      (s.lessons || []).forEach(function (t) {
        var key = String(t).trim();
        if (!seen[key]) {
          seen[key] = true;
          items.push({ text: key, sources: [path] });
        } else {
          items.forEach(function (it) {
            if (it.text === key && it.sources.indexOf(path) === -1) it.sources.push(path);
          });
        }
      });
    });

    if (!items.length) {
      view.appendChild(el("div", "notice", "暂无 lessons 数据,等待 site-src/summaries/*.json 接入。"));
    }
    items.forEach(function (it) {
      var card = el("div", "lesson-card reveal");
      card.appendChild(el("div", "lesson-text", it.text));
      var src = el("div", "lesson-src");
      src.appendChild(document.createTextNode("来源:"));
      it.sources.forEach(function (p, i) {
        if (i) src.appendChild(document.createTextNode(" · "));
        var a = el("a", null, p);
        a.href = itemLink(p);
        src.appendChild(a);
      });
      card.appendChild(src);
      view.appendChild(card);
    });
    FX.reveals(view);
    FX.marquee(["经验教训 × " + items.length, "花钱买来的教训", "Gate D NO-GO"]);
  }

  /* ================= 路由 ================= */

  function setActiveNav(name) {
    document.querySelectorAll(".nav a").forEach(function (a) {
      a.classList.toggle("active", a.getAttribute("data-nav") === name);
    });
  }

  function route() {
    var hash = location.hash || "#/";
    var body = hash.slice(1); // 去掉 #
    var qIdx = body.indexOf("?");
    var query = "";
    if (qIdx !== -1) { query = body.slice(qIdx + 1); body = body.slice(0, qIdx); }
    var parts = body.split("/").filter(Boolean);

    FX.transition(view, function () {
      if (parts.length === 0) { setActiveNav("home"); viewHome(); }
      else if (parts[0] === "guide") { setActiveNav("guide"); viewGuide(); }
      else if (parts[0] === "reports") { setActiveNav("reports"); viewReports(); }
      else if (parts[0] === "report" && parts[1]) { setActiveNav("reports"); viewReportDetail(decodeURIComponent(parts[1])); }
      else if (parts[0] === "docs") { setActiveNav("docs"); viewList("docs"); }
      else if (parts[0] === "configs") { setActiveNav("configs"); viewList("configs"); }
      else if (parts[0] === "experiments") { setActiveNav("experiments"); viewExperiments(); }
      else if (parts[0] === "browser") { setActiveNav("browser"); viewBrowser(); }
      else if (parts[0] === "lessons") { setActiveNav("lessons"); viewLessons(); }
      else if (parts[0] === "item") {
        setActiveNav("");
        var params = new URLSearchParams(query);
        viewItem(params.get("path") || "");
      }
      else { setActiveNav(""); view.innerHTML = '<div class="notice">未知路由:' + esc(hash) + ",试试顶部导航。</div>"; }
    });
  }

  /* ---------- 启动 ---------- */

  fetch("data/manifest.json")
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(function (m) {
      manifest = m;
      (m.files || []).forEach(function (f) { fileByRepoPath[f.repo_path] = f; });
      window.addEventListener("hashchange", route);
      route();
    })
    .catch(function (e) {
      view.innerHTML = '<div class="notice">manifest.json 加载失败(' + esc(e.message) +
        ')。请先运行 <code>python tools/build_site.py</code> 生成站点。</div>';
    });
})();
