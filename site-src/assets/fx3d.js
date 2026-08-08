/* ============================================================
   fx3d.js — 3D 与动效(纯手写,零依赖,离线可用)
   1. 首页 hero:Canvas 手写 3D 线框流水线(MoonViT → Projector → DeepSeek)
   2. 卡片鼠标 3D 倾斜
   3. 滚动 reveal(IntersectionObserver)
   4. 页面过渡辅助
   5. 跑马灯内容填充
   全部尊重 prefers-reduced-motion。
   ============================================================ */
(function (global) {
  "use strict";

  var REDUCED = global.matchMedia &&
    global.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- 1. Hero:3D 线框流水线 ---------- */

  // 立方体顶点
  function cubeVerts(s) {
    var v = [];
    [-1, 1].forEach(function (x) {
      [-1, 1].forEach(function (y) {
        [-1, 1].forEach(function (z) { v.push([x * s, y * s, z * s]); });
      });
    });
    return v;
  }
  var CUBE_EDGES = [
    [0,1],[2,3],[4,5],[6,7], [0,2],[1,3],[4,6],[5,7], [0,4],[1,5],[2,6],[3,7]
  ];
  // 四棱锥顶点(底面 4 点 + 顶点)
  function pyramidVerts(s) {
    return [
      [-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s], [0, -s * 1.2, 0]
    ];
  }
  var PYR_EDGES = [[0,1],[1,2],[2,3],[3,0],[0,4],[1,4],[2,4],[3,4]];

  // 场景节点:MoonViT(立方体) → Projector(棱锥) → DeepSeek(大立方体)
  // labelY:标签的纵向位置;中间节点放到下方,避免窄屏时三个标签叠在一起
  var NODES = [
    { label: "MoonViT",  x: -2.6, labelY: 1.35,  verts: cubeVerts(0.7),    edges: CUBE_EDGES, color: "#6B8CFF" },
    { label: "Projector", x: 0,   labelY: -1.45, verts: pyramidVerts(0.55), edges: PYR_EDGES,  color: "#FF6B6B" },
    { label: "DeepSeek", x: 2.6,  labelY: 1.35,  verts: cubeVerts(0.9),    edges: CUBE_EDGES, color: "#7FBC8C" }
  ];
  // 光点:沿两条连接线流动
  var LINKS = [[NODES[0], NODES[1]], [NODES[1], NODES[2]]];

  function hero(canvas) {
    var ctx = canvas.getContext("2d");
    var W, H, dpr;
    var mouseX = 0, mouseY = 0;
    var angle = 0;

    function resize() {
      dpr = global.devicePixelRatio || 1;
      W = canvas.clientWidth; H = canvas.clientHeight;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    global.addEventListener("resize", resize);

    canvas.parentElement.addEventListener("mousemove", function (e) {
      var r = canvas.getBoundingClientRect();
      mouseX = ((e.clientX - r.left) / r.width - 0.5) * 2;
      mouseY = ((e.clientY - r.top) / r.height - 0.5) * 2;
    });

    // 透视投影:绕 Y 轴旋转 + 简单透视除法
    function project(p, rotY, rotX) {
      var x = p[0], y = p[1], z = p[2];
      var c = Math.cos(rotY), s = Math.sin(rotY);
      var x1 = x * c - z * s, z1 = x * s + z * c;
      var c2 = Math.cos(rotX), s2 = Math.sin(rotX);
      var y1 = y * c2 - z1 * s2, z2 = y * s2 + z1 * c2;
      var fov = 4.2;
      var scale = fov / (fov + z2 + 2.5);
      return [W / 2 + x1 * scale * W * 0.16, H / 2 + y1 * scale * W * 0.16, scale];
    }

    function drawFrame(t) {
      ctx.clearRect(0, 0, W, H);
      var rotY = angle + mouseX * 0.35;
      var rotX = -0.25 + mouseY * 0.2;

      // 连接线
      LINKS.forEach(function (lk) {
        var a = project([lk[0].x, 0, 0], rotY, rotX);
        var b = project([lk[1].x, 0, 0], rotY, rotX);
        ctx.strokeStyle = "#111";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      });

      // 流动光点
      var dots = 6;
      LINKS.forEach(function (lk, li) {
        for (var i = 0; i < dots; i++) {
          var u = ((t * 0.0004 + i / dots + li * 0.5) % 1);
          var x = lk[0].x + (lk[1].x - lk[0].x) * u;
          var p = project([x, 0, 0], rotY, rotX);
          ctx.fillStyle = "#FFDC58";
          ctx.strokeStyle = "#111";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(p[0], p[1], 5 * p[2] + 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      });

      // 节点线框 + 标签
      NODES.forEach(function (nd) {
        var pts = nd.verts.map(function (v) {
          return project([v[0] + nd.x, v[1], v[2]], rotY, rotX);
        });
        ctx.strokeStyle = nd.color;
        ctx.lineWidth = 3;
        nd.edges.forEach(function (e) {
          ctx.beginPath();
          ctx.moveTo(pts[e[0]][0], pts[e[0]][1]);
          ctx.lineTo(pts[e[1]][0], pts[e[1]][1]);
          ctx.stroke();
        });
        // 顶点墨点
        ctx.fillStyle = "#111";
        pts.forEach(function (p) {
          ctx.fillRect(p[0] - 3, p[1] - 3, 6, 6);
        });
        // 标签(窄屏自动缩小字号)
        var c = project([nd.x, nd.labelY, 0], rotY, rotX);
        ctx.font = "900 " + (W < 560 ? 12 : 15) + "px ui-monospace, monospace";
        var tw = ctx.measureText(nd.label).width;
        ctx.fillStyle = "#F5F0E8";
        ctx.strokeStyle = "#111";
        ctx.lineWidth = 2.5;
        ctx.fillRect(c[0] - tw / 2 - 8, c[1] - 14, tw + 16, 24);
        ctx.strokeRect(c[0] - tw / 2 - 8, c[1] - 14, tw + 16, 24);
        ctx.fillStyle = "#111";
        ctx.fillText(nd.label, c[0] - tw / 2, c[1] + 4);
      });
    }

    if (REDUCED) {
      drawFrame(0); // 减少动效:只画一帧静态图
      return;
    }
    var raf;
    function loop(t) {
      // 页面切走后 canvas 被移除则停止
      if (!canvas.isConnected) { cancelAnimationFrame(raf); return; }
      angle += 0.008;
      drawFrame(t);
      raf = requestAnimationFrame(loop);
    }
    raf = requestAnimationFrame(loop);
  }

  /* ---------- 2. 卡片 3D 倾斜 ---------- */
  function tilt(el) {
    if (REDUCED || !global.matchMedia("(hover: hover)").matches) return;
    el.addEventListener("mousemove", function (e) {
      var r = el.getBoundingClientRect();
      var rx = ((e.clientY - r.top) / r.height - 0.5) * -8;
      var ry = ((e.clientX - r.left) / r.width - 0.5) * 8;
      el.style.transform =
        "perspective(700px) rotateX(" + rx + "deg) rotateY(" + ry + "deg) translate(-3px,-3px)";
    });
    el.addEventListener("mouseleave", function () { el.style.transform = ""; });
  }

  function tiltAll(root) {
    (root || document).querySelectorAll("[data-tilt]").forEach(tilt);
  }

  /* ---------- 3. 滚动 reveal ---------- */
  var io = null;
  function reveals(root) {
    if (REDUCED) {
      (root || document).querySelectorAll(".reveal").forEach(function (e) {
        e.classList.add("in");
      });
      return;
    }
    if (!io) {
      io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en, idx) {
          if (en.isIntersecting) {
            // 错落延迟
            en.target.style.transitionDelay = (idx % 6) * 60 + "ms";
            en.target.classList.add("in");
            io.unobserve(en.target);
          }
        });
      }, { threshold: 0.12 });
    }
    (root || document).querySelectorAll(".reveal:not(.in)").forEach(function (e) {
      io.observe(e);
    });
  }

  /* ---------- 4. 页面过渡 ---------- */
  function transition(viewEl, renderFn) {
    if (REDUCED) { renderFn(); viewEl.classList.add("page-enter"); return; }
    viewEl.classList.remove("page-enter");
    viewEl.classList.add("page-exit");
    setTimeout(function () {
      viewEl.classList.remove("page-exit");
      renderFn();
      viewEl.classList.add("page-enter");
      global.scrollTo(0, 0);
    }, 140);
  }

  /* ---------- 5. 跑马灯 ---------- */
  function marquee(items) {
    var track = document.getElementById("marquee-track");
    if (!track) return;
    var text = items.map(function (s) { return "◆ " + s; }).join("　");
    // 复制两份实现无缝循环
    track.textContent = text + "　" + text;
  }

  global.FX = {
    hero: hero,
    tilt: tilt,
    tiltAll: tiltAll,
    reveals: reveals,
    transition: transition,
    marquee: marquee,
    REDUCED: REDUCED
  };
})(window);
