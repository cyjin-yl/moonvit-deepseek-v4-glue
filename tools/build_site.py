#!/usr/bin/env python3
"""构建 GitHub Pages 静态展示网站。

用法:
    python tools/build_site.py [--summaries-dir <目录>]

把 site-src/ 的静态外壳 + 仓库内的报告/文档/配置/实验结果复制到 site/,
并生成 site/data/manifest.json 作为前端唯一数据源。
纯标准库,无第三方依赖。
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "site-src"
OUT = ROOT / "site"

# 体积规则
JSON_MAX = 8 * 1024 * 1024        # 单个 json 上限 8MB
LOG_MAX = 512 * 1024              # log/jsonl 上限 512KB
PNG_MAX = 2 * 1024 * 1024         # experiments 下 png 上限 2MB
TOTAL_WARN = 900 * 1024 * 1024    # 总大小超过 900MB 报警(Pages 上限 1GB)

# summaries 数据文件(由并行任务写入 site-src/summaries/)
SUMMARY_FILES = [
    "docs.json",
    "reports.json",
    "configs.json",
    "experiments-v100.json",
    "experiments-qwen3b.json",
    "guide.json",
]

# 扩展名 → kind
KIND_MAP = {
    ".md": "md",
    ".json": "json",
    ".png": "png",
    ".pdf": "pdf",
    ".svg": "svg",
    ".csv": "csv",
    ".log": "log",
    ".jsonl": "jsonl",
    ".typ": "typ",
}


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}GB"


class Builder:
    def __init__(self):
        self.files = []    # 已发布文件清单
        self.skipped = []  # 未发布文件清单

    def add_file(self, src_path: Path, repo_path: str, content_path: str):
        kind = KIND_MAP.get(src_path.suffix.lower())
        if kind is None:
            kind = src_path.suffix.lower().lstrip(".") or "bin"
        self.files.append(
            {
                "content_path": content_path,
                "repo_path": repo_path,
                "kind": kind,
                "size": src_path.stat().st_size,
            }
        )

    def skip(self, path: Path, repo_path: str, reason: str):
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        self.skipped.append({"repo_path": repo_path, "size": size, "reason": reason})

    def copy(self, src_path: Path, repo_path: str, dest_rel: str):
        """复制单个文件并登记。"""
        dest = OUT / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
        self.add_file(src_path, repo_path, dest_rel.replace("\\", "/"))


def clean_output():
    # 不删根目录本身,只清空内容:Windows 下 http.server 常驻预览会锁住根目录句柄
    OUT.mkdir(parents=True, exist_ok=True)
    for child in OUT.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_static(b: Builder):
    """site-src 的静态文件:index.html 与 assets/。"""
    shutil.copy2(SRC / "index.html", OUT / "index.html")
    assets_src = SRC / "assets"
    for p in sorted(assets_src.rglob("*")):
        if p.is_file():
            rel = p.relative_to(SRC).as_posix()
            b.copy(p, f"site-src/{rel}", rel)
    # GitHub Pages 404 兜底:hash 路由本来免疫,拷一份保险
    shutil.copy2(SRC / "index.html", OUT / "404.html")


def copy_docs(b: Builder):
    for p in sorted((ROOT / "docs").rglob("*.md")):
        rel = p.relative_to(ROOT).as_posix()
        b.copy(p, rel, f"content/{rel}")


def copy_configs(b: Builder):
    for p in sorted((ROOT / "configs").rglob("*.json")):
        rel = p.relative_to(ROOT).as_posix()
        b.copy(p, rel, f"content/{rel}")


def copy_report(b: Builder):
    """report/ 下全部 PNG + 名为 main.pdf / main.typ 的文件。"""
    for p in sorted((ROOT / "report").rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        name = p.name.lower()
        if p.suffix.lower() == ".png" or name in ("main.pdf", "main.typ"):
            b.copy(p, rel, f"content/{rel}")
        else:
            b.skip(p, rel, "报告目录中非发布类型(仅发布 PNG 与 main.pdf/main.typ)")


def copy_root_markdown(b: Builder):
    """根目录三个 md:HANDOFF.md、README.md、kimi-export 会话导出。"""
    targets = [ROOT / "HANDOFF.md", ROOT / "README.md"]
    targets += sorted(ROOT.glob("kimi-export*.md"))
    for p in targets:
        if p.exists():
            b.copy(p, p.name, f"content/{p.name}")
        else:
            print(f"  [警告] 根目录文件不存在: {p.name}")


def copy_experiments(b: Builder):
    """experiments/ 按体积与类型规则复制。"""
    for p in sorted((ROOT / "experiments").rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        ext = p.suffix.lower()
        size = p.stat().st_size

        if ext in (".safetensors", ".pt", ".gz"):
            b.skip(p, rel, "二进制权重/压缩包不发布")
            continue
        if ext in (".md", ".svg", ".csv"):
            b.copy(p, rel, f"content/{rel}")
            continue
        if ext == ".json":
            if size <= JSON_MAX:
                b.copy(p, rel, f"content/{rel}")
            else:
                b.skip(p, rel, f"json 超过 8MB({human_size(size)})")
            continue
        if ext in (".log", ".jsonl"):
            if size <= LOG_MAX:
                b.copy(p, rel, f"content/{rel}")
            else:
                b.skip(p, rel, f"{ext} 超过 512KB({human_size(size)})")
            continue
        if ext == ".png":
            if size <= PNG_MAX:
                b.copy(p, rel, f"content/{rel}")
            else:
                b.skip(p, rel, f"png 超过 2MB({human_size(size)})")
            continue
        b.skip(p, rel, f"未纳入发布规则的文件类型({ext or '无扩展名'})")


def load_summaries(summaries_dir: Path):
    """合并 summaries/*.json。文件缺失时警告并继续。

    每个文件的约定格式:
      {
        "overview": {...},                 // 仅 docs.json,作为 manifest.project
        "items": {"<repo相对路径>": {title, summary, conclusions, lessons, tags}},
        ... 其余附加字段(story_arc/top_lessons/tree_notes/groups/chapters 等)
      }
    附加字段在 manifest 顶层按来源存: manifest["<字段>"]["<文件名>"] = 值。
    """
    project = {}
    summaries = {}
    extras = {}
    guide = None
    for name in SUMMARY_FILES:
        path = summaries_dir / name
        if not path.exists():
            print(f"  [警告] summaries 文件缺失,跳过: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  [警告] summaries 文件解析失败 {name}: {e}")
            continue
        if name == "guide.json":
            # 导读页数据直通 manifest.guide,不进 extras(避免 chapters 等字段与其他来源串味)
            guide = data
            continue
        if name == "docs.json" and isinstance(data.get("overview"), dict):
            project = data["overview"]
        items = data.get("items") or data.get("summaries") or {}
        # 兼容两种写法:{"<path>": {...}} 字典,或 [{"path": "<path>", ...}] 列表
        if isinstance(items, list):
            normalized = {}
            for it in items:
                if isinstance(it, dict) and it.get("path"):
                    v = {k2: v2 for k2, v2 in it.items() if k2 != "path"}
                    normalized[it["path"]] = v
                else:
                    print(f"  [警告] {name} 中跳过无法识别的条目: {str(it)[:80]}")
            items = normalized
        for k, v in items.items():
            if k in summaries:
                print(f"  [警告] 摘要路径重复,后者覆盖前者: {k} ({name})")
            summaries[k] = v
        for key, value in data.items():
            if key in ("overview", "items", "summaries"):
                continue
            extras.setdefault(key, {})[name] = value
    return project, summaries, extras, guide


def repo_tree():
    """仓库结构概览,供前端展示。"""
    scratch_dirs = sorted(
        p.name for p in (ROOT / "scratch").iterdir() if p.is_dir()
    ) if (ROOT / "scratch").exists() else []
    tools_count = sum(1 for _ in (ROOT / "tools").rglob("*.py"))
    tests_count = sum(1 for _ in (ROOT / "tests").rglob("*.py"))
    src_files = sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")
    )
    return {
        "scratch": scratch_dirs,
        "tools_count": tools_count,
        "tests_count": tests_count,
        "src_files": src_files,
    }


def main():
    parser = argparse.ArgumentParser(description="构建静态展示网站到 site/")
    parser.add_argument(
        "--summaries-dir",
        default=str(SRC / "summaries"),
        help="summaries 数据目录(默认 site-src/summaries)",
    )
    args = parser.parse_args()

    # Windows 控制台默认 GBK,强制 UTF-8 输出避免乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    summaries_dir = Path(args.summaries_dir)
    if not summaries_dir.is_absolute():
        summaries_dir = ROOT / summaries_dir

    if not SRC.exists():
        sys.exit("site-src/ 不存在,无法构建")

    print("== 构建静态网站 ==")
    clean_output()
    b = Builder()

    copy_static(b)
    copy_docs(b)
    copy_configs(b)
    copy_report(b)
    copy_root_markdown(b)
    copy_experiments(b)

    project, summaries, extras, guide = load_summaries(summaries_dir)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "summaries": summaries,
        "files": b.files,
        "skipped": b.skipped,
        "repo_tree": repo_tree(),
    }
    if guide:
        manifest["guide"] = guide
    # 附加字段按来源存到顶层(tree_notes/groups/chapters/story_arc 等)
    for key, by_source in extras.items():
        manifest[key] = by_source

    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # 统计
    total_size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    file_count = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"总文件数: {file_count}")
    print(f"总大小:   {human_size(total_size)}")
    print(f"已发布:   {len(b.files)} 个仓库文件")
    print(f"已跳过:   {len(b.skipped)} 个")
    print(f"摘要条目: {len(summaries)} 条")
    if total_size > TOTAL_WARN:
        print(f"!! 警告: 总大小超过 900MB,接近 GitHub Pages 1GB 上限 !!")
    print(f"输出目录: {OUT}")


if __name__ == "__main__":
    main()
