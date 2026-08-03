#!/usr/bin/env python3
"""Relay v2: ModelScope -> local (8 threads) -> parallel scp pool -> workstation.

- One ssh call per dataset lists remote files (name size) to skip completes.
- Download one file at a time (8 ranged threads, per-chunk resume).
- scp uploads run in a pool (default 4) overlapping subsequent downloads.
- Final pass re-lists remote and reports any size mismatches.
"""
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

MSAPI = 'https://www.modelscope.cn/api/v1/datasets'
MSRES = 'https://www.modelscope.cn/datasets'
LOCAL_ROOT = 'D:/moonvit_on_dsv4f/scratch/mslocal'
REMOTE_HOST = 'doesworkstation'
REMOTE_ROOT = '/run/media/ezra/13D010B6FDBC1A06/staging/parquet'
CHUNK = 8 * 1024 * 1024
DL_WORKERS = 8
SCP_WORKERS = 8
MAX_SCP_BACKLOG = 6

DATASETS = [
    ('textvqa_val',    'lmms-lab/textvqa',       'data',                  'validation-'),
    ('mmmu_pro',       'AI-ModelScope/MMMU_Pro', 'standard (10 options)', 'test-'),
    ('docvqa_val',     'lmms-lab/DocVQA',        'DocVQA',                'validation-'),
    ('showui_desktop', 'showlab/ShowUI-desktop', 'data',                  'train-'),
    ('textvqa_train',  'lmms-lab/textvqa',       'data',                  'train-'),
    ('docvqa_train',   'lmms-lab/DocVQA',        'DocVQA',                'train-'),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tree(repo, root):
    url = f"{MSAPI}/{repo}/repo/tree?Revision=master&Root={urllib.parse.quote(root)}"
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            return [(f['Path'], f['Size']) for f in d['Data']['Files']
                    if f['Path'].endswith('.parquet')]
        except Exception as e:
            last = e
            time.sleep(3)
    raise RuntimeError(f"tree list failed for {repo} {root}: {last}")


def remote_listing(name):
    """Return {fname: size} for remote dataset dir; one ssh round trip."""
    cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', REMOTE_HOST,
           f"mkdir -p {REMOTE_ROOT}/{name} && cd {REMOTE_ROOT}/{name} && "
           "for f in *.parquet; do [ -f \"$f\" ] && [ ! -f \"$f.aria2\" ] && "
           "stat -c '%n %s' \"$f\"; done 2>/dev/null"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    res = {}
    for line in out.stdout.splitlines():
        parts = line.rsplit(' ', 1)
        if len(parts) == 2 and parts[1].isdigit():
            res[parts[0]] = int(parts[1])
    return res


def fetch_chunk(url, idx, expected, cpath):
    if os.path.exists(cpath) and os.path.getsize(cpath) == expected:
        return True
    start, end = idx * CHUNK, idx * CHUNK + expected - 1
    for attempt in range(12):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != expected:
                raise IOError(f"short read {len(data)} != {expected}")
            with open(cpath, 'wb') as f:
                f.write(data)
            return True
        except Exception:
            time.sleep(min(2 ** attempt, 30))
    log(f"  chunk {idx} FAILED after retries")
    return False


def download_file(url, fpath, size):
    nchunks = (size + CHUNK - 1) // CHUNK
    cdir = fpath + '.chunks'
    os.makedirs(cdir, exist_ok=True)
    tasks = [(i, min(CHUNK, size - i * CHUNK), os.path.join(cdir, f'{i:05d}'))
             for i in range(nchunks)]
    with concurrent.futures.ThreadPoolExecutor(DL_WORKERS) as ex:
        futs = [ex.submit(fetch_chunk, url, i, exp, cp) for i, exp, cp in tasks]
        if not all(f.result() for f in futs):
            return False
    with open(fpath, 'wb') as out:
        for i, exp, cp in tasks:
            with open(cp, 'rb') as f:
                shutil.copyfileobj(f, out)
    if os.path.getsize(fpath) != size:
        return False
    shutil.rmtree(cdir, ignore_errors=True)
    return True


def scp_one(name, fname, fpath, results, lock):
    r = subprocess.run(['scp', '-o', 'BatchMode=yes', fpath,
                        f'{REMOTE_HOST}:{REMOTE_ROOT}/{name}/{fname}'],
                       timeout=7200)
    ok = r.returncode == 0
    if ok:
        try:
            os.remove(fpath)
        except OSError:
            pass
    with lock:
        results.append((name, fname, ok))
    log(f"  scp {'OK' if ok else 'FAILED'}: {name}/{fname}")


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    import threading
    lock = threading.Lock()
    scp_results = []
    scp_pool = concurrent.futures.ThreadPoolExecutor(SCP_WORKERS)
    scp_futs = []
    dl_fail = []
    for name, repo, root, filt in DATASETS:
        if only and name not in only:
            continue
        log(f"=== {name} ({repo} {root} *{filt}*) ===")
        try:
            files = [(p, s) for p, s in tree(repo, root) if filt in p.split('/')[-1]]
        except RuntimeError as e:
            log(str(e))
            dl_fail.append((name, 'LISTING'))
            continue
        remote = remote_listing(name)
        todo = [(p, s) for p, s in files if remote.get(p.split('/')[-1]) != s]
        log(f"{name}: {len(files)} total, {len(files) - len(todo)} remote-complete, {len(todo)} to fetch")
        for p, size in todo:
            fname = p.split('/')[-1]
            url = f"{MSRES}/{repo}/resolve/master/{urllib.parse.quote(p)}"
            ldir = os.path.join(LOCAL_ROOT, name)
            os.makedirs(ldir, exist_ok=True)
            fpath = os.path.join(ldir, fname)
            while True:
                with lock:
                    backlog = sum(1 for f in scp_futs if not f.done())
                if backlog < MAX_SCP_BACKLOG:
                    break
                log(f"  scp backlog {backlog} >= {MAX_SCP_BACKLOG}, waiting...")
                time.sleep(20)
            if os.path.exists(fpath) and os.path.getsize(fpath) == size \
                    and not os.path.exists(fpath + '.chunks'):
                log(f"  local-complete, straight to scp: {fname}")
                scp_futs.append(scp_pool.submit(scp_one, name, fname, fpath, scp_results, lock))
                continue
            t0 = time.time()
            ok = download_file(url, fpath, size)
            dt = time.time() - t0
            if not ok:
                log(f"  DOWNLOAD FAILED: {fname}")
                dl_fail.append((name, fname))
                continue
            log(f"  dl {fname} {size / 1e6:.0f}MB {dt:.0f}s ({size / max(dt, 0.1) / 1e6:.1f}MB/s)")
            scp_futs.append(scp_pool.submit(scp_one, name, fname, fpath, scp_results, lock))
    for f in scp_futs:
        f.result()
    log("--- final remote verification ---")
    mism = []
    for name, repo, root, filt in DATASETS:
        if only and name not in only:
            continue
        try:
            files = [(p, s) for p, s in tree(repo, root) if filt in p.split('/')[-1]]
        except RuntimeError:
            continue
        remote = remote_listing(name)
        for p, s in files:
            fname = p.split('/')[-1]
            if remote.get(fname) != s:
                mism.append((name, fname, remote.get(fname), s))
    scp_bad = [(n, f) for n, f, ok in scp_results if not ok]
    log(f"RELAY_DONE scp_ok={sum(1 for *_, ok in scp_results if ok)} "
        f"scp_fail={len(scp_bad)} dl_fail={len(dl_fail)} remote_mismatch={len(mism)}")
    if dl_fail:
        log(f"  dl_fail: {dl_fail}")
    if scp_bad:
        log(f"  scp_fail: {scp_bad}")
    if mism:
        for m in mism:
            log(f"  mismatch: {m}")


if __name__ == '__main__':
    main()
