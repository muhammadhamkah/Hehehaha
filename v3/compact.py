"""Lossless recompression of CLOSED hourly recording files: ``HH.jsonl.gz`` -> ``HH.jsonl.zst``.

    python -m v3.compact --store /data/l2 [--level 19] [--dry-run]

The recorder runs this automatically in the background; the CLI is for existing stores.

Safety rules:
  * a file is compacted only when its hour ended more than ``grace_s`` ago AND it is not the
    file any writer currently has open (passed in by the recorder);
  * the zstd file is written to ``*.part``, then fully decompressed again and compared
    line-count + SHA-256 against the original decompressed content; only after an exact
    match is it renamed into place and the original deleted;
  * if a ``.zst`` already exists for that hour (e.g. late lines appended after a restart), the
    new data goes to ``HH.rN.jsonl.zst`` -- nothing is ever overwritten or merged in place;
  * every compaction is logged (original size, compressed size, ratio) to
    ``<store>/_health/compaction.jsonl``.
Readers (``data.event_store``) read .gz, .zst and .xz transparently.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

log = logging.getLogger("v3.compact")
DEFAULT_LEVEL = 19
GRACE_S = 180
_HOUR_RE = re.compile(r"(\d{8})[/\\](\d{2})\.jsonl\.gz$")


def hour_end_ms(path: str) -> int | None:
    m = _HOUR_RE.search(path)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + 3_600_000


def _target(path: str) -> str:
    base = path[: -len(".jsonl.gz")]
    t = base + ".jsonl.zst"
    n = 1
    while os.path.exists(t):
        t = f"{base}.r{n}.jsonl.zst"
        n += 1
    return t


def compact_file(path: str, level: int = DEFAULT_LEVEL) -> dict:
    import gzip

    import zstandard

    t0 = time.perf_counter()
    orig = os.path.getsize(path)
    h_src = hashlib.sha256()
    lines_src = 0
    target = _target(path)
    part = target + ".part"
    cctx = zstandard.ZstdCompressor(level=level, write_checksum=True)
    with gzip.open(path, "rb") as src, open(part, "wb") as dst, cctx.stream_writer(dst, closefd=False) as zw:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            h_src.update(chunk)
            lines_src += chunk.count(b"\n")
            zw.write(chunk)
    # verify: decompress the new file completely and compare with the original content
    h_dst = hashlib.sha256()
    lines_dst = 0
    with open(part, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        while True:
            chunk = zr.read(1 << 20)
            if not chunk:
                break
            h_dst.update(chunk)
            lines_dst += chunk.count(b"\n")
    if h_src.digest() != h_dst.digest() or lines_src != lines_dst:
        os.remove(part)
        raise RuntimeError(f"verification failed for {path}; original kept")
    os.replace(part, target)
    os.remove(path)
    comp = os.path.getsize(target)
    return {"file": path, "to": target, "lines": lines_src, "orig_bytes": orig, "zst_bytes": comp,
            "ratio": round(orig / comp, 3) if comp else None, "sha256": h_src.hexdigest(),
            "seconds": round(time.perf_counter() - t0, 1), "level": level}


def closed_files(store: str, open_paths: set[str], now_ms: int, grace_s: int = GRACE_S) -> list[str]:
    out = []
    for path in sorted(glob.glob(os.path.join(store, "*", "*", "*.jsonl.gz"))):
        end = hour_end_ms(path)
        if end is None or now_ms < end + grace_s * 1000:
            continue
        if os.path.abspath(path) in open_paths:
            continue
        out.append(path)
    return out


def compact_store(store: str, open_paths: set[str] | None = None, level: int = DEFAULT_LEVEL,
                  grace_s: int = GRACE_S, now_ms: int | None = None, dry_run: bool = False) -> list[dict]:
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    open_abs = {os.path.abspath(p) for p in (open_paths or set()) if p}
    done = []
    for path in closed_files(store, open_abs, now_ms, grace_s):
        if dry_run:
            done.append({"file": path, "orig_bytes": os.path.getsize(path), "dry_run": True})
            continue
        try:
            rec = compact_file(path, level)
        except Exception as exc:  # noqa: BLE001
            log.error("compaction of %s failed: %s (original kept)", path, exc)
            continue
        log.info("compacted %s: %.1f MB -> %.1f MB (ratio %.2f, %ss)", os.path.relpath(path, store),
                 rec["orig_bytes"] / 1e6, rec["zst_bytes"] / 1e6, rec["ratio"] or 0, rec["seconds"])
        d = os.path.join(store, "_health")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "compaction.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": int(time.time() * 1000), **rec}) + "\n")
        done.append(rec)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    ap.add_argument("--grace-s", type=int, default=GRACE_S)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    recs = compact_store(a.store, level=a.level, grace_s=a.grace_s, dry_run=a.dry_run)
    o = sum(r["orig_bytes"] for r in recs)
    z = sum(r.get("zst_bytes", 0) for r in recs)
    print(f"{len(recs)} files; {o / 1e9:.3f} GB" + ("" if a.dry_run else f" -> {z / 1e9:.3f} GB"))


if __name__ == "__main__":
    main()
