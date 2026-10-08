"""Storage projection from ACTUAL recorded bytes (run after >= 1 complete hour of recording).

    python -m v3.storage --store /data/l2 --max-gb 95 [--hours 3] [--days 14 21]

Measures, over the most recent ``--hours`` hourly files (each normalised by the time span it
actually covers, so a partial first or still-open last hour is exact):
  * on-disk bytes per symbol per hour (compressed .zst where the hour is already compacted;
    for hours still in .gz the zstd size is MEASURED by compressing that hour in memory at the
    recorder's level -- no assumed ratios)
  * bytes per stream type (decompressed share of each stream, applied to the compressed size)
  * projected GB/day and GB for 14 / 21 days (plus any --days), the current store size, and the
    remaining headroom under --max-gb
It only reports. If the projection does not fit, nothing is changed automatically: decide
first (the core diff-depth feed must not be reduced).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
from collections import defaultdict

from v3.compact import DEFAULT_LEVEL
from v3.recorder import store_size
from v3.store import store_symbols

_FILE_RE = re.compile(r"[/\\](\d{8})[/\\](\d{2})(?:\.r\d+)?\.jsonl\.(gz|zst|xz)$")
MIN_SECONDS = 3600
SAFETY = 0.9          # a plan "fits" only if it uses <= 90% of the budget


def _read_bytes(path: str) -> bytes:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return fh.read()
    if path.endswith(".zst"):
        import zstandard
        with open(path, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
            return zr.read()
    import lzma
    with lzma.open(path, "rb") as fh:
        return fh.read()


def _stream_kind(line: bytes) -> str:
    i = line.find(b'"s":"')
    if i < 0:
        return "?"
    j = line.find(b'"', i + 5)
    s = line[i + 5:j].decode(errors="replace")
    head, _, kind = s.partition("@")
    return head if head.startswith("__") else kind


def hour_files(store: str, symbol: str) -> dict[str, list[str]]:
    out = defaultdict(list)
    for p in glob.glob(os.path.join(store, symbol, "*", "*.jsonl.*")):
        m = _FILE_RE.search(p)
        if m:
            out[m.group(1) + m.group(2)].append(p)
    return dict(out)


def project(store: str, max_gb: float | None, hours: int = 3, days: tuple[int, ...] = (14, 21),
            level: int = DEFAULT_LEVEL, min_seconds: int = MIN_SECONDS) -> dict:
    import zstandard

    symbols = store_symbols(store)
    per_symbol, per_stream = {}, defaultdict(float)
    complete_hours_used = {}
    for sym in symbols:
        hf = hour_files(store, sym)
        keys = sorted(hf)[-hours:]
        comp_total, raw_total, secs = 0.0, 0.0, 0.0
        stream_raw = defaultdict(int)
        for k in keys:
            first = last = None
            for p in hf[k]:
                raw = _read_bytes(p)
                lines = raw.splitlines()
                if not lines:
                    continue
                raw_total += len(raw)
                for line in lines:
                    stream_raw[_stream_kind(line)] += len(line) + 1
                comp_total += (os.path.getsize(p) if not p.endswith(".gz")
                               else len(zstandard.ZstdCompressor(level=level).compress(raw)))
                r0, r1 = json.loads(lines[0])["r"], json.loads(lines[-1])["r"]
                first = r0 if first is None else min(first, r0)
                last = r1 if last is None else max(last, r1)
            if first is not None:
                secs += max(last - first, 1) / 1000
        if secs <= 0:
            continue
        per_hour = comp_total / secs * 3600
        per_symbol[sym] = {"seconds_measured": round(secs), "compressed_mb_per_hour": round(per_hour / 1e6, 2),
                           "raw_mb_per_hour": round(raw_total / secs * 3600 / 1e6, 2),
                           "compression_ratio_vs_raw": round(raw_total / comp_total, 2) if comp_total else None,
                           "gb_per_day": round(per_hour * 24 / 1e9, 3)}
        complete_hours_used[sym] = keys
        tot_raw = sum(stream_raw.values()) or 1
        for kind, b in stream_raw.items():
            per_stream[kind] += per_hour * b / tot_raw            # compressed bytes/hour attributed by raw share
    measured = min((v["seconds_measured"] for v in per_symbol.values()), default=0)
    if not per_symbol or measured < min_seconds:
        return {"error": f"need >= 1 hour of recording per symbol (measured {measured / 3600:.2f} h); rerun later"}
    other = sum(os.path.getsize(p) for d in ("_health", "_system") for p in glob.glob(os.path.join(store, d, "**", "*"),
                                                                                      recursive=True)
                if os.path.isfile(p))
    gb_day = sum(v["gb_per_day"] for v in per_symbol.values())
    current = store_size(store) / 1e9
    proj = {}
    for d in days:
        need = gb_day * d
        entry = {"projected_gb": round(need, 2)}
        if max_gb:
            entry["fits_with_10pct_margin"] = need + other / 1e9 <= max_gb * SAFETY
            entry["headroom_after_gb"] = round(max_gb - need - other / 1e9, 2)
        proj[f"{d}_days"] = entry
    out = {"symbols": per_symbol,
           "per_stream_compressed_mb_per_hour": {k: round(v / 1e6, 2) for k, v in sorted(per_stream.items(),
                                                                                          key=lambda kv: -kv[1])},
           "compressed_mb_per_hour_all_symbols": round(gb_day / 24 * 1e3, 1), "projected_gb_per_day": round(gb_day, 3),
           "projections": proj, "current_store_gb": round(current, 3), "max_gb": max_gb,
           "remaining_under_budget_gb": round(max_gb - current, 2) if max_gb else None,
           "days_until_budget_from_now": round((max_gb * 0.995 - current) / gb_day, 1) if max_gb and gb_day else None,
           "hours_used": complete_hours_used, "zstd_level": level}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--max-gb", type=float, default=95.0)
    ap.add_argument("--hours", type=int, default=3, help="most recent complete hours to measure")
    ap.add_argument("--days", nargs="+", type=int, default=[14, 21])
    ap.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    a = ap.parse_args()
    rep = project(a.store, a.max_gb, a.hours, tuple(a.days), a.level)
    with open(os.path.join(a.store, "storage_projection.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    if "error" in rep:
        print(rep["error"])
        return
    print(f"Measured (actual bytes, zstd level {rep['zstd_level']}):")
    for s, v in rep["symbols"].items():
        print(f"  {s:10s} {v['compressed_mb_per_hour']:8.1f} MB/h compressed  ({v['raw_mb_per_hour']:.0f} MB/h raw, "
              f"x{v['compression_ratio_vs_raw']})  -> {v['gb_per_day']:.2f} GB/day  [{v['seconds_measured'] / 3600:.1f} h measured]")
    print("  by stream (MB/h, compressed, all symbols):",
          ", ".join(f"{k} {v}" for k, v in rep["per_stream_compressed_mb_per_hour"].items()))
    print(f"Projected: {rep['projected_gb_per_day']:.2f} GB/day")
    for k, v in rep["projections"].items():
        fit = "" if "fits_with_10pct_margin" not in v else \
            (" -> FITS" if v["fits_with_10pct_margin"] else " -> DOES NOT FIT (stop and decide; do not cut diff depth)")
        print(f"  {k.replace('_', ' ')}: {v['projected_gb']:.1f} GB{fit}"
              + (f", headroom {v['headroom_after_gb']:.1f} GB" if "headroom_after_gb" in v else ""))
    if rep["max_gb"]:
        print(f"Store now {rep['current_store_gb']:.2f} GB of {rep['max_gb']} GB budget; "
              f"~{rep['days_until_budget_from_now']} days of recording left at this rate")


if __name__ == "__main__":
    main()
