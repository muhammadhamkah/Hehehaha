"""Recording-quality report for a V3 L2 store.

    python -m v3.qa --store data/l2 [--deep] [--out data/l2/QA.md]

Summarises the recorder's per-minute health log and -- with ``--deep`` -- independently
re-validates every recorded diff-depth message offline (sequence continuity, gaps by
reason, share of time with a valid book, diff-book vs depth20 agreement, longest silence
per stream). A day is marked USABLE for a symbol only if the book was valid >= 95% of the
day, no events were dropped by the writer and no resynchronisation failed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from data.event_store import EventReader, list_event_files
from v3.book import GAP, L2Book
from v3.store import manifest, store_symbols

USABLE_VALID_SHARE = 0.95


def _day(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def health_summary(store: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(store, "_health", "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    h = json.loads(line)
                except ValueError:
                    continue
                for s, v in h.get("symbols", {}).items():
                    rows.append({"day": _day(h["ts"]), "symbol": s, "ts": h["ts"], "valid": v["book_valid"],
                                 "msgs": sum(v["msgs"].values()), "gaps": v["gaps"], "resyncs": v["resyncs"],
                                 "resync_fail": v["resync_fail"], "d20_checked": v["d20_checked"],
                                 "d20_mismatch": v["d20_mismatch"], "dropped": v["writer_dropped"],
                                 "lat_p99": max((x["p99"] for x in v["latency_ms"].values()), default=None),
                                 "reconnects": h.get("reconnects", 0), "free_gb": h.get("free_gb")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ts")
    g = df.groupby(["day", "symbol"])
    out = g.agg(minutes=("ts", "count"), valid_minutes=("valid", "sum"), msgs=("msgs", "sum"),
                gaps=("gaps", "max"), resyncs=("resyncs", "max"), resync_fail=("resync_fail", "max"),
                d20_checked=("d20_checked", "max"), d20_mismatch=("d20_mismatch", "max"),
                dropped=("dropped", "max"), lat_p99_median=("lat_p99", "median"),
                reconnects=("reconnects", "max"), min_free_gb=("free_gb", "min")).reset_index()
    return out


def deep_check(store: str, symbol: str) -> list[dict]:
    """Offline continuity validation of one symbol, per day."""
    root = os.path.join(store, symbol)
    reader = EventReader(root, files=list_event_files(root), time_source="local")
    book = L2Book(symbol)
    per = defaultdict(lambda: {"diffs": 0, "gaps": defaultdict(int), "valid_ms": 0, "span": [None, None],
                               "d20_ok": 0, "d20_bad": 0, "snapshots": 0, "max_silence_ms": defaultdict(int),
                               "markers": defaultdict(int)})
    last_ts = None
    last_kind_ts: dict[str, int] = {}
    pending_d20: dict[int, dict] = {}
    for ev in reader:
        d = per[_day(ev.ts)]
        d["span"][0] = ev.ts if d["span"][0] is None else d["span"][0]
        d["span"][1] = ev.ts
        if last_ts is not None and book.valid and _day(last_ts) == _day(ev.ts):
            d["valid_ms"] += ev.ts - last_ts
        last_ts = ev.ts
        head, _, kind = ev.stream.partition("@")
        if head.startswith("__"):
            d["markers"][head] += 1
            if head in ("__snapshot__", "__snapshot_audit__") and not book.valid:
                d["snapshots"] += 1
                book.on_snapshot(ev.data, ev.ts)
            continue
        if kind in last_kind_ts:
            d["max_silence_ms"][kind] = max(d["max_silence_ms"][kind], ev.ts - last_kind_ts[kind])
        last_kind_ts[kind] = ev.ts
        if kind == "depth@100ms":
            d["diffs"] += 1
            status, _ = book.on_diff(ev.data, ev.ts)
            if status == GAP:
                d["gaps"][next(reversed(book.stats.gap_reasons))] += 1
            if book.valid and book.last_u in pending_d20:
                _cmp(book, pending_d20.pop(book.last_u), d)
        elif kind.startswith("depth20") and isinstance(ev.data, dict) and "u" in ev.data:
            pending_d20[ev.data["u"]] = ev.data
            if len(pending_d20) > 200:
                pending_d20.pop(next(iter(pending_d20)))
            if book.valid and book.last_u == ev.data["u"]:
                _cmp(book, pending_d20.pop(ev.data["u"]), d)
    out = []
    for day, d in sorted(per.items()):
        span = (d["span"][1] - d["span"][0]) if d["span"][0] is not None else 0
        chk = d["d20_ok"] + d["d20_bad"]
        out.append({"symbol": symbol, "day": day, "hours": round(span / 3.6e6, 2), "diffs": d["diffs"],
                    "gaps": dict(d["gaps"]), "valid_share": round(d["valid_ms"] / span, 4) if span else 0.0,
                    "d20_match": round(d["d20_ok"] / chk, 4) if chk else None, "snapshots_used": d["snapshots"],
                    "max_silence_s": {k: round(v / 1000, 1) for k, v in d["max_silence_ms"].items()},
                    "markers": dict(d["markers"])})
    return out


def _cmp(book: L2Book, d20: dict, d: dict) -> None:
    b, a = book.top(20)
    pb = [(float(p), float(q)) for p, q in d20.get("b", [])][:20]
    pa = [(float(p), float(q)) for p, q in d20.get("a", [])][:20]
    n = min(len(pb), len(b), len(pa), len(a))
    if pb[:n] == b[:n] and pa[:n] == a[:n]:
        d["d20_ok"] += 1
    else:
        d["d20_bad"] += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    m = manifest(a.store)
    L = [f"# L2 recording quality — {a.store}\n", f"Symbols: {', '.join(m['symbols'])}  \n"
         f"Sessions: {len(m.get('sessions', []))}\n"]
    hs = health_summary(a.store)
    if len(hs):
        hs["valid_share"] = (hs["valid_minutes"] / hs["minutes"]).round(4)
        hs["usable"] = (hs["valid_share"] >= USABLE_VALID_SHARE) & (hs["dropped"] == 0) & (hs["resync_fail"] == 0)
        L += ["## Recorder health (per day, per symbol)\n", hs.to_string(index=False), ""]
        usable = hs[hs["usable"]].groupby("day")["symbol"].apply(list)
        L += ["## Usable days\n", usable.to_string() if len(usable) else "_none yet_", ""]
    deep = []
    if a.deep:
        for s in store_symbols(a.store):
            deep += deep_check(a.store, s)
        L += ["## Offline continuity re-validation\n", pd.DataFrame(deep).to_string(index=False), ""]
    text = "\n".join(L) + "\n"
    out = a.out or os.path.join(a.store, "QA.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump({"health": hs.to_dict("records") if len(hs) else [], "deep": deep}, fh, indent=1, default=str)
    print(text)


if __name__ == "__main__":
    main()
