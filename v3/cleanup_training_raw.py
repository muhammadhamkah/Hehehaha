"""Delete raw L2 recordings of TRAINING days whose dataset is fully built and verified.

NEVER runs automatically. Default is a dry run that prints exactly which files would go:

    python -m v3.cleanup_training_raw --store /data/l2 --dataset data/v3/ds --split split.json
    python -m v3.cleanup_training_raw ... --delete --confirm DELETE-TRAINING-RAW

``split.json`` (written BEFORE looking at results):
    {"train": [...], "calibration": [...], "selection": [...], "test": [...]}
(``validation`` is accepted as well). Validation (calibration/selection) and test raw data are
never deleted -- they are needed for the execution replay -- nor is the last hour (23 UTC) of
a day that directly precedes a protected day (warm-up data for that day's replay/dataset).

A training day is deleted only if, for EVERY symbol with raw data that day:
  1  the day is in "train" and in no protected list (the split lists must not overlap)
  2  dataset_manifest.json has the (symbol, day) shard and its parquet exists
  3  the parquet's SHA-256 equals the manifest's checksum
  4  rows > 0, the parquet row count equals the manifest, and rows >= --min-row-fraction of the
     expected rows for the build window/sampling
  5  feature schema version + column hash and the label configuration are recorded
  6  QA passed: ``python -m v3.qa --deep`` reported the book valid >= 95% of that day
     (and the recorder health log, if present, marked the day usable)
Otherwise the whole day is kept. Every deletion is logged to ``_health/cleanup.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from v3.dataset import sha256_file
from v3.qa import USABLE_VALID_SHARE
from v3.store import store_symbols

CONFIRM = "DELETE-TRAINING-RAW"
PROTECTED_KEYS = ("calibration", "selection", "validation", "test")


def _ymd(day: str) -> str:
    return day.replace("-", "")


def _prev_day(day: str) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(day) - timedelta(days=1)).isoformat()


def load_split(path: str) -> tuple[list[str], set[str]]:
    with open(path, encoding="utf-8") as fh:
        sp = json.load(fh)
    train = list(sp.get("train", []))
    protected = set()
    for k in PROTECTED_KEYS:
        protected |= set(sp.get(k, []))
    if not train or not sp.get("test") or not (sp.get("calibration") or sp.get("selection") or sp.get("validation")):
        raise SystemExit("split must list train, validation (calibration/selection) and test days")
    overlap = set(train) & protected
    if overlap:
        raise SystemExit(f"split lists overlap between train and validation/test: {sorted(overlap)}")
    return train, protected


def check_day(store: str, dataset: str, manifest: dict, qa: dict, sym: str, day: str, min_frac: float) -> list[str]:
    """Empty list = all checks passed for (symbol, day); otherwise the reasons."""
    import pyarrow.parquet as pq

    bad = []
    e = manifest.get("shards", {}).get(f"{sym}_{day}")
    if e is None:
        return ["no dataset shard in dataset_manifest.json"]
    path = os.path.join(dataset, e.get("file", ""))
    if not os.path.exists(path):
        return [f"dataset file missing: {path}"]
    if not e.get("sha256"):
        bad.append("no checksum recorded")
    elif sha256_file(path) != e["sha256"]:
        bad.append("checksum mismatch (dataset changed since it was built)")
    try:
        rows = pq.ParquetFile(path).metadata.num_rows
    except Exception as exc:  # noqa: BLE001
        return bad + [f"dataset file unreadable: {exc}"]
    if rows <= 0 or rows != e.get("rows"):
        bad.append(f"row count invalid (file {rows}, manifest {e.get('rows')})")
    exp = e.get("expected_rows") or 0
    if exp and rows < min_frac * exp:
        bad.append(f"too few rows: {rows} < {min_frac:.0%} of expected {exp}")
    for k in ("feature_version", "columns_sha256", "label_config", "sample_ms"):
        if not e.get(k):
            bad.append(f"{k} not recorded")
    deep = [d for d in qa.get("deep", []) if d.get("symbol") == sym and d.get("day") == day]
    if not deep:
        bad.append("no deep QA for this day (run: python -m v3.qa --store ... --deep)")
    elif (deep[0].get("valid_share") or 0) < USABLE_VALID_SHARE:
        bad.append(f"QA failed: book valid {deep[0].get('valid_share')} < {USABLE_VALID_SHARE}")
    health = [h for h in qa.get("health", []) if h.get("symbol") == sym and h.get("day") == day]
    if health and not health[0].get("usable", False):
        bad.append("recorder health marked the day unusable")
    return bad


def plan(store: str, dataset: str, split: str, qa_path: str, days: list[str] | None, min_frac: float) -> dict:
    train, protected = load_split(split)
    targets = [d for d in (days or train) if d in train]
    refused = sorted(set(days or []) - set(train))
    with open(os.path.join(dataset, "dataset_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(qa_path, encoding="utf-8") as fh:
        qa = json.load(fh)
    keep_23 = {_prev_day(p) for p in protected}
    symbols = store_symbols(store)
    out = {"delete": [], "kept_days": {}, "refused_not_training": refused, "protected_days": sorted(protected)}
    for day in sorted(targets):
        syms = [s for s in symbols if os.path.isdir(os.path.join(store, s, _ymd(day)))]
        if not syms:
            continue
        reasons = {s: check_day(store, dataset, manifest, qa, s, day, min_frac) for s in syms}
        failed = {s: r for s, r in reasons.items() if r}
        if failed:
            out["kept_days"][day] = failed
            continue
        for s in syms:
            d = os.path.join(store, s, _ymd(day))
            for f in sorted(os.listdir(d)):
                if day in keep_23 and f.startswith("23."):
                    continue
                p = os.path.join(d, f)
                out["delete"].append({"file": p, "bytes": os.path.getsize(p), "symbol": s, "day": day})
    out["total_gb"] = round(sum(x["bytes"] for x in out["delete"]) / 1e9, 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--qa", help="QA json from `python -m v3.qa --deep` (default <store>/QA.json)")
    ap.add_argument("--days", nargs="+", help="subset of TRAINING days (default: all training days)")
    ap.add_argument("--min-row-fraction", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true", help="only list (this is also the default)")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--confirm", default="")
    a = ap.parse_args()
    p = plan(a.store, a.dataset, a.split, a.qa or os.path.join(a.store, "QA.json"), a.days, a.min_row_fraction)
    for day, why in p["kept_days"].items():
        print(f"KEEP {day}: " + "; ".join(f"{s}: {', '.join(r)}" for s, r in why.items()))
    if p["refused_not_training"]:
        print(f"REFUSED (not training days, never deleted): {', '.join(p['refused_not_training'])}")
    print(f"Protected (validation/test, never deleted): {', '.join(p['protected_days'])}")
    for x in p["delete"]:
        print(f"{'DELETE' if a.delete and not a.dry_run else 'would delete'} {x['file']} ({x['bytes'] / 1e6:.1f} MB)")
    print(f"{len(p['delete'])} files, {p['total_gb']:.3f} GB")
    if not a.delete or a.dry_run:
        print(f"dry run: nothing deleted (to delete: --delete --confirm {CONFIRM})")
        return
    if a.confirm != CONFIRM:
        raise SystemExit(f"refusing to delete without --confirm {CONFIRM}")
    log_path = os.path.join(a.store, "_health", "cleanup.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        for x in p["delete"]:
            os.remove(x["file"])
            fh.write(json.dumps({"ts": int(time.time() * 1000), "deleted": x["file"], "bytes": x["bytes"],
                                 "day": x["day"], "symbol": x["symbol"]}) + "\n")
    for x in {os.path.dirname(x["file"]) for x in p["delete"]}:
        if not os.listdir(x):
            os.rmdir(x)
    print(f"deleted {len(p['delete'])} files ({p['total_gb']:.3f} GB); log: {log_path}")


if __name__ == "__main__":
    main()
