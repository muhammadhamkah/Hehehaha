"""V3 forward labels on EXECUTABLE prices (no look-ahead into features).

Entry is at the quote prevailing ``latency_ms`` after the decision: LONG buys the ask,
SHORT sells the bid. Barriers are then measured on the exit side (LONG exits on the bid,
SHORT on the ask), so every touch is a price the trade could actually have been closed at.

The quote path is conflated into 100 ms buckets holding the max/min bid and ask, so a
touch is detected exactly, with first-touch time resolved to 100 ms.

Columns (times in ms after entry; INF = not touched within the horizon):
  t_up_{X} / t_dn_{X}       first touch of the +X (long) / -X (short) barrier, X in XS,
                            within HORIZON_MS. "which direction reaches X first" =
                            t_up < t_dn; "opposite touched first" follows directly.
  tp_{side}_{T}_{S}, sl_{side}_{T}_{S}
                            first touch of TP T / SL S for each barrier pair, within
                            PAIR_HORIZON_MS
  hret_{side}               executable return at PAIR_HORIZON_MS (time-stop exit)
  mfe_{side} / mae_{side}   max favourable / adverse excursion within HORIZON_MS
  mfe120_{side}/mae120_..   same within PAIR_HORIZON_MS
  fret_{h}s                 mid return from the DECISION mid at h seconds
  quote_gap_ms              longest gap in the quote path inside the pair horizon
                            (large -> the label crosses a disconnect; rows are dropped)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

XS = (10, 15, 20, 25, 30)
HORIZONS_S = (5, 10, 30, 60)
PAIRS = ((20, 12), (25, 15), (30, 15), (30, 20), (40, 20))
HORIZON_MS = 60_000
PAIR_HORIZON_MS = 120_000
FRET_S = (1, 5, 10, 30, 60)
BUCKET_MS = 100
INF = np.iinfo(np.int64).max
LABEL_CONFIG = {"version": "v3-labels-1", "xs_bps": list(XS), "horizons_s": list(HORIZONS_S),
                "pairs_tp_sl_bps": [list(p) for p in PAIRS], "horizon_ms": HORIZON_MS,
                "pair_horizon_ms": PAIR_HORIZON_MS, "fret_s": list(FRET_S), "bucket_ms": BUCKET_MS,
                "prices": "executable (long buys ask / exits bid; short sells bid / exits ask) after entry latency"}


def bucketize(q_ts: np.ndarray, q_bid: np.ndarray, q_ask: np.ndarray, t0: int, t1: int):
    """Per-100 ms bucket max/min of bid and ask, carried forward over empty buckets."""
    n = int((t1 - t0) // BUCKET_MS) + 1
    idx = ((q_ts - t0) // BUCKET_MS).astype(np.int64)
    ok = (idx >= 0) & (idx < n)
    idx, qb, qa = idx[ok], q_bid[ok], q_ask[ok]
    mxb = np.full(n, -np.inf)
    mnb = np.full(n, np.inf)
    mxa = np.full(n, -np.inf)
    mna = np.full(n, np.inf)
    np.maximum.at(mxb, idx, qb)
    np.minimum.at(mnb, idx, qb)
    np.maximum.at(mxa, idx, qa)
    np.minimum.at(mna, idx, qa)
    cnt = np.bincount(idx, minlength=n)
    last_b = np.full(n, np.nan)
    last_a = np.full(n, np.nan)
    order = np.argsort(idx, kind="stable")
    last_b[idx[order]] = qb[order]          # later quotes overwrite earlier ones in a bucket
    last_a[idx[order]] = qa[order]
    lb = pd.Series(last_b).ffill().to_numpy()
    la = pd.Series(last_a).ffill().to_numpy()
    empty = cnt == 0
    mxb[empty], mnb[empty], mxa[empty], mna[empty] = lb[empty], lb[empty], la[empty], la[empty]
    return {"t0": t0, "mxb": mxb, "mnb": mnb, "mxa": mxa, "mna": mna, "lb": lb, "la": la, "cnt": cnt}


def _first(cond: np.ndarray) -> int:
    k = int(np.argmax(cond)) if cond.size else 0
    return k if cond.size and cond[k] else -1


def labels(t: np.ndarray, mid0: np.ndarray, q_ts: np.ndarray, q_bid: np.ndarray, q_ask: np.ndarray,
           latency_ms: int = 100) -> pd.DataFrame:
    n = len(t)
    out: dict[str, np.ndarray] = {}
    for X in XS:
        out[f"t_up_{X}"] = np.full(n, INF, np.int64)
        out[f"t_dn_{X}"] = np.full(n, INF, np.int64)
    for side in ("long", "short"):
        for T, S in PAIRS:
            out[f"tp_{side}_{T}_{S}"] = np.full(n, INF, np.int64)
            out[f"sl_{side}_{T}_{S}"] = np.full(n, INF, np.int64)
        for c in ("hret", "mfe", "mae", "mfe120", "mae120"):
            out[f"{c}_{side}"] = np.full(n, np.nan)
    for h in FRET_S:
        out[f"fret_{h}s"] = np.full(n, np.nan)
    out["entry_ask"] = np.full(n, np.nan)
    out["entry_bid"] = np.full(n, np.nan)
    out["quote_gap_ms"] = np.full(n, np.nan)
    if n == 0 or len(q_ts) == 0:
        return pd.DataFrame(out)
    B = bucketize(q_ts, q_bid, q_ask, int(q_ts[0]), int(q_ts[-1]))
    t0 = B["t0"]
    nb_h = HORIZON_MS // BUCKET_MS
    nb_p = PAIR_HORIZON_MS // BUCKET_MS
    tp_lv = np.asarray(XS, float) / 1e4
    for i in range(n):
        te = int(t[i]) + latency_ms
        e = int(np.searchsorted(q_ts, te, side="right")) - 1
        if e < 0 or q_ts[e] < te - 5000:
            continue
        ask0, bid0 = float(q_ask[e]), float(q_bid[e])
        out["entry_ask"][i], out["entry_bid"][i] = ask0, bid0
        b0 = (te - t0) // BUCKET_MS + 1                   # first full bucket after entry
        bp = slice(b0, min(b0 + nb_p, len(B["mxb"])))
        if bp.stop - bp.start < nb_p:                    # path does not cover the horizon
            continue
        seg = q_ts[e:int(np.searchsorted(q_ts, te + PAIR_HORIZON_MS, side="right"))]
        out["quote_gap_ms"][i] = float(np.max(np.diff(seg))) if len(seg) > 1 else float(PAIR_HORIZON_MS)
        mxb, mnb, mxa, mna = B["mxb"][bp], B["mnb"][bp], B["mxa"][bp], B["mna"][bp]
        rel = (np.arange(len(mxb)) + 1) * BUCKET_MS       # bucket end time after entry (conservative)
        cmxb = np.maximum.accumulate(mxb)
        cmna = np.minimum.accumulate(mna)
        cmnb = np.minimum.accumulate(mnb)
        cmxa = np.maximum.accumulate(mxa)
        hz = nb_h
        for j, X in enumerate(XS):
            k = int(np.searchsorted(cmxb[:hz], ask0 * (1 + tp_lv[j]), side="left"))
            if k < hz:
                out[f"t_up_{X}"][i] = rel[k]
            k = int(np.searchsorted(-cmna[:hz], -bid0 * (1 - tp_lv[j]), side="left"))
            if k < hz:
                out[f"t_dn_{X}"][i] = rel[k]
        for T, S in PAIRS:
            k = int(np.searchsorted(cmxb, ask0 * (1 + T / 1e4), side="left"))
            out[f"tp_long_{T}_{S}"][i] = rel[k] if k < len(rel) else INF
            k = int(np.searchsorted(-cmnb, -ask0 * (1 - S / 1e4), side="left"))
            out[f"sl_long_{T}_{S}"][i] = rel[k] if k < len(rel) else INF
            k = int(np.searchsorted(-cmna, -bid0 * (1 - T / 1e4), side="left"))
            out[f"tp_short_{T}_{S}"][i] = rel[k] if k < len(rel) else INF
            k = int(np.searchsorted(cmxa, bid0 * (1 + S / 1e4), side="left"))
            out[f"sl_short_{T}_{S}"][i] = rel[k] if k < len(rel) else INF
        lb_end, la_end = B["lb"][bp.stop - 1], B["la"][bp.stop - 1]
        out["hret_long"][i] = (lb_end - ask0) / ask0 * 1e4
        out["hret_short"][i] = (bid0 - la_end) / bid0 * 1e4
        out["mfe_long"][i] = (cmxb[hz - 1] - ask0) / ask0 * 1e4
        out["mae_long"][i] = (cmnb[hz - 1] - ask0) / ask0 * 1e4
        out["mfe_short"][i] = (bid0 - cmna[hz - 1]) / bid0 * 1e4
        out["mae_short"][i] = (bid0 - cmxa[hz - 1]) / bid0 * 1e4
        out["mfe120_long"][i] = (cmxb[-1] - ask0) / ask0 * 1e4
        out["mae120_long"][i] = (cmnb[-1] - ask0) / ask0 * 1e4
        out["mfe120_short"][i] = (bid0 - cmna[-1]) / bid0 * 1e4
        out["mae120_short"][i] = (bid0 - cmxa[-1]) / bid0 * 1e4
        bd = (int(t[i]) - t0) // BUCKET_MS
        for h in FRET_S:
            kk = bd + h * 1000 // BUCKET_MS
            if kk < len(B["lb"]):
                out[f"fret_{h}s"][i] = (0.5 * (B["lb"][kk] + B["la"][kk]) - mid0[i]) / mid0[i] * 1e4
    return pd.DataFrame(out)


# ---------------------------------------------------------------------- derived targets
def large(df: pd.DataFrame, X: int, H: int) -> np.ndarray:
    return (np.minimum(df[f"t_up_{X}"].to_numpy(), df[f"t_dn_{X}"].to_numpy()) <= H * 1000).astype(np.int8)


def up_first(df: pd.DataFrame, X: int) -> np.ndarray:
    return (df[f"t_up_{X}"].to_numpy() < df[f"t_dn_{X}"].to_numpy()).astype(np.int8)


def tie(df: pd.DataFrame, X: int) -> np.ndarray:
    a, b = df[f"t_up_{X}"].to_numpy(), df[f"t_dn_{X}"].to_numpy()
    return (a == b) & (a != INF)


def pair_outcome(df: pd.DataFrame, side: str, T: int, S: int) -> tuple[np.ndarray, np.ndarray]:
    """(tp_first 0/1, gross bps): TP -> +T, SL -> -S (ties -> stop), else time-stop return."""
    tp = df[f"tp_{side}_{T}_{S}"].to_numpy()
    sl = df[f"sl_{side}_{T}_{S}"].to_numpy()
    win = (tp < sl) & (tp != INF)
    loss = (~win) & (sl != INF)
    hret = df[f"hret_{side}"].to_numpy(float)
    gross = np.where(win, float(T), np.where(loss, -float(S), np.nan_to_num(hret)))
    return win.astype(np.int8), gross


def hold_ms(df: pd.DataFrame, side: str, T: int, S: int) -> np.ndarray:
    tp = df[f"tp_{side}_{T}_{S}"].to_numpy().astype(float)
    sl = df[f"sl_{side}_{T}_{S}"].to_numpy().astype(float)
    return np.minimum(np.minimum(tp, sl), PAIR_HORIZON_MS)
