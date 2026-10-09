"""Forced (EV gate bypassed) top-1% non-overlapping trades on the SELECTION day -- diagnostic only."""
# Run from the repo root:  python -m research.v2_forced_top reports/v2_twostage/forced_top1.json
import json, sys, os, numpy as np, lightgbm as lgb
from research.v2_dataset import load_dataset
from research.v2_models import matrix, simulate
from strategy.v2 import V2Model

se = load_dataset('data/v2/ds_t15', ['2024-03-28'])
ca = load_dataset('data/v2/ds_t15', ['2024-03-27'])
out = {}

def forced(name, score_ca, score, side_long, T, S=8, frac=0.01):
    thr = np.quantile(score_ca, 1 - frac)          # threshold from the CAL day (no SEL peeking)
    m = score >= thr
    probs = {("long", T): np.where(m & side_long, 1.0, 0.0), ("short", T): np.where(m & ~side_long, 1.0, 0.0)}
    st, tr = simulate(se, probs, S, 150.0, 0.5, 0.10)
    if len(tr):
        lg = np.minimum(tr[f"ttp_long_{T}"], tr[f"ttp_short_{T}"]) <= 60000
        upl = tr[f"ttp_long_{T}"] < tr[f"ttp_short_{T}"]
        st["large_move_rate"] = round(float(lg.mean()), 4)
        st["dir_acc_given_move"] = round(float(((tr.side == "long") == upl)[lg].mean()), 4) if lg.any() else None
    st["sel_rows_flagged"] = int(m.sum())
    out[f"{name}_T{T}"] = st
    print(name, T, st)

for kind in ("lightgbm", "xgboost", "mlp"):
    d = f"models/v2/{kind}"
    if not os.path.exists(f"{d}/spec.json"):
        continue
    mdl = V2Model(d)
    x, xc = matrix(se, mdl.features), matrix(ca, mdl.features)
    for T in (20, 25):
        if ("long", T) not in mdl._fns:
            continue
        fl, fs = mdl._fns[("long", T)][0], mdl._fns[("short", T)][0]
        if kind == "lightgbm":
            bl = lgb.Booster(model_file=f"{d}/long_{T}.txt"); bs = lgb.Booster(model_file=f"{d}/short_{T}.txt")
            pl, ps, plc, psc = bl.predict(x), bs.predict(x), bl.predict(xc), bs.predict(xc)
        else:
            pl = np.array([fl(r[None]) for r in x]); ps = np.array([fs(r[None]) for r in x])
            plc = np.array([fl(r[None]) for r in xc]); psc = np.array([fs(r[None]) for r in xc])
        forced(f"direct_{kind}", np.maximum(plc, psc), np.maximum(pl, ps), pl >= ps, T)

for kind in ("lightgbm", "xgboost"):
    d = f"models/v2/twostage_{kind}"
    spec = json.load(open(f"{d}/spec.json"))
    x, xc = matrix(se, spec["features"]), matrix(ca, spec["features"])
    for T in (20, 25):
        if kind == "lightgbm":
            A = lgb.Booster(model_file=f"{d}/stagea_{T}.txt"); B = lgb.Booster(model_file=f"{d}/stageb_{T}.txt")
            pa, pac, pb = A.predict(x), A.predict(xc), B.predict(x)
        else:
            import xgboost as xgb
            A = xgb.Booster(); A.load_model(f"{d}/stagea_{T}.json"); B = xgb.Booster(); B.load_model(f"{d}/stageb_{T}.json")
            ia = (0, spec["stage_a"][str(T)]["best_iteration"] + 1); ib = (0, spec["stage_b"][str(T)]["best_iteration"] + 1)
            pa, pac, pb = A.inplace_predict(x, iteration_range=ia), A.inplace_predict(xc, iteration_range=ia), B.inplace_predict(x, iteration_range=ib)
        forced(f"twostage_{kind}", pac, pa, pb >= 0.5, T)
json.dump(out, open(sys.argv[1], "w"), indent=1)
