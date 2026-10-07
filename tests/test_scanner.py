from config import ScannerConfig
from market_data.scanner import MarketScanner


def _feed(sc: MarketScanner, sym: str, t0: int, n: int, price: float, qv: float, trades: int,
          vol_step: float, spread: float, price_jitter: float = 0.0):
    for i in range(n):
        ts = t0 + i * 1000
        p = price * (1 + (price_jitter if i % 2 else -price_jitter))
        sc.on_ticker({"s": sym, "c": str(p), "q": str(qv + i * vol_step), "n": trades + i * 10}, ts)
        sc.on_book_ticker({"s": sym, "b": str(p), "B": "10", "a": str(p + spread), "A": "10"}, ts)
    return t0 + (n - 1) * 1000


def test_filters_and_ranking():
    cfg = ScannerConfig(min_quote_volume_24h=1e6, max_spread_bps=5, top_n=2, min_age_for_ranking_s=0)
    sc = MarketScanner(cfg)
    sc.set_universe({"AAA", "BBB", "CCC", "DDD"})
    now = _feed(sc, "AAA", 0, 30, 100, 5e7, 1000, 1e5, 0.01, 0.001)      # liquid, active, volatile
    _feed(sc, "BBB", 0, 30, 100, 5e7, 1000, 1e3, 0.01, 0.0001)           # liquid, quiet
    _feed(sc, "CCC", 0, 30, 100, 1e5, 1000, 1e3, 0.01, 0.001)            # too little volume
    _feed(sc, "DDD", 0, 30, 100, 5e7, 1000, 1e5, 0.2, 0.001)             # spread too wide
    ranked = sc.rank(now)
    syms = [r.symbol for r in ranked]
    assert "CCC" not in syms and "DDD" not in syms
    assert syms[0] == "AAA"
    assert sc.select(now)[:2] == ["AAA", "BBB"]


def test_universe_excludes_unknown_symbols():
    sc = MarketScanner(ScannerConfig())
    sc.set_universe({"AAA"})
    sc.on_ticker({"s": "ZZZ", "c": "1", "q": "1", "n": 1}, 0)
    assert "ZZZ" not in sc.stats


def test_hysteresis_and_pins():
    cfg = ScannerConfig(min_quote_volume_24h=0, max_spread_bps=100, top_n=1, hysteresis=2.0,
                        min_age_for_ranking_s=0)
    sc = MarketScanner(cfg)
    sc.selected = ["BBB"]
    sc.last_ranking = []
    # Fake a ranking where BBB is 2nd: with hysteresis 2.0 it stays selected.
    sc.rank = lambda now: [type("R", (), {"symbol": s})() for s in ("AAA", "BBB", "CCC")]  # type: ignore
    assert sc.select(0) == ["BBB"]
    sc.selected = ["CCC"]           # CCC is 3rd >= keep limit 2 -> replaced by best
    assert sc.select(0) == ["AAA"]
    sc.pin("XYZ", "position")
    sc.pin("XYZ", "labels")
    assert "XYZ" in sc.active_symbols()
    sc.unpin("XYZ", "position")
    assert "XYZ" in sc.active_symbols()
    sc.unpin("XYZ", "labels")
    assert "XYZ" not in sc.active_symbols()
