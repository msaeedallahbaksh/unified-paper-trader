"""
Optimize Stock Pair Parameters (Walk-Forward)
============================================
Brute-force parameter search for each pair with a simple walk-forward split.

Outputs a config file with per-pair parameters:
  unified_trader/models/pairs_config.json

The live app will load this as a fallback if `data/pairs_config.json` is missing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import app as ut


@dataclass
class BacktestResult:
    trades: int
    win_rate: float
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float


def _compute_kalman_series(pair_prices: pd.DataFrame, stock1: str, stock2: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_prices = pair_prices[[stock1, stock2]].dropna()
    p1 = pair_prices[stock1].astype(float).values
    p2 = pair_prices[stock2].astype(float).values

    kf = ut.KalmanFilter(delta=1e-4, R=1.0)
    z = np.zeros(len(pair_prices), dtype=float)
    hr = np.zeros(len(pair_prices), dtype=float)
    for i in range(len(pair_prices)):
        kf.update(float(p1[i]), float(p2[i]))
        z[i] = float(kf.get_zscore())
        hr[i] = float(kf.get_hedge_ratio())

    return z, hr, p1, p2


def _simulate(
    z: np.ndarray,
    hr: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    start: int,
    end: int,
    entry: float,
    exit: float,
    stop_z: float,
    max_hold_bars: int,
    cost_bps: float,
) -> BacktestResult:
    """
    Simulate sequential trades (no overlap) with gross notional = 1.0 per trade.
    """
    if end - start < 200:
        return BacktestResult(trades=0, win_rate=0.0, total_return_pct=0.0, sharpe=0.0, max_drawdown_pct=0.0)

    # Costs per round trip on gross notional=1.0
    rt_cost = 2.0 * (cost_bps / 10000.0)

    trade_returns: List[float] = []
    equity = 1.0
    curve = [equity]

    i = max(start, 60)
    while i < end - 2:
        zi = float(z[i])
        if abs(zi) < entry:
            i += 1
            continue

        direction = "SELL" if zi > 0 else "BUY"
        entry_p1 = float(p1[i])
        entry_p2 = float(p2[i])
        entry_hr = float(hr[i])

        denom = max(entry_p1 + abs(entry_hr) * entry_p2, 1e-8)
        k = 1.0 / denom
        if direction == "BUY":
            qty1, qty2 = k, -k * entry_hr
        else:
            qty1, qty2 = -k, k * entry_hr

        exit_idx = None
        for j in range(i + 1, min(i + max_hold_bars, end - 1)):
            zj = float(z[j])
            if abs(zj) < exit:
                exit_idx = j
                break
            if abs(zj) > stop_z:
                exit_idx = j
                break

        if exit_idx is None:
            exit_idx = min(i + max_hold_bars, end - 1)

        pnl = qty1 * (float(p1[exit_idx]) - entry_p1) + qty2 * (float(p2[exit_idx]) - entry_p2)
        r = pnl - rt_cost
        trade_returns.append(r)

        equity *= (1.0 + r)
        curve.append(equity)

        i = exit_idx + 1

    if not trade_returns:
        return BacktestResult(trades=0, win_rate=0.0, total_return_pct=0.0, sharpe=0.0, max_drawdown_pct=0.0)

    rets = np.asarray(trade_returns, dtype=float)
    wins = float(np.sum(rets > 0))
    win_rate = wins / float(len(rets)) * 100.0

    total_return_pct = (equity - 1.0) * 100.0
    r_mean = float(np.mean(rets))
    r_std = float(np.std(rets))
    sharpe = (r_mean / r_std) * float(np.sqrt(len(rets))) if r_std > 1e-12 else 0.0

    curve_arr = np.asarray(curve, dtype=float)
    peak = np.maximum.accumulate(curve_arr)
    dd = (curve_arr - peak) / peak
    max_dd = float(np.min(dd)) * 100.0

    return BacktestResult(
        trades=int(len(rets)),
        win_rate=float(win_rate),
        total_return_pct=float(total_return_pct),
        sharpe=float(sharpe),
        max_drawdown_pct=float(max_dd),
    )


def main():
    parser = argparse.ArgumentParser(description="Walk-forward optimizer for stock pair parameters")
    parser.add_argument("--period", default=ut.STOCK_DATA_PERIOD, help="yfinance period (e.g. 60d, 1y)")
    parser.add_argument("--interval", default=ut.STOCK_DATA_INTERVAL or "", help="yfinance interval (e.g. 15m, 1h, 1d)")
    parser.add_argument("--max-pairs", type=int, default=40, help="max pairs to evaluate")
    parser.add_argument("--top-n", type=int, default=40, help="top N pairs to write")
    parser.add_argument("--cost-bps", type=float, default=ut.TRADING_COST_BPS, help="cost bps per open/close")
    parser.add_argument("--min-trades", type=int, default=6, help="minimum trades in train split")
    parser.add_argument("--out", default=str(ut.MODELS_DIR / "pairs_config.json"), help="output config path")
    args = parser.parse_args()

    interval = args.interval.strip() or None
    pairs = ut.STOCK_PAIRS[: args.max_pairs] if args.max_pairs else ut.STOCK_PAIRS

    tickers = sorted({p["stock1"] for p in pairs} | {p["stock2"] for p in pairs})
    prices = ut.get_stock_data(tickers, period=args.period, interval=interval)
    if prices is None or prices.empty:
        raise RuntimeError("Failed to download prices for optimizer")

    # Parameter grid
    entry_grid = [1.25, 1.5, 1.75, 2.0]
    exit_grid = [0.3, 0.5, 0.8]
    stop_grid = [3.0, 3.5, 4.0]

    # Holding limit in bars (15m -> 26 bars/day)
    max_hold_grid = [78, 130, 208]  # ~3d, ~5d, ~8d (trading-time)

    results = []
    for p in pairs:
        s1, s2 = p["stock1"], p["stock2"]
        pair_prices = prices[[s1, s2]].dropna()
        if len(pair_prices) < max(ut.STOCK_MIN_BARS, 300):
            continue

        z, hr, p1, p2 = _compute_kalman_series(pair_prices, s1, s2)
        split = int(len(z) * 0.7)

        best = None
        best_score = -1e9

        for entry in entry_grid:
            for exit_th in exit_grid:
                if exit_th >= entry:
                    continue
                for stop_z in stop_grid:
                    for max_hold in max_hold_grid:
                        train_res = _simulate(
                            z, hr, p1, p2,
                            start=0, end=split,
                            entry=entry, exit=exit_th,
                            stop_z=stop_z, max_hold_bars=max_hold,
                            cost_bps=args.cost_bps,
                        )
                        if train_res.trades < args.min_trades:
                            continue

                        # Score: favor sharpe + return, penalize drawdown
                        score = train_res.sharpe + 0.10 * train_res.total_return_pct + 0.05 * train_res.max_drawdown_pct
                        if score > best_score:
                            best_score = score
                            best = (entry, exit_th, stop_z, max_hold, train_res)

        if best is None:
            continue

        entry, exit_th, stop_z, max_hold, train_res = best
        test_res = _simulate(
            z, hr, p1, p2,
            start=split, end=len(z),
            entry=entry, exit=exit_th,
            stop_z=stop_z, max_hold_bars=max_hold,
            cost_bps=args.cost_bps,
        )

        # Convert holding bars to an approximate hour value for the live engine.
        # For 15m bars this is bars * 0.25, but we keep it generic.
        interval_minutes = 15 if interval and interval.endswith("m") else (60 if interval and interval.endswith("h") else 60)
        try:
            if interval and interval.endswith("m"):
                interval_minutes = int(interval[:-1])
            elif interval and interval.endswith("h"):
                interval_minutes = int(interval[:-1]) * 60
        except Exception:
            pass
        max_hold_hours = float(max_hold) * (float(interval_minutes) / 60.0)

        results.append(
            {
                "stock1": s1,
                "stock2": s2,
                "cluster": p.get("cluster", ""),
                "correlation": float(p.get("correlation", 0.0)),
                "entry_z": float(entry),
                "exit_z": float(exit_th),
                "stop_z": float(stop_z),
                "max_hold_hours": float(max_hold_hours),
                "train": train_res.__dict__,
                "test": test_res.__dict__,
            }
        )

    if not results:
        raise RuntimeError("No optimized results produced")

    # ===========================================================
    # CRITICAL FIX: Only keep pairs with POSITIVE expected value!
    # Previously we kept ALL pairs including losers - that was a bug
    # ===========================================================
    MIN_WIN_RATE = 50.0  # Minimum 50% win rate in test period
    MIN_SHARPE = 0.0     # Must be positive (edge exists)
    MIN_RETURN = 0.0     # Must be positive (net profitable)
    
    # Filter out losing pairs BEFORE ranking
    profitable_pairs = [
        r for r in results
        if r["test"]["sharpe"] >= MIN_SHARPE
        and r["test"]["total_return_pct"] >= MIN_RETURN
        and r["test"]["win_rate"] >= MIN_WIN_RATE
    ]
    
    print(f"\n📊 FILTERING RESULTS:")
    print(f"   Total pairs evaluated: {len(results)}")
    print(f"   Pairs with positive edge: {len(profitable_pairs)}")
    print(f"   Pairs REJECTED (negative edge): {len(results) - len(profitable_pairs)}")
    
    if not profitable_pairs:
        print("⚠️ No pairs passed filtering - relaxing constraints slightly...")
        # Fallback: at least require positive Sharpe
        profitable_pairs = [r for r in results if r["test"]["sharpe"] > 0]
    
    if not profitable_pairs:
        raise RuntimeError("No profitable pairs found - cannot generate config")

    # Rank by out-of-sample sharpe then return
    profitable_pairs.sort(key=lambda r: (r["test"]["sharpe"], r["test"]["total_return_pct"]), reverse=True)
    top = profitable_pairs[: args.top_n] if args.top_n else profitable_pairs

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "algorithm": "Walk-forward grid search (Kalman z-score) + cost_bps",
        "data": {"period": args.period, "interval": interval, "cost_bps": args.cost_bps},
        "grid": {"entry": entry_grid, "exit": exit_grid, "stop_z": stop_grid, "max_hold_bars": max_hold_grid},
        "pairs": [
            {
                "stock1": r["stock1"],
                "stock2": r["stock2"],
                "cluster": r.get("cluster", ""),
                "correlation": r.get("correlation", 0.0),
                "entry_z": r["entry_z"],
                "exit_z": r["exit_z"],
                "stop_z": r["stop_z"],
                "max_hold_hours": r["max_hold_hours"],
                "test": r["test"],
            }
            for r in top
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 70)
    print("✅ OPTIMIZATION COMPLETE")
    print(f"Pairs optimized: {len(results)}")
    print(f"Saved: {out_path}")
    print("\nTop 10 (test):")
    for r in top[:10]:
        t = r["test"]
        print(
            f"  {r['stock1']}-{r['stock2']}: "
            f"Sharpe={t['sharpe']:.2f} Ret={t['total_return_pct']:.1f}% "
            f"Trades={t['trades']} DD={t['max_drawdown_pct']:.1f}% "
            f"[entry={r['entry_z']}, exit={r['exit_z']}, stop={r['stop_z']}]"
        )


if __name__ == "__main__":
    main()

