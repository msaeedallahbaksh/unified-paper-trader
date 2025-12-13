"""
Train Stock Edge Model
======================
Generates a labeled dataset from historical pairs-trading simulations and trains a
lightweight logistic model that predicts P(profitable trade) at entry.

This is meant to be run manually (locally) to produce:
  unified_trader/models/stock_edge_model.json

Then the live app can load that file and filter/sized trades accordingly.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import app as ut
from stock_edge_model import FEATURE_NAMES, extract_features, save_model_json


@dataclass
class TradeExample:
    pair: str
    entry_time: str
    direction: str
    entry_z: float
    exit_z: float
    holding_bars: int
    pnl_pct: float
    label: int
    features: np.ndarray


def _simulate_pair_trades(
    stock1: str,
    stock2: str,
    pair_prices: pd.DataFrame,
    entry_th: float,
    exit_th: float,
    stop_z: float,
    max_hold_bars: int,
    cost_bps: float,
    lookback_bars: int = 2000,
) -> List[TradeExample]:
    """
    Simulate trades on a single pair using the same core mechanics as the live bot:
    - Kalman innovation z-score for entry/exit
    - BUY_SPREAD if z < -entry, SELL_SPREAD if z > entry
    - Exit if |z| < exit, stop if |z| > stop_z, or time stop
    """
    pair_prices = pair_prices[[stock1, stock2]].dropna()
    if len(pair_prices) < max(ut.STOCK_MIN_BARS, 200):
        return []

    if lookback_bars and len(pair_prices) > lookback_bars:
        pair_prices = pair_prices.tail(lookback_bars)

    p1 = pair_prices[stock1].astype(float).values
    p2 = pair_prices[stock2].astype(float).values
    idx = pair_prices.index

    kf = ut.KalmanFilter(delta=1e-4, R=1.0)
    z_hist: List[float] = []
    hr_hist: List[float] = []

    for i in range(len(pair_prices)):
        kf.update(float(p1[i]), float(p2[i]))
        z_hist.append(float(kf.get_zscore()))
        hr_hist.append(float(kf.get_hedge_ratio()))

    examples: List[TradeExample] = []

    i = 60  # warmup for features
    while i < len(pair_prices) - 2:
        z = z_hist[i]

        if abs(z) < entry_th:
            i += 1
            continue

        direction = "SELL_SPREAD" if z > 0 else "BUY_SPREAD"

        # Build features from history up to entry bar i
        context = {
            "z_hist": z_hist[: i + 1],
            "p1_hist": p1[: i + 1].tolist(),
            "p2_hist": p2[: i + 1].tolist(),
        }
        feats = extract_features(context, entry_threshold=entry_th)
        if feats is None:
            i += 1
            continue

        # Entry state
        entry_p1 = float(p1[i])
        entry_p2 = float(p2[i])
        entry_hr = float(hr_hist[i])

        # Normalize gross notional to 1.0 (labels are % returns)
        denom = max(entry_p1 + abs(entry_hr) * entry_p2, 1e-8)
        k = 1.0 / denom

        if direction == "BUY_SPREAD":
            qty1 = k
            qty2 = -k * entry_hr
        else:
            qty1 = -k
            qty2 = k * entry_hr

        # Costs on open + close (on gross notional=1.0)
        total_cost = 2.0 * (cost_bps / 10000.0)

        # Walk forward to exit
        exit_idx = None
        exit_reason = None
        for j in range(i + 1, min(i + max_hold_bars, len(pair_prices) - 1)):
            zj = float(z_hist[j])
            if abs(zj) < exit_th:
                exit_idx = j
                exit_reason = "EXIT_Z"
                break
            if abs(zj) > stop_z:
                exit_idx = j
                exit_reason = "STOP_Z"
                break

        if exit_idx is None:
            exit_idx = min(i + max_hold_bars, len(pair_prices) - 1)
            exit_reason = "TIME_STOP"

        exit_p1 = float(p1[exit_idx])
        exit_p2 = float(p2[exit_idx])

        pnl = qty1 * (exit_p1 - entry_p1) + qty2 * (exit_p2 - entry_p2)
        pnl_after_cost = pnl - total_cost
        pnl_pct = pnl_after_cost * 100.0  # since notional=1.0
        label = 1 if pnl_after_cost > 0 else 0

        examples.append(
            TradeExample(
                pair=f"{stock1}-{stock2}",
                entry_time=idx[i].isoformat(),
                direction=direction,
                entry_z=float(z_hist[i]),
                exit_z=float(z_hist[exit_idx]),
                holding_bars=int(exit_idx - i),
                pnl_pct=float(pnl_pct),
                label=int(label),
                features=feats,
            )
        )

        # jump forward past this trade (no overlap)
        i = exit_idx + 1

    return examples


def build_dataset(
    pairs: List[Dict],
    period: str,
    interval: Optional[str],
    entry_th: float,
    exit_th: float,
    stop_z: float,
    max_hold_bars: int,
    cost_bps: float,
    max_pairs: int,
) -> Tuple[np.ndarray, np.ndarray, List[TradeExample]]:
    pairs = pairs[:max_pairs] if max_pairs else pairs

    # Download data once for all tickers
    tickers = sorted({p["stock1"] for p in pairs} | {p["stock2"] for p in pairs})
    prices = ut.get_stock_data(tickers, period=period, interval=interval)
    if prices is None or prices.empty:
        raise RuntimeError("Failed to download stock data for training")

    examples: List[TradeExample] = []
    for p in pairs:
        s1 = p["stock1"]
        s2 = p["stock2"]
        pair_prices = prices[[s1, s2]].dropna()
        if pair_prices.empty:
            continue
        examples.extend(
            _simulate_pair_trades(
                s1,
                s2,
                pair_prices,
                entry_th=entry_th,
                exit_th=exit_th,
                stop_z=stop_z,
                max_hold_bars=max_hold_bars,
                cost_bps=cost_bps,
            )
        )

    if not examples:
        raise RuntimeError("No training examples generated (try lowering entry threshold or using more pairs)")

    X = np.vstack([e.features for e in examples])
    y = np.array([e.label for e in examples], dtype=int)
    return X, y, examples


def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[StandardScaler, LogisticRegression, Dict[str, float]]:
    # Time-ish split (dataset is already chronological per pair simulation)
    split = int(len(y) * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)

    metrics: Dict[str, float] = {}
    if len(np.unique(y_test)) >= 2:
        proba = clf.predict_proba(X_test_s)[:, 1]
        metrics["auc"] = float(roc_auc_score(y_test, proba))
    metrics["train_samples"] = float(len(y_train))
    metrics["test_samples"] = float(len(y_test))
    metrics["pos_rate"] = float(np.mean(y))
    return scaler, clf, metrics


def main():
    parser = argparse.ArgumentParser(description="Train stock edge model from historical intraday simulations")
    parser.add_argument("--period", default=ut.STOCK_DATA_PERIOD, help="yfinance period (e.g. 60d, 1y)")
    parser.add_argument("--interval", default=ut.STOCK_DATA_INTERVAL or "", help="yfinance interval (e.g. 15m, 1h, 1d)")
    parser.add_argument("--entry", type=float, default=ut.STOCK_ZSCORE_ENTRY, help="entry threshold")
    parser.add_argument("--exit", type=float, default=ut.STOCK_ZSCORE_EXIT, help="exit threshold")
    parser.add_argument("--stop-z", type=float, default=ut.STOCK_STOP_ZSCORE, help="z-score stop")
    parser.add_argument("--max-hold-bars", type=int, default=200, help="max holding bars (time stop)")
    parser.add_argument("--cost-bps", type=float, default=ut.TRADING_COST_BPS, help="cost bps per open/close (gross)")
    parser.add_argument("--max-pairs", type=int, default=30, help="max pairs to use for dataset")
    parser.add_argument("--out", default=str(ut.BASE_DIR / "models" / "stock_edge_model.json"), help="output model json path")
    args = parser.parse_args()

    interval = args.interval.strip() or None

    print("=" * 70)
    print("🧠 TRAIN STOCK EDGE MODEL")
    print("=" * 70)
    print(f"Pairs available: {len(ut.STOCK_PAIRS)}")
    print(f"Using up to: {args.max_pairs} pairs")
    print(f"Data: period={args.period}, interval={interval}")
    print(f"Rules: entry={args.entry}, exit={args.exit}, stop_z={args.stop_z}, max_hold_bars={args.max_hold_bars}, cost_bps={args.cost_bps}")
    print(f"Output: {args.out}")

    X, y, examples = build_dataset(
        pairs=ut.STOCK_PAIRS,
        period=args.period,
        interval=interval,
        entry_th=args.entry,
        exit_th=args.exit,
        stop_z=args.stop_z,
        max_hold_bars=args.max_hold_bars,
        cost_bps=args.cost_bps,
        max_pairs=args.max_pairs,
    )

    print(f"\n✅ Dataset: {len(y)} trades | win-rate={np.mean(y)*100:.1f}%")

    scaler, clf, metrics = train_model(X, y)
    print("\n📊 Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    save_model_json(
        path=out_path,
        feature_names=FEATURE_NAMES,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        coef=clf.coef_[0],
        intercept=float(clf.intercept_[0]),
        metrics=metrics,
        training={
            "period": args.period,
            "interval": interval,
            "entry": args.entry,
            "exit": args.exit,
            "stop_z": args.stop_z,
            "max_hold_bars": args.max_hold_bars,
            "cost_bps": args.cost_bps,
            "max_pairs": args.max_pairs,
            "feature_names": FEATURE_NAMES,
        },
    )

    print(f"\n✅ Saved model to: {out_path}")


if __name__ == "__main__":
    main()

