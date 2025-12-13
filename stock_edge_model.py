"""
Stock Edge Model (Lightweight)
=============================
This is a small, deployable trade-filter model for stock pairs trading.

Goal:
- Given recent z-score + price action context at a *potential entry*, estimate
  P(profitable trade) so we can filter bad mean-reversion signals and size up good ones.

Design constraints:
- Must be fast and lightweight (JSON-serialized coefficients)
- No heavy ML deps required at inference time
- Works with intraday or daily bars (timestamps are opaque to the model)

IMPORTANT:
- This cannot guarantee profitability. It reduces "dumb trades" by learning patterns
  from historical outcomes, but markets change and overfitting is always a risk.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


FEATURE_NAMES: List[str] = [
    "z",
    "abs_z",
    "z_mom_1",
    "z_mom_4",
    "z_mean_10",
    "z_std_10",
    "abs_z_mean_20",
    "time_in_signal",
    "ret1_1",
    "ret2_1",
    "ret_spread_1",
    "corr_50",
    "vol1_50",
    "vol2_50",
    "vol_spread_50",
]


def _sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_log_return(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return np.array([], dtype=float)
    prev = np.maximum(prices[:-1], 1e-12)
    curr = np.maximum(prices[1:], 1e-12)
    return np.log(curr / prev)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(context: Dict[str, Any], entry_threshold: float) -> Optional[np.ndarray]:
    """
    Build a feature vector from recent history.

    Expected context:
      - z_hist: list[float]
      - p1_hist: list[float]
      - p2_hist: list[float]
    """
    try:
        z = np.asarray(context.get("z_hist", []), dtype=float)
        p1 = np.asarray(context.get("p1_hist", []), dtype=float)
        p2 = np.asarray(context.get("p2_hist", []), dtype=float)

        n = min(len(z), len(p1), len(p2))
        if n < 60:
            return None

        z = z[-n:]
        p1 = p1[-n:]
        p2 = p2[-n:]

        z_now = float(z[-1])
        abs_z = abs(z_now)

        z_mom_1 = float(z[-1] - z[-2]) if n >= 2 else 0.0
        z_mom_4 = float(z[-1] - z[-5]) if n >= 5 else 0.0

        z_win10 = z[-10:] if n >= 10 else z
        z_mean_10 = float(np.mean(z_win10))
        z_std_10 = float(np.std(z_win10))

        z_win20 = z[-20:] if n >= 20 else z
        abs_z_mean_20 = float(np.mean(np.abs(z_win20)))

        # time-in-signal: consecutive bars beyond entry threshold
        tis = 0
        for v in z[::-1]:
            if abs(float(v)) >= float(entry_threshold):
                tis += 1
            else:
                break
        tis = float(min(tis, 200))

        r1 = _safe_log_return(p1)
        r2 = _safe_log_return(p2)
        if len(r1) < 55 or len(r2) < 55:
            return None

        ret1_1 = float(r1[-1])
        ret2_1 = float(r2[-1])
        ret_spread_1 = float(ret1_1 - ret2_1)

        corr_50 = _safe_corr(r1[-50:], r2[-50:])
        vol1_50 = float(np.std(r1[-50:]))
        vol2_50 = float(np.std(r2[-50:]))
        vol_spread_50 = float(np.std((r1[-50:] - r2[-50:])))

        feats = np.array(
            [
                z_now,
                abs_z,
                z_mom_1,
                z_mom_4,
                z_mean_10,
                z_std_10,
                abs_z_mean_20,
                tis,
                ret1_1,
                ret2_1,
                ret_spread_1,
                corr_50,
                vol1_50,
                vol2_50,
                vol_spread_50,
            ],
            dtype=float,
        )

        if not np.isfinite(feats).all():
            return None
        return feats
    except Exception:
        return None


@dataclass
class StockEdgeModel:
    feature_names: List[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "StockEdgeModel":
        feature_names = list(obj["feature_names"])
        return cls(
            feature_names=feature_names,
            scaler_mean=np.asarray(obj["scaler_mean"], dtype=float),
            scaler_scale=np.asarray(obj["scaler_scale"], dtype=float),
            coef=np.asarray(obj["coef"], dtype=float),
            intercept=float(obj["intercept"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StockEdgeModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(data)

    def predict_proba(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=float)
        if x.shape[0] != self.coef.shape[0]:
            raise ValueError(f"Feature length mismatch: got {x.shape[0]}, expected {self.coef.shape[0]}")

        scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
        xz = (x - self.scaler_mean) / scale

        logit = float(self.intercept + float(np.dot(self.coef, xz)))
        return _sigmoid(logit)


def save_model_json(
    path: str | Path,
    feature_names: List[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    metrics: Optional[Dict[str, Any]] = None,
    training: Optional[Dict[str, Any]] = None,
) -> None:
    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "feature_names": list(feature_names),
        "scaler_mean": [float(x) for x in np.asarray(scaler_mean, dtype=float).tolist()],
        "scaler_scale": [float(x) for x in np.asarray(scaler_scale, dtype=float).tolist()],
        "coef": [float(x) for x in np.asarray(coef, dtype=float).tolist()],
        "intercept": float(intercept),
        "metrics": metrics or {},
        "training": training or {},
    }
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")

