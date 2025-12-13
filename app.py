"""
Unified Paper Trading Simulator
================================
Combines Crypto (Cointegration) and Stocks (Kalman Filter) trading
in a single dashboard with switchable views.

Now with GATEKEEPER NEURAL NETWORK for trade filtering!
"""

import os
import sys
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.cluster import SpectralClustering

# Windows consoles can default to cp1252 and crash on emoji output.
# Make stdout/stderr UTF-8 (or at least non-crashing) for local runs.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Database Module (Supabase with JSON fallback)
try:
    from database import init_database, get_storage_mode, DATABASE_AVAILABLE
    init_database()
    print(f"📊 Storage mode: {get_storage_mode()}")
except Exception as e:
    print(f"⚠️ Database module not available: {e}")
    DATABASE_AVAILABLE = False

# Gatekeeper Neural Network (LAZY LOADED to reduce memory)
# Only loads when actually checking trades
GATEKEEPER_ENABLED = True
GATEKEEPER_THRESHOLD = 0.70  # V4 model with Focal Loss - balanced threshold
_gatekeeper = None  # Lazy loaded
_gatekeeper_path = Path(__file__).parent / "gatekeeper" / "gatekeeper.pth"
_gatekeeper_available = _gatekeeper_path.exists()

if _gatekeeper_available:
    print("🧠 Gatekeeper model found - will lazy-load when needed")
else:
    print("⚠️ Gatekeeper model not found - running without NN filter")
    GATEKEEPER_ENABLED = False


def get_gatekeeper():
    """Lazy-load Gatekeeper to reduce memory usage on startup."""
    global _gatekeeper
    
    if not GATEKEEPER_ENABLED or not _gatekeeper_available:
        return None
    
    if _gatekeeper is None:
        try:
            from gatekeeper import Gatekeeper
            _gatekeeper = Gatekeeper(str(_gatekeeper_path), threshold=GATEKEEPER_THRESHOLD)
            print("🧠 Gatekeeper loaded on-demand")
        except Exception as e:
            print(f"⚠️ Failed to load Gatekeeper: {e}")
            return None
    
    return _gatekeeper

# Circuit Breaker for Hybrid Gatekeeper Mode
# Normal Market (VIX < 20): Gatekeeper OFF → Maximize profit
# Stress Market (VIX > 20): Gatekeeper ON → Safety mode
circuit_breaker = None
try:
    from circuit_breaker import CircuitBreaker
    circuit_breaker = CircuitBreaker()
    print("🚨 Circuit Breaker loaded - Hybrid mode enabled!")
except ImportError:
    print("⚠️ Circuit Breaker not available - Gatekeeper always on")
except Exception as e:
    print(f"⚠️ Error loading Circuit Breaker: {e}")

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "10000"))
BASE_POSITION_SIZE = float(os.environ.get("BASE_POSITION_SIZE", "0.10"))  # fallback sizing

# Separate thresholds for crypto vs stocks (stocks often need lower thresholds or faster bars)
CRYPTO_ZSCORE_ENTRY = float(os.environ.get("CRYPTO_ZSCORE_ENTRY", "2.0"))
CRYPTO_ZSCORE_EXIT = float(os.environ.get("CRYPTO_ZSCORE_EXIT", "0.5"))
STOCK_ZSCORE_ENTRY = float(os.environ.get("STOCK_ZSCORE_ENTRY", "1.5"))
STOCK_ZSCORE_EXIT = float(os.environ.get("STOCK_ZSCORE_EXIT", "0.5"))

# Risk controls (paper trading realism + survival)
TRADING_COST_BPS = float(os.environ.get("TRADING_COST_BPS", "2.0"))  # per open/close on gross notional
CRYPTO_STOP_ZSCORE = float(os.environ.get("CRYPTO_STOP_ZSCORE", "4.0"))
STOCK_STOP_ZSCORE = float(os.environ.get("STOCK_STOP_ZSCORE", "3.5"))

CRYPTO_STOP_LOSS_PCT = float(os.environ.get("CRYPTO_STOP_LOSS_PCT", "0.06"))  # -6%
STOCK_STOP_LOSS_PCT = float(os.environ.get("STOCK_STOP_LOSS_PCT", "0.04"))    # -4%
CRYPTO_TAKE_PROFIT_PCT = float(os.environ.get("CRYPTO_TAKE_PROFIT_PCT", "0.05"))
STOCK_TAKE_PROFIT_PCT = float(os.environ.get("STOCK_TAKE_PROFIT_PCT", "0.03"))

CRYPTO_MAX_HOLDING_HOURS = float(os.environ.get("CRYPTO_MAX_HOLDING_HOURS", "168"))  # 7 days
STOCK_MAX_HOLDING_HOURS = float(os.environ.get("STOCK_MAX_HOLDING_HOURS", "72"))    # 3 days

MAX_OPEN_POSITIONS_CRYPTO = int(os.environ.get("MAX_OPEN_POSITIONS_CRYPTO", "6"))
MAX_OPEN_POSITIONS_STOCKS = int(os.environ.get("MAX_OPEN_POSITIONS_STOCKS", "8"))

# Data settings (yfinance)
CRYPTO_DATA_PERIOD = os.environ.get("CRYPTO_DATA_PERIOD", "60d")
CRYPTO_DATA_INTERVAL = os.environ.get("CRYPTO_DATA_INTERVAL", "").strip() or None
STOCK_DATA_PERIOD = os.environ.get("STOCK_DATA_PERIOD", "60d")
STOCK_DATA_INTERVAL = os.environ.get("STOCK_DATA_INTERVAL", "15m").strip() or None

CRYPTO_MIN_BARS = int(os.environ.get("CRYPTO_MIN_BARS", "30"))
_default_stock_min_bars = 200 if STOCK_DATA_INTERVAL not in (None, "1d") else 30
STOCK_MIN_BARS = int(os.environ.get("STOCK_MIN_BARS", str(_default_stock_min_bars)))

# Limit how much history we feed into the stock Kalman per update (keeps it responsive)
_default_stock_lookback = 800 if STOCK_DATA_INTERVAL not in (None, "1d") else 90
STOCK_SIGNAL_LOOKBACK_BARS = int(os.environ.get("STOCK_SIGNAL_LOOKBACK_BARS", str(_default_stock_lookback)))
UPDATE_INTERVAL = 180  # 3 minutes (faster updates)
STALE_THRESHOLD = 300  # 5 minutes (more responsive stale detection)

# Dynamic Position Sizing (CONSERVATIVE Kelly)
# RULE: Never bet more than 5% on ANY trade
# Why: A Z-score of 3.44 could be a golden setup OR a Luna collapse
# Survive by hitting singles, not home runs
MAX_POSITION_SIZE = 0.05  # HARD CAP: 5% max per trade (survive black swans)

POSITION_TIERS = {
    0.90: 0.05,  # 90%+ confidence: 5% of capital (max allowed)
    0.85: 0.04,  # 85%+ confidence: 4% of capital
    0.80: 0.03,  # 80%+ confidence: 3% of capital
    0.75: 0.02,  # 75%+ confidence: 2% of capital (minimum)
}

# Always store state relative to this file (avoid cwd-dependent bugs)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Models directory (tracked in repo)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Optional: Stock edge model (trained offline) to filter/sized entries
STOCK_EDGE_MODEL_FILE = MODELS_DIR / "stock_edge_model.json"
STOCK_EDGE_THRESHOLD = float(os.environ.get("STOCK_EDGE_THRESHOLD", "0.55"))
STOCK_EDGE_BLOCKING = os.environ.get("STOCK_EDGE_BLOCKING", "0").strip() in ("1", "true", "True", "yes", "YES")
_stock_edge_model = None
try:
    if STOCK_EDGE_MODEL_FILE.exists():
        from stock_edge_model import StockEdgeModel, extract_features

        _stock_edge_model = StockEdgeModel.load(STOCK_EDGE_MODEL_FILE)
        mode = "blocking" if STOCK_EDGE_BLOCKING else "sizing-only"
        print(f"🧠 Stock edge model loaded: {STOCK_EDGE_MODEL_FILE} ({mode}, threshold={STOCK_EDGE_THRESHOLD})")
except Exception as e:
    print(f"⚠️ Failed to load stock edge model: {e}")

# ============================================================
# CRYPTO PAIRS (Robinhood)
# ============================================================
CRYPTO_PAIRS = [
    {"coin1": "ETC", "coin2": "BCH", "pvalue": 0.0001},
    {"coin1": "SHIB", "coin2": "BCH", "pvalue": 0.0002},
    {"coin1": "XRP", "coin2": "XLM", "pvalue": 0.019},
    {"coin1": "LTC", "coin2": "XLM", "pvalue": 0.002},
    {"coin1": "ETC", "coin2": "XLM", "pvalue": 0.003},
    {"coin1": "SHIB", "coin2": "XLM", "pvalue": 0.004},
    {"coin1": "ETC", "coin2": "AAVE", "pvalue": 0.00001},
    {"coin1": "SHIB", "coin2": "AAVE", "pvalue": 0.0003},
    # Migrated from old paper trader
    {"coin1": "DOGE", "coin2": "LINK", "pvalue": 0.01},
]

# ============================================================
# STOCK UNIVERSE - Dynamically loaded from pairs_config.json
# Auto-refreshed weekly by background task
# ============================================================
PAIRS_CONFIG_FILE = DATA_DIR / "pairs_config.json"
FALLBACK_PAIRS_CONFIG_FILE = MODELS_DIR / "pairs_config.json"
PAIR_REFRESH_DAYS = 7  # Refresh pairs every 7 days

# Default pairs (used if no config file exists)
DEFAULT_STOCK_PAIRS = [
    {"stock1": "C", "stock2": "GS", "correlation": 0.99, "cluster": "Financials"},
    {"stock1": "AXP", "stock2": "BAC", "correlation": 0.97, "cluster": "Financials"},
    {"stock1": "MS", "stock2": "PH", "correlation": 0.96, "cluster": "Financials"},
    {"stock1": "GS", "stock2": "WFC", "correlation": 0.94, "cluster": "Financials"},
    {"stock1": "BAC", "stock2": "MS", "correlation": 0.98, "cluster": "Financials"},
    {"stock1": "COP", "stock2": "OXY", "correlation": 0.90, "cluster": "Energy"},
    {"stock1": "COP", "stock2": "EOG", "correlation": 0.86, "cluster": "Energy"},
    {"stock1": "KHC", "stock2": "PG", "correlation": 0.79, "cluster": "Consumer"},
    {"stock1": "MA", "stock2": "V", "correlation": 0.91, "cluster": "Payments"},
    {"stock1": "CVX", "stock2": "XOM", "correlation": 0.86, "cluster": "Energy"},
    {"stock1": "HD", "stock2": "LOW", "correlation": 0.89, "cluster": "Retail"},
    {"stock1": "KO", "stock2": "PEP", "correlation": 0.81, "cluster": "Beverages"},
]

MIN_STOCK_PAIRS = int(os.environ.get("MIN_STOCK_PAIRS", "25"))
MAX_STOCK_PAIRS = int(os.environ.get("MAX_STOCK_PAIRS", "60"))

# Optional: broaden the universe using the quant_system precomputed universe (more sectors).
QUANT_SYSTEM_UNIVERSE_FILE = BASE_DIR.parent / "quant_system" / "data" / "sp500_universe.json"


def _pair_dedupe_key(stock1: str, stock2: str) -> tuple:
    """Order-insensitive key for de-duping pairs across sources."""
    a, b = str(stock1).upper(), str(stock2).upper()
    return (a, b) if a <= b else (b, a)


def _load_quant_system_pairs() -> list:
    """Load correlation-cluster pairs from quant_system as a diversified fallback."""
    try:
        if not QUANT_SYSTEM_UNIVERSE_FILE.exists():
            return []

        with open(QUANT_SYSTEM_UNIVERSE_FILE, "r") as f:
            data = json.load(f)

        pairs = data.get("pairs", []) or []
        out = []
        for p in pairs:
            s1 = p.get("stock1")
            s2 = p.get("stock2")
            if not s1 or not s2 or s1 == s2:
                continue

            cluster_id = p.get("cluster")
            cluster_label = f"Cluster {cluster_id}" if cluster_id is not None else "QuantCluster"

            out.append(
                {
                    "stock1": str(s1).upper(),
                    "stock2": str(s2).upper(),
                    "correlation": float(p.get("correlation", 0.0)),
                    "cluster": cluster_label,
                }
            )

        if out:
            print(f"📊 Loaded {len(out)} fallback pairs from quant_system universe")
        return out

    except Exception as e:
        print(f"⚠️ Failed to load quant_system universe pairs: {e}")
        return []


def load_stock_pairs() -> list:
    """
    Load stock pairs.

    Priority:
    - `data/pairs_config.json` (cointegration + half-life filtered pairs)
    - If too few pairs, augment with quant_system diversified pairs
    - Always ensure we have a reasonable minimum by falling back to DEFAULT_STOCK_PAIRS
    """
    pairs = []

    try:
        config_path = None
        # Prefer optimized config shipped with the app; fall back to runtime-generated data config.
        if FALLBACK_PAIRS_CONFIG_FILE.exists():
            config_path = FALLBACK_PAIRS_CONFIG_FILE
        elif PAIRS_CONFIG_FILE.exists():
            config_path = PAIRS_CONFIG_FILE

        if config_path is not None:
            with open(config_path, 'r') as f:
                config = json.load(f)
                pairs = config.get('pairs', []) or []
                print(f"📊 Loaded {len(pairs)} stock pairs from config: {config_path.name} (generated: {config.get('generated_at', 'unknown')})")
    except Exception as e:
        print(f"⚠️ Error loading pairs config: {e}")

    if not pairs:
        print(f"📊 No pairs config found - starting with {len(DEFAULT_STOCK_PAIRS)} defaults")
        pairs = list(DEFAULT_STOCK_PAIRS)

    # If we have too few pairs, broaden across sectors.
    if len(pairs) < MIN_STOCK_PAIRS:
        fallback = _load_quant_system_pairs()
        if fallback:
            merged = []
            seen = set()

            for p in list(pairs) + list(fallback) + list(DEFAULT_STOCK_PAIRS):
                s1, s2 = p.get("stock1"), p.get("stock2")
                if not s1 or not s2 or s1 == s2:
                    continue
                key = _pair_dedupe_key(s1, s2)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(p)

            pairs = merged

    # Cap for performance (fetching too many tickers can slow yfinance significantly)
    if MAX_STOCK_PAIRS and len(pairs) > MAX_STOCK_PAIRS:
        pairs = pairs[:MAX_STOCK_PAIRS]

    print(f"📊 Using {len(pairs)} stock pairs (min={MIN_STOCK_PAIRS}, max={MAX_STOCK_PAIRS})")
    return pairs


def should_refresh_pairs() -> bool:
    """Check if pairs need refreshing (older than PAIR_REFRESH_DAYS)."""
    try:
        if not PAIRS_CONFIG_FILE.exists():
            return True
        
        with open(PAIRS_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            generated_at = config.get('generated_at')
            if not generated_at:
                return True
            
            generated_date = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            age_days = (datetime.now(generated_date.tzinfo) - generated_date).days
            return age_days >= PAIR_REFRESH_DAYS
    except Exception:
        return True


def refresh_stock_pairs():
    """Run pair finder to refresh stock pairs (background task)."""
    global STOCK_PAIRS
    
    print("\n🔬 Starting automated pair refresh...")
    
    try:
        # Import pair finder
        from pair_finder import run_pair_finder, save_pairs_config
        
        # Run the pair finder
        pairs = run_pair_finder()
        
        if pairs and len(pairs) >= 5:  # Only update if we found enough pairs
            save_pairs_config(pairs)
            STOCK_PAIRS = load_stock_pairs()
            print(f"✅ Pair refresh complete! Now using {len(STOCK_PAIRS)} pairs")
        else:
            print("⚠️ Pair refresh found too few pairs, keeping existing")
            
    except Exception as e:
        print(f"❌ Pair refresh failed: {e}")


# Load pairs on startup
STOCK_PAIRS = load_stock_pairs()


# ============================================================
# KALMAN FILTER (for Stocks)
# ============================================================
class KalmanFilter:
    """Simple Kalman Filter for dynamic hedge ratio."""
    
    def __init__(self, delta=1e-4, R=1.0):
        self.delta = delta
        self.R = R
        self.beta = np.array([0.0, 1.0])  # [intercept, hedge_ratio]
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * delta
        # Running stats for innovation z-score
        self.innovation_mean = 0.0
        self.innovation_var = 1.0
        self.n = 0
        self.last_innovation = 0.0
        self.last_zscore = 0.0
    
    def update(self, y, x):
        """Update filter with new observation."""
        H = np.array([1.0, x])
        
        # Predict
        P_pred = self.P + self.Q
        y_pred = H @ self.beta
        
        # Innovation
        innovation = y - y_pred
        S = H @ P_pred @ H.T + self.R
        
        # Update
        K = P_pred @ H.T / S
        self.beta = self.beta + K * innovation
        I_KH = np.eye(2) - np.outer(K, H)
        self.P = I_KH @ P_pred
        
        # Update innovation stats (EMA) + cache latest z-score
        self.n += 1
        alpha = min(0.1, 2.0 / (self.n + 1))
        self.innovation_mean = (1 - alpha) * self.innovation_mean + alpha * innovation
        self.innovation_var = (1 - alpha) * self.innovation_var + alpha * (innovation - self.innovation_mean)**2

        self.last_innovation = float(innovation)
        std = np.sqrt(max(self.innovation_var, 1e-8))
        self.last_zscore = float((innovation - self.innovation_mean) / std)
        
        return innovation
    
    def get_zscore(self):
        """Get z-score of the latest innovation (prediction error)."""
        return self.last_zscore
    
    def get_hedge_ratio(self):
        return self.beta[1]


# ============================================================
# PORTFOLIO MANAGEMENT (Database-backed with JSON fallback)
# ============================================================
try:
    from database import (
        get_portfolio as db_get_portfolio,
        update_portfolio_cash as db_update_cash,
        update_last_update as db_update_last_update,
        add_position as db_add_position,
        remove_position as db_remove_position,
        add_trade as db_add_trade,
        get_recent_trades as db_get_trades,
        update_signal as db_update_signal,
        clear_signals as db_clear_signals,
        DATABASE_AVAILABLE
    )
    USE_DATABASE = DATABASE_AVAILABLE
    print(f"📊 Database module loaded (Available: {DATABASE_AVAILABLE})")
except ImportError as e:
    print(f"⚠️ Database module not available: {e}")
    USE_DATABASE = False


def load_portfolio(market="crypto"):
    """Load portfolio state from database or JSON."""
    if USE_DATABASE:
        try:
            db_data = db_get_portfolio(market)
            # Convert positions to app format
            positions = {}
            for pair, pos in db_data.get('positions', {}).items():
                positions[pair] = {
                    'type': pos.get('type', pos.get('position_type')),
                    'entry_zscore': pos.get('entry_zscore', 0),
                    'entry_spread': pos.get('entry_price', 0),
                    'hedge_ratio': pos.get('hedge_ratio', 0),
                    'entry_prices': pos.get('entry_prices', {}),
                    'entry_time': pos.get('entry_time'),
                    'position_value': pos.get('size', pos.get('position_value', 0))
                }
            
            # Convert signals dict to list format for frontend
            db_signals = db_data.get('signals', {})
            if isinstance(db_signals, dict):
                signals = [
                    {
                        'pair': pair,
                        'action': sig.get('signal', 'NO_SIGNAL'),
                        'zscore': sig.get('zscore', 0),
                        'prices': sig.get('prices', {}),
                        'method': 'Kalman' if market == 'stocks' else 'OLS',
                        'gatekeeper_approved': sig.get('gatekeeper_approved', True),
                        'gatekeeper_reason': sig.get('gatekeeper_reason')
                    }
                    for pair, sig in db_signals.items()
                ]
            else:
                signals = db_signals or []
            
            return {
                "cash": db_data.get('cash', INITIAL_CAPITAL),
                "positions": positions,
                "start_time": db_data.get('created_at', datetime.now().isoformat()),
                "total_value": db_data.get('cash', INITIAL_CAPITAL) + sum(p.get('position_value', p.get('size', 0)) for p in positions.values()),
                "last_update": db_data.get('last_update'),
                "market": market,
                "signals": signals
            }
        except Exception as e:
            print(f"⚠️ Database load failed, falling back to JSON: {e}")
    
    # JSON fallback
    file = DATA_DIR / f"portfolio_{market}.json"
    if file.exists():
        with open(file, 'r') as f:
            return json.load(f)
    return {
        "cash": INITIAL_CAPITAL,
        "positions": {},
        "start_time": datetime.now().isoformat(),
        "total_value": INITIAL_CAPITAL,
        "last_update": None,
        "market": market
    }


def save_portfolio(portfolio, market="crypto"):
    """Save portfolio state to database and JSON."""
    if USE_DATABASE:
        try:
            # Update cash
            db_update_cash(market, portfolio.get('cash', INITIAL_CAPITAL))
            
            # Note: Positions are saved separately via db_add_position/db_remove_position
            # This is handled in execute_trade
        except Exception as e:
            print(f"⚠️ Database save failed: {e}")
    
    # Always save to JSON as backup
    with open(DATA_DIR / f"portfolio_{market}.json", 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)


def load_trades(market="crypto"):
    """Load trade history from database or JSON."""
    def _trade_time_key(t: dict) -> datetime:
        ts = t.get("time") or t.get("created_at") or t.get("timestamp")
        if ts is None:
            return datetime.min
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            try:
                return datetime.fromtimestamp(ts)
            except Exception:
                return datetime.min
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return datetime.min
        return datetime.min

    def _normalize_trade(t: dict) -> dict:
        # Copy so callers don't mutate DB row dicts
        d = dict(t or {})

        # Normalize time to ISO string
        if "time" not in d or d["time"] is None:
            ts = d.get("created_at") or d.get("timestamp")
            if isinstance(ts, datetime):
                d["time"] = ts.isoformat()
            elif ts is not None:
                d["time"] = str(ts)
        elif isinstance(d["time"], datetime):
            d["time"] = d["time"].isoformat()

        # Normalize trade type naming
        trade_type = d.get("type") or d.get("trade_type") or d.get("tradeType")
        if "type" not in d or d.get("type") is None:
            if trade_type is not None:
                d["type"] = trade_type

        # Normalize action (frontend expects OPEN/CLOSE)
        if not d.get("action"):
            tt = str(trade_type).upper() if trade_type is not None else ""
            d["action"] = "CLOSE" if tt == "CLOSE" else ("OPEN" if tt else "UNKNOWN")

        # Ensure pnl is JSON-serializable float when present
        if d.get("pnl") is not None:
            try:
                d["pnl"] = float(d["pnl"])
            except Exception:
                pass

        # Prices sometimes come back as JSON string from DB adapters
        if isinstance(d.get("prices"), str):
            try:
                d["prices"] = json.loads(d["prices"])
            except Exception:
                pass

        return d

    def _normalize_trades(trades: list) -> list:
        if not isinstance(trades, list):
            return []
        normalized = [_normalize_trade(t) for t in trades if isinstance(t, dict)]
        normalized.sort(key=_trade_time_key)
        return normalized

    if USE_DATABASE:
        try:
            return _normalize_trades(db_get_trades(market, limit=100))
        except Exception as e:
            print(f"⚠️ Database trades load failed: {e}")
    
    # JSON fallback
    file = DATA_DIR / f"trades_{market}.json"
    if file.exists():
        with open(file, 'r') as f:
            return _normalize_trades(json.load(f))
    return []


def save_trades(trades, market="crypto"):
    """Save trade history - database saves happen in execute_trade."""
    # Always save to JSON as backup
    with open(DATA_DIR / f"trades_{market}.json", 'w') as f:
        json.dump(trades, f, indent=2, default=str)


# ============================================================
# DATA FETCHING
# ============================================================
def get_crypto_data(symbols, period=None, interval=None):
    """Fetch crypto data."""
    if period is None:
        period = CRYPTO_DATA_PERIOD
    if interval is None:
        interval = CRYPTO_DATA_INTERVAL

    tickers = [f"{s}-USD" for s in symbols]
    try:
        download_kwargs = {
            "period": period,
            "progress": False,
            "threads": True,
            "auto_adjust": True,
        }
        if interval:
            download_kwargs["interval"] = interval

        data = yf.download(tickers, **download_kwargs)
        if data is None or data.empty:
            print(f"⚠️ Crypto data empty")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']
        else:
            prices = data[['Close']]
            prices.columns = [tickers[0]]
        prices.columns = [c.replace('-USD', '') for c in prices.columns]
        result = prices.dropna()
        print(f"✅ Crypto data fetched: {len(result)} rows")
        return result
    except Exception as e:
        print(f"❌ Crypto data error: {e}")
        return None


def get_stock_data(symbols, period=None, interval=None):
    """Fetch stock data."""
    if period is None:
        period = STOCK_DATA_PERIOD
    if interval is None:
        interval = STOCK_DATA_INTERVAL

    try:
        download_kwargs = {
            "period": period,
            "progress": False,
            "threads": True,
            "auto_adjust": True,
        }
        if interval:
            download_kwargs["interval"] = interval

        data = yf.download(symbols, **download_kwargs)
        if data is None or data.empty:
            print(f"⚠️ Stock data empty for {symbols}")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']
        else:
            prices = data[['Close']]
            prices.columns = [symbols[0]] if isinstance(symbols, list) else [symbols]
        # Do NOT drop rows across all symbols; pair-level logic handles missing values.
        result = prices.dropna(axis=1, how='all').dropna(how='all')
        print(f"✅ Stock data fetched: {len(result)} rows for {len(symbols)} symbols")
        return result
    except Exception as e:
        print(f"❌ Stock data error: {e}")
        return None


# ============================================================
# SIGNAL CALCULATION
# ============================================================
def calculate_crypto_zscore(prices, coin1, coin2, window=30):
    """Calculate rolling z-score for crypto pair (OLS)."""
    if coin1 not in prices.columns or coin2 not in prices.columns:
        return None, None, None

    pair_prices = prices[[coin1, coin2]].dropna()
    if len(pair_prices) < max(window, 10):
        return None, None, None

    y = pair_prices[coin1].values
    x = pair_prices[coin2].values
    
    x_const = add_constant(x)
    model = OLS(y, x_const).fit()
    hedge_ratio = model.params[1]
    
    spread = y - hedge_ratio * x
    spread_series = pd.Series(spread, index=pair_prices.index)
    
    rolling_mean = spread_series.rolling(window=window).mean()
    rolling_std = spread_series.rolling(window=window).std()
    zscore = (spread_series - rolling_mean) / rolling_std
    
    current_zscore = zscore.iloc[-1] if not pd.isna(zscore.iloc[-1]) else 0
    
    return current_zscore, hedge_ratio, {
        "price1": float(pair_prices[coin1].iloc[-1]),
        "price2": float(pair_prices[coin2].iloc[-1])
    }


def calculate_stock_zscore(prices, stock1, stock2):
    """
    Calculate Kalman filter z-score for stock pair.

    Returns:
        (zscore, hedge_ratio, current_prices, context)
    """
    if stock1 not in prices.columns or stock2 not in prices.columns:
        return None, None, None, None

    # Work on aligned, non-NaN data (Kalman can't handle missing obs)
    pair_prices = prices[[stock1, stock2]].dropna()
    if len(pair_prices) < STOCK_MIN_BARS:
        return None, None, None, None

    if STOCK_SIGNAL_LOOKBACK_BARS and len(pair_prices) > STOCK_SIGNAL_LOOKBACK_BARS:
        pair_prices = pair_prices.tail(STOCK_SIGNAL_LOOKBACK_BARS)

    # IMPORTANT: build z-score from the latest innovation, not the long-run mean.
    # Also: do NOT re-feed the same history into a persistent filter on every update,
    # otherwise z-scores collapse toward 0 and you get no trades.
    kf = KalmanFilter(delta=1e-4, R=1.0)
    z_hist = []
    hr_hist = []
    p1_vals = pair_prices[stock1].astype(float).values
    p2_vals = pair_prices[stock2].astype(float).values

    for i in range(len(pair_prices)):
        kf.update(float(p1_vals[i]), float(p2_vals[i]))
        z_hist.append(float(kf.get_zscore()))
        hr_hist.append(float(kf.get_hedge_ratio()))

    current_prices = {
        "price1": float(p1_vals[-1]),
        "price2": float(p2_vals[-1]),
    }
    context = {
        "z_hist": z_hist,
        "p1_hist": p1_vals.tolist(),
        "p2_hist": p2_vals.tolist(),
        "hr_hist": hr_hist,
    }

    return float(z_hist[-1]), float(hr_hist[-1]), current_prices, context


# ============================================================
# TRADE EXECUTION
# ============================================================
def check_gatekeeper(pair_key, zscore, prices, hedge_ratio, market, price_history=None):
    """
    Check if the Gatekeeper Neural Network approves this trade.
    
    HYBRID MODE (Circuit Breaker):
    - Normal Market (VIX < 20): Gatekeeper OFF → Full speed
    - Stress Market (VIX > 20 or BTC crash): Gatekeeper ON → Safety mode
    
    Returns:
        (approved, probability, reason)
    """
    stress_context = None

    # Check Circuit Breaker first
    if circuit_breaker is not None:
        cb_status = circuit_breaker.check_conditions()
        
        if not cb_status['gatekeeper_enabled']:
            # Normal market - Gatekeeper OFF, full speed mode
            return True, 1.0, "🟢 NORMAL MARKET - Full Speed"
        else:
            # Stress market - Gatekeeper ON, safety mode
            triggers = ', '.join(cb_status['triggers'][:2])  # First 2 triggers
            stress_context = f"({triggers})" if triggers else None
            gk = get_gatekeeper()  # Lazy load Gatekeeper only when needed
            
            if not GATEKEEPER_ENABLED or gk is None:
                # No Gatekeeper available, but market is stressed
                # Be conservative - reject high z-score trades
                if abs(zscore) > 3.0:
                    return False, 0.3, f"🔴 STRESS ({triggers}) - Extreme Z blocked"
                return True, 0.6, f"🟡 STRESS ({triggers}) - No NN, allowing moderate Z"
    
    # Gatekeeper check (when enabled by circuit breaker or no circuit breaker)
    gk = get_gatekeeper()  # Lazy load
    if not GATEKEEPER_ENABLED or gk is None:
        return True, 1.0, "Gatekeeper disabled"
    
    try:
        # If we have real history, use the Gatekeeper as designed.
        if isinstance(price_history, dict):
            spread = price_history.get("spread")
            zseries = price_history.get("zscore")
            p1 = price_history.get("price1")
            p2 = price_history.get("price2")
            v1 = price_history.get("volume1")
            v2 = price_history.get("volume2")

            if (
                isinstance(spread, pd.Series)
                and isinstance(zseries, pd.Series)
                and isinstance(p1, pd.Series)
                and isinstance(p2, pd.Series)
                and len(spread) >= 50
            ):
                prob = float(gk.should_trade(spread, zseries, p1, p2, v1, v2, return_probability=True))
                approved = prob > GATEKEEPER_THRESHOLD
                ctx = f" {stress_context}" if stress_context else ""
                reason = f"{'🟢' if approved else '🔴'} STRESS{ctx} - NN prob={prob:.2f}"
                return approved, prob, reason

        # No usable feature history wired in → DO NOT use random inputs.
        # Use a deterministic, conservative stress-mode heuristic.
        if abs(zscore) > 3.0:
            return False, 0.3, f"🔴 STRESS{(' ' + stress_context) if stress_context else ''} - Extreme Z blocked (no features)"
        return True, 0.6, f"🟡 STRESS{(' ' + stress_context) if stress_context else ''} - No features, allowing moderate Z"
    except Exception:
        # If Gatekeeper fails, be conservative in stress mode
        return abs(zscore) < 3.0, 0.5, f"🟡 NN Error, allowing moderate Z"


def get_position_size(confidence: float, portfolio_cash: float) -> float:
    """
    Dynamic position sizing based on model confidence (Kelly-inspired).
    
    Higher confidence = larger position.
    Sniper mode: Only take high-confidence shots with appropriate size.
    """
    size_pct = BASE_POSITION_SIZE
    for threshold, tier_pct in sorted(POSITION_TIERS.items(), reverse=True):
        if confidence >= threshold:
            size_pct = tier_pct
            break

    # Enforce hard cap (safety)
    size_pct = min(size_pct, MAX_POSITION_SIZE)
    return portfolio_cash * size_pct


def _safe_fromiso(dt_str: str) -> datetime:
    """Parse ISO datetime safely (no tz assumptions)."""
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return datetime.now()


def calculate_unrealized_pnl(pos: dict, current_prices: dict) -> tuple[float, float]:
    """
    Calculate unrealized PnL for a pairs position.

    Returns:
        (pnl_dollars, pnl_pct_of_position_value)
    """
    try:
        entry = pos.get("entry_prices", {}) or {}
        position_value = float(pos.get("position_value", 0.0) or 0.0)
        if position_value <= 0:
            return 0.0, 0.0

        qty1 = pos.get("qty1")
        qty2 = pos.get("qty2")

        # Preferred: leg-level PnL (realistic)
        if qty1 is not None and qty2 is not None:
            qty1 = float(qty1)
            qty2 = float(qty2)
            p1e = float(entry.get("price1", 0.0) or 0.0)
            p2e = float(entry.get("price2", 0.0) or 0.0)
            p1c = float(current_prices.get("price1", 0.0) or 0.0)
            p2c = float(current_prices.get("price2", 0.0) or 0.0)
            pnl = qty1 * (p1c - p1e) + qty2 * (p2c - p2e)
            return pnl, (pnl / position_value) * 100.0

        # Fallback: spread-return approximation (legacy)
        entry_spread = float(pos.get("entry_spread", 0.0) or 0.0)
        hedge_ratio = float(pos.get("hedge_ratio", 0.0) or 0.0)
        exit_spread = float(current_prices.get("price1", 0.0) or 0.0) - hedge_ratio * float(current_prices.get("price2", 0.0) or 0.0)

        if entry_spread == 0:
            return 0.0, 0.0

        if pos.get("type") == "BUY_SPREAD":
            spread_return = (exit_spread - entry_spread) / abs(entry_spread)
        else:
            spread_return = (entry_spread - exit_spread) / abs(entry_spread)

        pnl = position_value * spread_return
        return pnl, spread_return * 100.0
    except Exception:
        return 0.0, 0.0


def execute_trade(
    portfolio,
    trades,
    pair_key,
    trade_type,
    zscore,
    prices,
    hedge_ratio,
    market,
    close_reason=None,
    confidence_override=None,
):
    """Execute a paper trade with dynamic position sizing."""
    trade_record = {
        "time": datetime.now().isoformat(),
        "pair": pair_key,
        "type": trade_type,
        "zscore": float(zscore),
        "hedge_ratio": float(hedge_ratio),
        "price1": prices["price1"],
        "price2": prices["price2"],
        "market": market,
        "reason": f"Z={zscore:.2f} | Hedge={hedge_ratio:.4f}"
    }
    if close_reason:
        trade_record["close_reason"] = str(close_reason)
    
    gk_confidence = float(confidence_override) if confidence_override is not None else 0.5
    if confidence_override is not None:
        trade_record["edge_prob"] = gk_confidence
    
    # Check Gatekeeper for new positions (not for closing)
    if (
        confidence_override is None
        and trade_type in ["BUY_SPREAD", "SELL_SPREAD"]
        and GATEKEEPER_ENABLED
    ):
        approved, prob, gk_reason = check_gatekeeper(
            pair_key, zscore, prices, hedge_ratio, market
        )
        gk_confidence = prob
        trade_record["gatekeeper_prob"] = prob
        trade_record["gatekeeper_approved"] = approved
        
        if not approved:
            # Log the blocked trade but don't execute
            print(f"   🚫 BLOCKED {pair_key}: {gk_reason}")
            trade_record["action"] = "BLOCKED_BY_GATEKEEPER"
            trade_record["reason"] += f" | {gk_reason}"
            trades.append(trade_record)
            return portfolio, trades
    
    if trade_type in ["BUY_SPREAD", "SELL_SPREAD"]:
        # Position limit per market (avoid over-trading / over-exposure)
        max_pos = MAX_OPEN_POSITIONS_STOCKS if market == "stocks" else MAX_OPEN_POSITIONS_CRYPTO
        if len(portfolio.get("positions", {})) >= max_pos:
            print(f"   ⚠️ SKIP {pair_key}: Max open positions reached ({max_pos})")
            return portfolio, trades

        # Dynamic position sizing based on Gatekeeper confidence
        position_value = get_position_size(gk_confidence, portfolio["cash"])
        if position_value < 100:
            print(f"   ⚠️ SKIP {pair_key}: Position too small (${position_value:.0f} < $100, Cash=${portfolio['cash']:.0f})")
            return portfolio, trades
        
        print(f"   ✅ EXECUTING {trade_type} on {pair_key}: Z={zscore:.2f}, Size=${position_value:.0f}")
        
        trade_record["position_tier"] = f"Conf={gk_confidence:.2f} -> ${position_value:.0f}"
        spread_value = prices["price1"] - hedge_ratio * prices["price2"]

        # Realistic leg sizing: scale so gross notional ~= position_value
        p1 = float(prices["price1"])
        p2 = float(prices["price2"])
        hr = float(hedge_ratio)
        denom = max(p1 + abs(hr) * p2, 1e-8)
        k = float(position_value) / denom

        if trade_type == "BUY_SPREAD":
            qty1 = k
            qty2 = -k * hr
        else:  # SELL_SPREAD
            qty1 = -k
            qty2 = k * hr

        # Trading costs (open)
        open_cost = float(position_value) * (TRADING_COST_BPS / 10000.0)
        
        portfolio["positions"][pair_key] = {
            "type": trade_type,
            "entry_zscore": float(zscore),
            "entry_prices": prices,
            "entry_spread": float(spread_value),
            "hedge_ratio": float(hedge_ratio),
            "qty1": float(qty1),
            "qty2": float(qty2),
            "position_value": position_value,
            "open_cost": open_cost,
            "entry_time": datetime.now().isoformat()
        }
        portfolio["cash"] -= (position_value + open_cost)
        trade_record["position_value"] = position_value
        trade_record["cost"] = open_cost
        trade_record["action"] = "OPEN"
        
        # Save to database
        if USE_DATABASE:
            try:
                db_add_position(market, pair_key, trade_type, float(spread_value), 
                               float(zscore), float(hedge_ratio), position_value)
                db_update_cash(market, portfolio["cash"])
                db_add_trade(market, pair_key, trade_type, float(zscore), float(hedge_ratio),
                            float(spread_value), prices, trade_record.get("reason", ""),
                            pnl=None, gatekeeper_prob=gk_confidence)
            except Exception as e:
                print(f"⚠️ Database save error: {e}")
        
    elif trade_type == "CLOSE":
        if pair_key in portfolio["positions"]:
            pos = portfolio["positions"][pair_key]
            position_value = float(pos.get("position_value", 0.0) or 0.0)
            pnl, pnl_pct = calculate_unrealized_pnl(pos, prices)

            close_cost = position_value * (TRADING_COST_BPS / 10000.0)
            pnl_after_cost = pnl - close_cost

            portfolio["cash"] += position_value + pnl_after_cost
            
            trade_record["pnl"] = pnl_after_cost
            trade_record["pnl_gross"] = pnl
            trade_record["cost"] = close_cost
            trade_record["return_pct"] = pnl_pct
            trade_record["action"] = "CLOSE"
            trade_record["entry_zscore"] = pos["entry_zscore"]
            
            # Spread value for audit/logging/database (use the hedge ratio used for the position)
            hr_used = float(pos.get("hedge_ratio", hedge_ratio) or 0.0)
            exit_spread = float(prices.get("price1", 0.0) or 0.0) - hr_used * float(prices.get("price2", 0.0) or 0.0)

            del portfolio["positions"][pair_key]
            
            # Save to database
            if USE_DATABASE:
                try:
                    db_remove_position(market, pair_key)
                    db_update_cash(market, portfolio["cash"])
                    reason = f"Closed: {close_reason or 'EXIT'} | PnL=${pnl_after_cost:.2f} ({pnl_pct:.2f}%)"
                    db_add_trade(
                        market,
                        pair_key,
                        "CLOSE",
                        float(zscore),
                        float(hr_used),
                        float(exit_spread),
                        prices,
                        reason,
                        pnl=float(pnl_after_cost),
                        gatekeeper_prob=None,
                    )
                except Exception as e:
                    print(f"⚠️ Database save error: {e}")
    
    trades.append(trade_record)
    return portfolio, trades


# ============================================================
# UPDATE FUNCTIONS
# ============================================================
def update_crypto():
    """Update crypto portfolio."""
    print(f"\n📊 Crypto Update: {datetime.now().strftime('%H:%M:%S')}")
    
    portfolio = load_portfolio("crypto")
    trades = load_trades("crypto")
    
    # Always include any currently open positions, even if pair lists changed
    crypto_pairs_to_process = list(CRYPTO_PAIRS)
    known = {f"{p['coin1']}-{p['coin2']}" for p in CRYPTO_PAIRS}
    for pair_key in portfolio.get("positions", {}).keys():
        if pair_key not in known and "-" in pair_key:
            parts = pair_key.split("-")
            if len(parts) == 2:
                crypto_pairs_to_process.append({"coin1": parts[0], "coin2": parts[1], "pvalue": None})

    coins = set()
    for pair in crypto_pairs_to_process:
        coins.add(pair["coin1"])
        coins.add(pair["coin2"])
    
    prices = get_crypto_data(list(coins), period=CRYPTO_DATA_PERIOD, interval=CRYPTO_DATA_INTERVAL)
    if prices is None or len(prices) < CRYPTO_MIN_BARS:
        return None
    
    signals = []
    
    # Clear old signals from database before calculating new ones
    if USE_DATABASE:
        try:
            db_clear_signals("crypto")
        except Exception as e:
            print(f"⚠️ Failed to clear crypto signals: {e}")
    
    for pair in crypto_pairs_to_process:
        coin1, coin2 = pair["coin1"], pair["coin2"]
        pair_key = f"{coin1}-{coin2}"
        
        zscore, hedge_ratio, current_prices = calculate_crypto_zscore(prices, coin1, coin2)
        if zscore is None:
            continue
        
        signal = {"pair": pair_key, "zscore": float(zscore), "prices": current_prices, "method": "OLS"}
        
        if pair_key in portfolio["positions"]:
            pos = portfolio["positions"][pair_key]
            # Mark-to-market PnL + risk controls
            pnl, pnl_pct = calculate_unrealized_pnl(pos, current_prices)
            pos["unrealized_pnl"] = float(pnl)
            pos["unrealized_pnl_pct"] = float(pnl_pct)
            pos["last_prices"] = current_prices
            pos["last_seen"] = datetime.now().isoformat()

            age_hours = (datetime.now() - _safe_fromiso(pos.get("entry_time", ""))).total_seconds() / 3600.0

            close_reason = None
            if abs(zscore) < CRYPTO_ZSCORE_EXIT:
                close_reason = f"EXIT_Z (|z|<{CRYPTO_ZSCORE_EXIT})"
            elif abs(zscore) > CRYPTO_STOP_ZSCORE:
                close_reason = f"STOP_Z (|z|>{CRYPTO_STOP_ZSCORE})"
            elif pnl_pct <= -CRYPTO_STOP_LOSS_PCT * 100.0:
                close_reason = f"STOP_LOSS ({pnl_pct:.2f}%)"
            elif pnl_pct >= CRYPTO_TAKE_PROFIT_PCT * 100.0:
                close_reason = f"TAKE_PROFIT ({pnl_pct:.2f}%)"
            elif age_hours >= CRYPTO_MAX_HOLDING_HOURS:
                close_reason = f"TIME_STOP ({age_hours:.1f}h)"

            if close_reason:
                portfolio, trades = execute_trade(
                    portfolio,
                    trades,
                    pair_key,
                    "CLOSE",
                    zscore,
                    current_prices,
                    pos.get("hedge_ratio", hedge_ratio),
                    "crypto",
                    close_reason=close_reason,
                )
                signal["action"] = "CLOSED"
            else:
                signal["action"] = f"HOLDING ({pos['type']})"
        else:
            if zscore > CRYPTO_ZSCORE_ENTRY:
                portfolio, trades = execute_trade(
                    portfolio, trades, pair_key, "SELL_SPREAD", zscore, current_prices, hedge_ratio, "crypto"
                )
                signal["action"] = "SELL_SPREAD"
            elif zscore < -CRYPTO_ZSCORE_ENTRY:
                portfolio, trades = execute_trade(
                    portfolio, trades, pair_key, "BUY_SPREAD", zscore, current_prices, hedge_ratio, "crypto"
                )
                signal["action"] = "BUY_SPREAD"
            else:
                signal["action"] = "NO_SIGNAL"
        
        signals.append(signal)
        
        # Save signal to database
        if USE_DATABASE:
            try:
                db_update_signal(
                    "crypto", pair_key, signal.get("action", "NO_SIGNAL"),
                    float(zscore), float(hedge_ratio), current_prices,
                    gatekeeper_approved=True,  # Will be updated if blocked
                    gatekeeper_reason=None
                )
            except Exception as e:
                print(f"⚠️ Failed to save signal {pair_key}: {e}")
    
    total_value = float(portfolio.get("cash", 0.0) or 0.0)
    for pos in portfolio.get("positions", {}).values():
        pv = float(pos.get("position_value", 0.0) or 0.0)
        upnl = float(pos.get("unrealized_pnl", 0.0) or 0.0)
        total_value += pv + upnl
    
    portfolio["total_value"] = total_value
    portfolio["last_update"] = datetime.now().isoformat()
    portfolio["signals"] = signals
    
    save_portfolio(portfolio, "crypto")
    save_trades(trades, "crypto")
    
    print(f"   ✅ Processed {len(signals)} crypto pairs")
    return portfolio


def update_stocks():
    """Update stocks portfolio using Kalman Filter."""
    print(f"\n📈 Stocks Update: {datetime.now().strftime('%H:%M:%S')}")
    
    portfolio = load_portfolio("stocks")
    trades = load_trades("stocks")
    
    # Always include any currently open positions, even if pair lists changed
    stock_pairs_to_process = list(STOCK_PAIRS)
    known = {f"{p['stock1']}-{p['stock2']}" for p in STOCK_PAIRS}
    for pair_key in portfolio.get("positions", {}).keys():
        if pair_key not in known and "-" in pair_key:
            parts = pair_key.split("-")
            if len(parts) == 2:
                stock_pairs_to_process.append({"stock1": parts[0], "stock2": parts[1], "cluster": "OPEN_POSITION"})

    stocks = set()
    for pair in stock_pairs_to_process:
        stocks.add(pair["stock1"])
        stocks.add(pair["stock2"])
    
    print(f"   Fetching data for {len(stocks)} stocks...")
    prices = get_stock_data(list(stocks), period=STOCK_DATA_PERIOD, interval=STOCK_DATA_INTERVAL)
    if prices is None:
        print("   ⚠️ Failed to fetch stock data - market may be closed")
        # Still update timestamp so we know it tried
        portfolio["last_update"] = datetime.now().isoformat()
        portfolio["signals"] = []
        save_portfolio(portfolio, "stocks")
        return portfolio
    
    if len(prices) < STOCK_MIN_BARS:
        print(f"   ⚠️ Insufficient data: {len(prices)} rows (need {STOCK_MIN_BARS})")
        portfolio["last_update"] = datetime.now().isoformat()
        portfolio["signals"] = []
        save_portfolio(portfolio, "stocks")
        return portfolio
    
    signals = []
    
    # Clear old signals from database before calculating new ones
    if USE_DATABASE:
        try:
            db_clear_signals("stocks")
        except Exception as e:
            print(f"⚠️ Failed to clear stock signals: {e}")
    
    for pair in stock_pairs_to_process:
        stock1, stock2 = pair["stock1"], pair["stock2"]
        pair_key = f"{stock1}-{stock2}"
        
        zscore, hedge_ratio, current_prices, context = calculate_stock_zscore(prices, stock1, stock2)
        if zscore is None:
            continue

        # Per-pair overrides (if optimizer writes them into config)
        entry_th = float(pair.get("entry_z", pair.get("entry_threshold", STOCK_ZSCORE_ENTRY)))
        exit_th = float(pair.get("exit_z", pair.get("exit_threshold", STOCK_ZSCORE_EXIT)))
        stop_z = float(pair.get("stop_z", STOCK_STOP_ZSCORE))
        
        signal = {
            "pair": pair_key, 
            "zscore": float(zscore), 
            "prices": current_prices, 
            "method": "Kalman",
            "cluster": pair.get("cluster", "")
        }
        
        if pair_key in portfolio["positions"]:
            pos = portfolio["positions"][pair_key]
            # Ensure we keep the same parameters for the life of the position
            exit_th = float(pos.get("exit_threshold", exit_th))
            stop_z = float(pos.get("stop_z", stop_z))
            max_hold_hours = float(pos.get("max_holding_hours", pair.get("max_hold_hours", STOCK_MAX_HOLDING_HOURS)))

            pnl, pnl_pct = calculate_unrealized_pnl(pos, current_prices)
            pos["unrealized_pnl"] = float(pnl)
            pos["unrealized_pnl_pct"] = float(pnl_pct)
            pos["last_prices"] = current_prices
            pos["last_seen"] = datetime.now().isoformat()

            age_hours = (datetime.now() - _safe_fromiso(pos.get("entry_time", ""))).total_seconds() / 3600.0

            close_reason = None
            if abs(zscore) < exit_th:
                close_reason = f"EXIT_Z (|z|<{exit_th})"
            elif abs(zscore) > stop_z:
                close_reason = f"STOP_Z (|z|>{stop_z})"
            elif pnl_pct <= -STOCK_STOP_LOSS_PCT * 100.0:
                close_reason = f"STOP_LOSS ({pnl_pct:.2f}%)"
            elif pnl_pct >= STOCK_TAKE_PROFIT_PCT * 100.0:
                close_reason = f"TAKE_PROFIT ({pnl_pct:.2f}%)"
            elif age_hours >= max_hold_hours:
                close_reason = f"TIME_STOP ({age_hours:.1f}h)"

            if close_reason:
                portfolio, trades = execute_trade(
                    portfolio,
                    trades,
                    pair_key,
                    "CLOSE",
                    zscore,
                    current_prices,
                    pos.get("hedge_ratio", hedge_ratio),
                    "stocks",
                    close_reason=close_reason,
                )
                signal["action"] = "CLOSED"
            else:
                signal["action"] = f"HOLDING ({pos['type']})"
        else:
            if zscore > entry_th:
                # Optional edge model filter (trained offline)
                if _stock_edge_model is not None and "extract_features" in globals():
                    feats = extract_features(context, entry_threshold=entry_th) if context else None
                    if feats is not None:
                        edge_prob = float(_stock_edge_model.predict_proba(feats))
                        signal["edge_prob"] = edge_prob
                        if STOCK_EDGE_BLOCKING and edge_prob < STOCK_EDGE_THRESHOLD:
                            signal["action"] = f"BLOCKED (edge={edge_prob:.2f})"
                            signals.append(signal)
                            continue

                portfolio, trades = execute_trade(
                    portfolio,
                    trades,
                    pair_key,
                    "SELL_SPREAD",
                    zscore,
                    current_prices,
                    hedge_ratio,
                    "stocks",
                    confidence_override=signal.get("edge_prob"),
                )
                signal["action"] = "SELL_SPREAD"
                # Persist parameters used for this position
                if pair_key in portfolio.get("positions", {}):
                    portfolio["positions"][pair_key]["entry_threshold"] = float(entry_th)
                    portfolio["positions"][pair_key]["exit_threshold"] = float(exit_th)
                    portfolio["positions"][pair_key]["stop_z"] = float(stop_z)
                    portfolio["positions"][pair_key]["max_holding_hours"] = float(pair.get("max_hold_hours", STOCK_MAX_HOLDING_HOURS))
            elif zscore < -entry_th:
                # Optional edge model filter (trained offline)
                if _stock_edge_model is not None and "extract_features" in globals():
                    feats = extract_features(context, entry_threshold=entry_th) if context else None
                    if feats is not None:
                        edge_prob = float(_stock_edge_model.predict_proba(feats))
                        signal["edge_prob"] = edge_prob
                        if STOCK_EDGE_BLOCKING and edge_prob < STOCK_EDGE_THRESHOLD:
                            signal["action"] = f"BLOCKED (edge={edge_prob:.2f})"
                            signals.append(signal)
                            continue

                portfolio, trades = execute_trade(
                    portfolio,
                    trades,
                    pair_key,
                    "BUY_SPREAD",
                    zscore,
                    current_prices,
                    hedge_ratio,
                    "stocks",
                    confidence_override=signal.get("edge_prob"),
                )
                signal["action"] = "BUY_SPREAD"
                if pair_key in portfolio.get("positions", {}):
                    portfolio["positions"][pair_key]["entry_threshold"] = float(entry_th)
                    portfolio["positions"][pair_key]["exit_threshold"] = float(exit_th)
                    portfolio["positions"][pair_key]["stop_z"] = float(stop_z)
                    portfolio["positions"][pair_key]["max_holding_hours"] = float(pair.get("max_hold_hours", STOCK_MAX_HOLDING_HOURS))
            else:
                signal["action"] = "NO_SIGNAL"
        
        signals.append(signal)
        
        # Save signal to database
        if USE_DATABASE:
            try:
                db_update_signal(
                    "stocks", pair_key, signal.get("action", "NO_SIGNAL"),
                    float(zscore), float(hedge_ratio), current_prices,
                    gatekeeper_approved=True,  # Will be updated if blocked
                    gatekeeper_reason=None
                )
            except Exception as e:
                print(f"⚠️ Failed to save signal {pair_key}: {e}")
    
    total_value = float(portfolio.get("cash", 0.0) or 0.0)
    for pos in portfolio.get("positions", {}).values():
        pv = float(pos.get("position_value", 0.0) or 0.0)
        upnl = float(pos.get("unrealized_pnl", 0.0) or 0.0)
        total_value += pv + upnl
    
    portfolio["total_value"] = total_value
    portfolio["last_update"] = datetime.now().isoformat()
    portfolio["signals"] = signals
    
    save_portfolio(portfolio, "stocks")
    save_trades(trades, "stocks")
    
    print(f"   ✅ Processed {len(signals)} stock pairs")
    return portfolio


def background_updater():
    """Background thread to update both portfolios."""
    # NOTE: Pair refresh is DISABLED in background to save memory on Render
    # Use the /api/pairs/refresh endpoint to manually refresh pairs
    
    while True:
        try:
            update_crypto()
            update_stocks()
        except Exception as e:
            print(f"❌ Update error: {e}")
        time.sleep(UPDATE_INTERVAL)


# ============================================================
# FLASK ROUTES
# ============================================================
@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/portfolio/<market>')
def get_portfolio(market):
    """Get portfolio for specified market (crypto/stocks)."""
    portfolio = load_portfolio(market)
    trades = load_trades(market)
    
    start_value = INITIAL_CAPITAL
    current_value = portfolio.get("total_value", start_value)
    pnl = current_value - start_value
    pnl_pct = (pnl / start_value) * 100
    
    closed_trades = [t for t in trades if t.get("action") == "CLOSE"]
    wins = len([t for t in closed_trades if t.get("pnl", 0) > 0])
    win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0
    
    start_time = datetime.fromisoformat(portfolio.get("start_time", datetime.now().isoformat()))
    running_time = datetime.now() - start_time
    
    last_update = portfolio.get("last_update")
    is_stale = False
    seconds_since = 0
    if last_update:
        seconds_since = (datetime.now() - datetime.fromisoformat(last_update)).total_seconds()
        is_stale = seconds_since > STALE_THRESHOLD
    
    return jsonify({
        "market": market,
        "cash": portfolio.get("cash", INITIAL_CAPITAL),
        "total_value": current_value,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "positions": portfolio.get("positions", {}),
        "signals": portfolio.get("signals", []),
        "last_update": last_update,
        "running_time_str": str(running_time).split('.')[0],
        "total_trades": len(trades),
        "closed_trades": len(closed_trades),
        "win_rate": win_rate,
        "is_stale": is_stale,
        "seconds_since_update": seconds_since
    })


@app.route('/api/trades/<market>')
def get_trades(market):
    """Get trade history."""
    trades = load_trades(market)
    return jsonify(trades[-50:])


@app.route('/api/update/<market>')
def force_update(market):
    """Force update for specified market."""
    if market == "crypto":
        portfolio = update_crypto()
    else:
        portfolio = update_stocks()
    return jsonify({"status": "updated", "market": market})


@app.route('/api/reset/<market>')
def reset_portfolio(market):
    """Reset portfolio."""
    portfolio = {
        "cash": INITIAL_CAPITAL,
        "positions": {},
        "start_time": datetime.now().isoformat(),
        "total_value": INITIAL_CAPITAL,
        "last_update": None,
        "market": market
    }
    save_portfolio(portfolio, market)
    save_trades([], market)
    
    return jsonify({"status": "reset", "market": market})


@app.route('/api/summary')
def get_summary():
    """Get combined summary of both portfolios."""
    crypto = load_portfolio("crypto")
    stocks = load_portfolio("stocks")
    
    crypto_pnl = crypto.get("total_value", INITIAL_CAPITAL) - INITIAL_CAPITAL
    stocks_pnl = stocks.get("total_value", INITIAL_CAPITAL) - INITIAL_CAPITAL
    
    return jsonify({
        "crypto": {
            "total_value": crypto.get("total_value", INITIAL_CAPITAL),
            "pnl": crypto_pnl,
            "pnl_pct": (crypto_pnl / INITIAL_CAPITAL) * 100,
            "positions": len(crypto.get("positions", {}))
        },
        "stocks": {
            "total_value": stocks.get("total_value", INITIAL_CAPITAL),
            "pnl": stocks_pnl,
            "pnl_pct": (stocks_pnl / INITIAL_CAPITAL) * 100,
            "positions": len(stocks.get("positions", {}))
        },
        "combined": {
            "total_value": crypto.get("total_value", INITIAL_CAPITAL) + stocks.get("total_value", INITIAL_CAPITAL),
            "pnl": crypto_pnl + stocks_pnl,
            "initial": INITIAL_CAPITAL * 2
        }
    })


@app.route('/api/zscore_history/<market>/<pair>')
def get_zscore_history(market, pair):
    """Get z-score history for a specific pair (for charting)."""
    parts = pair.split('-')
    if len(parts) != 2:
        return jsonify({"error": "Invalid pair format"}), 400
    
    asset1, asset2 = parts[0], parts[1]

    if market == "crypto":
        prices = get_crypto_data([asset1, asset2], period=CRYPTO_DATA_PERIOD, interval=CRYPTO_DATA_INTERVAL)
        if prices is None:
            return jsonify({"error": "Failed to fetch data"}), 500

        prices = prices[[asset1, asset2]].dropna()
        if len(prices) < max(CRYPTO_MIN_BARS, 30):
            return jsonify({"error": "Insufficient data"}), 500

        # OLS for hedge ratio
        y = prices[asset1].values
        x = prices[asset2].values
        x_const = add_constant(x)
        model = OLS(y, x_const).fit()
        hedge_ratio = model.params[1]

        # Spread and rolling z-score
        spread = y - hedge_ratio * x
        spread_series = pd.Series(spread, index=prices.index)
        rolling_mean = spread_series.rolling(window=30).mean()
        rolling_std = spread_series.rolling(window=30).std()
        zscore = (spread_series - rolling_mean) / rolling_std

        zscore_data = []
        for date, z in zscore.items():
            if not pd.isna(z):
                zscore_data.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "zscore": float(z),
                    }
                )

        return jsonify(
            {
                "pair": pair,
                "market": market,
                "hedge_ratio": float(hedge_ratio),
                "current_zscore": float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0,
                "entry_threshold": CRYPTO_ZSCORE_ENTRY,
                "exit_threshold": CRYPTO_ZSCORE_EXIT,
                "data": zscore_data[-60:],  # last ~60 points
            }
        )

    # market == "stocks": use the same Kalman innovation z-score the bot trades on
    prices = get_stock_data([asset1, asset2], period=STOCK_DATA_PERIOD, interval=STOCK_DATA_INTERVAL)
    if prices is None:
        return jsonify({"error": "Failed to fetch data"}), 500

    prices = prices[[asset1, asset2]].dropna()
    if len(prices) < STOCK_MIN_BARS:
        return jsonify({"error": "Insufficient data"}), 500

    if STOCK_SIGNAL_LOOKBACK_BARS and len(prices) > STOCK_SIGNAL_LOOKBACK_BARS:
        prices = prices.tail(STOCK_SIGNAL_LOOKBACK_BARS)

    kf = KalmanFilter(delta=1e-4, R=1.0)
    zscores = []
    hedges = []
    for i in range(len(prices)):
        kf.update(float(prices[asset1].iloc[i]), float(prices[asset2].iloc[i]))
        zscores.append(kf.get_zscore())
        hedges.append(float(kf.get_hedge_ratio()))

    zscore_series = pd.Series(zscores, index=prices.index)
    hedge_series = pd.Series(hedges, index=prices.index)

    zscore_data = []
    for date, z in zscore_series.items():
        if pd.isna(z):
            continue
        # Use ISO timestamps so Plotly can render both daily and intraday.
        zscore_data.append({"date": date.isoformat(), "zscore": float(z)})

    tail_n = 300 if STOCK_DATA_INTERVAL not in (None, "1d") else 90

    return jsonify(
        {
            "pair": pair,
            "market": market,
            "hedge_ratio": float(hedge_series.iloc[-1]) if len(hedge_series) else 0.0,
            "current_zscore": float(zscore_series.iloc[-1]) if len(zscore_series) else 0.0,
            "entry_threshold": STOCK_ZSCORE_ENTRY,
            "exit_threshold": STOCK_ZSCORE_EXIT,
            "data": zscore_data[-tail_n:],
        }
    )


@app.route('/api/circuit_breaker')
def get_circuit_breaker_status():
    """
    Get Circuit Breaker status.
    
    The Hybrid System:
    - Normal Market (VIX < 20): Gatekeeper OFF → Maximize profit
    - Stress Market (VIX > 20 or crash): Gatekeeper ON → Safety mode
    """
    if circuit_breaker is None:
        return jsonify({
            "available": False,
            "gatekeeper_enabled": GATEKEEPER_ENABLED,
            "status": "Circuit Breaker not loaded",
            "mode": "ALWAYS_ON" if GATEKEEPER_ENABLED else "ALWAYS_OFF"
        })
    
    status = circuit_breaker.check_conditions()
    
    return jsonify({
        "available": True,
        "gatekeeper_enabled": status['gatekeeper_enabled'],
        "market_status": status['market_status'],
        "triggers": status['triggers'],
        "details": status['details'],
        "mode": "HYBRID",
        "status_string": circuit_breaker.get_status_string()
    })


@app.route('/api/circuit_breaker/override', methods=['POST'])
def override_circuit_breaker():
    """Manually override Circuit Breaker (force Gatekeeper on/off)."""
    if circuit_breaker is None:
        return jsonify({"error": "Circuit Breaker not available"}), 400
    
    data = request.get_json() or {}
    action = data.get('action', 'auto')
    
    if action == 'on':
        circuit_breaker.force_gatekeeper(True)
        return jsonify({"status": "Gatekeeper forced ON"})
    elif action == 'off':
        circuit_breaker.force_gatekeeper(False)
        return jsonify({"status": "Gatekeeper forced OFF"})
    elif action == 'auto':
        circuit_breaker.reset_override()
        return jsonify({"status": "Gatekeeper reset to AUTO mode"})
    else:
        return jsonify({"error": "Invalid action. Use: on, off, auto"}), 400


@app.route('/api/pairs')
def get_pairs_info():
    """Get current stock pairs and refresh status."""
    try:
        config_age = None
        generated_at = None
        
        if PAIRS_CONFIG_FILE.exists():
            with open(PAIRS_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                generated_at = config.get('generated_at')
                if generated_at:
                    gen_date = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                    config_age = (datetime.now(gen_date.tzinfo) - gen_date).days
        
        return jsonify({
            "pairs_count": len(STOCK_PAIRS),
            "pairs": STOCK_PAIRS,
            "generated_at": generated_at,
            "age_days": config_age,
            "refresh_threshold_days": PAIR_REFRESH_DAYS,
            "needs_refresh": should_refresh_pairs()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/pairs/refresh', methods=['POST'])
def trigger_pair_refresh():
    """Manually trigger pair refresh (runs in background)."""
    import threading
    
    try:
        print("📊 Manual pair refresh triggered!")
        refresh_thread = threading.Thread(target=refresh_stock_pairs, daemon=True)
        refresh_thread.start()
        
        return jsonify({
            "status": "Pair refresh started",
            "message": "This takes 2-3 minutes. Check /api/pairs for status."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Start background updater
def start_background():
    thread = threading.Thread(target=background_updater, daemon=True)
    thread.start()
    print("🔄 Background updater started")


# ============================================================
# AUTO-START FOR GUNICORN
# This runs when gunicorn imports the module (not just __main__)
# ============================================================
_background_started = False

def ensure_background_running():
    """Ensure background updater is running (works with gunicorn)."""
    global _background_started
    if not _background_started:
        _background_started = True
        print("🚀 Initializing on first request...")
        # Do initial updates
        try:
            update_crypto()
            update_stocks()
        except Exception as e:
            print(f"⚠️ Initial update error: {e}")
        # Start background thread
        start_background()


@app.before_request
def before_first_request():
    """Start background updater on first HTTP request."""
    ensure_background_running()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 UNIFIED TRADING SIMULATOR")
    print("="*60)
    print(f"   Crypto Pairs: {len(CRYPTO_PAIRS)} (OLS Z-score)")
    print(f"   Stock Pairs: {len(STOCK_PAIRS)} (Kalman Filter)")
    print(f"   Initial Capital: ${int(INITIAL_CAPITAL):,} per market")
    if GATEKEEPER_ENABLED and _gatekeeper_available:
        print(f"   🧠 Gatekeeper: ENABLED (lazy-load, threshold={GATEKEEPER_THRESHOLD})")
    else:
        print(f"   ⚠️ Gatekeeper: DISABLED")
    print("="*60)
    
    # Initial updates
    update_crypto()
    update_stocks()
    
    # Start background updater
    start_background()
    
    # Run Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

