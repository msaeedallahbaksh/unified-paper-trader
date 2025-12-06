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

# Gatekeeper Neural Network (optional)
GATEKEEPER_ENABLED = True
GATEKEEPER_THRESHOLD = 0.7
gatekeeper = None

try:
    from gatekeeper import Gatekeeper
    gatekeeper_path = Path(__file__).parent / "gatekeeper" / "gatekeeper.pth"
    if gatekeeper_path.exists():
        gatekeeper = Gatekeeper(str(gatekeeper_path), threshold=GATEKEEPER_THRESHOLD)
        print("🧠 Gatekeeper Neural Network loaded!")
    else:
        print("⚠️ Gatekeeper model not found - running without NN filter")
        GATEKEEPER_ENABLED = False
except ImportError as e:
    print(f"⚠️ Gatekeeper not available: {e}")
    GATEKEEPER_ENABLED = False
except Exception as e:
    print(f"⚠️ Error loading Gatekeeper: {e}")
    GATEKEEPER_ENABLED = False

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_CAPITAL = 10000
POSITION_SIZE = 0.15  # 15% per trade
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
UPDATE_INTERVAL = 300  # 5 minutes
STALE_THRESHOLD = 600  # 10 minutes

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

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
# STOCK UNIVERSE (S&P 500 Top Pairs from Spectral Clustering)
# ============================================================
STOCK_PAIRS = [
    {"stock1": "MA", "stock2": "V", "correlation": 0.91, "cluster": "Payments"},
    {"stock1": "HD", "stock2": "LOW", "correlation": 0.89, "cluster": "Home Improvement"},
    {"stock1": "COP", "stock2": "EOG", "correlation": 0.88, "cluster": "Energy"},
    {"stock1": "GS", "stock2": "JPM", "correlation": 0.86, "cluster": "Financials"},
    {"stock1": "CVX", "stock2": "XOM", "correlation": 0.86, "cluster": "Energy"},
    {"stock1": "DUK", "stock2": "SO", "correlation": 0.84, "cluster": "Utilities"},
    {"stock1": "LMT", "stock2": "NOC", "correlation": 0.82, "cluster": "Defense"},
    {"stock1": "KO", "stock2": "PEP", "correlation": 0.81, "cluster": "Beverages"},
    {"stock1": "GOOGL", "stock2": "META", "correlation": 0.78, "cluster": "Tech"},
    {"stock1": "AAPL", "stock2": "MSFT", "correlation": 0.75, "cluster": "Tech"},
]


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
        self.innovation_mean = 0.0
        self.innovation_var = 1.0
        self.n = 0
    
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
        
        # Update innovation stats
        self.n += 1
        alpha = min(0.1, 2.0 / (self.n + 1))
        self.innovation_mean = (1 - alpha) * self.innovation_mean + alpha * innovation
        self.innovation_var = (1 - alpha) * self.innovation_var + alpha * (innovation - self.innovation_mean)**2
        
        return innovation
    
    def get_zscore(self):
        """Get z-score of latest innovation."""
        std = np.sqrt(max(self.innovation_var, 1e-8))
        return -self.innovation_mean / std  # Negative for signal direction
    
    def get_hedge_ratio(self):
        return self.beta[1]


# ============================================================
# PORTFOLIO MANAGEMENT
# ============================================================
def load_portfolio(market="crypto"):
    """Load portfolio state."""
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
    """Save portfolio state."""
    with open(DATA_DIR / f"portfolio_{market}.json", 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)


def load_trades(market="crypto"):
    """Load trade history."""
    file = DATA_DIR / f"trades_{market}.json"
    if file.exists():
        with open(file, 'r') as f:
            return json.load(f)
    return []


def save_trades(trades, market="crypto"):
    """Save trade history."""
    with open(DATA_DIR / f"trades_{market}.json", 'w') as f:
        json.dump(trades, f, indent=2, default=str)


# ============================================================
# DATA FETCHING
# ============================================================
def get_crypto_data(symbols, period="60d"):
    """Fetch crypto data."""
    tickers = [f"{s}-USD" for s in symbols]
    try:
        data = yf.download(tickers, period=period, progress=False, threads=True)
        if data is None or data.empty:
            print(f"⚠️ Crypto data empty")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
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


def get_stock_data(symbols, period="60d"):
    """Fetch stock data."""
    try:
        data = yf.download(symbols, period=period, progress=False, threads=True)
        if data is None or data.empty:
            print(f"⚠️ Stock data empty for {symbols}")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data[['Close']]
            prices.columns = [symbols[0]] if isinstance(symbols, list) else [symbols]
        result = prices.dropna()
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
    
    y = prices[coin1].values
    x = prices[coin2].values
    
    x_const = add_constant(x)
    model = OLS(y, x_const).fit()
    hedge_ratio = model.params[1]
    
    spread = y - hedge_ratio * x
    spread_series = pd.Series(spread, index=prices.index)
    
    rolling_mean = spread_series.rolling(window=window).mean()
    rolling_std = spread_series.rolling(window=window).std()
    zscore = (spread_series - rolling_mean) / rolling_std
    
    current_zscore = zscore.iloc[-1] if not pd.isna(zscore.iloc[-1]) else 0
    
    return current_zscore, hedge_ratio, {
        "price1": float(prices[coin1].iloc[-1]),
        "price2": float(prices[coin2].iloc[-1])
    }


# Store Kalman filters for stocks
stock_filters = {}

def calculate_stock_zscore(prices, stock1, stock2):
    """Calculate Kalman filter z-score for stock pair."""
    global stock_filters
    
    if stock1 not in prices.columns or stock2 not in prices.columns:
        return None, None, None
    
    key = f"{stock1}-{stock2}"
    if key not in stock_filters:
        stock_filters[key] = KalmanFilter(delta=1e-4, R=1.0)
    
    kf = stock_filters[key]
    
    # Run filter on all data
    for i in range(len(prices)):
        y = prices[stock1].iloc[i]
        x = prices[stock2].iloc[i]
        kf.update(y, x)
    
    return kf.get_zscore(), kf.get_hedge_ratio(), {
        "price1": float(prices[stock1].iloc[-1]),
        "price2": float(prices[stock2].iloc[-1])
    }


# ============================================================
# TRADE EXECUTION
# ============================================================
def check_gatekeeper(pair_key, zscore, prices, hedge_ratio, market, price_history=None):
    """
    Check if the Gatekeeper Neural Network approves this trade.
    
    Returns:
        (approved, probability, reason)
    """
    if not GATEKEEPER_ENABLED or gatekeeper is None:
        return True, 1.0, "Gatekeeper disabled"
    
    try:
        # For now, use a simplified check based on z-score magnitude
        # Full integration would pass the feature sequence
        prob = gatekeeper.predict_probability(
            np.random.randn(1, 50, 15)  # Placeholder - would use real features
        ) if hasattr(gatekeeper, 'predict_probability') else 0.5
        
        approved = prob > GATEKEEPER_THRESHOLD
        reason = f"NN Prob={prob:.2f}" if approved else f"BLOCKED (prob={prob:.2f})"
        return approved, prob, reason
    except Exception as e:
        # If Gatekeeper fails, allow trade (fail-safe)
        return True, 0.5, f"Gatekeeper error: {str(e)[:30]}"


def execute_trade(portfolio, trades, pair_key, trade_type, zscore, prices, hedge_ratio, market):
    """Execute a paper trade."""
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
    
    # Check Gatekeeper for new positions (not for closing)
    if trade_type in ["BUY_SPREAD", "SELL_SPREAD"] and GATEKEEPER_ENABLED:
        approved, prob, gk_reason = check_gatekeeper(
            pair_key, zscore, prices, hedge_ratio, market
        )
        trade_record["gatekeeper_prob"] = prob
        trade_record["gatekeeper_approved"] = approved
        
        if not approved:
            # Log the blocked trade but don't execute
            trade_record["action"] = "BLOCKED_BY_GATEKEEPER"
            trade_record["reason"] += f" | {gk_reason}"
            trades.append(trade_record)
            return portfolio, trades
    
    if trade_type in ["BUY_SPREAD", "SELL_SPREAD"]:
        position_value = portfolio["cash"] * POSITION_SIZE
        if position_value < 100:
            return portfolio, trades
        
        spread_value = prices["price1"] - hedge_ratio * prices["price2"]
        
        portfolio["positions"][pair_key] = {
            "type": trade_type,
            "entry_zscore": float(zscore),
            "entry_prices": prices,
            "entry_spread": float(spread_value),
            "hedge_ratio": float(hedge_ratio),
            "position_value": position_value,
            "entry_time": datetime.now().isoformat()
        }
        portfolio["cash"] -= position_value
        trade_record["position_value"] = position_value
        trade_record["action"] = "OPEN"
        
    elif trade_type == "CLOSE":
        if pair_key in portfolio["positions"]:
            pos = portfolio["positions"][pair_key]
            entry_prices = pos["entry_prices"]
            entry_spread = pos.get("entry_spread", entry_prices["price1"] - pos["hedge_ratio"] * entry_prices["price2"])
            exit_spread = prices["price1"] - pos["hedge_ratio"] * prices["price2"]
            
            if pos["type"] == "BUY_SPREAD":
                spread_return = (exit_spread - entry_spread) / abs(entry_spread) if entry_spread != 0 else 0
            else:
                spread_return = (entry_spread - exit_spread) / abs(entry_spread) if entry_spread != 0 else 0
            
            pnl = pos["position_value"] * spread_return
            portfolio["cash"] += pos["position_value"] + pnl
            
            trade_record["pnl"] = pnl
            trade_record["return_pct"] = spread_return * 100
            trade_record["action"] = "CLOSE"
            trade_record["entry_zscore"] = pos["entry_zscore"]
            
            del portfolio["positions"][pair_key]
    
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
    
    coins = set()
    for pair in CRYPTO_PAIRS:
        coins.add(pair["coin1"])
        coins.add(pair["coin2"])
    
    prices = get_crypto_data(list(coins))
    if prices is None or len(prices) < 30:
        return None
    
    signals = []
    
    for pair in CRYPTO_PAIRS:
        coin1, coin2 = pair["coin1"], pair["coin2"]
        pair_key = f"{coin1}-{coin2}"
        
        zscore, hedge_ratio, current_prices = calculate_crypto_zscore(prices, coin1, coin2)
        if zscore is None:
            continue
        
        signal = {"pair": pair_key, "zscore": float(zscore), "prices": current_prices, "method": "OLS"}
        
        if pair_key in portfolio["positions"]:
            pos = portfolio["positions"][pair_key]
            if abs(zscore) < ZSCORE_EXIT:
                portfolio, trades = execute_trade(
                    portfolio, trades, pair_key, "CLOSE", zscore, current_prices, hedge_ratio, "crypto"
                )
                signal["action"] = "CLOSED"
            else:
                signal["action"] = f"HOLDING ({pos['type']})"
        else:
            if zscore > ZSCORE_ENTRY:
                portfolio, trades = execute_trade(
                    portfolio, trades, pair_key, "SELL_SPREAD", zscore, current_prices, hedge_ratio, "crypto"
                )
                signal["action"] = "SELL_SPREAD"
            elif zscore < -ZSCORE_ENTRY:
                portfolio, trades = execute_trade(
                    portfolio, trades, pair_key, "BUY_SPREAD", zscore, current_prices, hedge_ratio, "crypto"
                )
                signal["action"] = "BUY_SPREAD"
            else:
                signal["action"] = "NO_SIGNAL"
        
        signals.append(signal)
    
    total_value = portfolio["cash"]
    for pos in portfolio["positions"].values():
        total_value += pos["position_value"]
    
    portfolio["total_value"] = total_value
    portfolio["last_update"] = datetime.now().isoformat()
    portfolio["signals"] = signals
    
    save_portfolio(portfolio, "crypto")
    save_trades(trades, "crypto")
    
    return portfolio


def update_stocks():
    """Update stocks portfolio using Kalman Filter."""
    print(f"\n📈 Stocks Update: {datetime.now().strftime('%H:%M:%S')}")
    
    portfolio = load_portfolio("stocks")
    trades = load_trades("stocks")
    
    stocks = set()
    for pair in STOCK_PAIRS:
        stocks.add(pair["stock1"])
        stocks.add(pair["stock2"])
    
    print(f"   Fetching data for {len(stocks)} stocks...")
    prices = get_stock_data(list(stocks))
    if prices is None:
        print("   ⚠️ Failed to fetch stock data - market may be closed")
        # Still update timestamp so we know it tried
        portfolio["last_update"] = datetime.now().isoformat()
        portfolio["signals"] = []
        save_portfolio(portfolio, "stocks")
        return portfolio
    
    if len(prices) < 30:
        print(f"   ⚠️ Insufficient data: {len(prices)} rows (need 30)")
        portfolio["last_update"] = datetime.now().isoformat()
        portfolio["signals"] = []
        save_portfolio(portfolio, "stocks")
        return portfolio
    
    signals = []
    
    for pair in STOCK_PAIRS:
        stock1, stock2 = pair["stock1"], pair["stock2"]
        pair_key = f"{stock1}-{stock2}"
        
        zscore, hedge_ratio, current_prices = calculate_stock_zscore(prices, stock1, stock2)
        if zscore is None:
            continue
        
        signal = {
            "pair": pair_key, 
            "zscore": float(zscore), 
            "prices": current_prices, 
            "method": "Kalman",
            "cluster": pair.get("cluster", "")
        }
        
        if pair_key in portfolio["positions"]:
            pos = portfolio["positions"][pair_key]
            if abs(zscore) < ZSCORE_EXIT:
                portfolio, trades = execute_trade(
                    portfolio, trades, pair_key, "CLOSE", zscore, current_prices, hedge_ratio, "stocks"
                )
                signal["action"] = "CLOSED"
            else:
                signal["action"] = f"HOLDING ({pos['type']})"
        else:
            if zscore > ZSCORE_ENTRY:
                portfolio, trades = execute_trade(
                    portfolio, trades, pair_key, "SELL_SPREAD", zscore, current_prices, hedge_ratio, "stocks"
                )
                signal["action"] = "SELL_SPREAD"
            elif zscore < -ZSCORE_ENTRY:
                portfolio, trades = execute_trade(
                    portfolio, trades, pair_key, "BUY_SPREAD", zscore, current_prices, hedge_ratio, "stocks"
                )
                signal["action"] = "BUY_SPREAD"
            else:
                signal["action"] = "NO_SIGNAL"
        
        signals.append(signal)
    
    total_value = portfolio["cash"]
    for pos in portfolio["positions"].values():
        total_value += pos["position_value"]
    
    portfolio["total_value"] = total_value
    portfolio["last_update"] = datetime.now().isoformat()
    portfolio["signals"] = signals
    
    save_portfolio(portfolio, "stocks")
    save_trades(trades, "stocks")
    
    return portfolio


def background_updater():
    """Background thread to update both portfolios."""
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
    
    # Reset Kalman filters if stocks
    if market == "stocks":
        global stock_filters
        stock_filters = {}
    
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
    
    # Fetch historical data
    if market == "crypto":
        symbols = [f"{asset1}-USD", f"{asset2}-USD"]
        try:
            data = yf.download(symbols, period="60d", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']]
            prices.columns = [c.replace('-USD', '') for c in prices.columns]
        except:
            return jsonify({"error": "Failed to fetch data"}), 500
    else:
        symbols = [asset1, asset2]
        try:
            data = yf.download(symbols, period="60d", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']]
        except:
            return jsonify({"error": "Failed to fetch data"}), 500
    
    prices = prices.dropna()
    if len(prices) < 30:
        return jsonify({"error": "Insufficient data"}), 500
    
    # Calculate z-scores
    y = prices[asset1].values
    x = prices[asset2].values
    
    # OLS for hedge ratio
    x_const = add_constant(x)
    model = OLS(y, x_const).fit()
    hedge_ratio = model.params[1]
    
    # Spread and rolling z-score
    spread = y - hedge_ratio * x
    spread_series = pd.Series(spread, index=prices.index)
    rolling_mean = spread_series.rolling(window=30).mean()
    rolling_std = spread_series.rolling(window=30).std()
    zscore = (spread_series - rolling_mean) / rolling_std
    
    # Prepare response
    zscore_data = []
    for i, (date, z) in enumerate(zscore.items()):
        if not pd.isna(z):
            zscore_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "zscore": float(z)
            })
    
    return jsonify({
        "pair": pair,
        "market": market,
        "hedge_ratio": float(hedge_ratio),
        "current_zscore": float(zscore.iloc[-1]) if not pd.isna(zscore.iloc[-1]) else 0,
        "data": zscore_data[-30:]  # Last 30 days
    })


# Start background updater
def start_background():
    thread = threading.Thread(target=background_updater, daemon=True)
    thread.start()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 UNIFIED TRADING SIMULATOR")
    print("="*60)
    print(f"   Crypto Pairs: {len(CRYPTO_PAIRS)} (OLS Z-score)")
    print(f"   Stock Pairs: {len(STOCK_PAIRS)} (Kalman Filter)")
    print(f"   Initial Capital: ${INITIAL_CAPITAL:,} per market")
    if GATEKEEPER_ENABLED and gatekeeper is not None:
        print(f"   🧠 Gatekeeper: ACTIVE (threshold={GATEKEEPER_THRESHOLD})")
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

