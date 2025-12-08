"""
🔬 AUTOMATED PAIR FINDER
========================
Uses machine learning to find statistically valid pairs instead of manual selection.

Algorithm:
1. Download S&P 500 data
2. Cluster stocks by price movement (not sector) using OPTICS
3. Run cointegration tests within clusters
4. Filter by half-life (fast mean reversion only)
5. Output best pairs to pairs_config.json

Run weekly to keep pairs fresh!
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import combinations

# ML & Stats
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_FILE = Path(__file__).parent / "pairs_config.json"
MIN_DATA_POINTS = 200  # ~1 year of trading days
COINT_PVALUE_THRESHOLD = 0.05  # 95% confidence
MAX_HALF_LIFE_DAYS = 15  # Must revert within 15 days
MIN_CORRELATION = 0.5  # Minimum correlation within cluster
TOP_N_PAIRS = 25  # Number of pairs to output

# S&P 500 tickers (top ~100 by market cap for speed)
# Full S&P 500 would take too long - use largest/most liquid
SP500_TICKERS = [
    # Tech Giants
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "CRM", "ADBE", "AMD", "INTC", "QCOM", "TXN", "NOW", "IBM", "AMAT", "MU",
    "INTU", "PANW", "SNPS", "CDNS", "KLAC", "LRCX", "ADI", "NXPI", "MCHP", "FTNT",
    
    # Financials
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    "C", "SPGI", "CB", "MMC", "PGR", "CME", "ICE", "AON", "USB", "PNC",
    
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "CVS", "MDT", "ISRG", "GILD", "SYK", "VRTX", "REGN", "ZTS", "BDX",
    
    # Consumer
    "PG", "KO", "PEP", "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "LOW",
    "TGT", "TJX", "EL", "CL", "KMB", "MDLZ", "GIS", "HSY", "KHC", "SYY",
    
    # Industrials
    "CAT", "HON", "UNP", "UPS", "BA", "RTX", "LMT", "NOC", "GE", "DE",
    "MMM", "EMR", "ITW", "ETN", "PH", "ROK", "CMI", "IR", "PCAR", "NSC",
    
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "PXD",
    
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ED",
    
    # Real Estate
    "PLD", "AMT", "EQIX", "PSA", "CCI", "SPG", "O", "WELL", "DLR", "AVB",
    
    # Communications
    "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR",
    
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE",
    
    # Fintech/Payments
    "PYPL", "SQ", "COIN", "SOFI",
    
    # Auto
    "GM", "F", "RIVN",
]


def download_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Download price data for all tickers."""
    print(f"📥 Downloading data for {len(tickers)} tickers...")
    
    try:
        data = yf.download(tickers, period=period, progress=True, threads=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']
        else:
            prices = data['Close'] if 'Close' in data.columns else data
        
        # Remove tickers with insufficient data
        valid_cols = prices.columns[prices.count() >= MIN_DATA_POINTS]
        prices = prices[valid_cols].dropna(axis=1, how='any')
        
        print(f"✅ Got {len(prices.columns)} tickers with sufficient data")
        return prices
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return pd.DataFrame()


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns."""
    returns = prices.pct_change().dropna()
    return returns


def cluster_stocks(returns: pd.DataFrame, min_samples: int = 3) -> Dict[int, List[str]]:
    """
    Cluster stocks using OPTICS based on return correlations.
    
    OPTICS is better than DBSCAN for financial data because:
    - Handles varying density clusters
    - Doesn't require specifying number of clusters
    - Identifies outliers naturally
    """
    print("\n🔬 Clustering stocks by price movement patterns...")
    
    # Create correlation matrix as features
    corr_matrix = returns.corr()
    
    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(corr_matrix.values)
    
    # OPTICS clustering
    optics = OPTICS(
        min_samples=min_samples,
        xi=0.05,
        min_cluster_size=0.05,
        metric='euclidean'
    )
    
    labels = optics.fit_predict(features)
    
    # Group tickers by cluster
    clusters = {}
    tickers = returns.columns.tolist()
    
    for ticker, label in zip(tickers, labels):
        if label == -1:  # Noise/outliers
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(ticker)
    
    # Print cluster summary
    print(f"\n📊 Found {len(clusters)} clusters:")
    for cluster_id, members in sorted(clusters.items()):
        print(f"   Cluster {cluster_id}: {len(members)} stocks - {members[:5]}{'...' if len(members) > 5 else ''}")
    
    return clusters


def calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate half-life of mean reversion using Ornstein-Uhlenbeck.
    
    Half-life = ln(2) / lambda, where lambda is from AR(1) regression.
    Smaller half-life = faster mean reversion = better for trading.
    """
    spread = spread.dropna()
    if len(spread) < 20:
        return np.inf
    
    # Lagged spread
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    # Align lengths
    spread_lag = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]
    
    if len(spread_lag) < 10:
        return np.inf
    
    try:
        # AR(1) regression: spread_diff = lambda * (spread_lag - mean) + noise
        X = add_constant(spread_lag)
        model = OLS(spread_diff, X).fit()
        
        lambda_coef = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
        
        if lambda_coef >= 0:  # Not mean reverting
            return np.inf
        
        half_life = -np.log(2) / lambda_coef
        return half_life
        
    except Exception:
        return np.inf


def test_cointegration(prices: pd.DataFrame, stock1: str, stock2: str) -> Dict:
    """
    Test if two stocks are cointegrated using Engle-Granger test.
    
    Returns dict with:
    - pvalue: Lower = more likely cointegrated
    - half_life: Days for spread to revert halfway
    - correlation: Price correlation
    - hedge_ratio: Optimal ratio for spread
    """
    try:
        p1 = prices[stock1].dropna()
        p2 = prices[stock2].dropna()
        
        # Align indices
        common_idx = p1.index.intersection(p2.index)
        p1 = p1.loc[common_idx]
        p2 = p2.loc[common_idx]
        
        if len(p1) < MIN_DATA_POINTS:
            return None
        
        # Correlation check
        correlation = p1.corr(p2)
        if abs(correlation) < MIN_CORRELATION:
            return None
        
        # Cointegration test
        score, pvalue, _ = coint(p1, p2)
        
        if pvalue > COINT_PVALUE_THRESHOLD:
            return None  # Not cointegrated at 95% confidence
        
        # Calculate hedge ratio via OLS
        X = add_constant(p2)
        model = OLS(p1, X).fit()
        hedge_ratio = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
        
        # Calculate spread and half-life
        spread = p1 - hedge_ratio * p2
        half_life = calculate_half_life(spread)
        
        if half_life > MAX_HALF_LIFE_DAYS or half_life <= 0:
            return None  # Too slow to mean revert
        
        # Spread statistics
        spread_std = spread.std()
        spread_mean = spread.mean()
        current_zscore = (spread.iloc[-1] - spread_mean) / spread_std
        
        return {
            "stock1": stock1,
            "stock2": stock2,
            "pvalue": float(pvalue),
            "correlation": float(correlation),
            "hedge_ratio": float(hedge_ratio),
            "half_life": float(half_life),
            "spread_std": float(spread_std),
            "current_zscore": float(current_zscore),
        }
        
    except Exception as e:
        return None


def find_pairs_in_cluster(prices: pd.DataFrame, cluster_stocks: List[str]) -> List[Dict]:
    """Find all valid pairs within a cluster."""
    pairs = []
    
    for stock1, stock2 in combinations(cluster_stocks, 2):
        if stock1 not in prices.columns or stock2 not in prices.columns:
            continue
            
        result = test_cointegration(prices, stock1, stock2)
        if result:
            pairs.append(result)
    
    return pairs


def assign_cluster_name(stocks: List[str]) -> str:
    """Guess cluster sector based on stocks."""
    tech = {"AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AMD", "INTC", "AVGO", "QCOM", "CRM", "NOW", "ORCL", "ADBE"}
    finance = {"JPM", "GS", "MS", "BAC", "WFC", "V", "MA", "AXP", "C", "BLK", "SCHW"}
    healthcare = {"UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "CVS", "TMO", "ABT", "BMY"}
    energy = {"XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "PSX", "VLO"}
    consumer = {"PG", "KO", "PEP", "WMT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE"}
    industrial = {"CAT", "HON", "UNP", "BA", "LMT", "NOC", "RTX", "GE", "UPS", "DE"}
    
    stock_set = set(stocks)
    
    if len(stock_set & tech) >= 2:
        return "Tech"
    elif len(stock_set & finance) >= 2:
        return "Financials"
    elif len(stock_set & healthcare) >= 2:
        return "Healthcare"
    elif len(stock_set & energy) >= 2:
        return "Energy"
    elif len(stock_set & consumer) >= 2:
        return "Consumer"
    elif len(stock_set & industrial) >= 2:
        return "Industrials"
    else:
        return "Mixed"


def run_pair_finder() -> List[Dict]:
    """Main function to find and rank all valid pairs."""
    print("=" * 60)
    print("🔬 AUTOMATED PAIR FINDER")
    print("=" * 60)
    
    # Step 1: Download data
    prices = download_data(SP500_TICKERS)
    if prices.empty:
        print("❌ No data downloaded!")
        return []
    
    # Step 2: Calculate returns for clustering
    returns = calculate_returns(prices)
    
    # Step 3: Cluster stocks
    clusters = cluster_stocks(returns)
    
    if not clusters:
        print("⚠️ No clusters found - testing all combinations (slower)")
        clusters = {0: prices.columns.tolist()}
    
    # Step 4: Find pairs within each cluster
    print("\n🔍 Testing cointegration within clusters...")
    all_pairs = []
    
    for cluster_id, stocks in clusters.items():
        if len(stocks) < 2:
            continue
            
        print(f"\n   Cluster {cluster_id}: Testing {len(stocks)}C2 = {len(list(combinations(stocks, 2)))} pairs...")
        cluster_pairs = find_pairs_in_cluster(prices, stocks)
        
        # Add cluster info
        cluster_name = assign_cluster_name(stocks)
        for pair in cluster_pairs:
            pair["cluster"] = cluster_name
            pair["cluster_id"] = cluster_id
        
        all_pairs.extend(cluster_pairs)
        print(f"   ✅ Found {len(cluster_pairs)} valid pairs in cluster {cluster_id}")
    
    # Step 5: Rank by quality (lower pvalue + lower half_life = better)
    # Score = pvalue * half_life (lower is better)
    for pair in all_pairs:
        pair["quality_score"] = pair["pvalue"] * pair["half_life"]
    
    all_pairs.sort(key=lambda x: x["quality_score"])
    
    # Take top N
    top_pairs = all_pairs[:TOP_N_PAIRS]
    
    # Step 6: Print results
    print("\n" + "=" * 60)
    print(f"📊 TOP {len(top_pairs)} PAIRS (ranked by quality)")
    print("=" * 60)
    
    for i, pair in enumerate(top_pairs, 1):
        print(f"\n{i:2}. {pair['stock1']}-{pair['stock2']} ({pair['cluster']})")
        print(f"    p-value: {pair['pvalue']:.4f} | Corr: {pair['correlation']:.2f} | Half-life: {pair['half_life']:.1f} days")
        print(f"    Hedge Ratio: {pair['hedge_ratio']:.4f} | Current Z: {pair['current_zscore']:+.2f}")
    
    return top_pairs


def save_pairs_config(pairs: List[Dict]):
    """Save pairs to JSON config file."""
    # Convert to app.py format
    config = {
        "generated_at": datetime.now().isoformat(),
        "algorithm": "OPTICS clustering + Engle-Granger cointegration + Half-life filter",
        "thresholds": {
            "coint_pvalue": COINT_PVALUE_THRESHOLD,
            "max_half_life_days": MAX_HALF_LIFE_DAYS,
            "min_correlation": MIN_CORRELATION
        },
        "pairs": [
            {
                "stock1": p["stock1"],
                "stock2": p["stock2"],
                "correlation": round(p["correlation"], 2),
                "cluster": p["cluster"],
                "pvalue": round(p["pvalue"], 4),
                "half_life": round(p["half_life"], 1),
                "hedge_ratio": round(p["hedge_ratio"], 4),
            }
            for p in pairs
        ]
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Saved {len(pairs)} pairs to {OUTPUT_FILE}")
    
    # Also generate Python code for copy-paste
    print("\n📋 Python code for app.py:")
    print("-" * 40)
    print("STOCK_PAIRS = [")
    for p in pairs:
        print(f'    {{"stock1": "{p["stock1"]}", "stock2": "{p["stock2"]}", "correlation": {p["correlation"]:.2f}, "cluster": "{p["cluster"]}"}},')
    print("]")


if __name__ == "__main__":
    pairs = run_pair_finder()
    
    if pairs:
        save_pairs_config(pairs)
        print("\n✅ PAIR FINDER COMPLETE!")
        print("   Run this weekly to keep pairs fresh.")
    else:
        print("\n❌ No valid pairs found!")

