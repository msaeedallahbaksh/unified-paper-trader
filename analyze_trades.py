"""
Analyze Stock Trades - Deep Dive into Why Win Rate is 14%
=========================================================
Queries the live Supabase database to understand the trading problems.
"""

import os
import sys

# Fix Windows console encoding
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import json
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# Try to import psycopg2 for direct DB access
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# Alternatively, try to import requests to query the API
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

DATABASE_URL = os.environ.get("DATABASE_URL")
API_BASE = "https://unified-paper-trader.onrender.com"


def get_db_trades():
    """Get trades directly from Supabase database."""
    if not HAS_PSYCOPG2 or not DATABASE_URL:
        print("❌ Cannot connect to database (psycopg2 not available or DATABASE_URL not set)")
        return None
    
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, connect_timeout=10)
        cur = conn.cursor()
        
        # Get ALL stock trades
        cur.execute("""
            SELECT * FROM trades 
            WHERE market = 'stocks' 
            ORDER BY created_at DESC
        """)
        trades = [dict(row) for row in cur.fetchall()]
        
        cur.close()
        conn.close()
        return trades
    except Exception as e:
        print(f"❌ Database error: {e}")
        return None


def get_api_trades(market="stocks"):
    """Get trades from the deployed API."""
    if not HAS_REQUESTS:
        print("❌ requests module not available")
        return None
    
    try:
        resp = requests.get(f"{API_BASE}/api/trades/{market}", timeout=15)
        if resp.status_code != 200:
            print(f"❌ API error: {resp.status_code}")
            return None
        return resp.json()
    except Exception as e:
        print(f"❌ API error: {e}")
        return None


def get_api_portfolio(market="stocks"):
    """Get portfolio from the deployed API."""
    if not HAS_REQUESTS:
        return None
    
    try:
        resp = requests.get(f"{API_BASE}/api/portfolio/{market}", timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"❌ API error: {e}")
    return None


def analyze_trades(trades):
    """Analyze trade history for problems."""
    if not trades:
        print("No trades to analyze")
        return
    
    print("\n" + "=" * 70)
    print("📊 STOCK TRADE ANALYSIS")
    print("=" * 70)
    
    # Separate OPENs and CLOSEs
    opens = [t for t in trades if (t.get("trade_type") or t.get("type") or "").upper() in ("BUY_SPREAD", "SELL_SPREAD")]
    closes = [t for t in trades if (t.get("trade_type") or t.get("type") or "").upper() == "CLOSE"]
    
    print(f"\n📈 Total Records: {len(trades)}")
    print(f"   - OPENs: {len(opens)}")
    print(f"   - CLOSEs: {len(closes)}")
    
    # Analyze CLOSEs for PnL
    pnl_by_pair = defaultdict(list)
    all_pnl = []
    close_reasons = defaultdict(int)
    
    for t in closes:
        pair = t.get("pair", "UNKNOWN")
        pnl = t.get("pnl")
        reason = t.get("reason", "") or ""
        
        # Extract close reason from reason string
        if "STOP_LOSS" in reason:
            close_reasons["STOP_LOSS"] += 1
        elif "STOP_Z" in reason:
            close_reasons["STOP_Z"] += 1
        elif "EXIT_Z" in reason:
            close_reasons["EXIT_Z"] += 1
        elif "TAKE_PROFIT" in reason:
            close_reasons["TAKE_PROFIT"] += 1
        elif "TIME_STOP" in reason:
            close_reasons["TIME_STOP"] += 1
        else:
            close_reasons["UNKNOWN"] += 1
        
        if pnl is not None:
            try:
                pnl_val = float(pnl)
                pnl_by_pair[pair].append(pnl_val)
                all_pnl.append(pnl_val)
            except:
                pass
    
    # Overall stats
    if all_pnl:
        total_pnl = sum(all_pnl)
        wins = len([p for p in all_pnl if p > 0])
        losses = len([p for p in all_pnl if p <= 0])
        win_rate = (wins / len(all_pnl)) * 100 if all_pnl else 0
        avg_win = sum(p for p in all_pnl if p > 0) / wins if wins > 0 else 0
        avg_loss = sum(p for p in all_pnl if p <= 0) / losses if losses > 0 else 0
        
        print(f"\n💰 OVERALL PnL STATS:")
        print(f"   Total PnL: ${total_pnl:,.2f}")
        print(f"   Trades: {len(all_pnl)}")
        print(f"   Wins: {wins} ({win_rate:.1f}%)")
        print(f"   Losses: {losses} ({100-win_rate:.1f}%)")
        print(f"   Avg Win: ${avg_win:,.2f}")
        print(f"   Avg Loss: ${avg_loss:,.2f}")
        
        # Risk/Reward ratio
        if avg_loss != 0:
            rr = abs(avg_win / avg_loss)
            print(f"   Risk/Reward: {rr:.2f}x")
    
    # Close reasons breakdown
    if close_reasons:
        print(f"\n📋 CLOSE REASONS:")
        for reason, count in sorted(close_reasons.items(), key=lambda x: -x[1]):
            pct = (count / len(closes)) * 100 if closes else 0
            print(f"   {reason}: {count} ({pct:.1f}%)")
    
    # Per-pair breakdown
    print(f"\n📊 PER-PAIR BREAKDOWN:")
    pair_stats = []
    for pair, pnls in sorted(pnl_by_pair.items()):
        total = sum(pnls)
        trades_count = len(pnls)
        wins = len([p for p in pnls if p > 0])
        win_rate = (wins / trades_count) * 100 if trades_count > 0 else 0
        pair_stats.append((pair, total, trades_count, win_rate))
    
    # Sort by total PnL
    pair_stats.sort(key=lambda x: x[1])
    
    print("\n   🔴 WORST PERFORMING PAIRS:")
    for pair, total, count, wr in pair_stats[:10]:
        print(f"      {pair}: ${total:+,.2f} ({count} trades, {wr:.0f}% win)")
    
    print("\n   🟢 BEST PERFORMING PAIRS:")
    for pair, total, count, wr in pair_stats[-5:]:
        print(f"      {pair}: ${total:+,.2f} ({count} trades, {wr:.0f}% win)")
    
    # Check for duplicate trades (multiple opens on same pair)
    print(f"\n🔍 CHECKING FOR DUPLICATE TRADES:")
    pair_opens = defaultdict(list)
    for t in opens:
        pair = t.get("pair", "UNKNOWN")
        time_str = t.get("created_at") or t.get("time")
        pair_opens[pair].append(time_str)
    
    duplicates_found = False
    for pair, times in pair_opens.items():
        if len(times) > 10:  # More than 10 opens on same pair is suspicious
            duplicates_found = True
            print(f"   ⚠️ {pair}: {len(times)} OPEN trades!")
    
    if not duplicates_found:
        print("   ✅ No obvious duplicate trade patterns")
    
    # Check trade timing
    print(f"\n⏰ TRADE TIMING ANALYSIS:")
    for t in closes[:5]:
        pair = t.get("pair", "UNKNOWN")
        time_str = t.get("created_at") or t.get("time")
        pnl = t.get("pnl", 0)
        reason = t.get("reason", "")[:50] + "..." if len(t.get("reason", "")) > 50 else t.get("reason", "")
        print(f"   {time_str} | {pair}: ${pnl:+.2f} | {reason}")
    
    return {
        "total_trades": len(closes),
        "total_pnl": sum(all_pnl) if all_pnl else 0,
        "win_rate": win_rate if all_pnl else 0,
        "worst_pairs": pair_stats[:5],
        "close_reasons": dict(close_reasons),
    }


def check_pairs_config():
    """Check which pairs in the config have negative expected returns."""
    config_path = Path(__file__).parent / "models" / "pairs_config.json"
    if not config_path.exists():
        print("❌ pairs_config.json not found")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    pairs = config.get("pairs", [])
    
    print("\n" + "=" * 70)
    print("🔬 PAIRS CONFIG ANALYSIS")
    print("=" * 70)
    
    losers = []
    winners = []
    
    for p in pairs:
        test = p.get("test", {})
        ret = test.get("total_return_pct", 0)
        sharpe = test.get("sharpe", 0)
        win_rate = test.get("win_rate", 0)
        pair_name = f"{p['stock1']}-{p['stock2']}"
        
        if ret < 0 or sharpe < 0:
            losers.append((pair_name, ret, sharpe, win_rate))
        else:
            winners.append((pair_name, ret, sharpe, win_rate))
    
    print(f"\n📊 CONFIG PAIRS: {len(pairs)} total")
    print(f"   ✅ Expected Winners: {len(winners)}")
    print(f"   ❌ Expected Losers: {len(losers)}")
    
    if losers:
        print(f"\n🔴 LOSING PAIRS IN CONFIG (should be REMOVED):")
        losers.sort(key=lambda x: x[1])  # Sort by return
        for pair, ret, sharpe, wr in losers:
            print(f"   {pair}: Ret={ret:+.2f}%, Sharpe={sharpe:.2f}, WinRate={wr:.0f}%")
    
    return losers, winners


def main():
    print("🔍 Stock Trade Deep-Dive Analysis")
    print("=" * 70)
    
    # 1. Check pairs config for losing pairs
    losers, winners = check_pairs_config() or ([], [])
    
    # 2. Try to get trades from API
    print("\n" + "=" * 70)
    print("📡 Fetching live trade data...")
    print("=" * 70)
    
    # Try API first
    trades = get_api_trades("stocks")
    
    if trades:
        print(f"✅ Got {len(trades)} trades from API")
        analyze_trades(trades)
    else:
        print("⚠️ Could not fetch trades from API")
        # Try local file
        local_path = Path(__file__).parent / "data" / "trades_stocks.json"
        if local_path.exists():
            with open(local_path) as f:
                trades = json.load(f)
            print(f"📂 Loaded {len(trades)} trades from local file")
            analyze_trades(trades)
        else:
            print("❌ No trade data available")
    
    # 3. Get current portfolio state
    portfolio = get_api_portfolio("stocks")
    if portfolio:
        print("\n" + "=" * 70)
        print("💼 CURRENT PORTFOLIO STATE")
        print("=" * 70)
        print(f"   Cash: ${portfolio.get('cash', 0):,.2f}")
        print(f"   Total Value: ${portfolio.get('total_value', 0):,.2f}")
        print(f"   PnL: ${portfolio.get('pnl', 0):,.2f} ({portfolio.get('pnl_pct', 0):.2f}%)")
        print(f"   Win Rate: {portfolio.get('win_rate', 0):.1f}%")
        print(f"   Open Positions: {len(portfolio.get('positions', {}))}")
        print(f"   Total Trades: {portfolio.get('total_trades', 0)}")
        print(f"   Closed Trades: {portfolio.get('closed_trades', 0)}")
    
    # 4. Recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    if losers:
        print(f"\n1. ❌ REMOVE {len(losers)} LOSING PAIRS from pairs_config.json")
        print(f"   These pairs have negative expected returns in backtesting!")
    
    print("\n2. 📊 Fix the optimizer to FILTER OUT losing pairs")
    print("   The optimizer should only save pairs with positive Sharpe AND positive return")
    
    print("\n3. 🎯 Increase minimum win rate threshold")
    print("   Only trade pairs with >55% backtest win rate")
    
    print("\n4. 💰 Review position sizing")
    print("   Check if losses are outsized compared to wins")


if __name__ == "__main__":
    main()
