"""
Backtest the new mathematically-proven stock pairs from pair_finder.py
Uses realistic dollar P&L calculation for pairs trading
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.10  # 10% per trade (split between both legs)
LOOKBACK_DAYS = 50  # For calculating rolling stats
TEST_YEARS = 2  # How many years of data to test

# Load pairs from config
CONFIG_FILE = Path(__file__).parent / "data" / "pairs_config.json"


def load_pairs():
    """Load pairs from config file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get('pairs', [])
    return []


def download_data(tickers, years=2):
    """Download historical data for all tickers."""
    print(f"📥 Downloading {years} years of data for {len(tickers)} tickers...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True
    )['Close']
    
    print(f"   Got {len(data)} days of data")
    return data


def calculate_zscore(prices, stock1, stock2, lookback=50):
    """Calculate z-score using rolling OLS hedge ratio."""
    if stock1 not in prices.columns or stock2 not in prices.columns:
        return None, None
    
    p1 = prices[stock1].dropna()
    p2 = prices[stock2].dropna()
    
    # Align indices
    common = p1.index.intersection(p2.index)
    p1 = p1.loc[common]
    p2 = p2.loc[common]
    
    if len(p1) < lookback:
        return None, None
    
    # Rolling hedge ratio (OLS)
    hedge_ratios = []
    spreads = []
    
    for i in range(lookback, len(p1)):
        window_p1 = p1.iloc[i-lookback:i]
        window_p2 = p2.iloc[i-lookback:i]
        
        # Simple OLS: beta = cov(p1, p2) / var(p2)
        cov = np.cov(window_p1, window_p2)[0, 1]
        var = np.var(window_p2)
        hedge = cov / var if var > 0 else 1.0
        
        hedge_ratios.append(hedge)
        spreads.append(p1.iloc[i] - hedge * p2.iloc[i])
    
    # Convert to series
    dates = p1.index[lookback:]
    hedge_series = pd.Series(hedge_ratios, index=dates)
    spread_series = pd.Series(spreads, index=dates)
    
    # Z-score using rolling mean/std
    spread_mean = spread_series.rolling(lookback).mean()
    spread_std = spread_series.rolling(lookback).std()
    zscore = (spread_series - spread_mean) / spread_std
    
    return zscore, hedge_series


def backtest_pair(prices, stock1, stock2, pair_info=None):
    """
    Backtest a single pair using realistic dollar P&L.
    
    Pairs trade mechanics:
    - LONG_SPREAD (Z < -2): Buy stock1, Short stock2 * hedge_ratio
    - SHORT_SPREAD (Z > 2): Short stock1, Buy stock2 * hedge_ratio
    
    P&L is based on actual price changes in both legs.
    """
    zscore, hedge_ratio = calculate_zscore(prices, stock1, stock2)
    
    if zscore is None:
        return None
    
    # Clean data
    zscore = zscore.dropna()
    hedge_ratio = hedge_ratio.loc[zscore.index]
    
    p1 = prices[stock1].loc[zscore.index]
    p2 = prices[stock2].loc[zscore.index]
    
    # Trading simulation
    cash = INITIAL_CAPITAL
    position = None
    trades = []
    equity_curve = [INITIAL_CAPITAL]
    
    for i, date in enumerate(zscore.index[1:], 1):
        z = zscore.iloc[i]
        prev_z = zscore.iloc[i-1]
        hr = hedge_ratio.iloc[i]
        price1, price2 = p1.iloc[i], p2.iloc[i]
        
        if position is None:
            # Check for entry
            position_value = cash * POSITION_SIZE_PCT
            
            if z > ZSCORE_ENTRY and prev_z <= ZSCORE_ENTRY:
                # SHORT_SPREAD: Short stock1, Long stock2
                # Allocate half to each leg
                shares1 = (position_value * 0.5) / price1  # Short this many
                shares2 = (position_value * 0.5) / price2  # Long this many
                
                position = {
                    'type': 'SHORT_SPREAD',
                    'entry_z': z,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'shares1': shares1,  # Short
                    'shares2': shares2,  # Long
                    'total_value': position_value,
                    'entry_date': date
                }
                cash -= position_value
                
            elif z < -ZSCORE_ENTRY and prev_z >= -ZSCORE_ENTRY:
                # LONG_SPREAD: Long stock1, Short stock2
                shares1 = (position_value * 0.5) / price1  # Long this many
                shares2 = (position_value * 0.5) / price2  # Short this many
                
                position = {
                    'type': 'LONG_SPREAD',
                    'entry_z': z,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'shares1': shares1,  # Long
                    'shares2': shares2,  # Short
                    'total_value': position_value,
                    'entry_date': date
                }
                cash -= position_value
        else:
            # Check for exit
            if abs(z) < ZSCORE_EXIT:
                # Calculate actual dollar P&L
                if position['type'] == 'LONG_SPREAD':
                    # Long stock1: profit if price1 went up
                    # Short stock2: profit if price2 went down
                    pnl1 = position['shares1'] * (price1 - position['entry_price1'])
                    pnl2 = position['shares2'] * (position['entry_price2'] - price2)
                else:
                    # Short stock1: profit if price1 went down
                    # Long stock2: profit if price2 went up
                    pnl1 = position['shares1'] * (position['entry_price1'] - price1)
                    pnl2 = position['shares2'] * (price2 - position['entry_price2'])
                
                total_pnl = pnl1 + pnl2
                return_pct = total_pnl / position['total_value'] * 100
                
                cash += position['total_value'] + total_pnl
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': date,
                    'type': position['type'],
                    'entry_z': position['entry_z'],
                    'exit_z': z,
                    'pnl': total_pnl,
                    'pnl_pct': return_pct,
                    'duration': (date - position['entry_date']).days
                })
                
                position = None
        
        # Update equity (mark to market)
        if position:
            if position['type'] == 'LONG_SPREAD':
                mtm_pnl = (position['shares1'] * (price1 - position['entry_price1']) +
                          position['shares2'] * (position['entry_price2'] - price2))
            else:
                mtm_pnl = (position['shares1'] * (position['entry_price1'] - price1) +
                          position['shares2'] * (price2 - position['entry_price2']))
            equity_curve.append(cash + position['total_value'] + mtm_pnl)
        else:
            equity_curve.append(cash)
    
    # Close any remaining position at end
    if position:
        z = zscore.iloc[-1]
        price1, price2 = p1.iloc[-1], p2.iloc[-1]
        
        if position['type'] == 'LONG_SPREAD':
            pnl1 = position['shares1'] * (price1 - position['entry_price1'])
            pnl2 = position['shares2'] * (position['entry_price2'] - price2)
        else:
            pnl1 = position['shares1'] * (position['entry_price1'] - price1)
            pnl2 = position['shares2'] * (price2 - position['entry_price2'])
        
        total_pnl = pnl1 + pnl2
        return_pct = total_pnl / position['total_value'] * 100
        cash += position['total_value'] + total_pnl
        
        trades.append({
            'entry_date': position['entry_date'],
            'exit_date': zscore.index[-1],
            'type': position['type'],
            'entry_z': position['entry_z'],
            'exit_z': z,
            'pnl': total_pnl,
            'pnl_pct': return_pct,
            'duration': (zscore.index[-1] - position['entry_date']).days
        })
    
    # Calculate metrics
    if len(trades) == 0:
        return {
            'pair': f"{stock1}-{stock2}",
            'trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_trade_pnl': 0,
            'avg_trade_pct': 0,
            'avg_duration': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'final_equity': cash,
            'pair_info': pair_info
        }
    
    wins = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = wins / len(trades) * 100
    total_return = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    avg_pnl = np.mean([t['pnl'] for t in trades])
    avg_pct = np.mean([t['pnl_pct'] for t in trades])
    avg_duration = np.mean([t['duration'] for t in trades])
    
    # Sharpe ratio
    equity_series = pd.Series(equity_curve)
    daily_returns = equity_series.pct_change().dropna()
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    
    # Max drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min() * 100
    
    return {
        'pair': f"{stock1}-{stock2}",
        'trades': len(trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_trade_pnl': avg_pnl,
        'avg_trade_pct': avg_pct,
        'avg_duration': avg_duration,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'final_equity': cash,
        'pair_info': pair_info,
        'trade_details': trades
    }


def main():
    print("=" * 70)
    print("🔬 BACKTEST: NEW MATHEMATICALLY-PROVEN STOCK PAIRS")
    print("=" * 70)
    
    # Load pairs
    pairs = load_pairs()
    if not pairs:
        print("❌ No pairs found in config!")
        return
    
    print(f"\n📊 Testing {len(pairs)} pairs over {TEST_YEARS} years")
    print(f"   Entry: Z > {ZSCORE_ENTRY} or Z < -{ZSCORE_ENTRY}")
    print(f"   Exit: |Z| < {ZSCORE_EXIT}")
    print(f"   Position Size: {POSITION_SIZE_PCT*100}% per trade")
    
    # Get unique tickers
    tickers = set()
    for p in pairs:
        tickers.add(p['stock1'])
        tickers.add(p['stock2'])
    
    # Download data
    prices = download_data(list(tickers), years=TEST_YEARS)
    
    # Backtest each pair
    print("\n" + "=" * 70)
    print("📈 RESULTS BY PAIR")
    print("=" * 70)
    
    results = []
    for pair in pairs:
        stock1, stock2 = pair['stock1'], pair['stock2']
        print(f"\n🔄 Testing {stock1}-{stock2}...")
        
        result = backtest_pair(prices, stock1, stock2, pair_info=pair)
        if result:
            results.append(result)
            
            emoji = "✅" if result['total_return'] > 0 else "❌"
            print(f"   {emoji} Return: {result['total_return']:.1f}% | Trades: {result['trades']} | Win Rate: {result['win_rate']:.0f}%")
            print(f"      Sharpe: {result['sharpe']:.2f} | Max DD: {result['max_drawdown']:.1f}% | Avg Trade: {result['avg_trade_pct']:.1f}%")
            if pair.get('half_life'):
                print(f"      [p-value: {pair.get('pvalue', 'N/A')}, half-life: {pair.get('half_life', 'N/A')} days]")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    profitable = [r for r in results if r['total_return'] > 0]
    total_trades = sum(r['trades'] for r in results)
    avg_return = np.mean([r['total_return'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    avg_winrate = np.mean([r['win_rate'] for r in results if r['trades'] > 0])
    
    print(f"\n📈 Profitable Pairs: {len(profitable)}/{len(results)}")
    print(f"📊 Total Trades: {total_trades}")
    print(f"💰 Average Return: {avg_return:.1f}%")
    print(f"📉 Average Sharpe: {avg_sharpe:.2f}")
    print(f"🎯 Average Win Rate: {avg_winrate:.0f}%")
    
    # Best and worst pairs
    if results:
        results_sorted = sorted(results, key=lambda x: x['total_return'], reverse=True)
        
        print("\n🏆 TOP 3 PAIRS:")
        for r in results_sorted[:3]:
            print(f"   {r['pair']}: {r['total_return']:.1f}% ({r['trades']} trades, {r['win_rate']:.0f}% win rate, Sharpe={r['sharpe']:.2f})")
        
        print("\n⚠️ BOTTOM 3 PAIRS:")
        for r in results_sorted[-3:]:
            print(f"   {r['pair']}: {r['total_return']:.1f}% ({r['trades']} trades, {r['win_rate']:.0f}% win rate)")
    
    # Portfolio simulation (equal weight)
    print("\n" + "=" * 70)
    print("💼 COMBINED PORTFOLIO (if trading all pairs)")
    print("=" * 70)
    
    all_trades = []
    for r in results:
        if 'trade_details' in r:
            for t in r['trade_details']:
                t['pair'] = r['pair']
                all_trades.append(t)
    
    if all_trades:
        all_trades_sorted = sorted(all_trades, key=lambda x: x['exit_date'])
        
        portfolio_cash = INITIAL_CAPITAL
        for t in all_trades_sorted:
            portfolio_cash += t['pnl']
        
        portfolio_return = (portfolio_cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        portfolio_wins = sum(1 for t in all_trades if t['pnl'] > 0)
        portfolio_winrate = portfolio_wins / len(all_trades) * 100 if all_trades else 0
        avg_trade_return = np.mean([t['pnl_pct'] for t in all_trades])
        
        print(f"\n💰 Portfolio Final: ${portfolio_cash:,.2f}")
        print(f"📈 Portfolio Return: {portfolio_return:.1f}%")
        print(f"📊 Total Trades: {len(all_trades)}")
        print(f"🎯 Win Rate: {portfolio_winrate:.0f}%")
        print(f"📊 Avg Trade Return: {avg_trade_return:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ BACKTEST COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
