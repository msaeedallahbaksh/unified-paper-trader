"""Quick script to check recent trades."""
import sys
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except:
    pass

import requests
import json
from datetime import datetime, timedelta

SEP = "=" * 70

print(SEP)
print("TRADE ANALYSIS - Last 2 Days")
print(SEP)

# Get trades for both markets
for market in ['crypto', 'stocks']:
    print(f"\n{SEP}")
    print(f"{market.upper()} TRADES")
    print(SEP)
    
    r = requests.get(f'https://unified-paper-trader.onrender.com/api/trades/{market}', timeout=30)
    trades = r.json()
    
    # Filter last 2 days
    cutoff = datetime.now() - timedelta(days=2)
    recent = []
    for t in trades:
        time_str = t.get('time') or t.get('created_at') or ''
        try:
            if 'GMT' in str(time_str):
                dt = datetime.strptime(time_str, '%a, %d %b %Y %H:%M:%S GMT')
            else:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00').replace('+00:00', ''))
            if dt > cutoff:
                recent.append(t)
        except:
            recent.append(t)
    
    if not recent:
        print('  No trades in last 2 days')
        continue
    
    # Analyze
    closes = [t for t in recent if (t.get('trade_type') or t.get('type', '')).upper() == 'CLOSE']
    opens = [t for t in recent if (t.get('trade_type') or t.get('type', '')).upper() in ('BUY_SPREAD', 'SELL_SPREAD')]
    
    print(f'  Total recent trades: {len(recent)}')
    print(f'  Opens: {len(opens)}, Closes: {len(closes)}')
    
    if closes:
        pnls = [float(t.get('pnl', 0) or 0) for t in closes]
        wins = len([p for p in pnls if p > 0])
        total_pnl = sum(pnls)
        
        print(f'\n  CLOSED TRADES SUMMARY:')
        print(f'  Total PnL: ${total_pnl:,.2f}')
        print(f'  Wins: {wins}/{len(closes)} ({100*wins/len(closes):.0f}% win rate)')
        if wins > 0:
            avg_win = sum(p for p in pnls if p > 0) / wins
            print(f'  Avg Win: ${avg_win:,.2f}')
        if len(closes) - wins > 0:
            avg_loss = sum(p for p in pnls if p <= 0) / (len(closes) - wins)
            print(f'  Avg Loss: ${avg_loss:,.2f}')
        
        print(f'\n  Recent closed trades:')
        for t in closes[:15]:
            pair = t.get('pair', '?')
            pnl = float(t.get('pnl', 0) or 0)
            reason = t.get('reason', '')
            # Extract close reason
            if 'Closed:' in reason:
                reason = reason.split('Closed:')[1].split('|')[0].strip()
            elif 'EXIT_Z' in reason or 'STOP' in reason or 'TAKE' in reason or 'TIME' in reason:
                reason = reason.split('|')[0].strip() if '|' in reason else reason[:30]
            else:
                reason = reason[:30]
            time_str = (t.get('time') or t.get('created_at', '?'))[:19]
            icon = '✓' if pnl > 0 else '✗' if pnl < 0 else ' '
            print(f'    {icon} {time_str} | {pair:12} | ${pnl:>10,.2f} | {reason}')

# Portfolio summary
print(f"\n{SEP}")
print("CURRENT PORTFOLIO STATUS")
print(SEP)

for market in ['crypto', 'stocks']:
    r = requests.get(f'https://unified-paper-trader.onrender.com/api/portfolio/{market}', timeout=30)
    d = r.json()
    pnl = d.get('pnl', 0)
    positions = d.get('positions', {})
    print(f"\n{market.upper()}:")
    print(f"  Total Value: ${d.get('total_value', 0):,.2f}")
    print(f"  Cash: ${d.get('cash', 0):,.2f}")
    print(f"  PnL: ${pnl:,.2f} ({d.get('pnl_pct', 0):.2f}%)")
    print(f"  Win Rate: {d.get('win_rate', 0):.1f}%")
    print(f"  Open Positions: {len(positions)}")
    
    if positions:
        print(f"  Current positions:")
        for pair, pos in positions.items():
            ptype = pos.get('type', '?')
            entry_z = pos.get('entry_zscore', 0)
            upnl = pos.get('unrealized_pnl', 0)
            print(f"    {pair}: {ptype} | Entry Z: {entry_z:.2f} | Unrealized: ${upnl:,.2f}")
