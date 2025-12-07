"""
📊 DATABASE MODULE - Supabase PostgreSQL (with JSON fallback)

Replaces in-memory storage with persistent database.
This significantly reduces memory usage and prevents data loss on restarts.

Falls back to JSON storage if database is unavailable (for local development).

Tables:
- portfolios: Current portfolio state (cash, last_update)
- positions: Open positions
- trades: Trade history
- signals: Current signals (ephemeral, recalculated)
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import psycopg2
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not available - using JSON storage")

# Database connection string from environment
# Note: Special characters in password must be URL encoded (# = %23)
DATABASE_URL = os.environ.get('DATABASE_URL')

# Fallback to Supabase direct connection if not set
if not DATABASE_URL:
    DATABASE_URL = 'postgresql://postgres:musl1m%23ntre@db.linfgykvqzixrmaxlrho.supabase.co:5432/postgres'

# Database availability flag
DATABASE_AVAILABLE = False

# JSON fallback directory
JSON_DATA_DIR = Path(__file__).parent / "data"
JSON_DATA_DIR.mkdir(exist_ok=True)


def get_connection():
    """Get database connection."""
    global DATABASE_AVAILABLE
    
    if not PSYCOPG2_AVAILABLE:
        DATABASE_AVAILABLE = False
        return None
    
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, connect_timeout=5)
        DATABASE_AVAILABLE = True
        return conn
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        DATABASE_AVAILABLE = False
        return None


def _load_json(filename: str) -> dict:
    """Load data from JSON file."""
    filepath = JSON_DATA_DIR / filename
    if filepath.exists():
        try:
            return json.loads(filepath.read_text())
        except:
            return {}
    return {}


def _save_json(filename: str, data: dict):
    """Save data to JSON file."""
    filepath = JSON_DATA_DIR / filename
    filepath.write_text(json.dumps(data, indent=2, default=str))


def init_database():
    """Initialize database tables (or JSON files)."""
    global DATABASE_AVAILABLE
    
    conn = get_connection()
    
    if conn is None:
        # Fallback to JSON
        logger.info("📁 Using JSON storage (database unavailable)")
        DATABASE_AVAILABLE = False
        
        # Initialize JSON files if needed
        for market in ['crypto', 'stocks']:
            portfolio_file = f"portfolio_{market}.json"
            if not (JSON_DATA_DIR / portfolio_file).exists():
                _save_json(portfolio_file, {
                    'cash': 10000,
                    'positions': {},
                    'trades': [],
                    'signals': {},
                    'last_update': None
                })
        return
    
    DATABASE_AVAILABLE = True
    cur = conn.cursor()
    
    # Create tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id SERIAL PRIMARY KEY,
            market VARCHAR(20) NOT NULL UNIQUE,
            cash DECIMAL(15, 2) DEFAULT 10000,
            last_update TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id SERIAL PRIMARY KEY,
            market VARCHAR(20) NOT NULL,
            pair VARCHAR(50) NOT NULL,
            position_type VARCHAR(10) NOT NULL,
            entry_price DECIMAL(15, 6),
            entry_zscore DECIMAL(10, 4),
            hedge_ratio DECIMAL(15, 6),
            size DECIMAL(15, 2),
            entry_time TIMESTAMP DEFAULT NOW(),
            UNIQUE(market, pair)
        );
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            market VARCHAR(20) NOT NULL,
            pair VARCHAR(50) NOT NULL,
            trade_type VARCHAR(20) NOT NULL,
            zscore DECIMAL(10, 4),
            hedge_ratio DECIMAL(15, 6),
            spread_value DECIMAL(15, 6),
            prices JSONB,
            reason VARCHAR(255),
            pnl DECIMAL(15, 2),
            gatekeeper_prob DECIMAL(5, 4),
            circuit_breaker_status VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY,
            market VARCHAR(20) NOT NULL,
            pair VARCHAR(50) NOT NULL,
            signal_type VARCHAR(20),
            zscore DECIMAL(10, 4),
            hedge_ratio DECIMAL(15, 6),
            prices JSONB,
            gatekeeper_approved BOOLEAN DEFAULT TRUE,
            gatekeeper_reason VARCHAR(255),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(market, pair)
        );
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market);
        CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market);
        CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market);
    """)
    
    # Initialize portfolios if not exist
    cur.execute("""
        INSERT INTO portfolios (market, cash) 
        VALUES ('crypto', 10000), ('stocks', 10000)
        ON CONFLICT (market) DO NOTHING;
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    
    logger.info("✅ Database initialized")


# ============================================================
# PORTFOLIO OPERATIONS
# ============================================================
def get_portfolio(market: str) -> Dict[str, Any]:
    """Get portfolio for a market."""
    # JSON fallback
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        # Ensure all required fields exist (backward compatibility)
        defaults = {
            'cash': 10000,
            'positions': {},
            'trades': [],
            'signals': {},
            'last_update': None
        }
        for key, value in defaults.items():
            if key not in data:
                data[key] = value
        return data
    
    # Database mode
    conn = get_connection()
    if conn is None:
        return get_portfolio(market)  # Retry with JSON
    
    cur = conn.cursor()
    
    # Get portfolio
    cur.execute("SELECT * FROM portfolios WHERE market = %s", (market,))
    portfolio = cur.fetchone()
    
    if not portfolio:
        cur.execute(
            "INSERT INTO portfolios (market, cash) VALUES (%s, 10000) RETURNING *",
            (market,)
        )
        portfolio = cur.fetchone()
        conn.commit()
    
    # Get positions
    cur.execute("SELECT * FROM positions WHERE market = %s", (market,))
    positions_rows = cur.fetchall()
    positions = {
        row['pair']: {
            'type': row['position_type'],
            'entry_price': float(row['entry_price']) if row['entry_price'] else 0,
            'entry_zscore': float(row['entry_zscore']) if row['entry_zscore'] else 0,
            'hedge_ratio': float(row['hedge_ratio']) if row['hedge_ratio'] else 0,
            'size': float(row['size']) if row['size'] else 0,
            'entry_time': row['entry_time'].isoformat() if row['entry_time'] else None
        }
        for row in positions_rows
    }
    
    # Get recent trades
    cur.execute(
        "SELECT * FROM trades WHERE market = %s ORDER BY created_at DESC LIMIT 50",
        (market,)
    )
    trades = [dict(row) for row in cur.fetchall()]
    for t in trades:
        t['time'] = t['created_at'].isoformat() if t['created_at'] else None
        t['prices'] = t['prices'] or {}
    
    # Get signals
    cur.execute("SELECT * FROM signals WHERE market = %s", (market,))
    signals = {
        row['pair']: {
            'signal': row['signal_type'],
            'zscore': float(row['zscore']) if row['zscore'] else 0,
            'hedge_ratio': float(row['hedge_ratio']) if row['hedge_ratio'] else 0,
            'prices': row['prices'] or {},
            'gatekeeper_approved': row['gatekeeper_approved'],
            'gatekeeper_reason': row['gatekeeper_reason']
        }
        for row in cur.fetchall()
    }
    
    cur.close()
    conn.close()
    
    return {
        'cash': float(portfolio['cash']),
        'positions': positions,
        'trades': trades,
        'signals': signals,
        'last_update': portfolio['last_update'].isoformat() if portfolio['last_update'] else None
    }


def update_portfolio_cash(market: str, cash: float):
    """Update portfolio cash."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        data['cash'] = cash
        data['last_update'] = datetime.now().isoformat()
        _save_json(f"portfolio_{market}.json", data)
        return
    
    conn = get_connection()
    if conn is None:
        return update_portfolio_cash(market, cash)
    
    cur = conn.cursor()
    cur.execute(
        "UPDATE portfolios SET cash = %s, last_update = NOW() WHERE market = %s",
        (cash, market)
    )
    conn.commit()
    cur.close()
    conn.close()


def update_last_update(market: str):
    """Update last_update timestamp."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        data['last_update'] = datetime.now().isoformat()
        _save_json(f"portfolio_{market}.json", data)
        return
    
    conn = get_connection()
    if conn is None:
        return update_last_update(market)
    
    cur = conn.cursor()
    cur.execute(
        "UPDATE portfolios SET last_update = NOW() WHERE market = %s",
        (market,)
    )
    conn.commit()
    cur.close()
    conn.close()


# ============================================================
# POSITION OPERATIONS
# ============================================================
def add_position(market: str, pair: str, position_type: str, entry_price: float,
                 entry_zscore: float, hedge_ratio: float, size: float):
    """Add or update a position."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        data['positions'][pair] = {
            'type': position_type,
            'entry_price': entry_price,
            'entry_zscore': entry_zscore,
            'hedge_ratio': hedge_ratio,
            'size': size,
            'entry_time': datetime.now().isoformat()
        }
        _save_json(f"portfolio_{market}.json", data)
        return
    
    conn = get_connection()
    if conn is None:
        return add_position(market, pair, position_type, entry_price, entry_zscore, hedge_ratio, size)
    
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO positions (market, pair, position_type, entry_price, entry_zscore, hedge_ratio, size)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (market, pair) DO UPDATE SET
            position_type = EXCLUDED.position_type,
            entry_price = EXCLUDED.entry_price,
            entry_zscore = EXCLUDED.entry_zscore,
            hedge_ratio = EXCLUDED.hedge_ratio,
            size = EXCLUDED.size,
            entry_time = NOW()
    """, (market, pair, position_type, entry_price, entry_zscore, hedge_ratio, size))
    conn.commit()
    cur.close()
    conn.close()


def remove_position(market: str, pair: str):
    """Remove a position."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        if pair in data.get('positions', {}):
            del data['positions'][pair]
            _save_json(f"portfolio_{market}.json", data)
        return
    
    conn = get_connection()
    if conn is None:
        return remove_position(market, pair)
    
    cur = conn.cursor()
    cur.execute("DELETE FROM positions WHERE market = %s AND pair = %s", (market, pair))
    conn.commit()
    cur.close()
    conn.close()


def get_position(market: str, pair: str) -> Optional[Dict]:
    """Get a specific position."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        return data.get('positions', {}).get(pair)
    
    conn = get_connection()
    if conn is None:
        return get_position(market, pair)
    
    cur = conn.cursor()
    cur.execute("SELECT * FROM positions WHERE market = %s AND pair = %s", (market, pair))
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if row:
        return {
            'type': row['position_type'],
            'entry_price': float(row['entry_price']) if row['entry_price'] else 0,
            'entry_zscore': float(row['entry_zscore']) if row['entry_zscore'] else 0,
            'hedge_ratio': float(row['hedge_ratio']) if row['hedge_ratio'] else 0,
            'size': float(row['size']) if row['size'] else 0,
            'entry_time': row['entry_time'].isoformat() if row['entry_time'] else None
        }
    return None


# ============================================================
# TRADE OPERATIONS
# ============================================================
def add_trade(market: str, pair: str, trade_type: str, zscore: float,
              hedge_ratio: float, spread_value: float, prices: dict,
              reason: str, pnl: float = None, gatekeeper_prob: float = None,
              circuit_breaker_status: str = None):
    """Record a trade."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        trade = {
            'time': datetime.now().isoformat(),
            'pair': pair,
            'type': trade_type,
            'zscore': zscore,
            'hedge_ratio': hedge_ratio,
            'spread_value': spread_value,
            'prices': prices,
            'reason': reason,
            'pnl': pnl,
            'gatekeeper_prob': gatekeeper_prob,
            'circuit_breaker_status': circuit_breaker_status
        }
        if 'trades' not in data:
            data['trades'] = []
        data['trades'].insert(0, trade)
        data['trades'] = data['trades'][:100]  # Keep last 100
        _save_json(f"portfolio_{market}.json", data)
        return
    
    conn = get_connection()
    if conn is None:
        return add_trade(market, pair, trade_type, zscore, hedge_ratio, spread_value, 
                        prices, reason, pnl, gatekeeper_prob, circuit_breaker_status)
    
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trades (market, pair, trade_type, zscore, hedge_ratio, spread_value,
                           prices, reason, pnl, gatekeeper_prob, circuit_breaker_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (market, pair, trade_type, zscore, hedge_ratio, spread_value,
          json.dumps(prices), reason, pnl, gatekeeper_prob, circuit_breaker_status))
    conn.commit()
    cur.close()
    conn.close()


def get_recent_trades(market: str, limit: int = 50) -> List[Dict]:
    """Get recent trades."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        return data.get('trades', [])[:limit]
    
    conn = get_connection()
    if conn is None:
        return get_recent_trades(market, limit)
    
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM trades WHERE market = %s ORDER BY created_at DESC LIMIT %s",
        (market, limit)
    )
    trades = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    
    for t in trades:
        t['time'] = t['created_at'].isoformat() if t['created_at'] else None
    
    return trades


# ============================================================
# SIGNAL OPERATIONS
# ============================================================
def update_signal(market: str, pair: str, signal_type: str, zscore: float,
                  hedge_ratio: float, prices: dict, gatekeeper_approved: bool = True,
                  gatekeeper_reason: str = None):
    """Update or insert a signal."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        if 'signals' not in data:
            data['signals'] = {}
        data['signals'][pair] = {
            'signal': signal_type,
            'zscore': zscore,
            'hedge_ratio': hedge_ratio,
            'prices': prices,
            'gatekeeper_approved': gatekeeper_approved,
            'gatekeeper_reason': gatekeeper_reason
        }
        _save_json(f"portfolio_{market}.json", data)
        return
    
    conn = get_connection()
    if conn is None:
        return update_signal(market, pair, signal_type, zscore, hedge_ratio, prices, 
                            gatekeeper_approved, gatekeeper_reason)
    
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO signals (market, pair, signal_type, zscore, hedge_ratio, prices,
                            gatekeeper_approved, gatekeeper_reason, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (market, pair) DO UPDATE SET
            signal_type = EXCLUDED.signal_type,
            zscore = EXCLUDED.zscore,
            hedge_ratio = EXCLUDED.hedge_ratio,
            prices = EXCLUDED.prices,
            gatekeeper_approved = EXCLUDED.gatekeeper_approved,
            gatekeeper_reason = EXCLUDED.gatekeeper_reason,
            updated_at = NOW()
    """, (market, pair, signal_type, zscore, hedge_ratio, json.dumps(prices),
          gatekeeper_approved, gatekeeper_reason))
    conn.commit()
    cur.close()
    conn.close()


def clear_signals(market: str):
    """Clear all signals for a market."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        data['signals'] = {}
        _save_json(f"portfolio_{market}.json", data)
        return
    
    conn = get_connection()
    if conn is None:
        return clear_signals(market)
    
    cur = conn.cursor()
    cur.execute("DELETE FROM signals WHERE market = %s", (market,))
    conn.commit()
    cur.close()
    conn.close()


def get_signals(market: str) -> Dict[str, Dict]:
    """Get all signals for a market."""
    if not DATABASE_AVAILABLE:
        data = _load_json(f"portfolio_{market}.json")
        return data.get('signals', {})
    
    conn = get_connection()
    if conn is None:
        return get_signals(market)
    
    cur = conn.cursor()
    cur.execute("SELECT * FROM signals WHERE market = %s", (market,))
    signals = {
        row['pair']: {
            'signal': row['signal_type'],
            'zscore': float(row['zscore']) if row['zscore'] else 0,
            'hedge_ratio': float(row['hedge_ratio']) if row['hedge_ratio'] else 0,
            'prices': row['prices'] or {},
            'gatekeeper_approved': row['gatekeeper_approved'],
            'gatekeeper_reason': row['gatekeeper_reason']
        }
        for row in cur.fetchall()
    }
    cur.close()
    conn.close()
    return signals


# ============================================================
# UTILITY
# ============================================================
def get_portfolio_value(market: str) -> float:
    """Get total portfolio value (cash + positions)."""
    portfolio = get_portfolio(market)
    total = portfolio['cash']
    # Positions are marked to market elsewhere
    for pair, pos in portfolio['positions'].items():
        total += pos.get('size', 0)
    return total


def health_check() -> bool:
    """Check database connectivity."""
    if not DATABASE_AVAILABLE:
        return True  # JSON mode always works
    
    try:
        conn = get_connection()
        if conn is None:
            return False
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def get_storage_mode() -> str:
    """Get current storage mode."""
    return "PostgreSQL" if DATABASE_AVAILABLE else "JSON"


if __name__ == "__main__":
    print("🔧 Initializing database...")
    init_database()
    print(f"✅ Storage mode: {get_storage_mode()}")
    
    # Test
    print("\n📊 Testing portfolio operations...")
    portfolio = get_portfolio('crypto')
    print(f"   Crypto cash: ${portfolio.get('cash', 10000):,.2f}")
    print(f"   Positions: {len(portfolio.get('positions', {}))}")
    print(f"   Trades: {len(portfolio.get('trades', []))}")
    print(f"   Signals: {len(portfolio.get('signals', {}))}")

