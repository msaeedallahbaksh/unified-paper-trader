"""
🚨 CIRCUIT BREAKER - Hybrid Gatekeeper System

The smart money move:
- Normal Market (VIX < 20): Gatekeeper OFF → Maximize profit
- Stress Market (VIX > 20): Gatekeeper ON → Safety mode

Additional triggers:
- BTC drops 5% in a day → Enable Gatekeeper
- ETH drops 7% in a day → Enable Gatekeeper
- S&P 500 drops 3% in a day → Enable Gatekeeper

"A race car without brakes wins the race fastest.
But the brakes stop you from hitting the wall."
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
VIX_THRESHOLD = 20.0  # Normal < 20, Stress > 20
BTC_CRASH_THRESHOLD = -0.05  # -5% in a day
ETH_CRASH_THRESHOLD = -0.07  # -7% in a day
SPY_CRASH_THRESHOLD = -0.03  # -3% in a day

# Cache to avoid hitting API too often
_cache = {}
CACHE_DURATION = timedelta(minutes=5)


def _is_cache_valid(key):
    """Check if cached value is still valid."""
    time_key = f"{key}_time"
    if time_key not in _cache or _cache[time_key] is None:
        return False
    return datetime.now() - _cache[time_key] < CACHE_DURATION


def get_vix():
    """Get current VIX level."""
    if _is_cache_valid('vix'):
        return _cache['vix']
    
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period='1d')
        if not hist.empty:
            _cache['vix'] = hist['Close'].iloc[-1]
            _cache['vix_time'] = datetime.now()
            return _cache['vix']
    except Exception as e:
        logger.warning(f"Failed to get VIX: {e}")
    
    return 15.0  # Default to normal market


def get_daily_change(symbol):
    """Get daily percentage change for a symbol."""
    cache_key = symbol.lower().replace('-', '_').replace('^', '')
    
    if _is_cache_valid(cache_key):
        return _cache[cache_key]
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='2d')
        if len(hist) >= 2:
            yesterday = hist['Close'].iloc[-2]
            today = hist['Close'].iloc[-1]
            change = (today - yesterday) / yesterday
            _cache[cache_key] = change
            _cache[f"{cache_key}_time"] = datetime.now()
            return change
    except Exception as e:
        logger.warning(f"Failed to get {symbol} change: {e}")
    
    return 0.0  # Default to no change


class CircuitBreaker:
    """
    Circuit Breaker for the trading system.
    
    Monitors market conditions and activates/deactivates the Gatekeeper
    based on stress indicators.
    """
    
    def __init__(self):
        self.vix_threshold = VIX_THRESHOLD
        self.btc_crash_threshold = BTC_CRASH_THRESHOLD
        self.eth_crash_threshold = ETH_CRASH_THRESHOLD
        self.spy_crash_threshold = SPY_CRASH_THRESHOLD
        self.manual_override = None  # None = auto, True = force on, False = force off
        
        self.last_check = None
        self.last_status = None
        self.triggers = []
    
    def check_conditions(self):
        """
        Check all market stress conditions.
        
        Returns:
            dict with:
            - gatekeeper_enabled: bool
            - triggers: list of active triggers
            - market_status: 'NORMAL' or 'STRESS'
            - details: dict of indicator values
        """
        self.triggers = []
        
        # Manual override
        if self.manual_override is not None:
            return {
                'gatekeeper_enabled': self.manual_override,
                'triggers': ['MANUAL_OVERRIDE'],
                'market_status': 'OVERRIDE',
                'details': {}
            }
        
        # Get indicators
        vix = get_vix()
        btc_change = get_daily_change('BTC-USD')
        eth_change = get_daily_change('ETH-USD')
        spy_change = get_daily_change('SPY')
        
        details = {
            'vix': vix,
            'btc_change': btc_change * 100,  # Convert to %
            'eth_change': eth_change * 100,
            'spy_change': spy_change * 100,
        }
        
        # Check triggers
        if vix > self.vix_threshold:
            self.triggers.append(f'VIX_HIGH ({vix:.1f} > {self.vix_threshold})')
        
        if btc_change < self.btc_crash_threshold:
            self.triggers.append(f'BTC_CRASH ({btc_change*100:.1f}%)')
        
        if eth_change < self.eth_crash_threshold:
            self.triggers.append(f'ETH_CRASH ({eth_change*100:.1f}%)')
        
        if spy_change < self.spy_crash_threshold:
            self.triggers.append(f'SPY_CRASH ({spy_change*100:.1f}%)')
        
        # Determine status
        gatekeeper_enabled = len(self.triggers) > 0
        market_status = 'STRESS' if gatekeeper_enabled else 'NORMAL'
        
        self.last_check = datetime.now()
        self.last_status = {
            'gatekeeper_enabled': gatekeeper_enabled,
            'triggers': self.triggers.copy(),
            'market_status': market_status,
            'details': details
        }
        
        return self.last_status
    
    def should_use_gatekeeper(self):
        """
        Simple boolean check for whether to use Gatekeeper.
        
        Returns:
            True if Gatekeeper should be enabled (stress mode)
            False if Gatekeeper should be disabled (normal mode)
        """
        status = self.check_conditions()
        return status['gatekeeper_enabled']
    
    def get_status_string(self):
        """Get human-readable status string."""
        if self.last_status is None:
            self.check_conditions()
        
        status = self.last_status
        
        if status['market_status'] == 'NORMAL':
            return "🟢 NORMAL - Gatekeeper OFF (Full Speed)"
        elif status['market_status'] == 'STRESS':
            triggers = ', '.join(status['triggers'])
            return f"🔴 STRESS - Gatekeeper ON ({triggers})"
        else:
            return "⚪ OVERRIDE - Manual Control"
    
    def force_gatekeeper(self, enabled):
        """Manually override Gatekeeper status."""
        self.manual_override = enabled
        logger.info(f"Gatekeeper manually {'enabled' if enabled else 'disabled'}")
    
    def reset_override(self):
        """Reset to automatic mode."""
        self.manual_override = None
        logger.info("Gatekeeper reset to automatic mode")


# Global instance
circuit_breaker = CircuitBreaker()


def get_circuit_breaker():
    """Get the global circuit breaker instance."""
    return circuit_breaker


if __name__ == "__main__":
    print("="*70)
    print("🚨 CIRCUIT BREAKER TEST")
    print("="*70)
    
    cb = CircuitBreaker()
    status = cb.check_conditions()
    
    print(f"\n{cb.get_status_string()}")
    print(f"\nDetails:")
    for key, value in status['details'].items():
        print(f"   {key}: {value:.2f}")
    
    print(f"\nGatekeeper Enabled: {status['gatekeeper_enabled']}")
    print(f"Triggers: {status['triggers']}")

