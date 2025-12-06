"""
Feature Engineering Module for the Gatekeeper Neural Network.

This module extracts sophisticated quantitative features from price/spread data
to feed into the LSTM classifier.

Features:
- Spread Velocity (Rate of change)
- Volume Ratio (vs 20-day average)
- Market Volatility (VIX or rolling std)
- Mean Reversion Speed (Theta from OU process)
- Half-Life of Spread
- Hurst Exponent
- RSI of Spread
- Bollinger Band Width
- Correlation Stability
- Z-Score itself (current and lagged)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Extracts features for the Gatekeeper LSTM model.
    All features are normalized to [0, 1] range.
    """
    
    def __init__(self, lookback: int = 50, normalize: bool = True):
        """
        Args:
            lookback: Number of historical periods to include in sequences
            normalize: Whether to apply MinMaxScaler (CRITICAL for NN!)
        """
        self.lookback = lookback
        self.normalize = normalize
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False
    
    def calculate_spread_velocity(self, spread: pd.Series, window: int = 5) -> pd.Series:
        """
        Rate of change of the spread.
        Positive = spreading apart, Negative = converging.
        """
        return spread.diff(window) / window
    
    def calculate_spread_acceleration(self, spread: pd.Series, window: int = 5) -> pd.Series:
        """
        Second derivative - is the divergence accelerating or decelerating?
        """
        velocity = self.calculate_spread_velocity(spread, window)
        return velocity.diff(window) / window
    
    def calculate_volume_ratio(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Current volume / Average volume.
        High ratio = unusual activity (could be good or bad signal).
        """
        avg_volume = volume.rolling(window=window).mean()
        return volume / avg_volume
    
    def calculate_rolling_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Rolling standard deviation of returns.
        High volatility = more uncertainty.
        """
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def calculate_ou_theta(self, spread: pd.Series, window: int = 50) -> pd.Series:
        """
        Estimate mean-reversion speed (theta) from Ornstein-Uhlenbeck process.
        
        dS = theta * (mu - S) * dt + sigma * dW
        
        Higher theta = faster reversion = better for pairs trading.
        """
        theta_series = pd.Series(index=spread.index, dtype=float)
        
        for i in range(window, len(spread)):
            window_data = spread.iloc[i-window:i].values
            
            # Simple AR(1) estimation: S_t = a + b * S_{t-1} + e
            # theta ≈ -log(b) if |b| < 1
            y = window_data[1:]
            x = window_data[:-1]
            
            if len(x) > 1 and np.std(x) > 1e-10:
                b = np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x)
                if 0 < abs(b) < 1:
                    theta_series.iloc[i] = -np.log(abs(b))
                else:
                    theta_series.iloc[i] = 0.0
            else:
                theta_series.iloc[i] = 0.0
        
        return theta_series.fillna(0)
    
    def calculate_half_life(self, spread: pd.Series, window: int = 50) -> pd.Series:
        """
        Half-life of mean reversion.
        
        Half-life = ln(2) / theta
        
        Shorter half-life = faster reversion = better signal.
        """
        theta = self.calculate_ou_theta(spread, window)
        half_life = np.log(2) / theta.replace(0, np.nan)
        return half_life.clip(upper=100).fillna(100)  # Cap at 100 days
    
    def calculate_hurst_exponent(self, spread: pd.Series, window: int = 50) -> pd.Series:
        """
        Hurst Exponent estimation using R/S analysis.
        
        H < 0.5: Mean-reverting (GOOD for pairs trading)
        H = 0.5: Random walk
        H > 0.5: Trending (BAD for pairs trading)
        """
        hurst_series = pd.Series(index=spread.index, dtype=float)
        
        for i in range(window, len(spread)):
            ts = spread.iloc[i-window:i].values
            
            # R/S Analysis
            mean = np.mean(ts)
            std = np.std(ts)
            
            if std < 1e-10:
                hurst_series.iloc[i] = 0.5
                continue
            
            # Cumulative deviations from mean
            cumdev = np.cumsum(ts - mean)
            R = np.max(cumdev) - np.min(cumdev)
            S = std
            
            if S > 0 and R > 0:
                # H = log(R/S) / log(n)
                RS = R / S
                n = len(ts)
                hurst_series.iloc[i] = np.log(RS) / np.log(n)
            else:
                hurst_series.iloc[i] = 0.5
        
        return hurst_series.clip(0, 1).fillna(0.5)
    
    def calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index of the spread.
        
        RSI < 30: Oversold (expect upward reversion)
        RSI > 70: Overbought (expect downward reversion)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) / 100  # Normalize to [0, 1]
    
    def calculate_bollinger_width(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Bollinger Band Width = (Upper - Lower) / Middle
        
        High width = high volatility regime
        Low width = low volatility regime (potential breakout coming)
        """
        middle = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        
        width = (upper - lower) / middle.replace(0, np.nan)
        return width.fillna(0)
    
    def calculate_correlation_stability(
        self, 
        price1: pd.Series, 
        price2: pd.Series, 
        window: int = 20,
        lookback: int = 5
    ) -> pd.Series:
        """
        Rolling standard deviation of correlation.
        
        High instability = relationship is breaking down = DANGER.
        """
        rolling_corr = price1.rolling(window=window).corr(price2)
        corr_std = rolling_corr.rolling(window=lookback).std()
        return corr_std.fillna(0)
    
    def calculate_zscore_momentum(self, zscore: pd.Series, window: int = 5) -> pd.Series:
        """
        Momentum of z-score.
        
        If z-score is increasing rapidly, the divergence might continue.
        """
        return zscore.diff(window)
    
    def calculate_time_in_signal(self, zscore: pd.Series, threshold: float = 2.0) -> pd.Series:
        """
        How many consecutive days has z-score been above/below threshold?
        
        Longer time in signal = might be structural change, not temporary.
        """
        above_threshold = (abs(zscore) > threshold).astype(int)
        
        # Count consecutive days
        time_in_signal = pd.Series(index=zscore.index, dtype=float)
        count = 0
        
        for i in range(len(above_threshold)):
            if above_threshold.iloc[i] == 1:
                count += 1
            else:
                count = 0
            time_in_signal.iloc[i] = count
        
        return time_in_signal / 20  # Normalize by typical max
    
    def extract_features(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        price1: pd.Series,
        price2: pd.Series,
        volume1: Optional[pd.Series] = None,
        volume2: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Extract all features for the Gatekeeper model.
        
        Returns:
            DataFrame with all features, normalized to [0, 1] if self.normalize=True
        """
        features = pd.DataFrame(index=spread.index)
        
        # Core Features
        features['zscore'] = zscore
        features['zscore_abs'] = abs(zscore)
        features['zscore_momentum'] = self.calculate_zscore_momentum(zscore)
        features['spread_velocity'] = self.calculate_spread_velocity(spread)
        features['spread_acceleration'] = self.calculate_spread_acceleration(spread)
        
        # Volume Features (if available)
        if volume1 is not None and volume2 is not None:
            combined_volume = volume1 + volume2
            features['volume_ratio'] = self.calculate_volume_ratio(combined_volume)
        else:
            features['volume_ratio'] = 1.0
        
        # Volatility Features
        returns = spread.pct_change().replace([np.inf, -np.inf], 0)
        features['volatility'] = self.calculate_rolling_volatility(returns)
        features['bollinger_width'] = self.calculate_bollinger_width(spread)
        
        # VIX (Market Volatility) if available
        if vix is not None:
            features['vix'] = vix / 100  # Normalize VIX
        else:
            features['vix'] = features['volatility']  # Use spread vol as proxy
        
        # Mean Reversion Features
        features['ou_theta'] = self.calculate_ou_theta(spread)
        features['half_life'] = self.calculate_half_life(spread)
        features['hurst'] = self.calculate_hurst_exponent(spread)
        
        # Technical Indicators
        features['rsi'] = self.calculate_rsi(spread)
        
        # Correlation Stability
        features['corr_stability'] = self.calculate_correlation_stability(price1, price2)
        
        # Time Features
        features['time_in_signal'] = self.calculate_time_in_signal(zscore)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        # Replace infinities
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def fit_scaler(self, features: pd.DataFrame):
        """Fit the MinMaxScaler on training data."""
        self.scaler.fit(features.values)
        self.fitted = True
    
    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """
        Transform features to normalized values.
        
        CRITICAL: Neural networks are stupid about scale!
        """
        if not self.normalize:
            return features.values
        
        if not self.fitted:
            raise ValueError("Scaler not fitted! Call fit_scaler() first.")
        
        return self.scaler.transform(features.values)
    
    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform in one step."""
        if self.normalize:
            self.fit_scaler(features)
            return self.transform(features)
        return features.values
    
    def create_sequences(
        self, 
        features: np.ndarray, 
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input.
        
        Args:
            features: (N, num_features) array
            labels: (N,) array of binary labels
        
        Returns:
            X: (N - lookback, lookback, num_features)
            y: (N - lookback,) if labels provided
        """
        X = []
        y = [] if labels is not None else None
        
        for i in range(self.lookback, len(features)):
            X.append(features[i - self.lookback:i])
            if labels is not None:
                y.append(labels[i])
        
        X = np.array(X)
        y = np.array(y) if labels is not None else None
        
        return X, y


def get_feature_names() -> list:
    """Return list of feature names in order."""
    return [
        'zscore',
        'zscore_abs', 
        'zscore_momentum',
        'spread_velocity',
        'spread_acceleration',
        'volume_ratio',
        'volatility',
        'bollinger_width',
        'vix',
        'ou_theta',
        'half_life',
        'hurst',
        'rsi',
        'corr_stability',
        'time_in_signal'
    ]


if __name__ == "__main__":
    # Test feature extraction
    import yfinance as yf
    
    print("Testing Feature Engineering Module...")
    
    # Download test data
    data = yf.download(['AAPL', 'MSFT'], period='1y', progress=False)
    price1 = data['Close']['AAPL']
    price2 = data['Close']['MSFT']
    volume1 = data['Volume']['AAPL']
    volume2 = data['Volume']['MSFT']
    
    # Calculate spread and z-score
    spread = price1 - 0.5 * price2  # Simple hedge ratio
    zscore = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
    
    # Extract features
    fe = FeatureEngineer(lookback=50, normalize=True)
    features = fe.extract_features(spread, zscore, price1, price2, volume1, volume2)
    
    print(f"\nExtracted {len(features.columns)} features:")
    for col in features.columns:
        print(f"  - {col}: range [{features[col].min():.3f}, {features[col].max():.3f}]")
    
    # Test normalization
    normalized = fe.fit_transform(features)
    print(f"\nNormalized shape: {normalized.shape}")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Create sequences
    X, _ = fe.create_sequences(normalized)
    print(f"\nSequence shape: {X.shape}")
    print("✅ Feature Engineering Module Working!")

