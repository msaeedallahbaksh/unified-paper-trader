"""
Inference Module for the Gatekeeper LSTM.

This module provides the interface between the trading bot and the trained model.

Usage:
    gatekeeper = Gatekeeper('models/gatekeeper.pth')
    
    # Before executing a trade:
    probability = gatekeeper.predict(features)
    
    if probability > 0.7:
        EXECUTE_TRADE()
    else:
        PASS()
"""

import torch
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple, Dict
import os

import torch.nn as nn

# Simplified model matching training
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
    def forward(self, lstm_output):
        scores = self.attention(lstm_output).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights

class GatekeeperLSTM(nn.Module):
    """V3 optimized architecture - smaller and better regularized"""
    def __init__(self, input_size=15, hidden_size=32, num_layers=1, dropout=0.5, **kwargs):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 16)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 1)
    
    def forward(self, x, return_attention=False):
        b, s, f = x.shape
        x = x.transpose(1, 2)
        x = self.bn_input(x)
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        context, attn = self.attention(lstm_out)
        context = self.bn1(context)
        out = torch.relu(self.fc1(context))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        if return_attention:
            return out, attn
        return out, attn

def load_model(path: str, device: str = 'cpu') -> GatekeeperLSTM:
    """Load a trained model from disk."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    model = GatekeeperLSTM(
        input_size=config['input_size'],
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
from features import FeatureEngineer, get_feature_names


class Gatekeeper:
    """
    The Gatekeeper: Neural network filter for trade signals.
    
    This class wraps the trained LSTM model and provides a simple interface
    for making trading decisions.
    
    Usage:
        gatekeeper = Gatekeeper('models/gatekeeper.pth')
        
        # Get probability that signal will be profitable
        prob = gatekeeper.should_trade(spread, zscore, price1, price2)
        
        if prob > 0.7:
            execute_trade()
    """
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str = None,
        threshold: float = 0.7,
        device: str = None
    ):
        """
        Args:
            model_path: Path to saved model (gatekeeper.pth)
            scaler_path: Path to saved scaler (scaler.pkl)
            threshold: Probability threshold for trade execution
            device: 'cuda' or 'cpu'
        """
        self.threshold = threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"🧠 Loading Gatekeeper from {model_path}...")
        self.model = load_model(model_path, self.device)
        self.model.eval()
        
        # Load scaler
        if scaler_path is None:
            scaler_path = model_path.replace('gatekeeper.pth', 'scaler.pkl')
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"📏 Loaded scaler from {scaler_path}")
        else:
            self.scaler = None
            print("⚠️ No scaler found - features should be pre-normalized!")
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer(lookback=50, normalize=False)
        
        # Get model config
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        self.metrics = checkpoint.get('metrics', {})
        
        print(f"✅ Gatekeeper loaded!")
        print(f"   Threshold: {self.threshold}")
        print(f"   Model AUC: {self.metrics.get('auc', 'N/A')}")
    
    def extract_features_from_data(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        price1: pd.Series,
        price2: pd.Series,
        volume1: Optional[pd.Series] = None,
        volume2: Optional[pd.Series] = None
    ) -> Optional[np.ndarray]:
        """
        Extract features from raw market data.
        
        Returns feature sequence ready for model input.
        """
        if len(spread) < 50:
            return None
        
        features = self.feature_engineer.extract_features(
            spread, zscore, price1, price2, volume1, volume2
        )
        
        # Get last 50 time steps
        feature_seq = features.iloc[-50:].values
        
        if len(feature_seq) != 50:
            return None
        
        return feature_seq
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using the saved scaler."""
        if self.scaler is None:
            return features
        
        # Reshape for scaler
        original_shape = features.shape
        flat = features.reshape(-1, original_shape[-1])
        normalized = self.scaler.transform(flat)
        return normalized.reshape(original_shape)
    
    def predict_probability(self, features: np.ndarray) -> float:
        """
        Get probability that the signal will be profitable.
        
        Args:
            features: Feature sequence (50, num_features) or (1, 50, num_features)
        
        Returns:
            Probability between 0 and 1
        """
        # Ensure 3D shape
        if len(features.shape) == 2:
            features = features[np.newaxis, ...]  # Add batch dimension
        
        # Normalize
        features = self.normalize_features(features)
        
        # Convert to tensor
        x = torch.FloatTensor(features).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            probability, attention = self.model(x, return_attention=True)
        
        return probability.item()
    
    def should_trade(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        price1: pd.Series,
        price2: pd.Series,
        volume1: Optional[pd.Series] = None,
        volume2: Optional[pd.Series] = None,
        return_probability: bool = False
    ) -> bool:
        """
        Main interface: Should we execute this trade?
        
        Args:
            spread: Spread time series
            zscore: Z-score time series
            price1: Price of asset 1
            price2: Price of asset 2
            volume1: Volume of asset 1 (optional)
            volume2: Volume of asset 2 (optional)
            return_probability: Return probability instead of bool
        
        Returns:
            True if trade should be executed, False otherwise
            (or probability if return_probability=True)
        """
        # Extract features
        features = self.extract_features_from_data(
            spread, zscore, price1, price2, volume1, volume2
        )
        
        if features is None:
            # Not enough data - default to not trading
            return 0.0 if return_probability else False
        
        # Get probability
        probability = self.predict_probability(features)
        
        if return_probability:
            return probability
        
        return probability > self.threshold
    
    def get_position_size_multiplier(self, probability: float) -> float:
        """
        Convert probability to position size multiplier.
        
        Higher confidence = larger position.
        
        Example:
            prob 0.7 -> 0.5x position
            prob 0.8 -> 0.75x position
            prob 0.9 -> 1.0x position
        
        Returns:
            Multiplier between 0 and 1
        """
        if probability < self.threshold:
            return 0.0
        
        # Linear scaling from threshold to 1.0
        # threshold -> 0.25, 1.0 -> 1.0
        multiplier = 0.25 + 0.75 * (probability - self.threshold) / (1 - self.threshold)
        return min(1.0, max(0.0, multiplier))
    
    def predict_with_explanation(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        price1: pd.Series,
        price2: pd.Series,
        volume1: Optional[pd.Series] = None,
        volume2: Optional[pd.Series] = None
    ) -> Dict:
        """
        Get prediction with feature importance explanation.
        
        Returns dict with probability and feature analysis.
        """
        features = self.extract_features_from_data(
            spread, zscore, price1, price2, volume1, volume2
        )
        
        if features is None:
            return {'error': 'Insufficient data'}
        
        # Get prediction
        probability = self.predict_probability(features)
        decision = probability > self.threshold
        
        # Get current feature values
        current_features = features[-1]  # Last time step
        feature_names = get_feature_names()
        
        # Build explanation
        explanation = {
            'probability': probability,
            'decision': 'EXECUTE' if decision else 'PASS',
            'threshold': self.threshold,
            'confidence': abs(probability - 0.5) * 2,  # 0 to 1
            'position_multiplier': self.get_position_size_multiplier(probability),
            'features': {
                name: float(current_features[i]) 
                for i, name in enumerate(feature_names) 
                if i < len(current_features)
            }
        }
        
        # Key factors
        explanation['key_factors'] = []
        
        if current_features[0] > 0:  # zscore positive
            explanation['key_factors'].append(f"Z-Score: +{current_features[0]:.2f} (SELL signal)")
        else:
            explanation['key_factors'].append(f"Z-Score: {current_features[0]:.2f} (BUY signal)")
        
        if len(current_features) > 11 and current_features[11] < 0.5:  # hurst
            explanation['key_factors'].append(f"Hurst: {current_features[11]:.2f} (Mean-reverting ✓)")
        elif len(current_features) > 11:
            explanation['key_factors'].append(f"Hurst: {current_features[11]:.2f} (Trending ✗)")
        
        return explanation


class MockGatekeeper:
    """
    Mock Gatekeeper for testing without a trained model.
    
    Always returns a configurable probability.
    """
    
    def __init__(self, default_probability: float = 0.75):
        self.default_probability = default_probability
        self.threshold = 0.7
    
    def should_trade(self, *args, return_probability: bool = False, **kwargs):
        if return_probability:
            return self.default_probability
        return self.default_probability > self.threshold
    
    def predict_probability(self, features: np.ndarray) -> float:
        return self.default_probability


def integrate_with_trader(gatekeeper: Gatekeeper, signal: dict, market_data: dict) -> dict:
    """
    Integration helper for the trading bot.
    
    Args:
        gatekeeper: Trained Gatekeeper instance
        signal: Trade signal from the bot {'pair': ..., 'zscore': ..., 'action': ...}
        market_data: Market data dict with prices and spread
    
    Returns:
        Modified signal with Gatekeeper decision
    """
    if signal['action'] in ['NO_SIGNAL', 'HOLDING']:
        return signal
    
    # Get Gatekeeper prediction
    explanation = gatekeeper.predict_with_explanation(
        spread=market_data['spread'],
        zscore=market_data['zscore'],
        price1=market_data['price1'],
        price2=market_data['price2'],
        volume1=market_data.get('volume1'),
        volume2=market_data.get('volume2')
    )
    
    # Add to signal
    signal['gatekeeper_probability'] = explanation['probability']
    signal['gatekeeper_decision'] = explanation['decision']
    signal['position_multiplier'] = explanation['position_multiplier']
    
    # Override action if Gatekeeper says PASS
    if explanation['decision'] == 'PASS':
        signal['original_action'] = signal['action']
        signal['action'] = 'BLOCKED_BY_GATEKEEPER'
    
    return signal


if __name__ == "__main__":
    print("="*60)
    print("🔮 GATEKEEPER INFERENCE TEST")
    print("="*60)
    
    # Check if model exists
    model_path = 'models/gatekeeper.pth'
    
    if not os.path.exists(model_path):
        print(f"\n⚠️ Model not found at {model_path}")
        print("Using MockGatekeeper for testing...")
        gatekeeper = MockGatekeeper(default_probability=0.75)
    else:
        gatekeeper = Gatekeeper(model_path, threshold=0.7)
    
    # Test with synthetic data
    print("\n🧪 Testing with synthetic data...")
    
    import yfinance as yf
    
    # Download test data
    data = yf.download(['AAPL', 'MSFT'], period='3mo', progress=False)
    price1 = data['Close']['AAPL']
    price2 = data['Close']['MSFT']
    
    spread = price1 - 0.5 * price2
    zscore = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
    
    # Test should_trade
    if isinstance(gatekeeper, Gatekeeper):
        result = gatekeeper.predict_with_explanation(
            spread, zscore, price1, price2
        )
        
        print(f"\n📊 Prediction Results:")
        print(f"   Probability: {result['probability']:.3f}")
        print(f"   Decision: {result['decision']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Position Multiplier: {result['position_multiplier']:.2f}x")
        
        print(f"\n🔑 Key Factors:")
        for factor in result['key_factors']:
            print(f"   • {factor}")
    else:
        prob = gatekeeper.should_trade(
            spread, zscore, price1, price2, return_probability=True
        )
        print(f"   Mock probability: {prob:.3f}")
    
    print("\n✅ Inference module test complete!")

