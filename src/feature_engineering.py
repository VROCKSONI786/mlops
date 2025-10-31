import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create technical indicators and features for stock prediction"""
    
    def __init__(self):
        self.feature_columns = None
        
    def create_technical_indicators(self, df):
        """
        Create technical indicators from stock data
        
        Args:
            df: DataFrame with OHLCV data
        """
        df = df.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # RSI (Relative Strength Index)
        df['RSI'] = self.calculate_rsi(df['Close'], period=14)
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Volume-based features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Price momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # High-Low range
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Lag features
        for i in [1, 2, 3, 5]:
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
            df[f'Returns_Lag_{i}'] = df['Returns'].shift(i)
        
        # Target variable: 1 if price goes up tomorrow, 0 if down
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        return df
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_features(self, df, drop_na=True):
        """
        Prepare final feature set for training
        
        Args:
            df: DataFrame with technical indicators
            drop_na: Whether to drop rows with NaN values
        """
        # Select feature columns (exclude target and date)
        exclude_cols = ['Date', 'Target', 'Dividends', 'Stock Splits']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Store original length
        original_len = len(df)
        
        if drop_na:
            df = df.dropna()
            logger.info(f"Dropped {original_len - len(df)} rows with NaN values")
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df['Target'] if 'Target' in df.columns else None
        
        logger.info(f"Feature shape: {X.shape}")
        if y is not None:
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, df
    
    def get_feature_names(self):
        """Get list of feature column names"""
        return self.feature_columns


if __name__ == "__main__":
    # Test feature engineering
    df = pd.read_csv('data/raw/stock_data.csv')
    
    engineer = FeatureEngineer()
    df_features = engineer.create_technical_indicators(df)
    X, y, df_processed = engineer.prepare_features(df_features)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns:\n{engineer.get_feature_names()}")