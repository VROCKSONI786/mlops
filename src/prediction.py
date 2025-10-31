import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    """Make predictions using trained model"""
    
    def __init__(self, model_path='models', model_name='stock_model'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to model directory
            model_name: Base name of model files
        """
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
        
    def load_model(self):
        """Load model, scaler, and feature names"""
        try:
            model_file = os.path.join(self.model_path, f'{self.model_name}.pkl')
            scaler_file = os.path.join(self.model_path, f'{self.model_name}_scaler.pkl')
            feature_file = os.path.join(self.model_path, f'{self.model_name}_features.pkl')
            
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            self.feature_names = joblib.load(feature_file)
            
            logger.info(f"Model loaded successfully from {model_file}")
            logger.info(f"Model expects {len(self.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Make predictions on feature data
        
        Args:
            X: Feature DataFrame or array
            
        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            # Ensure X has correct columns
            if isinstance(X, pd.DataFrame):
                # Reorder columns to match training
                X = X[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            results = {
                'prediction': int(predictions[0]),
                'prediction_label': 'UP' if predictions[0] == 1 else 'DOWN',
                'probability_down': float(probabilities[0][0]),
                'probability_up': float(probabilities[0][1]),
                'confidence': float(max(probabilities[0])),
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_from_stock_data(self, df_features):
        """
        Make prediction from processed stock data
        
        Args:
            df_features: DataFrame with technical indicators
            
        Returns:
            Prediction results
        """
        # Get the most recent data point
        if df_features.empty:
            raise ValueError("No data available for prediction")
        
        latest_data = df_features.iloc[[-1]].copy()
        
        # Select only feature columns and fill any NaN
        X = latest_data[self.feature_names]
        
        # Check for NaN values and fill them
        if X.isnull().any().any():
            logger.warning("NaN values detected in features, filling with 0")
            X = X.fillna(0)
        
        return self.predict(X)
    
    def batch_predict(self, X):
        """
        Make predictions for multiple samples
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Ensure correct column order
            if isinstance(X, pd.DataFrame):
                X = X[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            results = []
            for i in range(len(predictions)):
                result = {
                    'prediction': int(predictions[i]),
                    'prediction_label': 'UP' if predictions[i] == 1 else 'DOWN',
                    'probability_down': float(probabilities[i][0]),
                    'probability_up': float(probabilities[i][1]),
                    'confidence': float(max(probabilities[i]))
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df.to_dict('records')
        else:
            return []


if __name__ == "__main__":
    # Test prediction
    from data_collection import StockDataCollector
    from feature_engineering import FeatureEngineer
    
    # Collect data
    collector = StockDataCollector(ticker='AAPL')
    df = collector.fetch_real_time_data(days=60)
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.create_technical_indicators(df)
    X, _, df_processed = engineer.prepare_features(df_features, drop_na=True)
    
    # Make prediction
    predictor = StockPredictor()
    result = predictor.predict_from_stock_data(df_processed)
    
    print(f"\nPrediction Results:")
    print(f"Prediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probability UP: {result['probability_up']:.2%}")
    print(f"Probability DOWN: {result['probability_down']:.2%}")