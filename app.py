#!/usr/bin/env python3
"""
Flask REST API for real-time stock price prediction
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import StockDataCollector
from feature_engineering import FeatureEngineer
from prediction import StockPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = None

def load_predictor():
    """Load the trained model"""
    global predictor
    try:
        predictor = StockPredictor(model_path='models', model_name='stock_model')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        predictor = None

# Load model on startup
load_predictor()


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make prediction for a stock ticker
    
    Request body:
    {
        "ticker": "AAPL",
        "days": 60  # optional, days of historical data to use
    }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get request data
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL').upper()
        days = data.get('days', 100)  # Increased to 100 for better feature calculation
        
        logger.info(f"Prediction request for {ticker}")
        
        # Collect real-time data
        collector = StockDataCollector(ticker=ticker)
        df = collector.fetch_real_time_data(days=days)
        
        if df.empty:
            return jsonify({
                'error': f'No data available for ticker {ticker}'
            }), 404
        
        logger.info(f"Collected {len(df)} records for {ticker}")
        
        # Get stock info
        stock_info = collector.get_stock_info()
        
        # Engineer features
        engineer = FeatureEngineer()
        df_features = engineer.create_technical_indicators(df)
        
        # Prepare features - keep the last valid row even with some NaN
        X_all, _, df_processed = engineer.prepare_features(df_features, drop_na=False)
        
        # Fill any remaining NaN with forward fill then backward fill
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        # Drop rows that still have NaN after filling
        df_processed = df_processed.dropna()
        
        if df_processed.empty or len(df_processed) < 1:
            return jsonify({
                'error': 'Insufficient data for prediction. Try a ticker with more trading history.'
            }), 400
        
        # Make prediction
        result = predictor.predict_from_stock_data(df_processed)
        
        # Get latest stock data
        latest = df_processed.iloc[-1]
        
        # Prepare response
        response = {
            'ticker': ticker,
            'stock_name': stock_info.get('name', 'N/A'),
            'current_price': float(latest['Close']),
            'prediction': result['prediction_label'],
            'confidence': round(result['confidence'] * 100, 2),
            'probability_up': round(result['probability_up'] * 100, 2),
            'probability_down': round(result['probability_down'] * 100, 2),
            'latest_data': {
                'date': str(latest['Date']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'close': float(latest['Close']),
                'volume': int(latest['Volume'])
            },
            'technical_indicators': {
                'sma_20': float(latest.get('SMA_20', 0)),
                'rsi': float(latest.get('RSI', 0)),
                'macd': float(latest.get('MACD', 0)),
                'volatility': float(latest.get('Volatility', 0))
            },
            'timestamp': result['timestamp']
        }
        
        logger.info(f"Prediction: {result['prediction_label']} with {result['confidence']:.2%} confidence")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple tickers
    
    Request body:
    {
        "tickers": ["AAPL", "GOOGL", "MSFT"]
    }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({
                'error': 'No tickers provided'
            }), 400
        
        results = []
        
        for ticker in tickers:
            try:
                # Make prediction for each ticker
                collector = StockDataCollector(ticker=ticker)
                df = collector.fetch_real_time_data(days=100)
                
                engineer = FeatureEngineer()
                df_features = engineer.create_technical_indicators(df)
                X, _, df_processed = engineer.prepare_features(df_features, drop_na=False)
                
                # Fill NaN values
                df_processed = df_processed.fillna(method='ffill').fillna(method='bfill').dropna()
                
                if df_processed.empty:
                    results.append({
                        'ticker': ticker,
                        'error': 'Insufficient data'
                    })
                    continue
                
                result = predictor.predict_from_stock_data(df_processed)
                latest = df_processed.iloc[-1]
                
                results.append({
                    'ticker': ticker,
                    'prediction': result['prediction_label'],
                    'confidence': round(result['confidence'] * 100, 2),
                    'current_price': float(latest['Close'])
                })
                
            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        feature_importance = predictor.get_feature_importance(top_n=15)
        
        return jsonify({
            'model_loaded': True,
            'num_features': len(predictor.feature_names),
            'feature_importance': feature_importance,
            'model_path': predictor.model_path
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/reload-model', methods=['POST'])
def reload_model():
    """Reload the model (useful after retraining)"""
    try:
        load_predictor()
        return jsonify({
            'status': 'success',
            'message': 'Model reloaded successfully',
            'model_loaded': predictor is not None
        })
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # For development
    app.run(host='0.0.0.0', port=5000, debug=True)