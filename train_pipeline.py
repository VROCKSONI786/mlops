#!/usr/bin/env python3
"""
Complete training pipeline for stock prediction model
This script can be run daily by Jenkins for automated retraining
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import StockDataCollector
from feature_engineering import FeatureEngineer
from model_training import StockPriceModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training_pipeline(ticker='AAPL', period='2y', model_type='xgboost'):
    """
    Run complete training pipeline
    
    Args:
        ticker: Stock ticker symbol
        period: Historical data period
        model_type: Type of model to train
        
    Returns:
        Dictionary with training results
    """
    logger.info("="*70)
    logger.info(f"Starting Training Pipeline - {datetime.now()}")
    logger.info(f"Ticker: {ticker}, Period: {period}, Model: {model_type}")
    logger.info("="*70)
    
    try:
        # Step 1: Data Collection
        logger.info("\n[Step 1/4] Collecting stock data...")
        collector = StockDataCollector(ticker=ticker, period=period)
        df = collector.fetch_historical_data()
        
        # Save raw data
        os.makedirs('data/raw', exist_ok=True)
        data_file = f'data/raw/stock_data_{ticker}_{datetime.now().strftime("%Y%m%d")}.csv'
        collector.save_data(df, data_file)
        
        logger.info(f"✓ Collected {len(df)} records")
        
        # Step 2: Feature Engineering
        logger.info("\n[Step 2/4] Engineering features...")
        engineer = FeatureEngineer()
        df_features = engineer.create_technical_indicators(df)
        X, y, df_processed = engineer.prepare_features(df_features)
        
        logger.info(f"✓ Created {X.shape[1]} features from {X.shape[0]} samples")
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        processed_file = f'data/processed/processed_data_{ticker}_{datetime.now().strftime("%Y%m%d")}.csv'
        df_processed.to_csv(processed_file, index=False)
        
        # Step 3: Model Training
        logger.info("\n[Step 3/4] Training model...")
        trainer = StockPriceModel(model_type=model_type, experiment_name=f'stock_prediction_{ticker}')
        metrics = trainer.train(X, y, test_size=0.2)
        
        logger.info(f"✓ Model trained successfully")
        logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  - Precision: {metrics['precision']:.4f}")
        logger.info(f"  - Recall: {metrics['recall']:.4f}")
        logger.info(f"  - F1-Score: {metrics['f1']:.4f}")
        logger.info(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Step 4: Save Model
        logger.info("\n[Step 4/4] Saving model...")
        trainer.save_model(model_path='models', model_name='stock_model')
        logger.info(f"✓ Model saved to models/stock_model.pkl")
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Features: {X.shape[1]}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Model saved: models/stock_model.pkl")
        logger.info("="*70 + "\n")
        
        return {
            'status': 'success',
            'ticker': ticker,
            'samples': len(df),
            'features': X.shape[1],
            'metrics': metrics,
            'model_path': 'models/stock_model.pkl',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main entry point for the training pipeline"""
    parser = argparse.ArgumentParser(description='Train stock prediction model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='2y', help='Historical data period')
    parser.add_argument('--model', type=str, default='xgboost', 
                       choices=['xgboost', 'random_forest', 'gradient_boosting'],
                       help='Model type')
    
    args = parser.parse_args()
    
    result = run_training_pipeline(
        ticker=args.ticker,
        period=args.period,
        model_type=args.model
    )
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)


if __name__ == "__main__":
    main()