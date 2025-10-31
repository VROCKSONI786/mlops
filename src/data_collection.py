import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collect historical and real-time stock data using yfinance"""
    
    def __init__(self, ticker='AAPL', period='2y'):
        """
        Initialize data collector
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Historical period ('1y', '2y', '5y', 'max')
        """
        self.ticker = ticker
        self.period = period
        self.stock = yf.Ticker(ticker)
        
    def fetch_historical_data(self):
        """Fetch historical stock data"""
        try:
            logger.info(f"Fetching historical data for {self.ticker}")
            df = self.stock.history(period=self.period)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Reset index to have Date as a column
            df.reset_index(inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def fetch_real_time_data(self, days=5):
        """
        Fetch recent data for real-time predictions
        
        Args:
            days: Number of recent days to fetch
        """
        try:
            logger.info(f"Fetching real-time data for {self.ticker}")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = self.stock.history(start=start_date, end=end_date)
            df.reset_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching real-time data: {str(e)}")
            raise
    
    def get_stock_info(self):
        """Get stock information"""
        try:
            info = self.stock.info
            return {
                'symbol': info.get('symbol', self.ticker),
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('currentPrice', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching stock info: {str(e)}")
            return {'symbol': self.ticker, 'name': 'N/A'}
    
    def save_data(self, df, filepath):
        """Save data to CSV"""
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise


if __name__ == "__main__":
    # Test the data collector
    collector = StockDataCollector(ticker='AAPL', period='2y')
    df = collector.fetch_historical_data()
    print(f"Data shape: {df.shape}")
    print(df.head())
    
    # Save data
    collector.save_data(df, 'data/raw/stock_data.csv')