import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPriceModel:
    """Train and evaluate stock price prediction models with MLflow tracking"""
    
    def __init__(self, model_type='xgboost', experiment_name='stock_prediction'):
        """
        Initialize model trainer
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'xgboost'
            experiment_name: MLflow experiment name
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        
    def get_model(self):
        """Get model based on type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model with MLflow tracking
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set proportion
        """
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples", X_train.shape[0])
            
            # Train model
            logger.info(f"Training {self.model_type} model...")
            self.model = self.get_model()
            self.model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            y_pred_proba_test = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
            train_metrics = self.calculate_metrics(y_train, y_pred_train, 
                                                   self.model.predict_proba(X_train_scaled)[:, 1])
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
            
            # Log feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance plot
                feature_importance.head(20).to_csv('feature_importance.csv', index=False)
                mlflow.log_artifact('feature_importance.csv')
                os.remove('feature_importance.csv')
            
            # Log model
            if self.model_type == 'xgboost':
                mlflow.xgboost.log_model(self.model, "model")
            else:
                mlflow.sklearn.log_model(self.model, "model")
            
            # Log scaler
            joblib.dump(self.scaler, 'scaler.pkl')
            mlflow.log_artifact('scaler.pkl')
            os.remove('scaler.pkl')
            
            # Log classification report
            report = classification_report(y_test, y_pred_test)
            logger.info(f"\nClassification Report:\n{report}")
            
            # Print results
            logger.info(f"\n{'='*50}")
            logger.info(f"Model: {self.model_type}")
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test Precision: {metrics['precision']:.4f}")
            logger.info(f"Test Recall: {metrics['recall']:.4f}")
            logger.info(f"Test F1-Score: {metrics['f1']:.4f}")
            logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"{'='*50}\n")
            
            return metrics
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def save_model(self, model_path='models', model_name='stock_model'):
        """Save model and scaler to disk"""
        os.makedirs(model_path, exist_ok=True)
        
        model_file = os.path.join(model_path, f'{model_name}.pkl')
        scaler_file = os.path.join(model_path, f'{model_name}_scaler.pkl')
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        
        # Save feature names
        feature_file = os.path.join(model_path, f'{model_name}_features.pkl')
        joblib.dump(self.feature_names, feature_file)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Scaler saved to {scaler_file}")
        
    def load_model(self, model_path='models', model_name='stock_model'):
        """Load model and scaler from disk"""
        model_file = os.path.join(model_path, f'{model_name}.pkl')
        scaler_file = os.path.join(model_path, f'{model_name}_scaler.pkl')
        feature_file = os.path.join(model_path, f'{model_name}_features.pkl')
        
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.feature_names = joblib.load(feature_file)
        
        logger.info(f"Model loaded from {model_file}")


if __name__ == "__main__":
    # Test model training
    from feature_engineering import FeatureEngineer
    
    df = pd.read_csv('data/raw/stock_data.csv')
    engineer = FeatureEngineer()
    df_features = engineer.create_technical_indicators(df)
    X, y, _ = engineer.prepare_features(df_features)
    
    # Train model
    trainer = StockPriceModel(model_type='xgboost')
    metrics = trainer.train(X, y)
    trainer.save_model()