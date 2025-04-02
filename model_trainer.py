import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.cv_scores = None
        self.best_params = None
        
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using grid search with cross-validation."""
        logger.info("Starting hyperparameter optimization...")
        
        # Simplified parameter grid focusing on most important aspects
        # Based on feature importance analysis showing recent_form, avg_points, 
        # and avg_finish_position as key features
        param_grid = {
            'n_estimators': [100],        # Simplified as model converges well with 100
            'max_depth': [3, 4],          # Shallow trees work well for this data
            'learning_rate': [0.05, 0.1], # Focus on faster learning rates
            'min_child_weight': [1],      # Default works well
            'subsample': [0.8],           # Good default for preventing overfitting
            'colsample_bytree': [0.8]     # Good default for feature sampling
        }
        
        best_score = float('inf')
        best_params = None
        
        # Perform grid search with cross-validation
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for learning_rate in param_grid['learning_rate']:
                    params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'min_child_weight': param_grid['min_child_weight'][0],
                        'subsample': param_grid['subsample'][0],
                        'colsample_bytree': param_grid['colsample_bytree'][0],
                        'objective': 'reg:squarederror',
                        'random_state': 42
                    }
                    
                    model = XGBRegressor(**params)
                    cv_scores = cross_val_score(
                        model,
                        X,
                        y,
                        cv=5,
                        scoring='neg_mean_squared_error'
                    )
                    
                    mse = -cv_scores.mean()
                    if mse < best_score:
                        best_score = mse
                        best_params = params
                        logger.info(f"New best MSE: {mse:.4f} with params: {params}")
        
        logger.info(f"Best parameters found: {best_params}")
        return best_params
    
    def train(self, X: pd.DataFrame, y: pd.Series, optimize: bool = True) -> Tuple[float, float]:
        """Train the model and return performance metrics."""
        logger.info("Starting model training...")
        
        try:
            # Remove points from features if present
            if 'points' in X.columns:
                logger.info("Removing points from training features")
                X = X.drop('points', axis=1)
            
            # Memory optimization
            X = X.astype('float32')
            y = y.astype('float32')
            
            if optimize:
                self.best_params = self._optimize_hyperparameters(X, y)
                self.model = XGBRegressor(**self.best_params)
            else:
                # Default parameters if not optimizing
                self.model = XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    objective='reg:squarederror',
                    random_state=42
                )
            
            # Perform cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_predictions = []
            cv_actuals = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                self.model.fit(X_train, y_train)
                predictions = self.model.predict(X_val)
                
                cv_predictions.extend(predictions)
                cv_actuals.extend(y_val)
            
            # Calculate performance metrics
            cv_mse = mean_squared_error(cv_actuals, cv_predictions)
            cv_rmse = np.sqrt(cv_mse)
            cv_r2 = r2_score(cv_actuals, cv_predictions)
            
            # Store cross-validation scores
            self.cv_scores = {
                'mse': cv_mse,
                'rmse': cv_rmse,
                'r2': cv_r2
            }
            
            # Train final model on full dataset
            self.model.fit(X, y)
            
            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Clean up memory
            import gc
            gc.collect()
            
            logger.info(f"Training completed with CV RMSE: {cv_rmse:.4f}, R²: {cv_r2:.4f}")
            logger.info("\nTop 10 most important features:")
            logger.info(self.feature_importance.head(10))
            
            return cv_rmse, cv_r2
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Remove points from prediction features if present
        if 'points' in X.columns:
            X = X.drop('points', axis=1)
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No model to save!")
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from a file."""
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Test model training
    from data_fetcher import F1DataFetcher
    from feature_engineering import FeatureEngineer
    
    fetcher = F1DataFetcher()
    results = fetcher.get_current_season_results()
    standings = fetcher.get_driver_standings()
    qualifying = fetcher.get_qualifying_results()
    sprint = fetcher.get_sprint_results()
    
    engineer = FeatureEngineer()
    X, y = engineer.prepare_features(
        results,
        standings,
        qualifying_df=qualifying,
        sprint_df=sprint
    )
    
    trainer = ModelTrainer()
    cv_rmse, cv_r2 = trainer.train(X, y, optimize=True)
    
    print(f"\nFinal Cross-Validation Metrics:")
    print(f"RMSE: {cv_rmse:.4f}")
    print(f"R²: {cv_r2:.4f}")
    
    if trainer.feature_importance is not None:
        print("\nFeature Importance:")
        print(trainer.feature_importance.head(10))