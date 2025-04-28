import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, n_races=5):
        self.n_races = n_races
        self.driver_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.training_columns = None

    def _calculate_recent_performance(self, results_df: pd.DataFrame, driver_id: str) -> pd.Series:
        """Calculate simple performance metrics."""
        driver_results = results_df[results_df['DriverId'] == driver_id].sort_values('EventDate', ascending=False)
        recent_races = driver_results.head(self.n_races).copy()  # Use the user-selected number!

        if recent_races.empty:
            return pd.Series({
                'avg_recent_position': 20.0,
                'avg_recent_grid': 20.0,
                'recent_dnf_rate': 1.0,
                'recent_overtakes': 0.0
            })

        recent_races['Position'] = pd.to_numeric(recent_races['Position'], errors='coerce')
        recent_races['GridPosition'] = pd.to_numeric(recent_races['GridPosition'], errors='coerce')

        avg_pos = recent_races['Position'].mean()
        avg_grid = recent_races['GridPosition'].mean()
        dnf_rate = recent_races['Status'].apply(lambda x: 'Finished' not in str(x) and '+' not in str(x)).mean()
        overtakes = (recent_races['GridPosition'] - recent_races['Position']).mean()

        return pd.Series({
            'avg_recent_position': avg_pos if pd.notna(avg_pos) else 20.0,
            'avg_recent_grid': avg_grid if pd.notna(avg_grid) else 20.0,
            'recent_dnf_rate': dnf_rate if pd.notna(dnf_rate) else 1.0,
            'recent_overtakes': overtakes if pd.notna(overtakes) else 0.0
        })

    def prepare_features(self, race_results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare simple features for training."""
        logger.info(f"Preparing simple features from {len(race_results_df)} race results...")

        if race_results_df.empty:
            raise ValueError("Input race_results_df cannot be empty.")

        race_results_df['EventDate'] = pd.to_datetime(race_results_df['EventDate'])
        race_results_df['Position'] = pd.to_numeric(race_results_df['Position'], errors='coerce')

        unique_drivers = race_results_df['DriverId'].unique()
        all_features = []

        for driver_id in unique_drivers:
            driver_features = self._calculate_recent_performance(race_results_df, driver_id)
            driver_features['DriverId'] = driver_id
            driver_features['TargetPosition'] = race_results_df[race_results_df['DriverId'] == driver_id].sort_values('EventDate').iloc[-1]['Position']
            all_features.append(driver_features)

        features_df = pd.DataFrame(all_features)

        y = features_df['TargetPosition']
        X = features_df.drop(columns=['TargetPosition', 'DriverId'])

        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)

        self.training_columns = X.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=self.training_columns)

        logger.info(f"Generated simple features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    def prepare_prediction_features(self, last_5_races_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare simple prediction features."""
        logger.info("Preparing simple features for prediction...")

        if self.training_columns is None:
            raise ValueError("Model has not been trained yet.")

        unique_drivers = last_5_races_df['DriverId'].unique()
        prediction_features_list = []

        for driver_id in unique_drivers:
            driver_perf = self._calculate_recent_performance(last_5_races_df, driver_id)
            prediction_features_list.append(driver_perf)

        X_pred = pd.DataFrame(prediction_features_list)
        X_pred = X_pred.reindex(columns=self.training_columns, fill_value=0)

        X_pred_scaled = self.scaler.transform(X_pred)
        X_pred = pd.DataFrame(X_pred_scaled, columns=self.training_columns)

        logger.info(f"Generated simple prediction features shape: {X_pred.shape}")
        return X_pred
