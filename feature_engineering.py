import pandas as pd
import numpy as np
from data_fetcher import F1DataFetcher  # Add this import at the top
from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {
            'driver_id': LabelEncoder(),
            'constructor': LabelEncoder(),
            'circuit': LabelEncoder()
        }
        # Remove unused feature_groups dictionary since it's not being used anywhere
        self.feature_groups = {
            'driver': ['avg_quali_position', 'q3_rate', 'finishing_rate'],
            'constructor': ['constructor_points', 'constructor_wins'],
            'circuit': ['circuit_type', 'track_length'],
            'race': ['grid_position', 'weather_condition']
        }
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        df_encoded = df.copy()
        
        for column, encoder in self.label_encoders.items():
            if column in df.columns:
                df_encoded[column] = encoder.fit_transform(df[column])
        
        return df_encoded
    
    def _calculate_qualifying_stats(self, qualifying_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate qualifying-specific statistics."""
        stats = []
        
        for driver_id in qualifying_df['driver_id'].unique():
            driver_quali = qualifying_df[qualifying_df['driver_id'] == driver_id]
            
            # Convert position to numeric and invert (P1 should be highest)
            positions = -pd.to_numeric(driver_quali['position'], errors='coerce')
            
            # Calculate Q3 appearances
            q3_times = driver_quali['q3_time'].notna().sum()
            
            stats.append({
                'driver_id': driver_id,
                'avg_quali_position': positions.mean(),  # Now higher is better
                'best_quali_position': positions.max(),  # Changed from min to max due to inversion
                'q3_appearances': q3_times,
                'q3_rate': q3_times / len(driver_quali) if len(driver_quali) > 0 else 0,
                'quali_consistency': positions.std() if len(positions) > 1 else 0
            })
        
        return pd.DataFrame(stats)
    
    def _calculate_team_trend(self, constructor_results: pd.DataFrame) -> float:
        """Calculate team's performance trend over the season."""
        if len(constructor_results) < 2:
            return 0.0
            
        # Sort by date to ensure chronological order
        constructor_results = constructor_results.sort_values('date')
        
        # Convert positions to numeric and invert (so negative slope means getting worse)
        positions = -pd.to_numeric(constructor_results['position'], errors='coerce')
        
        # Create race index (0, 1, 2, ...) for trend calculation
        races = range(len(positions))
        
        # Calculate trend using linear regression
        if len(positions.dropna()) < 2:
            return 0.0
            
        z = np.polyfit(races, positions.fillna(positions.mean()), 1)
        return z[0]  # Positive slope means improving trend (higher positions are better)

    def _calculate_constructor_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        stats = []
        
        for constructor in results_df['constructor'].unique():
            constructor_results = results_df[results_df['constructor'] == constructor]
            
            # Invert positions so higher is better
            positions = -pd.to_numeric(constructor_results['position'], errors='coerce')
            
            stats.append({
                'constructor': constructor,
                'team_points_per_race': constructor_results['points'].sum() / len(constructor_results),
                'team_avg_position': positions.mean(),  # Now higher is better
                'team_reliability': 1 - constructor_results['status'].str.contains('DNF|DNS', case=False, na=False).mean(),
                'team_development_trend': self._calculate_team_trend(constructor_results),
                'team_qualifying_strength': -constructor_results['grid'].astype(float).mean(),  # Invert grid position
                'team_race_pace': positions.mean()  # Now higher is better
            })
        
        return pd.DataFrame(stats)
    
    def _calculate_driver_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        stats = []
        
        for driver_id in results_df['driver_id'].unique():
            driver_results = results_df[results_df['driver_id'] == driver_id]
            
            # Focus on recent performance and consistency
            recent_results = driver_results.sort_values('date').tail(5)
            fastest_laps = pd.to_numeric(recent_results['fastest_lap_time'], errors='coerce')
            lap_speeds = pd.to_numeric(recent_results['fastest_lap_speed'], errors='coerce')
            
            stats.append({
                'driver_id': driver_id,
                # Give more weight to recent performance
                'recent_race_completion': (recent_results['status'] != 'DNF').mean(),
                'recent_avg_speed': lap_speeds.mean(),
                'recent_speed_consistency': lap_speeds.std() if len(lap_speeds) > 1 else 0,
                'recent_laps_completed': recent_results['laps'].mean(),
                'recent_fastest_laps': (recent_results['fastest_lap_rank'] == '1').sum(),
                # Add season-long consistency metrics
                'season_reliability': (driver_results['status'] != 'DNF').mean(),
                'season_fastest_laps': (driver_results['fastest_lap_rank'] == '1').sum(),
                'qualifying_performance': driver_results['grid'].astype(float).mean()
            })
        
        return pd.DataFrame(stats)
    
    def _calculate_constructor_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        stats = []
        
        for constructor in results_df['constructor'].unique():
            constructor_results = results_df[results_df['constructor'] == constructor]
            
            # Focus on reliability and performance metrics
            stats.append({
                'constructor': constructor,
                'team_reliability': 1 - constructor_results['status'].str.contains('DNF|DNS', case=False, na=False).mean(),
                'team_fastest_laps': (constructor_results['fastest_lap_rank'] == '1').sum(),
                'avg_team_speed': pd.to_numeric(constructor_results['fastest_lap_speed'], errors='coerce').mean(),
                'laps_completed_rate': constructor_results['laps'].mean() / constructor_results['laps'].max(),
                'mechanical_failures': constructor_results['status'].str.contains('Technical|Mechanical', case=False, na=False).mean()
            })
        
        return pd.DataFrame(stats)
    
    def _calculate_qualifying_stats(self, qualifying_df: pd.DataFrame) -> pd.DataFrame:
        stats = []
        
        for driver_id in qualifying_df['driver_id'].unique():
            driver_quali = qualifying_df[qualifying_df['driver_id'] == driver_id]
            
            # Focus on timing-based metrics
            stats.append({
                'driver_id': driver_id,
                'q3_appearances': driver_quali['q3_time'].notna().sum(),
                'q3_rate': driver_quali['q3_time'].notna().sum() / len(driver_quali) if len(driver_quali) > 0 else 0,
                'avg_q1_time': pd.to_numeric(driver_quali['q1_time'].str.replace(':', '.'), errors='coerce').mean(),
                'avg_q2_time': pd.to_numeric(driver_quali['q2_time'].str.replace(':', '.'), errors='coerce').mean(),
                'avg_q3_time': pd.to_numeric(driver_quali['q3_time'].str.replace(':', '.'), errors='coerce').mean()
            })
        
        return pd.DataFrame(stats)
    
    def _calculate_race_specific_features(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate race-specific features."""
        def convert_lap_time_to_seconds(time_str):
            if pd.isna(time_str):
                return np.nan
            try:
                if ':' in str(time_str):
                    minutes, rest = str(time_str).split(':')
                    seconds = float(minutes) * 60 + float(rest)
                else:
                    seconds = float(time_str)
                return seconds
            except (ValueError, TypeError):
                return np.nan

        stats = []
        
        for race_name in results_df['race_name'].unique():
            race_results = results_df[results_df['race_name'] == race_name]
            
            # Calculate overtaking difficulty (higher means more difficult)
            grid_positions = pd.to_numeric(race_results['grid'], errors='coerce')
            finish_positions = pd.to_numeric(race_results['position'], errors='coerce')
            position_changes = abs(grid_positions - finish_positions).mean()
            
            # Convert lap times to seconds before calculating statistics
            lap_times = race_results['fastest_lap_time'].apply(convert_lap_time_to_seconds)
            
            # Calculate track characteristics
            stats.append({
                'race_name': race_name,
                'overtaking_difficulty': 1 / (position_changes + 1),  # Normalize to 0-1
                'track_position_importance': (grid_positions.corr(finish_positions) 
                                           if len(grid_positions) > 1 else 0),
                'dnf_rate': race_results['status'].str.contains('DNF', case=False, na=False).mean(),
                'avg_pit_stops': race_results['pit_stops'].mean() if 'pit_stops' in race_results.columns else 0,
                'weather_impact': lap_times.std() if not lap_times.empty else 0
            })
        
        return pd.DataFrame(stats)

    def prepare_features(self, results_df: pd.DataFrame, standings_df: pd.DataFrame, 
                        qualifying_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare features for model training and prediction."""
        # Convert dict to DataFrame if necessary
        if isinstance(results_df, dict):
            results_df = pd.DataFrame(results_df['results'] if 'results' in results_df else [results_df])
                
        if isinstance(standings_df, dict):
            standings_df = pd.DataFrame(standings_df['standings'] if 'standings' in standings_df else [standings_df])
                
        if isinstance(qualifying_df, dict) and qualifying_df is not None:
            qualifying_df = pd.DataFrame(qualifying_df['qualifying'] if 'qualifying' in qualifying_df else [qualifying_df])
        
        # Validate required columns
        required_columns = ['driver_id', 'constructor', 'circuit', 'position']
        for col in required_columns:
            if col not in results_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate all features first
        driver_stats = self._calculate_driver_stats(results_df)
        constructor_stats = self._calculate_constructor_stats(results_df)
        race_stats = self._calculate_race_specific_features(results_df)
        qualifying_stats = self._calculate_qualifying_stats(qualifying_df) if qualifying_df is not None else pd.DataFrame()
        
        # Merge all features into features_df
        features_df = results_df.copy()
        features_df = features_df.merge(driver_stats, on='driver_id', how='left')
        features_df = features_df.merge(constructor_stats, on='constructor', how='left')
        features_df = features_df.merge(race_stats, on='race_name', how='left')
        if not qualifying_stats.empty:
            features_df = features_df.merge(qualifying_stats, on='driver_id', how='left')
        
        # Get target variable
        y = -pd.to_numeric(features_df['position'], errors='coerce') + 20
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'driver_id': features_df['driver_id'],
            'constructor': features_df['constructor'],
            'race_name': features_df['race_name']
        })
        
        # Select and prepare features
        exclude_cols = ['position', 'race_name', 'date', 'status']
        feature_cols = [col for col in features_df.columns 
                       if col not in exclude_cols and features_df[col].dtype in ['int64', 'float64', 'bool']]
        # Modify feature scaling approach
        X = features_df[feature_cols].copy()
        
        # Better handling of missing values
        for col in X.columns:
            if X[col].isnull().any():
                if 'rate' in col or 'reliability' in col:
                    X[col] = X[col].fillna(X[col].mean())  # Use mean for rate/reliability metrics
                elif 'speed' in col or 'time' in col:
                    X[col] = X[col].fillna(X[col].median())  # Use median for performance metrics
                else:
                    X[col] = X[col].fillna(0)
        
        # Scale features while preserving relative importance
        for col in X.columns:
            if X[col].std() > 0:
                if 'reliability' in col or 'rate' in col:
                    X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min())  # 0-1 scaling for percentages
                else:
                    X[col] = (X[col] - X[col].mean()) / X[col].std()  # Standard scaling for other metrics
        
        return X, y, predictions

# Update main test section
if __name__ == "__main__":
    # Test feature engineering
    from data_fetcher import F1DataFetcher
    
    fetcher = F1DataFetcher()
    results = fetcher.get_current_season_results()
    standings = fetcher.get_driver_standings()
    qualifying = fetcher.get_qualifying_results()
    
    engineer = FeatureEngineer()
    X, y = engineer.prepare_features(  # Updated to unpack 2 values
        results,
        standings,
        qualifying_df=qualifying
    )
    
    print("Feature engineering completed!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("\nFeature columns:")
    print(X.columns.tolist())