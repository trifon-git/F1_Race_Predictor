import streamlit as st
import pandas as pd
import numpy as np
from data_fetcher import F1DataFetcher
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = None
    if 'next_race' not in st.session_state:
        st.session_state.next_race = None
    if 'cv_scores' not in st.session_state:
        st.session_state.cv_scores = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None

def train_model():
    """Train a new model and store it in session state."""
    try:
        with st.spinner('Fetching F1 data...'):
            fetcher = F1DataFetcher()
            # Get last 10 races results
            results_df = fetcher.get_last_n_races(20)  # Changed from n=10 to 20 for consistency
            
            # Get qualifying and sprint results for these races
            race_dates = results_df['date'].unique().tolist()
            qualifying_df = fetcher.get_qualifying_results_for_races(race_dates)
            
            # Check if qualifying data is empty
            if qualifying_df.empty:
                st.warning("No qualifying data available. Training model without qualifying features.")
                qualifying_df = None
            
            standings_df = fetcher.get_driver_standings()
            
            # Get next race information
            st.session_state.next_race = fetcher.get_next_race()
        
        with st.spinner('Preparing features...'):
            st.session_state.feature_engineer = FeatureEngineer()
            
            # Prepare features with combined race data
            X, y, predictions = st.session_state.feature_engineer.prepare_features(
                results_df,
                standings_df,
                qualifying_df=qualifying_df  # Will be None if empty
            )
            
        with st.spinner('Training model...'):
            trainer = ModelTrainer()
            cv_rmse, cv_r2 = trainer.train(X, y, optimize=True)
            
            st.session_state.model = trainer
            st.session_state.cv_scores = trainer.cv_scores
            st.session_state.feature_importance = trainer.feature_importance
        
        st.success('Model trained successfully!')
        
        # Display model performance metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Cross-Validation RMSE', f"{cv_rmse:.4f}")
        with col2:
            st.metric('Cross-Validation R²', f"{cv_r2:.4f}")
        
        # Display feature importance plot
        if st.session_state.feature_importance is not None:
            top_features = st.session_state.feature_importance.head(10)
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Most Important Features'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f'Error during model training: {str(e)}')
        logger.error(f'Model training error: {str(e)}', exc_info=True)

def make_predictions(circuit: str):
    """Make predictions for a given circuit."""
    try:
        if st.session_state.model is None or st.session_state.feature_engineer is None:
            st.warning('Please train the model first.')
            return
        
        fetcher = F1DataFetcher()
        current_drivers = fetcher.get_current_drivers()
        results = fetcher.get_current_season_results()
        standings = fetcher.get_driver_standings()
        qualifying = fetcher.get_qualifying_results()
        sprint = fetcher.get_sprint_results()
        
        # Prepare features for prediction
        X_pred = pd.DataFrame()
        for driver in current_drivers:
            driver_data = {
                'driver_id': driver['driver_id'],
                'constructor': driver['constructor'],
                'circuit': circuit
            }
            X_pred = pd.concat([X_pred, pd.DataFrame([driver_data])], ignore_index=True)
        
        # Engineer features (fixed argument order)
        # In make_predictions function
        race_data = {
            'results': results,
            'qualifying': qualifying,
            'sprint': sprint
        }
        
        X_pred_full = st.session_state.feature_engineer.prepare_features(
            race_data,
            standings
        )[0].head(len(current_drivers))
        
        # Make predictions
        predictions = st.session_state.model.predict(X_pred_full)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Position': range(1, len(predictions) + 1),
            'Driver': [f"{d['firstname']} {d['lastname']}" for d in current_drivers],
            'Constructor': [d['constructor'] for d in current_drivers],
            'Predicted Position': predictions
        })
        results_df = results_df.sort_values('Predicted Position')
        results_df['Position'] = range(1, len(results_df) + 1)
        
        # Display results
        is_next_race = (st.session_state.next_race is not None and 
                       st.session_state.next_race['circuit'] == circuit)
        
        if is_next_race:
            st.header(f"🏁 Prediction for Next Race: {st.session_state.next_race['name']}")
            st.subheader(f"Date: {st.session_state.next_race['date']}")
        else:
            st.header(f"Race Prediction for {circuit}")
        
        # Style the DataFrame
        st.dataframe(
            results_df.style
            .set_properties(**{
                'background-color': '#f0f2f6',
                'color': 'black',
                'border-color': 'white'
            })
            .set_table_styles([
                {'selector': 'th',
                 'props': [('background-color', '#1f77b4'),
                          ('color', 'white'),
                          ('font-weight', 'bold')]},
                {'selector': 'td',
                 'props': [('text-align', 'center')]}
            ])
        )
        
        # Display prediction confidence
        if st.session_state.cv_scores:
            st.info(f"Prediction Confidence Metrics:\n"
                   f"- Average Error (RMSE): ±{st.session_state.cv_scores['rmse']:.2f} positions\n"
                   f"- Model Accuracy (R²): {st.session_state.cv_scores['r2']:.2%}")
        
        # Create position distribution plot
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=predictions,
            name='Predicted Positions',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        fig.update_layout(
            title='Distribution of Predicted Positions',
            yaxis_title='Position',
            showlegend=False
        )
        st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f'Error making predictions: {str(e)}')
        logger.error(f'Prediction error: {str(e)}', exc_info=True)

def main():
    st.title("F1 Race Prediction")
    
    # Initialize data fetcher and engineer
    fetcher = F1DataFetcher()
    engineer = FeatureEngineer()
    
    # Fetch data
    with st.spinner('Fetching race data...'):
        results = fetcher.get_current_season_results()
        standings = fetcher.get_driver_standings()
        qualifying = fetcher.get_qualifying_results()
    
    # Process features
    with st.spinner('Processing features...'):
        X, y, predictions = engineer.prepare_features(
            results,
            standings,
            qualifying_df=qualifying
        )
    
    # Display results
    st.success('Data processed successfully!')
    st.write("Feature Statistics:")
    st.dataframe(X.describe())

if __name__ == "__main__":
    main()