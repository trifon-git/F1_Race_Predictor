import streamlit as st
import pandas as pd
from data_fetcher import get_last_n_races_results, get_next_race_info
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer

st.title("🏁 F1 Simple Race Predictor")

n_races = st.number_input("🔢 How many past races should the model use?", min_value=1, max_value=10, value=5)

if st.button(f"Fetch Last {n_races} Race Results"):
    race_results = get_last_n_races_results(n_races)

    if not race_results.empty:
        st.session_state['race_results'] = race_results
        st.success("✅ Race results fetched!")
        st.dataframe(race_results)

        unique_circuits = race_results['CircuitName'].unique()
        st.info(f"ℹ️ Using results from **{len(unique_circuits)} different circuits**.")
    else:
        st.error("❌ No race results found.")

if 'race_results' in st.session_state:
    fe = FeatureEngineer(n_races=n_races)

    try:
        features, labels = fe.prepare_features(st.session_state['race_results'])
        st.session_state['features'] = features
        st.session_state['labels'] = labels
    except Exception as e:
        st.error(f"❌ Error preparing features: {e}")

    if st.button("Train Model"):
        trainer = ModelTrainer()

        try:
            X = st.session_state['features']
            y = st.session_state['labels']

            trainer.train(X, y)
            st.session_state['trainer'] = trainer
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Error during training: {e}")

    if 'trainer' in st.session_state and st.button("Predict Next Race"):
        next_race = get_next_race_info()

        if next_race is not None and 'EventName' in next_race:
            race_name = next_race['EventName']
            st.subheader(f"🏎️ Predictions for: {race_name}")
            st.info(f"🔮 Predicting race results for **{race_name}**!")
        else:
            st.subheader("🏎️ Predictions for: (Unknown Upcoming Race)")
            st.warning("⚠️ Next race information is missing.")

        try:
            prediction_features = fe.prepare_prediction_features(st.session_state['race_results'])
            preds = st.session_state['trainer'].predict(prediction_features)

            drivers = st.session_state['race_results']['DriverId'].unique()
            prediction_df = pd.DataFrame({
                'Driver': drivers,
                'Predicted Finish Position': preds
            }).sort_values('Predicted Finish Position')

            st.write("### 🏁 Predicted Race Results")
            st.dataframe(prediction_df)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
