# F1 Race Predictor

A machine learning-powered Formula 1 race predictor that forecasts the final grid positions for F1 races. The application uses historical race data from the current season via the Jolpica F1 API to make predictions.

## Features

- Fetch real-time F1 data using the Jolpica F1 API
- Train machine learning models using XGBoost
- Interactive web interface built with Streamlit
- Predict race outcomes for any circuit in the current season
- Visualize predictions using interactive tables

## Setup

1. Create and activate the virtual environment:
```bash
python -m venv f1_venv
source f1_venv/bin/activate  # On Windows: f1_venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Click the "Train New Model" button in the sidebar to fetch current season data and train the model

4. Select a circuit from the dropdown menu to get race predictions

## Project Structure

- `data_fetcher.py`: Handles data retrieval from the Jolpica F1 API
- `feature_engineering.py`: Processes and transforms raw data into model features
- `model_trainer.py`: Implements the XGBoost model for race predictions
- `app.py`: Streamlit web application
- `requirements.txt`: Python package dependencies

## Model Features

The prediction model takes into account various factors including:
- Driver's historical performance
- Constructor performance
- Circuit characteristics
- Grid position
- Recent form and consistency
- DNF (Did Not Finish) rate

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- plotly
- requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) for providing F1 data
- [Streamlit](https://streamlit.io/) for the web application framework
- Formula 1 for the amazing sport! 