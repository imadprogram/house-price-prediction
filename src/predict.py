import os
import joblib
import pandas as pd
from src.features import add_engineered_features

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "best_model.joblib"
)


def load_model(path: str = MODEL_PATH):
  """Loads the saved joblib model pipeline."""
  if not os.path.exists(path):
    raise FileNotFoundError(f"Model file not found at: {path}")
  return joblib.load(path)


def predict_price(input_df: pd.DataFrame, model=None) -> float:
  """Prepares input data with engineered features and returns the predicted price."""
  if model is None:
    model = load_model()

  # apply feature engineering
  processed_df = add_engineered_features(input_df)

  # run prediction through the pipeline (preprocessor + regressor)
  prediction = model.predict(processed_df)

  return float(prediction[0])