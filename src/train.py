"""Script for training, optimizing, and serializing the House Price Prediction model."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import add_engineered_features

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "House_Prices.csv"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")


def train_and_save():
  print("1. Loading dataset...")
  df = pd.read_csv(DATA_PATH).iloc[:1460]

  print("2. Handling missing values and feature engineering...")
  
  df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
  df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

  cat_cols = df.select_dtypes(include=["object", "string"]).columns
  num_cols = df.select_dtypes(include="number").columns
  df[cat_cols] = df[cat_cols].fillna("none")
  df[num_cols] = df[num_cols].fillna(0)

  # apply engineered features
  df = add_engineered_features(df)

  # separate X and y
  X = df.drop(columns=["Id", "SalePrice"])
  y = df["SalePrice"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # build Preprocessor
  num_features = X.select_dtypes(include="number").columns
  cat_features = X.select_dtypes(include=["object", "string"]).columns

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), num_features),
          ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
      ]
  )

  # Pipeline & Hyperparameter Optimization
  pipeline = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          (
              "regressor",
              GradientBoostingRegressor(random_state=42),
          ),
      ]
  )

  param_grid = {
      "regressor__n_estimators": [100, 150],
      "regressor__learning_rate": [0.05, 0.1],
      "regressor__max_depth": [3, 4],
  }

  print("3. Running 5-Fold Cross-Validation & GridSearchCV...")
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  grid = GridSearchCV(
      pipeline, param_grid, cv=kf, scoring="r2", n_jobs=-1, verbose=1
  )
  grid.fit(X_train, y_train)

  best_model = grid.best_estimator_
  print(f" Best Hyperparameters: {grid.best_params_}")

  # Evaluation on unseen test set
  print("4. Evaluating on test set...")
  y_pred = best_model.predict(X_test)
  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)

  print(f"   MAE:  ${mae:,.2f}")
  print(f"   RMSE: ${rmse:,.2f}")
  print(f"   R²:   {r2:.4f}")

  # Save model artifact
  print(f"5. Saving model to {MODEL_PATH}...")
  os.makedirs(MODEL_DIR, exist_ok=True)
  joblib.dump(best_model, MODEL_PATH)
  print(" Model training and serialization complete!")


if __name__ == "__main__":
  train_and_save()