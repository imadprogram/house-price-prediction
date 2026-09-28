import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
  """Calculates domain-specific engineered features:

  - TotalSF: Total interior square footage
  - TotalBath: Total full and half bathrooms
  - HouseAge: Age of property at time of sale
  - YearsSinceRemodel: Years elapsed since last remodel
  """
  data = df.copy()

  # total square footage (1st floor + 2nd floor + Basement)
  data["TotalSF"] = (
      data.get("1stFlrSF", 0)
      + data.get("2ndFlrSF", 0)
      + data.get("TotalBsmtSF", 0)
  )

  # total bathrooms (Full + 0.5 * Half)
  data["TotalBath"] = (
      data.get("FullBath", 0)
      + 0.5 * data.get("HalfBath", 0)
      + data.get("BsmtFullBath", 0)
      + 0.5 * data.get("BsmtHalfBath", 0)
  )

  # house age at the time of sale (prevent negative values with clip)
  if "YrSold" in data.columns and "YearBuilt" in data.columns:
    data["HouseAge"] = (data["YrSold"] - data["YearBuilt"]).clip(lower=0)

  # 4. Years since last remodel at time of sale
  if "YrSold" in data.columns and "YearRemodAdd" in data.columns:
    data["YearsSinceRemodel"] = (data["YrSold"] - data["YearRemodAdd"]).clip(
        lower=0
    )

  return data