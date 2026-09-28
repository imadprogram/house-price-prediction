import pandas as pd
from src.features import add_engineered_features


def test_add_engineered_features():
  sample_df = pd.DataFrame([{
      "1stFlrSF": 1000,
      "2ndFlrSF": 500,
      "TotalBsmtSF": 800,
      "FullBath": 2,
      "HalfBath": 1,
      "BsmtFullBath": 1,
      "BsmtHalfBath": 0,
      "YearBuilt": 2000,
      "YearRemodAdd": 2010,
      "YrSold": 2020,
  }])

  result = add_engineered_features(sample_df)

  # check TotalSF = 1000 + 500 + 800 = 2300
  assert result["TotalSF"].iloc[0] == 2300

  # check TotalBath = 2 + (0.5*1) + 1 + 0 = 3.5
  assert result["TotalBath"].iloc[0] == 3.5

  # check HouseAge = 2020 - 2000 = 20
  assert result["HouseAge"].iloc[0] == 20

  # check YearsSinceRemodel = 2020 - 2010 = 10
  assert result["YearsSinceRemodel"].iloc[0] == 10