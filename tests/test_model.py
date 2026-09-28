import pandas as pd
from src.predict import load_model, predict_price


def test_model_loading_and_prediction():
  model = load_model()
  assert model is not None

  # load baseline row
  df = pd.read_csv("data/House_Prices.csv").iloc[:1].drop(
      columns=["Id", "SalePrice"]
  )
  price = predict_price(df, model=model)

  # price must be a positive float within reasonable housing bounds
  assert isinstance(price, float)
  assert 30000 < price < 1000000







# python -m pytest test