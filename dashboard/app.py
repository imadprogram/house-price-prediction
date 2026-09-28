import sys
from pathlib import Path

# Add project root to sys.path so 'src' imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from src.predict import load_model, predict_price

# page Configuration
st.set_page_config(
    page_title="House Price Estimator", page_icon="🏡", layout="wide"
)


# cache Model Loading (Runs once, never retrains)
@st.cache_resource
def get_model():
  return load_model()


# cache Dataset Loading (For baseline values and comparison charts)
@st.cache_data
def get_data():
  data_path = ROOT_DIR / "data" / "House_Prices.csv"
  df = pd.read_csv(data_path).iloc[:1460]  # Original training subset
  return df


model = get_model()
df_baseline = get_data()

# header & Overview
st.title("🏡 House Price Estimation Tool")
st.markdown(
    "Estimate market sale prices for residential properties based on key"
    " architectural and physical features."
)

st.divider()

# user Input Form
st.subheader("1. Enter Property Characteristics")

col1, col2, col3 = st.columns(3)

with col1:
  st.markdown("#### 📐 Size & Space")
  gr_liv_area = st.number_input(
      "Above Ground Living Area (sq ft)",
      min_value=300,
      max_value=6000,
      value=1500,
      step=50,
  )
  total_bsmt_sf = st.number_input(
      "Total Basement Area (sq ft)",
      min_value=0,
      max_value=4000,
      value=800,
      step=50,
  )
  lot_area = st.number_input(
      "Lot Size (sq ft)",
      min_value=1000,
      max_value=50000,
      value=9000,
      step=500,
  )

with col2:
  st.markdown("#### ⭐ Quality & Age")
  overall_qual = st.slider(
      "Overall Material & Finish Quality (1-10)",
      min_value=1,
      max_value=10,
      value=6,
  )
  overall_cond = st.slider(
      "Overall Condition Rating (1-10)", min_value=1, max_value=10, value=5
  )
  year_built = st.number_input(
      "Year Built", min_value=1870, max_value=2024, value=2000
  )
  year_remod = st.number_input(
      "Year Remodeled / Added", min_value=1950, max_value=2024, value=2005
  )

with col3:
  st.markdown("#### 📍 Location & Amenities")
  neighborhoods = sorted(df_baseline["Neighborhood"].dropna().unique().tolist())
  neighborhood = st.selectbox(
      "Neighborhood",
      options=neighborhoods,
      index=neighborhoods.index("CollgCr") if "CollgCr" in neighborhoods else 0,
  )
  full_bath = st.selectbox("Full Bathrooms", options=[1, 2, 3, 4], index=1)
  half_bath = st.selectbox("Half Bathrooms", options=[0, 1, 2], index=1)
  garage_cars = st.selectbox(
      "Garage Car Capacity", options=[0, 1, 2, 3, 4], index=2
  )

st.divider()

# predict Button & Results Display
if st.button("🚀 Calculate Estimated Price", type="primary", use_container_width=True):
  # Create a baseline input row using median/mode of the training set
  input_data = df_baseline.drop(columns=["Id", "SalePrice"]).iloc[[0]].copy()

  # fill the baseline row with numeric medians and categorical modes
  for col in input_data.columns:
    if col in df_baseline.select_dtypes(include="number").columns:
      input_data[col] = df_baseline[col].median()
    else:
      input_data[col] = df_baseline[col].mode()[0]

  # overwrite with the user's specific inputs
  input_data["GrLivArea"] = gr_liv_area
  input_data["1stFlrSF"] = gr_liv_area * 0.6  # Realistic breakdown
  input_data["2ndFlrSF"] = gr_liv_area * 0.4
  input_data["TotalBsmtSF"] = total_bsmt_sf
  input_data["LotArea"] = lot_area
  input_data["OverallQual"] = overall_qual
  input_data["OverallCond"] = overall_cond
  input_data["YearBuilt"] = year_built
  input_data["YearRemodAdd"] = max(year_remod, year_built)
  input_data["YrSold"] = 2024
  input_data["Neighborhood"] = neighborhood
  input_data["FullBath"] = full_bath
  input_data["HalfBath"] = half_bath
  input_data["BsmtFullBath"] = 1 if total_bsmt_sf > 0 else 0
  input_data["BsmtHalfBath"] = 0
  input_data["GarageCars"] = garage_cars
  input_data["GarageArea"] = garage_cars * 250

  # make prediction
  estimated_price = predict_price(input_data, model=model)

  # display the Estimated Price
  st.subheader("2. Valuation Result")
  res_col1, res_col2 = st.columns([1, 2])

  with res_col1:
    st.metric(
        label="Estimated Market Value",
        value=f"${estimated_price:,.0f}",
        delta=f"Based on {neighborhood} market dynamics",
    )
    st.info(
        f"**Computed Total Area:** {gr_liv_area + total_bsmt_sf:,} sq ft  \n"
        f"**Effective Property Age:** {2024 - year_built} years"
    )

  with res_col2:
    # ontextual Visualization: Compare prediction against the neighborhood distribution
    neighborhood_prices = df_baseline[
        df_baseline["Neighborhood"] == neighborhood
    ]["SalePrice"]
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.kdeplot(
        neighborhood_prices, fill=True, color="skyblue", ax=ax, label="Historical Prices in Neighborhood"
    )
    ax.axvline(
        estimated_price,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Your Estimate (${estimated_price:,.0f})",
    )
    ax.set_title(f"Price Position within {neighborhood}", fontsize=11)
    ax.set_xlabel("Sale Price ($)")
    ax.legend(fontsize=9)
    st.pyplot(fig)