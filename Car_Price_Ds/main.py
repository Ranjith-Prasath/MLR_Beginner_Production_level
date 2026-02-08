import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. SETUP & LOADING ---
st.title("🚗 Car Price Predictor (Robust)")
st.write("Tweak the values below to estimate the car's price.")

# --- PATH FIX: Get the absolute path of the current file ---
# This ensures we find the .pkl files whether running locally or on Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'car_price_model_safe.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

# Load the files using the smart paths
try:
    artifacts = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error(f"Error: Could not find model files. Looked in: {model_path}")
    st.stop()

params = artifacts["params"]
model_features = artifacts["columns"]

# --- 2. INPUTS (The UI) ---
st.sidebar.header("Car Specs")

# Numeric Inputs
horsepower = st.sidebar.slider("Horsepower", 50, 300, 150)
curbweight = st.sidebar.slider("Curb Weight (lbs)", 1500, 5000, 2500)
enginesize = st.sidebar.slider("Engine Size (cu in)", 60, 350, 120)
highwaympg = st.sidebar.slider("Highway MPG", 15, 55, 30)
carlength = st.sidebar.slider("Car Length (inches)", 140, 210, 170)
carwidth = st.sidebar.slider("Car Width (inches)", 60, 75, 65)

# Categorical Inputs
brand = st.sidebar.selectbox("Brand", ["Toyota", "BMW", "Porsche", "Jaguar", "Volvo", "Buick", "Other"])
engine_loc = st.sidebar.radio("Engine Location", ["Front", "Rear"])
is_rotary = st.sidebar.checkbox("Rotary Engine?")


# --- 3. PREDICTION LOGIC ---
def predict_price_detailed():
    base_price = params.get('const', 0)
    current_price = base_price
    breakdown = []

    # B. Handle Numeric Features (Using Mean Defaults)
    scaler_cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                   'enginesize', 'boreratio', 'stroke', 'compressionratio',
                   'horsepower', 'peakrpm', 'citympg', 'highwaympg']

    default_values = scaler.mean_
    input_df = pd.DataFrame([default_values], columns=scaler_cols)

    input_df['horsepower'] = horsepower
    input_df['curbweight'] = curbweight
    input_df['enginesize'] = enginesize
    input_df['highwaympg'] = highwaympg
    input_df['carlength'] = carlength
    input_df['carwidth'] = carwidth

    input_scaled = scaler.transform(input_df)
    input_df_scaled = pd.DataFrame(input_scaled, columns=scaler_cols)

    for col in model_features:
        if col in input_df_scaled.columns:
            coef = params.get(col, 0)
            val = input_df_scaled[col].values[0]
            contribution = coef * val
            current_price += contribution
            breakdown.append((col, contribution))

    # C. Handle Categorical Features
    if 'enginelocation' in model_features and engine_loc == "Rear":
        contrib = params['enginelocation']
        current_price += contrib
        breakdown.append(("Rear Engine Bonus", contrib))

    if 'enginetype_rotor' in model_features and is_rotary:
        contrib = params['enginetype_rotor']
        current_price += contrib
        breakdown.append(("Rotary Engine Rareness", contrib))

    brand_key = f"Brand_{brand.lower()}"
    if brand_key in params:
        contrib = params[brand_key]
        current_price += contrib
        breakdown.append((f"{brand} Brand", contrib))

    return current_price, breakdown, base_price


# --- 4. DISPLAY ---
if st.button('Estimate Price'):
    price, breakdown, base = predict_price_detailed()

    st.success(f"### Estimated Price: ${price:,.2f}")

    with st.expander("See Price Breakdown", expanded=True):
        st.write(f"**Base Price:** ${base:,.2f}")
        st.write("---")
        for name, amount in breakdown:
            color = ":green" if amount >= 0 else ":red"
            st.write(f"{color}[{name}]: **${amount:,.2f}**")