import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("mpg_model.pkl")

# Page config
st.set_page_config(page_title="Car MPG Predictor", page_icon="🚗", layout="centered")

# Title
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color:#2E86C1;">🚗 Car MPG Prediction App</h1>
        <p style="font-size:18px;">Enter car details below to estimate fuel efficiency (MPG)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input form
with st.form("mpg_form"):
    col1, col2 = st.columns(2)

    with col1:
        cylinders = st.number_input("Cylinders", 3, 12, 4, step=1)
        displacement = st.number_input("Displacement (cu in)", 50.0, 500.0, 150.0, step=1.0)
        horsepower = st.number_input("Horsepower", 40.0, 250.0, 100.0, step=1.0)

    with col2:
        weight = st.number_input("Weight (lbs)", 1500.0, 6000.0, 2500.0, step=10.0)
        acceleration = st.number_input("Acceleration (0-60 mph)", 8.0, 30.0, 15.0, step=0.1)
        model_year = st.number_input("Model Year", 70, 83, 76, step=1)

    origin = st.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1: "🇺🇸 USA", 2: "🇪🇺 Europe", 3: "🇯🇵 Japan"}[x])

    submitted = st.form_submit_button("🔮 Predict MPG")

# Prediction
if submitted:
    input_data = pd.DataFrame(
        [[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]],
        columns=['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
    )
    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; background-color:#F4F6F6; padding:20px; border-radius:15px;">
            <h2 style="color:#117A65;">✨ Predicted Fuel Efficiency ✨</h2>
            <h1 style="color:#D35400;">{prediction:.2f} MPG</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
