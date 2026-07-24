import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Calories Burnt Prediction",
    page_icon="🔥",
    layout="centered"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("🔥 Calories Burnt Prediction")
st.write("Enter your details below to estimate calories burnt during exercise.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.slider("Age", 10, 100, 25)

height = st.number_input(
    "Height (cm)",
    min_value=120.0,
    max_value=220.0,
    value=170.0,
    step=0.1
)

weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=70.0,
    step=0.1
)

duration = st.slider(
    "Exercise Duration (Minutes)",
    1,
    180,
    30
)

heart_rate = st.slider(
    "Heart Rate (bpm)",
    50,
    220,
    95
)

body_temp = st.number_input(
    "Body Temperature (°C)",
    min_value=35.0,
    max_value=42.0,
    value=37.5,
    step=0.1
)

# Encode Gender
gender_value = 0 if gender == "Male" else 1

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔥 Predict Calories Burnt"):

    input_data = np.array([[
        gender_value,
        age,
        height,
        weight,
        duration,
        heart_rate,
        body_temp
    ]])

    try:
        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = float(model.predict(input_scaled)[0])
        prediction = max(0, prediction)

        st.divider()

        st.metric(
            label="Estimated Calories Burnt",
            value=f"{prediction:.2f} kcal"
        )

        # -----------------------------
        # BMI
        # -----------------------------
        bmi = weight / ((height / 100) ** 2)

        st.subheader(f"📊 BMI: {bmi:.2f}")

        if bmi < 18.5:
            st.info("BMI Status: Underweight")
        elif bmi < 25:
            st.success("BMI Status: Normal Weight")
        elif bmi < 30:
            st.warning("BMI Status: Overweight")
        else:
            st.error("BMI Status: Obese")

        # -----------------------------
        # Water Recommendation After Exercise
        # -----------------------------
        if prediction < 150:
            water_range = "300 - 500 ml"
        elif prediction < 350:
            water_range = "500 - 1000 ml"
        else:
            water_range = "1000 - 1500 ml"

        st.info(f"💧 Water After Exercise: {water_range}")

        # -----------------------------
        # Daily Water Requirement
        # -----------------------------
        daily_water = weight * 35          # ml/day
        exercise_water = duration * 10     # extra ml
        total_water = daily_water + exercise_water

        st.info(f"💦 Total Water Needed Today: {int(total_water)} ml")

        # -----------------------------
        # Health Tips
        # -----------------------------
        st.subheader("🏃 Health Tips")

        if prediction < 150:
            st.success("Light workout completed. Stay hydrated and eat a balanced meal.")
        elif prediction < 350:
            st.success("Moderate workout! Drink enough water and include protein in your meal.")
        else:
            st.warning("Intense workout! Rest well, drink electrolytes, and consume enough protein.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")