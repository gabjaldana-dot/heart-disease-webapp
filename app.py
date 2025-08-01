import streamlit as st
import numpy as np
import joblib

# Page setup
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="centered"
)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.markdown("""
    <h1 style='text-align: center; font-size: 42px;'>💓 Heart Disease Predictor</h1>
    <p style='text-align: center; font-size: 20px;'>Please enter the patient's details below.</p>
    <hr style='margin-top: 10px;'>
""", unsafe_allow_html=True)

# CSS styles
st.markdown("""
    <style>
        label {
            font-size: 18px !important;
            font-weight: bold !important;
        }
        .stNumberInput input {
            font-size: 18px !important;
        }
        .stTextInput input {
            font-size: 18px !important;
        }
        .stSelectbox div, .stRadio div {
            font-size: 18px !important;
        }
        button[kind="primary"] {
            font-size: 20px !important;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120)
        sex = st.radio("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", {
            "Typical Angina (0)": 0,
            "Atypical Angina (1)": 1,
            "Non-anginal Pain (2)": 2,
            "Asymptomatic (3)": 3
        }.keys())
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
        chol = st.number_input("Cholesterol", min_value=100, max_value=600)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["Yes", "No"])

    with col2:
        restecg = st.selectbox("Resting ECG", {
            "Normal (0)": 0,
            "ST-T Wave Abnormality (1)": 1,
            "Left Ventricular Hypertrophy (2)": 2
        }.keys())
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250)
        exang = st.radio("Exercise Induced Angina?", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", {
            "Upsloping (0)": 0,
            "Flat (1)": 1,
            "Downsloping (2)": 2
        }.keys())
        ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", {
            "Normal (0)": 0,
            "Fixed Defect (1)": 1,
            "Reversible Defect (2)": 2,
            "Unknown (3)": 3
        }.keys())

    submitted = st.form_submit_button("🔍 Predict")

# Processing
if submitted:
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0
    cp_val = {"Typical Angina (0)": 0, "Atypical Angina (1)": 1, "Non-anginal Pain (2)": 2, "Asymptomatic (3)": 3}[cp]
    restecg_val = {"Normal (0)": 0, "ST-T Wave Abnormality (1)": 1, "Left Ventricular Hypertrophy (2)": 2}[restecg]
    slope_val = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}[slope]
    thal_val = {"Normal (0)": 0, "Fixed Defect (1)": 1, "Reversible Defect (2)": 2, "Unknown (3)": 3}[thal]

    features = np.array([[age, sex, cp_val, trestbps, chol, fbs, restecg_val,
                          thalach, exang, oldpeak, slope_val, ca, thal_val]])

    features_scaled = scaler.transform(features)
    result = model.predict(features_scaled)

    st.markdown("<hr>", unsafe_allow_html=True)
    if result[0] == 1:
        st.error("⚠️ The patient is likely to have heart disease.", icon="⚠️")
    else:
        st.success("✅ The patient is unlikely to have heart disease.", icon="✅")
