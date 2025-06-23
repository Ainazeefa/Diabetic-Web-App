import streamlit as st
import pandas as pd
import joblib

# ========== Load Model ==========
model = joblib.load("model.pkl")

# ========== Page Config ==========
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========== Default Values ==========
defaults = {
    "gender": "Female",
    "smoking_history": "never",
    "hypertension": "No",
    "heart_disease": "No",
    "age": 30,
    "bmi": 22.0,
    "hba1c": 5.5,
    "glucose": 100.0
}

# ========== Reset Trigger ==========
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

# ========== Handle Reset ==========
if st.session_state.reset_triggered:
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state.reset_triggered = False
    st.rerun()

# ========== Title ==========
st.title("🩺 Patient Diabetes Risk Predictor")
st.write("Enter patient details below to predict the likelihood of diabetes.")

# ========== Input Fields ==========
# Must prefill from session state or defaults (before widget call)
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Create widgets
st.selectbox("Gender", ["Female", "Male"], key="gender")
st.selectbox("Smoking History", ["never", "former", "current", "not current", "ever", "No Info"], key="smoking_history")
st.radio("Hypertension", ["No", "Yes"], key="hypertension")
st.radio("Heart Disease", ["No", "Yes"], key="heart_disease")
st.number_input("Age (years)", min_value=0, max_value=120, key="age")
st.number_input("BMI", min_value=10.0, max_value=60.0, help="Body Mass Index. Normal: 18.5–24.9", key="bmi")
st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, help="Avg. blood sugar over past 2–3 months. Normal: <5.7%", key="hba1c")
st.number_input("Blood Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, help="Normal fasting: 70–99 mg/dL", key="glucose")

# ========== Buttons ==========
col1, col2 = st.columns([1, 1])
with col1:
    submit = st.button("🔍 Predict")
with col2:
    if st.button("🔄 Reset"):
        st.session_state.reset_triggered = True

# ========== Prediction Logic ==========
if submit:
    hypertension = 1 if st.session_state.hypertension == "Yes" else 0
    heart_disease = 1 if st.session_state.heart_disease == "Yes" else 0

    if st.session_state.age > 80:
        st.warning("⚠️ Model was trained mostly on patients aged 80 or younger. Result may be less accurate.")

    input_df = pd.DataFrame({
        "gender": [st.session_state.gender],
        "smoking_history": [st.session_state.smoking_history],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "age": [st.session_state.age],
        "bmi": [st.session_state.bmi],
        "HbA1c_level": [st.session_state.hba1c],
        "blood_glucose_level": [st.session_state.glucose]
    })

    # ===== Summary =====
    st.markdown("### 📋 Input Summary")
    summary = {
        "Feature": ["Gender", "Smoking History", "Hypertension", "Heart Disease", "Age", "BMI", "HbA1c", "Glucose"],
        "Value": [
            st.session_state.gender, st.session_state.smoking_history,
            st.session_state.hypertension, st.session_state.heart_disease,
            st.session_state.age, st.session_state.bmi,
            st.session_state.hba1c, st.session_state.glucose
        ]
    }
    st.table(pd.DataFrame(summary))

    # ===== Prediction =====
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.markdown("### 🧪 Prediction Result")
    st.metric("Diabetes Risk", f"{prob:.1%}")

    if pred == 1:
        st.error("⚠️ The patient is **likely to have diabetes**.")
    else:
        st.success("✅ The patient is **unlikely to have diabetes**.")

    st.caption("This tool provides informational predictions and does not replace medical advice.")
