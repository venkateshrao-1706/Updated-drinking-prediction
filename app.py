import pandas as pd
import streamlit as st

from predictor import FEATURES, Predictor

st.set_page_config(page_title="Drinking Status Prediction", page_icon="🍷", layout="wide")


# Loaded once per server process, not on every click/rerun
@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    return Predictor("model_weights.npz")


predictor = load_predictor()

# (key, label, min, max, default, step) - all floats so Streamlit doesn't complain about mixed types
NUMERIC_FIELDS = [
    ("age", "Age", 18.0, 100.0, 30.0, 1.0),
    ("height", "Height (cm)", 100.0, 220.0, 170.0, 1.0),
    ("weight", "Weight (kg)", 30.0, 200.0, 70.0, 1.0),
    ("waistline", "Waistline (cm)", 40.0, 150.0, 80.0, 1.0),
    ("sight_left", "Sight Left", 0.1, 9.9, 1.0, 0.1),
    ("sight_right", "Sight Right", 0.1, 9.9, 1.0, 0.1),
    ("SBP", "Systolic Blood Pressure", 70.0, 250.0, 120.0, 1.0),
    ("DBP", "Diastolic Blood Pressure", 40.0, 150.0, 80.0, 1.0),
    ("BLDS", "Blood Sugar", 50.0, 500.0, 100.0, 1.0),
    ("tot_chole", "Total Cholesterol", 50.0, 600.0, 180.0, 1.0),
    ("HDL_chole", "HDL Cholesterol", 5.0, 300.0, 50.0, 1.0),
    ("LDL_chole", "LDL Cholesterol", 5.0, 500.0, 100.0, 1.0),
    ("triglyceride", "Triglyceride", 5.0, 2000.0, 120.0, 1.0),
    ("hemoglobin", "Hemoglobin", 3.0, 25.0, 14.0, 0.1),
    ("urine_protein", "Urine Protein", 1.0, 6.0, 1.0, 1.0),
    ("serum_creatinine", "Serum Creatinine", 0.1, 20.0, 1.0, 0.1),
    ("SGOT_AST", "SGOT AST", 1.0, 10000.0, 25.0, 1.0),
    ("SGOT_ALT", "SGOT ALT", 1.0, 10000.0, 25.0, 1.0),
    ("gamma_GTP", "Gamma GTP", 1.0, 2000.0, 30.0, 1.0),
]

SMOKING_LABELS = {1.0: "1 - Never smoked", 2.0: "2 - Used to smoke, quit", 3.0: "3 - Current smoker"}
HEARING_LABELS = {1.0: "Normal", 2.0: "Abnormal"}

st.title("Drinking Status Prediction")
st.write("Enter the health information below.")
st.caption("For demonstration only. Not medical advice.")

values = {}
with st.form("prediction_form"):
    cols = st.columns(3)
    for i, (key, label, lo, hi, default, step) in enumerate(NUMERIC_FIELDS):
        values[key] = cols[i % 3].number_input(
            label, min_value=lo, max_value=hi, value=default, step=step
        )
    values["hear_left"] = cols[0].selectbox(
        "Hearing Left", list(HEARING_LABELS), format_func=HEARING_LABELS.get
    )
    values["hear_right"] = cols[1].selectbox(
        "Hearing Right", list(HEARING_LABELS), format_func=HEARING_LABELS.get
    )
    values["SMK_stat_type_cd"] = cols[2].selectbox(
        "Smoking Status", list(SMOKING_LABELS), format_func=SMOKING_LABELS.get
    )
    sex = cols[0].selectbox("Sex", ["Female", "Male"])
    submitted = st.form_submit_button("Predict")

if submitted:
    if values["SBP"] <= values["DBP"]:
        st.warning("Systolic BP is normally expected to be higher than Diastolic BP.")

    values["sex_Female"] = 1.0 if sex == "Female" else 0.0
    values["sex_Male"] = 1.0 - values["sex_Female"]

    row = pd.DataFrame([values])[FEATURES]  # enforce training column order
    prob = predictor.predict_proba(row.to_numpy())

    if prob >= 0.5:
        st.error("Drinking habit detected: Drinker")
    else:
        st.success("Drinking habit not detected: Non-Drinker")
    st.metric("Drinking probability", f"{prob * 100:.2f}%")

    tab1, tab2 = st.tabs(["Prediction", "Input Summary"])
    with tab1:
        st.progress(min(max(prob, 0.0), 1.0))
        chart_data = pd.DataFrame(
            {"status": ["Drinker", "Non-Drinker"], "probability": [prob * 100, (1 - prob) * 100]}
        )
        st.bar_chart(chart_data, x="status", y="probability")
    with tab2:
        st.subheader("Entered Information")
        st.dataframe(row, hide_index=True)
