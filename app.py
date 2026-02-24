import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "artifacts/model.joblib"

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📉 Customer Churn Prediction (Telco)")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --- IMPORTANT: expected columns from training data ---
expected_cols = list(model.feature_names_in_)

st.caption("This app automatically matches the model's required input columns.")

# Helper: infer which columns are numeric/categorical from telco dataset (local file)
# If dataset not present, fallback to simple guessing.
# Helper: infer which columns are numeric/categorical from telco dataset (local file)
try:
    df_ref = pd.read_csv("data/telco_churn.csv")

    # match training cleaning
    if "TotalCharges" in df_ref.columns:
        df_ref["TotalCharges"] = pd.to_numeric(df_ref["TotalCharges"], errors="coerce")

    if "customerID" in df_ref.columns:
        df_ref = df_ref.drop(columns=["customerID"])
    if "Churn" in df_ref.columns:
        df_ref = df_ref.drop(columns=["Churn"])

except Exception:
    df_ref = pd.DataFrame(columns=expected_cols)

# Determine categorical vs numeric from reference dataframe
cat_cols = [c for c in expected_cols if c in df_ref.columns and df_ref[c].dtype == "object"]
num_cols = [c for c in expected_cols if c not in cat_cols]

st.subheader("Input Features")

inputs = {}

colA, colB, colC = st.columns(3)
all_cols = expected_cols

# Create inputs dynamically
for i, col in enumerate(all_cols):
    target_col = [colA, colB, colC][i % 3]

    with target_col:
        if col in cat_cols:
            # dropdown using unique values from dataset
            options = sorted(df_ref[col].dropna().unique().tolist()) if col in df_ref.columns else ["Yes", "No"]
            inputs[col] = st.selectbox(col, options)
        else:
            # numeric field
            if col in df_ref.columns:
                s = pd.to_numeric(df_ref[col], errors="coerce")
                default_val = float(s.dropna().median()) if not s.dropna().empty else 0.0
            else:
                default_val = 0.0
            inputs[col] = st.number_input(col, value=float(default_val))

if st.button("Predict"):
    X = pd.DataFrame([inputs])[expected_cols]  # enforce correct order
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    st.metric("Churn Probability", f"{proba:.3f}")
    st.write("Prediction:", "✅ Churn" if pred == 1 else "❌ No Churn")