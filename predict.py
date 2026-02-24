import joblib
import pandas as pd

MODEL_PATH = "artifacts/model.joblib"

def predict_one(features: dict) -> dict:
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features])
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    return {"churn_probability": proba, "churn_pred": pred}

if __name__ == "__main__":
    sample = {
        "tenure": 5,
        "MonthlyCharges": 89.1,
        "TotalCharges": 445.5,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "Partner": "Yes",
        "Dependents": "No",
        "SeniorCitizen": 0,
    }
    print(predict_one(sample))