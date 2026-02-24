import os, json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

ART_DIR = "artifacts"
DATA_PATH = os.path.join("data", "telco_churn.csv")

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df

def build_pipeline(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    baseline = LogisticRegression(max_iter=2000)
    strong = GradientBoostingClassifier(random_state=42)

    baseline_pipe = Pipeline([("prep", preprocessor), ("model", baseline)])
    strong_pipe = Pipeline([("prep", preprocessor), ("model", strong)])
    return baseline_pipe, strong_pipe

def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    return metrics

def main():
    os.makedirs(ART_DIR, exist_ok=True)
    df = load_data(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline_pipe, strong_pipe = build_pipeline(X)

    baseline_pipe.fit(X_train, y_train)
    strong_pipe.fit(X_train, y_train)

    baseline_metrics = evaluate(baseline_pipe, X_test, y_test)
    strong_metrics = evaluate(strong_pipe, X_test, y_test)

    best_model = strong_pipe if strong_metrics["roc_auc"] >= baseline_metrics["roc_auc"] else baseline_pipe
    best_name = "GradientBoosting" if best_model is strong_pipe else "LogisticRegression"

    metrics_out = {
        "baseline": baseline_metrics,
        "strong": strong_metrics,
        "best_model": best_name,
    }

    joblib.dump(best_model, os.path.join(ART_DIR, "model.joblib"))
    with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("Training done.")
    print("Best:", best_name)
    print("Saved to artifacts/model.joblib and artifacts/metrics.json")

if __name__ == "__main__":
    main()