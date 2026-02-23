# churn-e2e
End-to-end customer churn prediction with preprocessing, model training, evaluation, and Streamlit app.
# Customer Churn Prediction (Telco) — End-to-End

An end-to-end machine learning project to predict customer churn using the Telco Customer Churn dataset.
Includes data preprocessing, model training/evaluation, and a Streamlit web app for inference.

## Features
- Cleaning + preprocessing (numeric scaling, categorical one-hot encoding)
- Models: Logistic Regression (baseline), Gradient Boosting (best)
- Metrics: ROC-AUC, F1, Confusion Matrix (saved in `artifacts/metrics.json`)
- Streamlit app for predictions

## Dataset
Telco Customer Churn (IBM/Kaggle).  
Download the dataset and place it as: `data/telco_churn.csv`

## How to Run
```bash
pip install -r requirements.txt
python train.py
python -m streamlit run app.py
