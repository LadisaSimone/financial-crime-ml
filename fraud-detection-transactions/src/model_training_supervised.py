# fraud-detection-transactions/src/model_training_supervised.py

import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from collections import Counter

# Load the dataset
df = pd.read_csv("../data/transactions.csv")

# Define features and target variable
X = df.drop(columns=['transaction_id', 'customer_id', 'is_fraud'])  # Drop IDs and target
y = df['is_fraud']

# Adjust SMOTE to ensure fraud cases are properly represented
smote = SMOTE(sampling_strategy=0.15, random_state=42)
X, y = smote.fit_resample(X, y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model (Generalized Model)
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("\nLogistic Regression Results:\n", classification_report(y_test, y_pred_log))

# Train Random Forest Classifier with Balanced Regularization
rf_model = RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_split=15, min_samples_leaf=5, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Results:\n", classification_report(y_test, y_pred_rf))

# Train XGBoost Classifier with Adjusted Regularization
xgb_model = XGBClassifier(n_estimators=50, max_depth=3, min_child_weight=5, gamma=5, subsample=0.7, colsample_bytree=0.7, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost Results:\n", classification_report(y_test, y_pred_xgb))

# Save trained models
joblib.dump(log_reg, "../models/logistic_regression.pkl")
joblib.dump(rf_model, "../models/random_forest_model.pkl")
joblib.dump(xgb_model, "../models/xgboost_model.pkl")
print("Saved trained models.")

if __name__ == "__main__":
    print("Supervised fraud detection model training complete with feature scaling and improved balance.")