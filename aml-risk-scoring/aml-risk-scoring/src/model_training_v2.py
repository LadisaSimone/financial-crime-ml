# aml-risk-scoring/src/model_training_v2.py

import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from collections import Counter

# Load the dataset
df = pd.read_csv("../data/aml_risk_data.csv")

# Define features and target variable
X = df.drop(columns=['customer_id', 'risk_label'])  # Drop IDs and target
y = df['risk_label']

# Print class distribution before SMOTE
print("Original class distribution:", Counter(y))

# Dynamically oversample minority classes to match the majority class
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

# Print new class distribution after SMOTE
print("New class distribution after SMOTE:", Counter(y))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("\nGradient Boosting Results:\n", classification_report(y_test, y_pred_gb))

# Train LightGBM Classifier
lgbm_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)
print("\nLightGBM Results:\n", classification_report(y_test, y_pred_lgbm))

# Save trained models
joblib.dump(gb_model, "../models/gradient_boosting_model.pkl")
joblib.dump(lgbm_model, "../models/lightgbm_model.pkl")
print("Saved trained models.")

if __name__ == "__main__":
    print("AML Risk Scoring Model training complete with SMOTE and enhanced classifiers.")
