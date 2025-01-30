# aml-risk-scoring/src/model_training.py

import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("../data/aml_risk_data.csv")
print('******************')
print(df.columns)
print('******************')

# Define features and target variable
X = df.drop(columns=['customer_id', 'risk_label'])  # Drop IDs and target
y = df['risk_label']

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier with Hyperparameter Tuning
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=4, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Results:\n", classification_report(y_test, y_pred_rf))

# Train XGBoost Classifier with Regularization
xgb_model = XGBClassifier(n_estimators=100, max_depth=4, min_child_weight=6, gamma=2, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost Results:\n", classification_report(y_test, y_pred_xgb))

# Train Gradient Boosting Classifier as an Alternative Model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_split=10, min_samples_leaf=4, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("\nGradient Boosting Results:\n", classification_report(y_test, y_pred_gb))

# Save trained models
joblib.dump(rf_model, "../models/random_forest_model.pkl")
joblib.dump(xgb_model, "../models/xgboost_model.pkl")
joblib.dump(gb_model, "../models/gradient_boosting_model.pkl")
print("Saved trained models.")

if __name__ == "__main__":
    print("Supervised AML risk scoring model training complete with improved class balance and optimized models.")

