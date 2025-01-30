# fraud-detection-transactions/src/model_training.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("../data/transactions.csv")

# Scale numerical features
scaler = StandardScaler()
df[['amount', 'time_of_day']] = scaler.fit_transform(df[['amount', 'time_of_day']])

# Separate features and labels
X = df.drop(columns=['transaction_id', 'is_fraud'])  # Drop ID and target
y = df['is_fraud']

# Train Isolation Forest model
isolation_forest = IsolationForest(contamination=0.02, random_state=42)
df['anomaly_score_if'] = isolation_forest.fit_predict(X)
df['fraud_prediction_if'] = (df['anomaly_score_if'] == -1).astype(int)

# Train Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df['fraud_prediction_lof'] = lof.fit_predict(X)
df['fraud_prediction_lof'] = (df['fraud_prediction_lof'] == -1).astype(int)

# Print classification reports
print("\nIsolation Forest Results:\n", classification_report(y, df['fraud_prediction_if']))
print("\nLocal Outlier Factor Results:\n", classification_report(y, df['fraud_prediction_lof']))

# Visualize fraud predictions
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['amount'], y=df['time_of_day'], hue=df['fraud_prediction_if'], palette='coolwarm', alpha=0.6)
plt.title("Fraud Detection using Isolation Forest")
plt.xlabel("Transaction Amount")
plt.ylabel("Time of Day")
plt.show()

# Save best performing model
import joblib
joblib.dump(isolation_forest, "../models/isolation_forest.pkl")
print("Saved Isolation Forest model.")

if __name__ == "__main__":
    print("Fraud detection model training complete.")