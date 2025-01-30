import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=1000):
    """ Generate synthetic customer transaction data for AML risk scoring. """
    
    customer_ids = np.arange(1, n_samples + 1)
    
    # Generate random features
    transaction_amount = np.random.exponential(scale=1000, size=n_samples)
    transaction_frequency = np.random.randint(1, 50, n_samples)
    geographic_risk_score = np.random.uniform(0, 1, n_samples)
    kyc_completeness = np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8])
    sar_flag = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])  # Suspicious Activity Reports (SAR)

    # Categorical Features
    merchant_category = np.random.choice(["Retail", "Crypto", "Luxury", "Gambling", "Tech"], n_samples)

    # Assign risk labels (Low, Medium, High)
    risk_labels = np.random.choice(["Low", "Medium", "High"], size=n_samples, p=[0.7, 0.2, 0.1])

    # Create DataFrame
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "transaction_amount": transaction_amount,
        "transaction_frequency": transaction_frequency,
        "geographic_risk_score": geographic_risk_score,
        "kyc_completeness": kyc_completeness,
        "sar_flag": sar_flag,
        "merchant_category": merchant_category,
        "risk_label": risk_labels
    })

    return df

def preprocess_data(df):
    """ Preprocess the dataset: encode categorical data and scale numerical features. """

    # One-hot encode categorical feature
    df = pd.get_dummies(df, columns=["merchant_category"], drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ["transaction_amount", "transaction_frequency", "geographic_risk_score"]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Map risk labels to numerical values
    risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
    df["risk_label"] = df["risk_label"].map(risk_mapping)

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df = preprocess_data(df)
    df.to_csv("../data/aml_risk_data.csv", index=False)
    print("✅ Synthetic AML risk dataset created and saved.")
