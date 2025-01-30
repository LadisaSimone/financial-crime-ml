# fraud-detection-transactions/src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def generate_synthetic_data(n_samples=1000):

    np.random.seed(42)
    
    # Normal transactions
    normal_transactions = {
        'transaction_id': range(1, n_samples+1),
        'customer_id': np.random.randint(1, 200, n_samples),  # 200 unique customers
        'amount': np.random.normal(50, 20, n_samples),  # Avg transaction amount
        'merchant_category': np.random.choice(['Retail', 'Food', 'Travel', 'Tech'], n_samples),
        'time_of_day': np.random.uniform(0, 24, n_samples),  # Hours in a day
        'is_fraud': 0  # Normal transactions
    }
    
    # Fraudulent transactions
    fraud_transactions = {
        'transaction_id': range(n_samples+1, n_samples+51),
        'customer_id': np.random.randint(1, 200, 50),
        'amount': np.random.normal(300, 100, 50),  # Fraudulent transactions are much larger
        'merchant_category': np.random.choice(['Luxury', 'Crypto', 'Gambling'], 50),
        'time_of_day': np.random.uniform(0, 24, 50),
        'is_fraud': 1  # Fraudulent transactions
    }
    
    # Combine datasets
    df_normal = pd.DataFrame(normal_transactions)
    df_fraud = pd.DataFrame(fraud_transactions)
    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    
    # Feature Engineering
    # Encode categorical variables
    df['merchant_category'] = LabelEncoder().fit_transform(df['merchant_category'])
    
    # Transaction Velocity: Count transactions per customer per day
    df['transaction_count'] = df.groupby('customer_id')['customer_id'].transform('count')
    
    # Mean Transaction Amount Per Customer
    df['mean_transaction_amount'] = df.groupby('customer_id')['amount'].transform('mean')
    
    # Convert categorical to numerical encoding
    df = pd.get_dummies(df, columns=['merchant_category'], drop_first=True)
    
    # Save dataset
    df.to_csv("../data/transactions.csv", index=False)
    print("Synthetic dataset created with additional features and saved as transactions.csv")
    return df

if __name__ == "__main__":
    generate_synthetic_data()