# Fraud Detection in Financial Transactions

## 📌 Introduction
Fraud detection is a crucial task in financial systems to prevent fraudulent transactions. This project explores different machine learning approaches to detect anomalies and classify fraudulent transactions. Despite extensive tuning, the models exhibited **overfitting**, highlighting the challenges of fraud detection with synthetic data.

## 📝 Dataset & Preprocessing
- **Synthetic Data**: Generated a dataset with **normal transactions** and **fraudulent transactions**.
- **Features Used**:
  - `transaction_amount`, `merchant_category`, `time_of_day`
- **Preprocessing**:
  - Applied **Standard Scaling**
  - Handled categorical data via **one-hot encoding**
  - **SMOTE** (Synthetic Minority Over-sampling) was used to balance the dataset.

## ⚡ Machine Learning Models Used
We trained three models to detect fraud:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

## 📊 Experiments & Results
### **First Attempt - Strong Overfitting**
- **All models achieved 100% precision, recall, and F1-score.**
- Overfitting was due to **synthetic fraud samples being too easy to classify**.

### **Adjustments Made to Reduce Overfitting**
1. **Reduced SMOTE sampling from 0.15 → 0.07** to prevent excessive resampling.
2. **Increased Regularization**:
   - Logistic Regression: `C=0.5`, `max_iter=100`
   - Random Forest: Fewer trees, shallower depth
   - XGBoost: More aggressive regularization (`gamma`, `subsample`)
3. **Shuffled the dataset to avoid data leakage.**

### **Final Results - Still Overfitting**
Despite adjustments, models still overfit:
- **Logistic Regression, Random Forest, XGBoost all achieved perfect 1.00 scores**.
- **Why?**
  - Fraud cases were **too artificially distinct**.
  - Synthetic data lacks **real-world complexity (correlations, noise, user behaviors, etc.)**.

## 🔍 Conclusions & Next Steps
### **Why is Fraud Detection Hard?**
- **Fraudulent transactions are rare**, making real-world imbalances much larger.
- **Fraud is dynamic**: Fraudsters adapt, making fixed patterns ineffective.
- **Real-world fraud features** (IP address, location, device info) were missing in our dataset.

### **Next Steps for Improvement**
✔️ **Use real-world datasets** instead of synthetic data.  
✔️ **Introduce additional features** (merchant reputation, transaction history).  
✔️ **Use anomaly detection models** instead of pure classification.  
✔️ **Consider an ensemble approach** to combine anomaly detection + supervised learning.  

## 🚀 How to Run the Project
```bash
# Clone the repository
git clone <repo-link>
cd fraud-detection-transactions

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run model training
python3 src/model_training_supervised.py
```

## 📂 Project Structure
```
fraud-detection-transactions/
│── data/
│   ├── transactions.csv  # Synthetic dataset
│── src/
│   ├── data_preprocessing.py  # Data cleaning & processing
│   ├── model_training_supervised.py  # Supervised model training
│── notebooks/
│── models/  # Saved trained models
│── README.md  # Project documentation
│── requirements.txt  # Dependencies
```

## 💡 Final Thoughts
This project highlights the **challenges of fraud detection**, especially with limited and synthetic data. While the models showed **high accuracy**, real-world fraud detection requires **better feature engineering, real datasets, and anomaly detection methods**.

---
🚀 **Next step?** Consider integrating this into a real-time fraud detection API!

