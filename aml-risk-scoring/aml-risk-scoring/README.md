# AML Risk Scoring Model

## 📌 Introduction
Anti-Money Laundering (AML) risk scoring is a critical component in financial institutions to assess customer risk levels and prevent illicit activities. This project leverages machine learning techniques to classify customers into different risk categories based on transaction behaviors, geographical risk, KYC completeness, and other financial indicators.

## 📊 Dataset & Preprocessing
- **Features Used:**
  - `transaction_amount`, `transaction_frequency`, `geographic_risk_score`, `kyc_completeness`, `sar_flag`, `merchant_category`
- **Preprocessing Steps:**
  - Applied **Standard Scaling** for numerical features
  - Applied **One-Hot Encoding** for categorical variables
  - Used **SMOTE** for balancing the dataset

## 🤖 Machine Learning Models Used
The following models were trained and evaluated:
1. **XGBoost Classifier** - Initial model
2. **Gradient Boosting Classifier** - Improved version
3. **LightGBM Classifier** - Final optimized model

## 📈 Experiments & Results
### **First Attempt - Baseline Model**
- XGBoost achieved an accuracy of **59%**, with weak recall and precision for riskier classes.
- The model struggled due to imbalanced data.

### **Second Attempt - Gradient Boosting Improvements**
- Applied better hyperparameter tuning and feature selection.
- Improved accuracy to **65%**, but still had issues with minority class recall.

### **Final Model - LightGBM Optimization**
- Achieved **65% accuracy** with improved class balance.
- Used a refined SMOTE strategy to ensure oversampling does not distort the distribution.
- Overall, LightGBM provided the best precision-recall tradeoff.

## 📂 Repository Structure
```
aml-risk-scoring/
│── data/                   # Contains AML dataset
│── models/                 # Trained ML models stored here
│── notebooks/              # Jupyter notebooks for EDA & experimentation
│── src/                    # Main scripts for processing & training
│   │── data_preprocessing.py  # Prepares and cleans the dataset
│   │── model_training.py      # Baseline model training
│   │── model_training_v2.py   # Improved LightGBM model
│   │── risk_scoring.py        # Inference script for new customers
│── README.md               # Project documentation
│── requirements.txt        # Dependencies
```

## 📌 Conclusion
- This project demonstrated the challenges of AML risk scoring due to **class imbalance and complex feature relationships**.
- **LightGBM performed best**, but further improvements could be made with additional feature engineering and external risk indicators.
- Future work may involve integrating **unsupervised anomaly detection** and **explainable AI methods** to increase transparency.

## 🚀 How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocess Data:**
   ```bash
   python src/data_preprocessing.py
   ```
3. **Train the Model:**
   ```bash
   python src/model_training_v2.py
   ```
4. **Use the Model for Risk Scoring:**
   ```bash
   python src/risk_scoring.py
   ```

## 📧 Contact
If you have any questions or suggestions, feel free to reach out!

---
This project is part of the **Financial Crime ML Repository**.

