# AI-Powered Customer Churn Prediction System

An end-to-end Machine Learning application that predicts whether a telecom customer is likely to churn based on customer demographics, account information, and service usage patterns. The project demonstrates the complete ML lifecycle, from data preprocessing and feature engineering to model deployment with an interactive Streamlit dashboard.

**🌐 Live Demo:** https://ai-customer-churn-prediction-ykqrabvby54bjzzdr3b2zm.streamlit.app/

---

## Project Overview

Customer churn is one of the biggest business challenges for subscription-based companies. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project helps businesses:

* Predict customers who are likely to churn.
* Understand the key factors driving churn.
* Estimate customer risk using probability scores.
* Support proactive customer retention strategies.
* Provide explainable predictions using SHAP.

---

## Key Features

* End-to-end Machine Learning pipeline
* Data preprocessing and cleaning
* Advanced feature engineering
* Multiple ML algorithms
* Hyperparameter tuning using GridSearchCV
* Model comparison and evaluation
* Explainable AI with SHAP
* Interactive Streamlit dashboard
* Downloadable prediction reports
* Automatic synthetic data generation when the dataset is unavailable

---

## Tech Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Scikit-learn
* XGBoost

### Explainable AI

* SHAP

### Web Application

* Streamlit

### Model Persistence

* Joblib

### Development Tools

* Jupyter Notebook
* Git
* GitHub

---

## Project Structure

```text
customer_churn_prediction/
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── churn_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.pkl
│
├── app/
│   └── streamlit_app.py
│
├── reports/
│   ├── plots/
│   └── metrics/
│
├── requirements.txt
├── README.md
└── main.py
```

---

## Dataset

This project uses the IBM Telco Customer Churn dataset.

Dataset includes customer information such as:

* Customer demographics
* Service subscriptions
* Billing details
* Contract information
* Payment methods
* Customer tenure
* Monthly and total charges
* Churn status

Place the dataset inside:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

If the dataset is not found, the application automatically generates synthetic telecom-like customer data for demonstration purposes.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/<your-username>/customer_churn_prediction.git

cd customer_churn_prediction
```

### Create a Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the Model

```bash
python main.py
```

This will:

* Load the dataset
* Clean and preprocess data
* Engineer new features
* Train multiple ML models
* Perform hyperparameter tuning
* Select the best-performing model
* Save trained artifacts
* Generate evaluation reports

---

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

### Exploratory Data Analysis

```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## Machine Learning Pipeline

```
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
SHAP Explainability
      │
      ▼
Prediction
      │
      ▼
Streamlit Dashboard
```

---

## Models Used

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier

The final model is selected based on the highest ROC-AUC score.

---

## Evaluation Metrics

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

Evaluation reports are automatically saved inside:

```text
reports/metrics/
```

Visualizations are saved inside:

```text
reports/plots/
```

---

## Streamlit Dashboard Features

The dashboard includes:

* Customer Information Form
* Churn Prediction
* Churn Probability
* Risk Level Indicator
* SHAP Feature Importance
* Downloadable Prediction Report

Risk Categories

| Probability | Risk Level |
| ----------- | ---------- |
| Below 30%   | Low        |
| 30%–60%     | Medium     |
| Above 60%   | High       |

---

## Sample Prediction

Input:

```json
{
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "TechSupport": "No",
  "tenure": 12,
  "MonthlyCharges": 85
}
```

Output:

```text
Prediction : Churn

Probability : 91%

Risk Level : High

Recommended Action :
Offer a retention discount and proactive customer support.
```

---

## Live Demo

Try the deployed application:

[**https://ai-customer-churn-prediction-ykqrabvby54bjzzdr3b2zm.streamlit.app/**](https://ai-customer-churn-prediction-ykqrabvby54bjzzdr3b2zm.streamlit.app/)

You can:

* Predict customer churn in real time.
* View churn probability.
* Explore SHAP explanations.
* Test different customer profiles.
* Download prediction reports.

---

## Future Improvements

* FastAPI REST API
* Docker deployment
* MLflow experiment tracking
* Customer segmentation using K-Means
* Retention recommendation engine
* PostgreSQL database integration
* Authentication and user management
* Cloud deployment on AWS or Azure

---

## Learning Outcomes

This project demonstrates practical experience in:

* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis
* Machine Learning
* Hyperparameter Optimization
* Model Evaluation
* Explainable AI
* Dashboard Development
* Model Deployment
* Software Project Structure
* Git and GitHub

---

## License

This project is licensed under the MIT License.

---

## Author

**Deepak K Kushwaha**

B.Tech CSE (AI & ML)

Machine Learning • Data Science • AI Engineering

GitHub: https://github.com/<your-username>

---

⭐ If you found this project helpful, consider giving it a star on GitHub!
