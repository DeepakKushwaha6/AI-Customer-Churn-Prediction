# AI-Powered Customer Churn Prediction System

A complete end-to-end machine learning system that predicts whether a telecom customer is likely to churn (leave the company) based on demographics, account information, and service usage patterns.

## Features

- **Data Preprocessing** — Missing value imputation, duplicate removal, categorical encoding, numerical scaling, stratified train-test split
- **Feature Engineering** — Tenure groups, average monthly spending, service count, contract duration encoding
- **Multiple ML Models** — Logistic Regression, Random Forest, XGBoost, Gradient Boosting
- **Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation
- **Model Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
- **Explainable AI** — SHAP feature contribution analysis
- **Streamlit Dashboard** — Interactive prediction interface with risk indicators and downloadable reports
- **Synthetic Data Fallback** — Automatically generates Telco-like data if the dataset is missing

## Project Structure

```
customer_churn_prediction/
├── data/
│   ├── raw/                  # Raw Telco dataset CSV
│   └── processed/            # Engineered dataset
├── notebooks/
│   └── EDA.ipynb             # Exploratory data analysis
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
├── models/
│   ├── churn_model.pkl       # Best trained model
│   ├── preprocessor.pkl      # Fitted preprocessor
│   └── model_metadata.pkl    # Feature names and tuning results
├── app/
│   └── streamlit_app.py      # Interactive dashboard
├── reports/
│   ├── plots/                # Generated visualizations
│   └── metrics/              # Evaluation metrics JSON
├── requirements.txt
├── README.md
└── main.py                   # Pipeline entry point
```

## Environment Setup

### Prerequisites

- Python 3.12 or higher
- pip

### Installation

```bash
# Clone or navigate to the project directory
cd customer_churn_prediction

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Place the Telco Customer Churn dataset at:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or [IBM Sample Data](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

If the dataset is not present, the pipeline automatically generates synthetic Telco-like data.

## Usage

### 1. Run the Full Pipeline

Trains all models, selects the best performer, saves artifacts, and generates reports:

```bash
python main.py
```

### 2. Launch the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

### 3. Exploratory Data Analysis

```bash
jupyter notebook notebooks/EDA.ipynb
```

## Model Training Details

| Model              | Hyperparameters Tuned                          |
|--------------------|------------------------------------------------|
| Logistic Regression| C, penalty, solver                             |
| Random Forest      | n_estimators, max_depth, min_samples_split     |
| XGBoost            | n_estimators, max_depth, learning_rate         |
| Gradient Boosting  | n_estimators, max_depth, learning_rate         |

Selection criterion: highest ROC-AUC on the held-out test set.

## Evaluation Metrics

The system reports:

- **Accuracy** — Overall correct predictions
- **Precision** — True positives / predicted positives
- **Recall** — True positives / actual positives
- **F1 Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under the receiver operating characteristic curve
- **Confusion Matrix** — True/false positive and negative counts

Results are saved to `reports/metrics/evaluation_metrics.json` and visualizations to `reports/plots/`.

## Streamlit Dashboard

The dashboard provides:

1. **Customer Input Form** — All Telco features with sensible defaults
2. **Churn Prediction** — Binary label and probability
3. **Risk Level Indicator** — Low (<30%), Medium (30–60%), High (>60%)
4. **SHAP Feature Chart** — Top contributors to the prediction
5. **Download Report** — Text summary of the prediction

## API Usage (Programmatic)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src")))
from predict import ChurnPredictor

predictor = ChurnPredictor()
predictor.load_artifacts()

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0,
}

result = predictor.predict(customer)
print(f"Churn: {result.churn_prediction} ({result.churn_probability:.1%})")
print(f"Risk: {result.risk_level}")
```

## Deployment

For production deployment:

1. Train the model: `python main.py`
2. Copy `models/`, `src/`, and `app/` to your server
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run app/streamlit_app.py --server.port 8501`

For containerized deployment, wrap the Streamlit app in a Docker image with the trained model artifacts included.

## License

MIT License — free to use for educational and commercial purposes.
