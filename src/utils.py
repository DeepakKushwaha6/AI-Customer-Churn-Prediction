"""
Utility functions for the Customer Churn Prediction System.

Provides logging setup, path management, and synthetic data generation
for the Telco Customer Churn dataset when raw data is unavailable.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def setup_logging(name: str = "churn_prediction", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_directories() -> dict[str, Path]:
    """
    Create required project directories if they do not exist.

    Returns:
        Dictionary mapping directory names to Path objects.
    """
    root = get_project_root()
    dirs = {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "models": root / "models",
        "plots": root / "reports" / "plots",
        "metrics": root / "reports" / "metrics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def generate_synthetic_telco_data(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Telco Customer Churn dataset.

    Mimics the structure and distributions of the real Telco dataset
    when the original CSV is not available.

    Args:
        n_samples: Number of customer records to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with Telco-like columns and Churn target.
    """
    rng = np.random.default_rng(random_state)

    gender = rng.choice(["Male", "Female"], n_samples)
    senior = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, n_samples)
    phone = rng.choice(["Yes", "No"], n_samples, p=[0.90, 0.10])
    multiple = rng.choice(["Yes", "No", "No phone service"], n_samples, p=[0.42, 0.48, 0.10])
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
    )

    def service_choice(has_internet: bool) -> str:
        if not has_internet:
            return "No internet service"
        return rng.choice(["Yes", "No"], p=[0.28, 0.72])

    online_security = [service_choice(i != "No") for i in internet]
    online_backup = [service_choice(i != "No") for i in internet]
    device_protection = [service_choice(i != "No") for i in internet]
    tech_support = [service_choice(i != "No") for i in internet]
    streaming_tv = [service_choice(i != "No") for i in internet]
    streaming_movies = [service_choice(i != "No") for i in internet]

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.55, 0.21, 0.24],
    )
    paperless = rng.choice(["Yes", "No"], n_samples, p=[0.59, 0.41])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n_samples,
        p=[0.34, 0.19, 0.24, 0.23],
    )

    monthly = np.round(rng.uniform(18, 120, n_samples), 2)
    monthly[internet == "Fiber optic"] += rng.uniform(10, 30, sum(internet == "Fiber optic"))
    monthly[internet == "No"] = np.round(rng.uniform(18, 45, sum(internet == "No")), 2)
    total = np.round(monthly * np.maximum(tenure, 1) + rng.normal(0, 50, n_samples), 2)
    total = np.maximum(total, monthly)

    # Churn probability based on behavioral signals
    churn_prob = np.full(n_samples, 0.15)
    churn_prob[contract == "Month-to-month"] += 0.25
    churn_prob[contract == "One year"] += 0.05
    churn_prob[tenure < 12] += 0.20
    churn_prob[monthly > 70] += 0.10
    churn_prob[internet == "Fiber optic"] += 0.08
    churn_prob[payment == "Electronic check"] += 0.12
    churn_prob[tech_support == np.array(["No"] * n_samples)] += 0.05
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = rng.binomial(1, churn_prob)
    churn_label = np.where(churn == 1, "Yes", "No")

    customer_ids = [f"{i:04d}-XXXXX" for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total.astype(str),
            "Churn": churn_label,
        }
    )

    # Introduce ~0.1% missing TotalCharges for realism
    missing_idx = rng.choice(n_samples, size=max(1, n_samples // 1000), replace=False)
    df.loc[missing_idx, "TotalCharges"] = " "

    return df


def load_or_generate_data(raw_path: Path | None = None) -> pd.DataFrame:
    """
    Load Telco dataset from disk or generate synthetic data.

    Args:
        raw_path: Optional path to raw CSV file.

    Returns:
        Loaded or generated DataFrame.
    """
    logger = setup_logging()
    dirs = ensure_directories()
    path = raw_path or dirs["raw"] / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if path.exists():
        logger.info("Loading dataset from %s", path)
        return pd.read_csv(path)

    logger.warning("Dataset not found at %s. Generating synthetic data.", path)
    df = generate_synthetic_telco_data()
    df.to_csv(path, index=False)
    logger.info("Synthetic dataset saved to %s", path)
    return df
