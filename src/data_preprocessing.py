"""
Data preprocessing module for Customer Churn Prediction.

Handles missing values, duplicates, encoding, scaling, and train-test splitting.
"""

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from utils import setup_logging


@dataclass
class PreprocessingResult:
    """Container for preprocessed data and fitted transformers."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder
    feature_names: list[str] = field(default_factory=list)


class DataPreprocessor:
    """Preprocess raw Telco customer data for machine learning."""

    CATEGORICAL_COLS = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "TenureGroup",
    ]
    NUMERICAL_COLS = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AvgMonthlySpending",
        "ServiceCount",
        "ContractDuration",
    ]
    DROP_COLS = ["customerID", "Churn"]

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Initialize the preprocessor.

        Args:
            test_size: Fraction of data for testing.
            random_state: Random seed for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.logger = setup_logging(self.__class__.__name__)
        self.preprocessor: ColumnTransformer | None = None
        self.label_encoder = LabelEncoder()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw dataset: handle missing values and remove duplicates.

        Args:
            df: Raw input DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        data = df.copy()
        initial_rows = len(data)

        data = data.drop_duplicates()
        if len(data) < initial_rows:
            self.logger.info("Removed %d duplicate rows", initial_rows - len(data))

        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
            missing_count = data["TotalCharges"].isna().sum()
            if missing_count > 0:
                median_val = data["TotalCharges"].median()
                data["TotalCharges"] = data["TotalCharges"].fillna(median_val)
                self.logger.info("Imputed %d missing TotalCharges with median", missing_count)

        data = data.dropna()
        self.logger.info("Cleaned dataset shape: %s", data.shape)
        return data

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode Churn target (Yes/No) to binary labels."""
        return self.label_encoder.fit_transform(y)

    def build_preprocessor(
        self, categorical_cols: list[str], numerical_cols: list[str]
    ) -> ColumnTransformer:
        """
        Build sklearn ColumnTransformer for encoding and scaling.

        Args:
            categorical_cols: Columns to one-hot encode.
            numerical_cols: Columns to standardize.

        Returns:
            Unfitted ColumnTransformer.
        """
        return ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                ),
                ("num", StandardScaler(), numerical_cols),
            ],
            remainder="drop",
        )

    def fit_transform(
        self, df: pd.DataFrame, engineered_cols: list[str] | None = None
    ) -> PreprocessingResult:
        """
        Full preprocessing pipeline: clean, split, encode, scale.

        Args:
            df: Input DataFrame with engineered features.
            engineered_cols: Optional extra columns to include.

        Returns:
            PreprocessingResult with train/test splits and fitted transformers.
        """
        data = self.clean_data(df)

        if "Churn" not in data.columns:
            raise ValueError("Target column 'Churn' not found in dataset.")

        y = self.encode_target(data["Churn"])
        X = data.drop(columns=self.DROP_COLS, errors="ignore")

        cat_cols = [c for c in self.CATEGORICAL_COLS if c in X.columns]
        num_cols = [c for c in self.NUMERICAL_COLS if c in X.columns]

        if engineered_cols:
            for col in engineered_cols:
                if col in X.columns and col not in cat_cols + num_cols:
                    if X[col].dtype == "object":
                        cat_cols.append(col)
                    else:
                        num_cols.append(col)

        self.preprocessor = self.build_preprocessor(cat_cols, num_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        feature_names = list(
            self.preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
        ) + num_cols

        self.logger.info(
            "Train set: %s, Test set: %s, Features: %d",
            X_train_processed.shape,
            X_test_processed.shape,
            len(feature_names),
        )

        return PreprocessingResult(
            X_train=pd.DataFrame(X_train_processed, columns=feature_names),
            X_test=pd.DataFrame(X_test_processed, columns=feature_names),
            y_train=pd.Series(y_train, name="Churn"),
            y_test=pd.Series(y_test, name="Churn"),
            preprocessor=self.preprocessor,
            label_encoder=self.label_encoder,
            feature_names=feature_names,
        )

    def transform_single(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a single customer record using fitted preprocessor.

        Args:
            df: Single-row DataFrame with raw features.

        Returns:
            Processed feature array.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not fitted. Run fit_transform first.")
        X = df.drop(columns=self.DROP_COLS, errors="ignore")
        return self.preprocessor.transform(X)

    def save(self, path: str) -> None:
        """Save fitted preprocessor and label encoder."""
        if self.preprocessor is None:
            raise RuntimeError("Nothing to save. Preprocessor not fitted.")
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "label_encoder": self.label_encoder,
                "categorical_cols": self.CATEGORICAL_COLS,
                "numerical_cols": self.NUMERICAL_COLS,
            },
            path,
        )
        self.logger.info("Preprocessor saved to %s", path)

    def load(self, path: str) -> None:
        """Load fitted preprocessor and label encoder."""
        artifacts = joblib.load(path)
        self.preprocessor = artifacts["preprocessor"]
        self.label_encoder = artifacts["label_encoder"]
        self.logger.info("Preprocessor loaded from %s", path)
