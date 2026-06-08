"""
Feature engineering module for Customer Churn Prediction.

Creates tenure groups, spending metrics, service counts, and contract encoding.
"""

import pandas as pd

from utils import setup_logging


class FeatureEngineer:
    """Engineer domain-specific features from raw Telco customer data."""

    SERVICE_COLS = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    CONTRACT_MAP = {
        "Month-to-month": 1,
        "One year": 12,
        "Two year": 24,
    }

    TENURE_BINS = [0, 12, 24, 48, 72]
    TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]

    def __init__(self) -> None:
        """Initialize feature engineer with logger."""
        self.logger = setup_logging(self.__class__.__name__)

    def create_tenure_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bin tenure into meaningful customer lifecycle groups.

        Args:
            df: Input DataFrame with 'tenure' column.

        Returns:
            DataFrame with 'TenureGroup' column added.
        """
        data = df.copy()
        data["TenureGroup"] = pd.cut(
            data["tenure"],
            bins=self.TENURE_BINS,
            labels=self.TENURE_LABELS,
            include_lowest=True,
        )
        data["TenureGroup"] = data["TenureGroup"].astype(str)
        return data

    def create_avg_monthly_spending(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute average monthly spending from total and tenure.

        Args:
            df: Input DataFrame with MonthlyCharges and TotalCharges.

        Returns:
            DataFrame with 'AvgMonthlySpending' column.
        """
        data = df.copy()
        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        tenure_safe = data["tenure"].replace(0, 1)
        data["AvgMonthlySpending"] = (data["TotalCharges"] / tenure_safe).round(2)
        data["AvgMonthlySpending"] = data["AvgMonthlySpending"].fillna(
            data["MonthlyCharges"]
        )
        return data

    def create_service_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count number of subscribed add-on services.

        Args:
            df: Input DataFrame with service columns.

        Returns:
            DataFrame with 'ServiceCount' column.
        """
        data = df.copy()
        available = [c for c in self.SERVICE_COLS if c in data.columns]
        data["ServiceCount"] = sum(
            (data[col] == "Yes").astype(int) for col in available
        )
        return data

    def encode_contract_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode contract type as approximate duration in months.

        Args:
            df: Input DataFrame with 'Contract' column.

        Returns:
            DataFrame with 'ContractDuration' column.
        """
        data = df.copy()
        data["ContractDuration"] = data["Contract"].map(self.CONTRACT_MAP).fillna(1)
        return data

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: Raw input DataFrame.

        Returns:
            DataFrame with engineered features.
        """
        self.logger.info("Starting feature engineering on %d records", len(df))
        data = self.create_tenure_groups(df)
        data = self.create_avg_monthly_spending(data)
        data = self.create_service_count(data)
        data = self.encode_contract_duration(data)
        self.logger.info("Feature engineering complete. Columns: %d", len(data.columns))
        return data
