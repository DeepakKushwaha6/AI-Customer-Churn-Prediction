"""
Prediction module for Customer Churn Prediction.

Loads trained model and preprocessor for single and batch predictions.
Includes SHAP-based explainability.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from utils import ensure_directories, get_project_root, setup_logging


@dataclass
class PredictionResult:
    """Container for a single customer prediction."""

    churn_probability: float
    churn_prediction: str
    risk_level: str
    shap_values: dict[str, float] | None = None


class ChurnPredictor:
    """Load artifacts and predict customer churn with explanations."""

    RISK_THRESHOLDS = {"Low": 0.30, "Medium": 0.60, "High": 1.0}

    def __init__(
        self,
        model_path: str | None = None,
        preprocessor_path: str | None = None,
    ) -> None:
        """
        Initialize predictor with optional model paths.

        Args:
            model_path: Path to saved model pickle.
            preprocessor_path: Path to saved preprocessor pickle.
        """
        self.logger = setup_logging(self.__class__.__name__)
        root = get_project_root()
        self.model_path = model_path or str(root / "models" / "churn_model.pkl")
        self.preprocessor_path = preprocessor_path or str(
            root / "models" / "preprocessor.pkl"
        )
        self.model: Any = None
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.feature_names: list[str] = []
        self._shap_explainer: Any = None

    def load_artifacts(self) -> None:
        """Load model, preprocessor, and feature names from disk."""
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor.load(self.preprocessor_path)

            meta_path = Path(self.model_path).parent / "model_metadata.pkl"
            if meta_path.exists():
                meta = joblib.load(meta_path)
                self.feature_names = meta.get("feature_names", [])
                self.logger.info("Loaded metadata with %d features", len(self.feature_names))

            self.logger.info("Artifacts loaded successfully")
        except FileNotFoundError as exc:
            self.logger.error("Artifact not found: %s", exc)
            raise

    def _classify_risk(self, probability: float) -> str:
        """Map churn probability to risk level."""
        if probability < self.RISK_THRESHOLDS["Low"]:
            return "Low"
        if probability < self.RISK_THRESHOLDS["Medium"]:
            return "Medium"
        return "High"

    def _prepare_input(self, customer_data: dict[str, Any] | pd.DataFrame) -> np.ndarray:
        """
        Apply feature engineering and preprocessing to input.

        Args:
            customer_data: Raw customer record(s).

        Returns:
            Processed feature array.
        """
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()

        df = self.feature_engineer.transform(df)
        return self.preprocessor.transform_single(df)

    def predict(self, customer_data: dict[str, Any] | pd.DataFrame) -> PredictionResult:
        """
        Predict churn for a single customer.

        Args:
            customer_data: Raw customer features.

        Returns:
            PredictionResult with probability, label, and risk level.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_artifacts() first.")

        X = self._prepare_input(customer_data)
        if self.feature_names and X.ndim == 2:
            X_df = pd.DataFrame(X, columns=self.feature_names[: X.shape[1]])
            proba = float(self.model.predict_proba(X_df)[0, 1])
        else:
            proba = float(self.model.predict_proba(X)[0, 1])
        pred_label = "Yes" if proba >= 0.5 else "No"
        risk = self._classify_risk(proba)

        shap_contributions = self.explain_prediction(X)

        return PredictionResult(
            churn_probability=proba,
            churn_prediction=pred_label,
            risk_level=risk,
            shap_values=shap_contributions,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn for multiple customers.

        Args:
            df: DataFrame of raw customer records.

        Returns:
            DataFrame with predictions appended.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_artifacts() first.")

        engineered = self.feature_engineer.transform(df)
        drop_cols = DataPreprocessor.DROP_COLS
        X_raw = engineered.drop(columns=drop_cols, errors="ignore")

        if self.preprocessor.preprocessor is None:
            raise RuntimeError("Preprocessor not loaded.")

        X = self.preprocessor.preprocessor.transform(X_raw)
        probas = self.model.predict_proba(X)[:, 1]
        preds = np.where(probas >= 0.5, "Yes", "No")
        risks = [self._classify_risk(p) for p in probas]

        result = df.copy()
        result["ChurnProbability"] = probas.round(4)
        result["ChurnPrediction"] = preds
        result["RiskLevel"] = risks
        return result

    def explain_prediction(self, X_processed: np.ndarray) -> dict[str, float] | None:
        """
        Compute SHAP feature contributions for a prediction.

        Args:
            X_processed: Preprocessed feature array.

        Returns:
            Dictionary mapping feature names to SHAP values, or None.
        """
        try:
            import shap
        except ImportError:
            self.logger.warning("SHAP not installed. Skipping explainability.")
            return None

        if not self.feature_names:
            self.logger.warning("Feature names unavailable for SHAP.")
            return None

        try:
            if self._shap_explainer is None:
                if hasattr(self.model, "feature_importances_"):
                    self._shap_explainer = shap.TreeExplainer(self.model)
                else:
                    self._shap_explainer = shap.Explainer(
                        self.model.predict_proba, X_processed
                    )

            shap_values = self._shap_explainer.shap_values(X_processed)
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0]

            contributions = {
                name: float(val)
                for name, val in zip(self.feature_names, values, strict=False)
            }
            return dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
        except Exception as exc:
            self.logger.warning("SHAP explanation failed: %s", exc)
            return None

    def generate_report(self, result: PredictionResult, customer_id: str = "N/A") -> str:
        """
        Generate a text prediction report for download.

        Args:
            result: PredictionResult object.
            customer_id: Customer identifier.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 50,
            "CUSTOMER CHURN PREDICTION REPORT",
            "=" * 50,
            f"Customer ID: {customer_id}",
            f"Churn Prediction: {result.churn_prediction}",
            f"Churn Probability: {result.churn_probability:.2%}",
            f"Risk Level: {result.risk_level}",
            "",
        ]

        if result.shap_values:
            lines.append("Top Feature Contributions (SHAP):")
            for feat, val in list(result.shap_values.items())[:10]:
                direction = "increases" if val > 0 else "decreases"
                lines.append(f"  - {feat}: {val:+.4f} ({direction} churn risk)")

        lines.append("=" * 50)
        return "\n".join(lines)
