"""
Model evaluation module with metrics and visualization.

Computes classification metrics and generates publication-quality plots.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils import ensure_directories, setup_logging


class ModelEvaluator:
    """Evaluate churn prediction models and generate reports."""

    def __init__(self) -> None:
        """Initialize evaluator with logging and plot style."""
        self.logger = setup_logging(self.__class__.__name__)
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
            }
        )

    def compute_metrics(
        self, y_true: np.ndarray | pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities for positive class.

        Returns:
            Dictionary of metric names and values.
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if y_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        return metrics

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
    ) -> dict[str, Any]:
        """
        Full evaluation: metrics, confusion matrix, classification report.

        Args:
            model: Fitted classifier.
            X_test: Test features.
            y_test: Test labels.
            model_name: Name for reporting.

        Returns:
            Evaluation results dictionary.
        """
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        metrics = self.compute_metrics(y_test, y_pred, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.logger.info(
            "%s - Accuracy: %.4f, F1: %.4f, ROC-AUC: %.4f",
            model_name,
            metrics["accuracy"],
            metrics["f1_score"],
            metrics.get("roc_auc", 0.0),
        )

        return {
            "model_name": model_name,
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    def plot_confusion_matrix(
        self, cm: np.ndarray, model_name: str, save: bool = True
    ) -> plt.Figure:
        """Plot and optionally save confusion matrix heatmap."""
        dirs = ensure_directories()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()

        if save:
            path = dirs["plots"] / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(path, bbox_inches="tight")
            self.logger.info("Confusion matrix saved to %s", path)
        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        save: bool = True,
    ) -> plt.Figure:
        """Plot ROC curve with AUC annotation."""
        dirs = ensure_directories()
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#2E86AB", linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_name}")
        ax.legend(loc="lower right")
        plt.tight_layout()

        if save:
            path = dirs["plots"] / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(path, bbox_inches="tight")
            self.logger.info("ROC curve saved to %s", path)
        return fig

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: list[str],
        model_name: str,
        top_n: int = 15,
        save: bool = True,
    ) -> plt.Figure | None:
        """Plot feature importance for tree-based models."""
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        if importances is None:
            self.logger.warning("Model does not support feature importance.")
            return None

        dirs = ensure_directories()
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_values = importances[indices]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.color_palette("viridis", len(top_features))
        ax.barh(range(len(top_features)), top_values[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance - {model_name}")
        plt.tight_layout()

        if save:
            path = dirs["plots"] / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(path, bbox_inches="tight")
            self.logger.info("Feature importance plot saved to %s", path)
        return fig

    def plot_model_comparison(
        self, results: dict[str, dict], save: bool = True
    ) -> plt.Figure:
        """Bar chart comparing metrics across models."""
        dirs = ensure_directories()
        metrics_df = pd.DataFrame(
            {name: res["metrics"] for name, res in results.items()}
        ).T

        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot(
            kind="bar", ax=ax, rot=0
        )
        ax.set_title("Model Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        plt.tight_layout()

        if save:
            path = dirs["plots"] / "model_comparison.png"
            fig.savefig(path, bbox_inches="tight")
            self.logger.info("Model comparison plot saved to %s", path)
        return fig

    def save_metrics(self, results: dict[str, Any], filename: str = "evaluation_metrics.json") -> Path:
        """Save evaluation metrics to JSON."""
        dirs = ensure_directories()
        path = dirs["metrics"] / filename

        serializable = {}
        for name, res in results.items():
            serializable[name] = {
                "metrics": res["metrics"],
                "confusion_matrix": res["confusion_matrix"],
                "classification_report": res["classification_report"],
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        self.logger.info("Metrics saved to %s", path)
        return path

    def generate_full_report(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str],
        all_models: dict[str, Any],
        best_name: str,
    ) -> dict[str, Any]:
        """
        Generate complete evaluation report with plots and metrics.

        Args:
            model: Best fitted model.
            X_test: Test features.
            y_test: Test labels.
            feature_names: List of feature names.
            all_models: All trained models.
            best_name: Name of best model.

        Returns:
            Evaluation results for all models.
        """
        results = {}
        for name, mdl in all_models.items():
            results[name] = self.evaluate_model(mdl, X_test, y_test, name)

        best_result = results[best_name]
        self.plot_confusion_matrix(
            np.array(best_result["confusion_matrix"]), best_name
        )
        if best_result["y_proba"] is not None:
            self.plot_roc_curve(y_test, best_result["y_proba"], best_name)
        self.plot_feature_importance(model, feature_names, best_name)
        self.plot_model_comparison(results)
        self.save_metrics(results)

        plt.close("all")
        return results
