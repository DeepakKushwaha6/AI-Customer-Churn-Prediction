"""
Model training module with hyperparameter tuning and cross-validation.

Trains Logistic Regression, Random Forest, XGBoost, and Gradient Boosting.
"""

from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier

from utils import ensure_directories, setup_logging


@dataclass
class TrainingResult:
    """Container for trained models and tuning results."""

    models: dict[str, Any] = field(default_factory=dict)
    best_model_name: str = ""
    best_model: Any = None
    cv_scores: dict[str, float] = field(default_factory=dict)
    grid_results: dict[str, Any] = field(default_factory=dict)


class ModelTrainer:
    """Train and tune multiple classifiers for churn prediction."""

    def __init__(self, random_state: int = 42, cv_folds: int = 5) -> None:
        """
        Initialize model trainer.

        Args:
            random_state: Random seed.
            cv_folds: Number of cross-validation folds.
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.logger = setup_logging(self.__class__.__name__)
        self.result = TrainingResult()

    def _base_models(self) -> dict[str, Any]:
        """Define base model configurations."""
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                random_state=self.random_state,
                eval_metric="logloss",
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200,
                random_state=self.random_state,
            ),
        }

    def _param_grids(self) -> dict[str, dict]:
        """Define hyperparameter search grids."""
        return {
            "LogisticRegression": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs"],
            },
            "RandomForest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
            },
            "GradientBoosting": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        }

    def train_baseline(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> dict[str, Any]:
        """
        Train all base models without hyperparameter tuning.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Dictionary of fitted models.
        """
        models = {}
        for name, model in self._base_models().items():
            self.logger.info("Training baseline %s...", name)
            model.fit(X_train, y_train)
            scores = cross_val_score(
                model, X_train, y_train, cv=self.cv_folds, scoring="roc_auc"
            )
            self.result.cv_scores[name] = float(np.mean(scores))
            models[name] = model
            self.logger.info("%s CV ROC-AUC: %.4f (+/- %.4f)", name, scores.mean(), scores.std())
        self.result.models = models
        return models

    def tune_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> dict[str, Any]:
        """
        Perform GridSearchCV for each model.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Dictionary of best tuned models.
        """
        tuned_models = {}
        param_grids = self._param_grids()

        for name, base_model in self._base_models().items():
            self.logger.info("Tuning %s with GridSearchCV...", name)
            grid = GridSearchCV(
                base_model,
                param_grids[name],
                cv=self.cv_folds,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
            )
            grid.fit(X_train, y_train)
            tuned_models[name] = grid.best_estimator_
            self.result.grid_results[name] = {
                "best_params": grid.best_params_,
                "best_score": float(grid.best_score_),
            }
            self.logger.info(
                "%s best ROC-AUC: %.4f, params: %s",
                name,
                grid.best_score_,
                grid.best_params_,
            )

        self.result.models = tuned_models
        return tuned_models

    def select_best_model(
        self, X_test: pd.DataFrame, y_test: pd.Series, metric: str = "roc_auc"
    ) -> tuple[str, Any]:
        """
        Select best model based on test set performance.

        Args:
            X_test: Test features.
            y_test: Test labels.
            metric: Scoring metric name.

        Returns:
            Tuple of (best_model_name, best_model).
        """
        from sklearn.metrics import get_scorer

        scorer = get_scorer(metric)
        best_score = -1.0
        best_name = ""
        best_model = None

        for name, model in self.result.models.items():
            score = scorer(model, X_test, y_test)
            self.logger.info("%s test %s: %.4f", name, metric, score)
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model

        self.result.best_model_name = best_name
        self.result.best_model = best_model
        self.logger.info("Best model: %s (%s=%.4f)", best_name, metric, best_score)
        return best_name, best_model

    def save_model(self, model: Any, path: str | None = None) -> str:
        """
        Persist best model to disk.

        Args:
            model: Fitted model to save.
            path: Optional save path.

        Returns:
            Path where model was saved.
        """
        dirs = ensure_directories()
        save_path = path or str(dirs["models"] / "churn_model.pkl")
        joblib.dump(model, save_path)
        self.logger.info("Model saved to %s", save_path)
        return save_path

    def load_model(self, path: str) -> Any:
        """Load a persisted model."""
        model = joblib.load(path)
        self.logger.info("Model loaded from %s", path)
        return model
