"""
Main entry point for the Customer Churn Prediction System.

Orchestrates data loading, preprocessing, training, evaluation, and artifact persistence.
"""

import sys
from pathlib import Path

# Add src to path for module imports
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from data_preprocessing import DataPreprocessor  # noqa: E402
from evaluate_model import ModelEvaluator  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402
from train_model import ModelTrainer  # noqa: E402
from utils import ensure_directories, load_or_generate_data, setup_logging  # noqa: E402


def run_eda_plots(df: pd.DataFrame, logger) -> None:
    """Generate EDA visualizations and save to reports/plots."""
    dirs = ensure_directories()
    sns.set_theme(style="whitegrid")

    # Churn distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    churn_counts = df["Churn"].value_counts()
    colors = ["#2E86AB", "#E94F37"]
    ax.pie(
        churn_counts,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Customer Churn Distribution")
    fig.savefig(dirs["plots"] / "churn_distribution.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved churn distribution plot")

    # Correlation heatmap (numerical features)
    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    num_df = df.copy()
    num_df["TotalCharges"] = pd.to_numeric(num_df["TotalCharges"], errors="coerce")
    num_df["ChurnBinary"] = (num_df["Churn"] == "Yes").astype(int)
    corr = num_df[num_cols + ["ChurnBinary"]].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.savefig(dirs["plots"] / "correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap")

    # Customer segmentation by contract and churn
    fig, ax = plt.subplots(figsize=(8, 5))
    seg = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    seg.plot(kind="bar", ax=ax, color=["#2E86AB", "#E94F37"])
    ax.set_title("Churn by Contract Type")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Count")
    ax.legend(title="Churn")
    plt.xticks(rotation=45)
    fig.savefig(dirs["plots"] / "segmentation_contract.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved segmentation chart")

    # Tenure vs Monthly Charges scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df.copy()
    plot_df["TotalCharges"] = pd.to_numeric(plot_df["TotalCharges"], errors="coerce")
    sns.scatterplot(
        data=plot_df,
        x="tenure",
        y="MonthlyCharges",
        hue="Churn",
        alpha=0.5,
        palette={"Yes": "#E94F37", "No": "#2E86AB"},
        ax=ax,
    )
    ax.set_title("Tenure vs Monthly Charges by Churn")
    fig.savefig(dirs["plots"] / "tenure_charges_scatter.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved tenure/charges scatter plot")


def main() -> None:
    """Execute the full churn prediction pipeline."""
    logger = setup_logging("main")
    logger.info("Starting Customer Churn Prediction Pipeline")

    dirs = ensure_directories()

    # 1. Load data
    df = load_or_generate_data()
    logger.info("Dataset loaded: %d records, %d columns", len(df), len(df.columns))

    # 2. EDA plots
    run_eda_plots(df, logger)

    # 3. Feature engineering
    engineer = FeatureEngineer()
    df_engineered = engineer.transform(df)
    df_engineered.to_csv(dirs["processed"] / "processed_data.csv", index=False)

    # 4. Preprocessing
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    result = preprocessor.fit_transform(df_engineered)

    # 5. Train models with hyperparameter tuning
    trainer = ModelTrainer(random_state=42, cv_folds=5)
    trainer.train_baseline(result.X_train, result.y_train)
    trainer.tune_hyperparameters(result.X_train, result.y_train)
    best_name, best_model = trainer.select_best_model(result.X_test, result.y_test)

    # 6. Evaluate models
    evaluator = ModelEvaluator()
    evaluation = evaluator.generate_full_report(
        best_model,
        result.X_test,
        result.y_test,
        result.feature_names,
        trainer.result.models,
        best_name,
    )

    # 7. Save artifacts
    model_path = trainer.save_model(best_model)
    preprocessor.save(str(dirs["models"] / "preprocessor.pkl"))

    joblib.dump(
        {
            "best_model_name": best_name,
            "feature_names": result.feature_names,
            "cv_scores": trainer.result.cv_scores,
            "grid_results": trainer.result.grid_results,
        },
        dirs["models"] / "model_metadata.pkl",
    )

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("Best model: %s", best_name)
    logger.info(
        "Test metrics - Accuracy: %.4f, F1: %.4f, ROC-AUC: %.4f",
        evaluation[best_name]["metrics"]["accuracy"],
        evaluation[best_name]["metrics"]["f1_score"],
        evaluation[best_name]["metrics"].get("roc_auc", 0),
    )
    logger.info("Model saved to: %s", model_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
