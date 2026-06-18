"""Baseline classification helpers from the original notebook."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_train_test_split(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create the stratified train/test split used by the notebook."""
    return train_test_split(
        df[features],
        df[target],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )


def build_models(random_state: int = 42) -> dict[str, object]:
    """Return the baseline models used by the notebook."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        ),
    }


def evaluate_models(models: dict[str, object], x_train, x_test, y_train, y_test) -> pd.DataFrame:
    """Fit models and return core metrics plus fitted model objects."""
    results = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
                "Average Precision": average_precision_score(y_test, probabilities),
                "Fitted Model": model,
            }
        )

    return pd.DataFrame(results)


def classification_report_table(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return a classification report as a Streamlit-friendly table."""
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def confusion_matrix_table(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[int] | None = None,
) -> pd.DataFrame:
    """Return a confusion matrix as a labeled DataFrame."""
    if labels is None:
        labels = [0, 1]

    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    return pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )


def permutation_importance_table(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Return permutation importance results for a fitted model."""
    importance = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
    )

    return pd.DataFrame(
        {
            "Feature": x_test.columns,
            "Importance": importance.importances_mean,
            "Std": importance.importances_std,
        }
    ).sort_values("Importance", ascending=False)
