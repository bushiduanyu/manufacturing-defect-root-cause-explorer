"""Reusable analysis calculations from the original notebook."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def missing_values(df: pd.DataFrame) -> pd.Series:
    """Return missing-value counts by column."""
    return df.isna().sum()


def class_balance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return target class counts and percentages."""
    counts = df[target].value_counts().sort_index()
    percents = df[target].value_counts(normalize=True).sort_index() * 100
    return pd.DataFrame({"Count": counts, "Percent": percents})


def grouped_means(df: pd.DataFrame, target: str, columns: list[str]) -> pd.DataFrame:
    """Return feature means grouped by target class."""
    return df.groupby(target)[columns].mean()


def group_difference_summary(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    low_label: int = 0,
    high_label: int = 1,
) -> pd.DataFrame:
    """Compare low-defect and high-defect groups with raw and standardized differences."""
    means = df.groupby(target)[columns].mean()
    stds = df.groupby(target)[columns].std()
    difference = means.loc[high_label] - means.loc[low_label]
    percent_difference = (difference / means.loc[low_label]) * 100
    pooled_std = np.sqrt((stds.loc[low_label] ** 2 + stds.loc[high_label] ** 2) / 2)

    return pd.DataFrame(
        {
            "Low Defect Avg": means.loc[low_label],
            "High Defect Avg": means.loc[high_label],
            "Difference": difference,
            "Percent Difference": percent_difference,
            "Standardized Difference": difference / pooled_std,
        }
    )


def sort_by_standardized_difference(
    summary: pd.DataFrame,
    ascending: bool = False,
    use_absolute: bool = True,
) -> pd.DataFrame:
    """Sort a difference summary by standardized difference."""
    sort_values = summary["Standardized Difference"]
    if use_absolute:
        sort_values = sort_values.abs()

    return summary.assign(_sort_value=sort_values).sort_values(
        "_sort_value",
        ascending=ascending,
    ).drop(columns="_sort_value")


def top_standardized_differences(
    summary: pd.DataFrame,
    limit: int = 10,
    use_absolute: bool = True,
) -> pd.DataFrame:
    """Return the largest standardized differences for display."""
    return sort_by_standardized_difference(
        summary,
        ascending=False,
        use_absolute=use_absolute,
    ).head(limit)


def summary_statistics_by_status(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> pd.DataFrame:
    """Return mean, median, standard deviation, and count by defect status."""
    summary = df.groupby(target)[columns].agg(["mean", "median", "std", "count"])
    return summary.round(4)


def effect_size_label(cohens_d: float) -> str:
    """Label Cohen's d effect size using common rough thresholds."""
    absolute_effect = abs(cohens_d)

    if absolute_effect < 0.2:
        return "negligible"
    if absolute_effect < 0.5:
        return "small"
    if absolute_effect < 0.8:
        return "medium"
    return "large"


def statistical_tests(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    low_label: int = 0,
    high_label: int = 1,
) -> pd.DataFrame:
    """Run Mann-Whitney U tests and Cohen's d calculations."""
    results = []

    for column in columns:
        low = df.loc[df[target] == low_label, column]
        high = df.loc[df[target] == high_label, column]
        _, p_value = mannwhitneyu(low, high, alternative="two-sided")
        pooled_std = np.sqrt((low.std() ** 2 + high.std() ** 2) / 2)
        cohens_d = (high.mean() - low.mean()) / pooled_std

        results.append(
            {
                "Variable": column,
                "Low Defect Median": low.median(),
                "High Defect Median": high.median(),
                "Mean Difference": high.mean() - low.mean(),
                "Cohen's d": cohens_d,
                "Effect Size": effect_size_label(cohens_d),
                "Mann-Whitney p-value": p_value,
                "Statistically Significant": p_value < 0.05,
            }
        )

    return pd.DataFrame(results).sort_values(
        "Cohen's d",
        key=lambda values: values.abs(),
        ascending=False,
    )


def format_statistical_tests_for_display(results: pd.DataFrame) -> pd.DataFrame:
    """Return statistical test results rounded and labeled for Streamlit tables."""
    display_results = results.copy()
    numeric_columns = [
        "Low Defect Median",
        "High Defect Median",
        "Mean Difference",
        "Cohen's d",
        "Mann-Whitney p-value",
    ]

    for column in numeric_columns:
        if column in display_results.columns:
            display_results[column] = display_results[column].round(4)

    if "Statistically Significant" in display_results.columns:
        display_results["Significant at 0.05"] = display_results[
            "Statistically Significant"
        ].map({True: "Yes", False: "No"})
        display_results = display_results.drop(columns="Statistically Significant")

    return display_results


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a correlation matrix for selected numeric columns."""
    return df[columns].corr()
