"""Domain-specific Plotly charts for the manufacturing defect app."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


STATUS_LABELS = {
    0: "Low Defect",
    1: "High Defect",
}


def defect_status_distribution_chart(
    class_balance: pd.DataFrame,
    count_column: str = "Count",
) -> go.Figure:
    """Create a bar chart showing defect-status counts and percentages."""
    chart_data = class_balance.reset_index().rename(columns={"index": "DefectStatus"})
    chart_data["Defect Status"] = chart_data["DefectStatus"].map(STATUS_LABELS)
    chart_data["Label"] = chart_data.apply(
        lambda row: f"{int(row[count_column]):,}<br>{row['Percent']:.1f}%",
        axis=1,
    )

    fig = px.bar(
        chart_data,
        x="Defect Status",
        y=count_column,
        text="Label",
        title="Production Records by Defect Status",
        color="Defect Status",
        hover_data={"Percent": ":.1f", count_column: ":,"},
        color_discrete_map={
            "Low Defect": "#2A9D8F",
            "High Defect": "#E76F51",
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="Record Count")
    return fig


def defect_rate_histogram_by_status(df: pd.DataFrame) -> go.Figure:
    """Create a defect-rate histogram split by defect status."""
    chart_data = df.copy()
    chart_data["Defect Status"] = chart_data["DefectStatus"].map(STATUS_LABELS)

    return px.histogram(
        chart_data,
        x="DefectRate",
        color="Defect Status",
        nbins=30,
        barmode="overlay",
        marginal="box",
        title="Defect Rate Profile by Defect Status",
        color_discrete_map={
            "Low Defect": "#2A9D8F",
            "High Defect": "#E76F51",
        },
    )


def standardized_difference_bar_chart(
    difference_summary: pd.DataFrame,
    limit: int = 10,
) -> go.Figure:
    """Create a horizontal bar chart of standardized group differences."""
    chart_data = difference_summary.copy()
    chart_data = chart_data.reindex(
        chart_data["Standardized Difference"].abs().sort_values(ascending=False).index
    ).head(limit)
    chart_data = chart_data.reset_index().rename(columns={"index": "Feature"})
    chart_data = chart_data.sort_values("Standardized Difference")

    return px.bar(
        chart_data,
        x="Standardized Difference",
        y="Feature",
        orientation="h",
        title="Top Process Separators: High Defect vs Low Defect",
        color="Standardized Difference",
        color_continuous_scale="RdBu_r",
    )


def selected_feature_boxplot_by_status(
    df: pd.DataFrame,
    feature: str,
) -> go.Figure:
    """Create a boxplot for one feature split by defect status."""
    chart_data = df.copy()
    chart_data["Defect Status"] = chart_data["DefectStatus"].map(STATUS_LABELS)

    return px.box(
        chart_data,
        x="Defect Status",
        y=feature,
        color="Defect Status",
        points="outliers",
        title=f"{feature}: Low vs High Defect Distribution",
        color_discrete_map={
            "Low Defect": "#2A9D8F",
            "High Defect": "#E76F51",
        },
    )


def correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """Create a heatmap for a correlation matrix."""
    return px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Process Signal Correlation Map",
    )


def model_metrics_bar_chart(
    model_results: pd.DataFrame,
) -> go.Figure:
    """Create a grouped bar chart comparing model metrics."""
    metric_columns = ["Accuracy", "ROC-AUC", "Average Precision"]
    chart_data = model_results.drop(columns=["Fitted Model"], errors="ignore").melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    return px.bar(
        chart_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        range_y=[0, 1],
        title="Classifier Performance Benchmark",
    )


def confusion_matrix_heatmap(confusion_matrix_df: pd.DataFrame) -> go.Figure:
    """Create a labeled heatmap for a confusion matrix DataFrame."""
    return px.imshow(
        confusion_matrix_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Random Forest Prediction Matrix",
        labels={"x": "Predicted Label", "y": "Actual Label", "color": "Count"},
    )


def feature_importance_chart(
    importance_df: pd.DataFrame,
    limit: int = 10,
) -> go.Figure:
    """Create a horizontal bar chart for model feature importance."""
    chart_data = importance_df.head(limit).sort_values("Importance")

    return px.bar(
        chart_data,
        x="Importance",
        y="Feature",
        orientation="h",
        error_x="Std" if "Std" in chart_data.columns else None,
        title="Random Forest Driver Importance",
        color="Importance",
        color_continuous_scale="Teal",
    )
