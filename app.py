"""Streamlit app for the Manufacturing Defect Root Cause Explorer."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.analysis import (
    class_balance,
    correlation_matrix,
    format_statistical_tests_for_display,
    group_difference_summary,
    sort_by_standardized_difference,
    statistical_tests,
    summary_statistics_by_status,
    top_standardized_differences,
)
from src.data_loader import load_dataset, validate_dataset, validation_summary
from src.feature_groups import (
    COMPARISON_FEATURES,
    FEATURE_GROUP_DESCRIPTIONS,
    FEATURE_GROUP_DISPLAY_NAMES,
    FEATURE_GROUPS,
    MODEL_FEATURES,
    TARGET_COLUMN,
)
from src.modeling import (
    build_models,
    classification_report_table,
    confusion_matrix_table,
    evaluate_models,
    make_train_test_split,
    permutation_importance_table,
)
from src.visualizations import (
    confusion_matrix_heatmap,
    correlation_heatmap,
    defect_rate_histogram_by_status,
    defect_status_distribution_chart,
    feature_importance_chart,
    model_metrics_bar_chart,
    selected_feature_boxplot_by_status,
    standardized_difference_bar_chart,
)


st.set_page_config(
    page_title="Manufacturing Defect Root Cause Explorer",
    page_icon="Factory",
    layout="wide",
)


STATUS_OPTIONS = {
    "All records": None,
    "Low defect only": 0,
    "High defect only": 1,
}


def apply_page_styles() -> None:
    """Apply compact product-dashboard styling."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h1 {
            margin-bottom: 0.25rem;
        }
        h2 {
            border-top: 1px solid #e2e8f0;
            padding-top: 1.2rem;
            margin-top: 1.6rem;
        }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
        }
        .section-note {
            color: #475569;
            font-size: 0.98rem;
            line-height: 1.45;
            margin-bottom: 0.85rem;
        }
        .small-note {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        .product-label {
            color: #0f766e;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, note: str) -> None:
    """Render a consistent section heading with engineering context."""
    st.markdown(f"## {title}")
    st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def status_label(value: int) -> str:
    """Return a readable defect status label."""
    return "High Defect" if value == 1 else "Low Defect"


@st.cache_data
def load_validated_data() -> tuple[pd.DataFrame, dict]:
    """Load and validate the manufacturing dataset."""
    data = load_dataset()
    validate_dataset(data)
    return data, validation_summary(data)


@st.cache_data
def build_analysis_outputs(data: pd.DataFrame) -> dict:
    """Create reusable analysis tables for the dashboard."""
    balance = class_balance(data, TARGET_COLUMN)
    difference_summary = group_difference_summary(
        data,
        TARGET_COLUMN,
        COMPARISON_FEATURES,
    )
    stats = statistical_tests(data, TARGET_COLUMN, COMPARISON_FEATURES)
    corr = correlation_matrix(data, COMPARISON_FEATURES + [TARGET_COLUMN])
    status_summary = summary_statistics_by_status(
        data,
        TARGET_COLUMN,
        COMPARISON_FEATURES,
    )

    return {
        "balance": balance,
        "difference_summary": difference_summary,
        "top_differences": top_standardized_differences(difference_summary),
        "sorted_differences": sort_by_standardized_difference(difference_summary),
        "statistical_tests": stats,
        "display_statistical_tests": format_statistical_tests_for_display(stats),
        "correlation": corr,
        "status_summary": status_summary,
    }


@st.cache_resource
def build_model_outputs(data: pd.DataFrame) -> dict:
    """Train baseline models and create model insight tables."""
    x_train, x_test, y_train, y_test = make_train_test_split(
        data,
        MODEL_FEATURES,
        TARGET_COLUMN,
    )
    model_results = evaluate_models(build_models(), x_train, x_test, y_train, y_test)
    random_forest = model_results.loc[
        model_results["Model"] == "Random Forest",
        "Fitted Model",
    ].iloc[0]

    return {
        "results": model_results,
        "random_forest": random_forest,
        "classification_report": classification_report_table(
            random_forest,
            x_test,
            y_test,
            target_names=["Low Defect", "High Defect"],
        ),
        "confusion_matrix": confusion_matrix_table(random_forest, x_test, y_test),
        "importance": permutation_importance_table(random_forest, x_test, y_test),
    }


def render_executive_summary(analysis_outputs: dict, model_outputs: dict) -> None:
    """Render a concise portfolio-style summary of the dashboard findings."""
    section_header(
        "Executive Summary",
        "The dashboard points to a focused set of process signals that deserve engineering attention before deeper plant-floor validation.",
    )

    balance = analysis_outputs["balance"]
    high_defect_share = balance.loc[1, "Percent"]
    top_driver = analysis_outputs["top_differences"].index[0]
    rf_metrics = model_outputs["results"].set_index("Model").loc["Random Forest"]
    top_model_features = ", ".join(model_outputs["importance"].head(3)["Feature"])

    c1, c2, c3 = st.columns(3)
    c1.metric("High-Defect Share", f"{high_defect_share:.1f}%")
    c2.metric("Largest Group Separator", top_driver)
    c3.metric("Random Forest ROC-AUC", f"{rf_metrics['ROC-AUC']:.3f}")

    st.markdown(
        f"""
        - High-defect records are the dominant operating condition in this dataset, so portfolio interpretation should emphasize separation patterns and class-aware model metrics.
        - The strongest process signals are concentrated around **{top_model_features}**, aligning model importance with the exploratory root-cause view.
        - Results should be treated as an engineering triage layer: they identify where to investigate, not what has been causally proven.
        """
    )


def render_overview(data: pd.DataFrame, validation: dict, analysis_outputs: dict) -> None:
    """Render dataset summary, validation, balance, and KPI overview."""
    section_header(
        "1. Operating Overview",
        "A plant-quality snapshot that checks data readiness, class balance, and the headline process indicators before deeper diagnosis.",
    )

    kpi_left, kpi_mid, kpi_right, kpi_extra = st.columns(4)
    kpi_left.metric("Production Records", f"{validation['row_count']:,}")
    kpi_mid.metric("Process Signals", validation["column_count"])
    kpi_right.metric("Missing Cells", validation["total_missing_values"])
    kpi_extra.metric("Data Readiness", "Ready" if validation["is_valid"] else "Review")

    st.markdown(
        '<div class="small-note">The data readiness check confirms the expected manufacturing signals, numeric analysis fields, and binary defect-status target are available for diagnosis.</div>',
        unsafe_allow_html=True,
    )

    balance_col, profile_col = st.columns([1.1, 1])
    with balance_col:
        st.plotly_chart(
            defect_status_distribution_chart(analysis_outputs["balance"]),
            width="stretch",
            key="overview_defect_status_distribution",
        )

    with profile_col:
        st.markdown("### Process Health Indicators")
        k1, k2 = st.columns(2)
        k3, k4 = st.columns(2)
        k1.metric("Mean Defect Rate", f"{data['DefectRate'].mean():.2f}")
        k2.metric("Mean Quality Score", f"{data['QualityScore'].mean():.1f}")
        k3.metric("Mean Maintenance Hours", f"{data['MaintenanceHours'].mean():.1f}")
        k4.metric("Mean Production Volume", f"{data['ProductionVolume'].mean():.0f}")
        st.info(
            "High-defect records dominate the sample, so accuracy and raw counts are treated as supporting indicators rather than standalone proof.",
        )

    with st.expander("Validation details"):
        validation_table = pd.DataFrame(
            {
                "Check": [
                    "Missing required columns",
                    "Non-numeric analysis columns",
                    "Binary DefectStatus",
                    "Total missing values",
                ],
                "Result": [
                    ", ".join(validation["missing_columns"]) or "None",
                    ", ".join(validation["non_numeric_columns"]) or "None",
                    "Yes" if validation["target_is_binary"] else "No",
                    str(validation["total_missing_values"]),
                ],
            }
        )
        st.dataframe(validation_table, width="stretch", hide_index=True)


def render_root_cause_explorer(data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Render sidebar controls and interactive feature comparison."""
    section_header(
        "2. Root Cause Workbench",
        "Focus on one process area at a time and compare how its signals behave across low-defect and high-defect production records.",
    )

    st.sidebar.header("Diagnosis Controls")
    group_key = st.sidebar.selectbox(
        "Feature group",
        options=list(FEATURE_GROUPS.keys()),
        format_func=lambda key: FEATURE_GROUP_DISPLAY_NAMES[key],
    )
    status_choice = st.sidebar.radio("Defect-status filter", list(STATUS_OPTIONS.keys()))
    selected_status = STATUS_OPTIONS[status_choice]

    filtered_data = data.copy()
    if selected_status is not None:
        filtered_data = filtered_data[filtered_data[TARGET_COLUMN] == selected_status]

    selected_features = [
        feature for feature in FEATURE_GROUPS[group_key] if feature != TARGET_COLUMN
    ]
    selected_feature = st.selectbox("Feature to compare", selected_features)

    st.markdown(f"### {FEATURE_GROUP_DISPLAY_NAMES[group_key]}")
    st.markdown(
        f'<div class="small-note">{FEATURE_GROUP_DESCRIPTIONS[group_key]}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records in View", f"{len(filtered_data):,}")
    c2.metric("Mean Defect Rate", f"{filtered_data['DefectRate'].mean():.2f}")
    c3.metric("Mean Quality Score", f"{filtered_data['QualityScore'].mean():.1f}")
    c4.metric(f"Mean {selected_feature}", f"{filtered_data[selected_feature].mean():.2f}")

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(
            selected_feature_boxplot_by_status(data, selected_feature),
            width="stretch",
            key=f"root_cause_boxplot_{selected_feature}",
        )
    with right:
        status_key = "all" if selected_status is None else str(selected_status)
        st.plotly_chart(
            defect_rate_histogram_by_status(filtered_data),
            width="stretch",
            key=f"root_cause_defect_rate_histogram_{status_key}",
        )

    st.markdown(
        '<div class="small-note">The boxplot preserves the low-vs-high defect comparison; the histogram reflects the current operating slice from the sidebar.</div>',
        unsafe_allow_html=True,
    )

    return filtered_data, selected_feature


def render_feature_difference_analysis(data: pd.DataFrame, analysis_outputs: dict) -> None:
    """Render standardized differences, boxplots, and comparison tables."""
    section_header(
        "3. Driver Separation Analysis",
        "Standardized differences put unlike units on one scale, making it easier to see which process signals separate high-defect records from low-defect records.",
    )

    chart_col, table_col = st.columns([1.15, 1])
    with chart_col:
        st.plotly_chart(
            standardized_difference_bar_chart(analysis_outputs["top_differences"]),
            width="stretch",
            key="driver_separation_standardized_differences",
        )
    with table_col:
        st.markdown("### Ranked Process Separators")
        display_difference = analysis_outputs["sorted_differences"].round(3).reset_index()
        display_difference = display_difference.rename(columns={"index": "Feature"})
        st.dataframe(display_difference, width="stretch", hide_index=True)

    box_feature = st.selectbox(
        "Boxplot feature",
        COMPARISON_FEATURES,
        index=COMPARISON_FEATURES.index("MaintenanceHours"),
        key="difference_box_feature",
    )
    st.plotly_chart(
        selected_feature_boxplot_by_status(data, box_feature),
        width="stretch",
        key=f"feature_difference_boxplot_{box_feature}",
    )

    st.markdown("### Operating Summary by Defect Status")
    st.dataframe(analysis_outputs["status_summary"], width="stretch")


def render_statistical_evidence(analysis_outputs: dict) -> None:
    """Render statistical test results and interpretation notes."""
    section_header(
        "4. Evidence Strength",
        "Mann-Whitney U tests flag distribution shifts, while Cohen's d translates those shifts into practical engineering effect sizes.",
    )

    st.dataframe(
        analysis_outputs["display_statistical_tests"],
        width="stretch",
        hide_index=True,
    )

    label_col, note_col = st.columns([0.9, 1.1])
    with label_col:
        st.markdown("### Effect Size Guide")
        st.dataframe(
            pd.DataFrame(
                {
                    "Label": ["negligible", "small", "medium", "large"],
                    "Absolute Cohen's d": ["< 0.20", "0.20 to 0.49", "0.50 to 0.79", ">= 0.80"],
                    "Engineering Read": [
                        "Little visible separation",
                        "Possible process signal",
                        "Meaningful process separation",
                        "Strong candidate driver",
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with note_col:
        st.warning(
            "Association is not causation. Higher maintenance hours may indicate unstable equipment, but they may also be a response to defects rather than the original cause.",
        )
        st.markdown(
            "The strongest operational signal appears when several views agree: standardized differences, statistical tests, and model importance all pointing to the same process variables."
        )


def render_correlation_explorer(data: pd.DataFrame, analysis_outputs: dict) -> None:
    """Render correlation heatmap and target-focused correlation tables."""
    section_header(
        "5. Correlation Explorer",
        "Correlation highlights linear relationships between process signals and helps identify where two measurements may be telling a similar story.",
    )

    st.plotly_chart(
        correlation_heatmap(analysis_outputs["correlation"]),
        width="stretch",
        key="correlation_explorer_heatmap",
    )

    corr = analysis_outputs["correlation"]
    left, right = st.columns(2)
    with left:
        st.markdown("### Correlation with DefectRate")
        defect_rate_corr = corr["DefectRate"].drop("DefectRate").sort_values(ascending=False)
        st.dataframe(
            defect_rate_corr.rename("Correlation").reset_index().rename(columns={"index": "Feature"}),
            width="stretch",
            hide_index=True,
        )
    with right:
        st.markdown("### Correlation with DefectStatus")
        status_corr = corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
        st.dataframe(
            status_corr.rename("Correlation").reset_index().rename(columns={"index": "Feature"}),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        '<div class="small-note">Weak single-variable correlations do not rule out multivariable or nonlinear process behavior; they indicate that no single linear relationship is enough to explain defect status.</div>',
        unsafe_allow_html=True,
    )


def render_model_insights(model_outputs: dict) -> None:
    """Render model metrics, confusion matrix, and feature importance."""
    section_header(
        "6. Model Insights",
        "Baseline classifiers test whether the process signals work together well enough to distinguish high-defect production records.",
    )

    model_results = model_outputs["results"]
    metrics_display = model_results.drop(columns=["Fitted Model"], errors="ignore").round(3)

    chart_col, table_col = st.columns([1.15, 1])
    with chart_col:
        st.plotly_chart(
            model_metrics_bar_chart(model_results),
            width="stretch",
            key="model_insights_metrics",
        )
    with table_col:
        st.markdown("### Classifier Benchmark")
        st.dataframe(metrics_display, width="stretch", hide_index=True)
        st.markdown(
            '<div class="small-note">Logistic regression provides a transparent linear benchmark; random forest adds nonlinear interactions that are often useful in process data.</div>',
            unsafe_allow_html=True,
        )

    confusion_col, importance_col = st.columns(2)
    with confusion_col:
        st.plotly_chart(
            confusion_matrix_heatmap(model_outputs["confusion_matrix"]),
            width="stretch",
            key="model_insights_confusion_matrix",
        )
    with importance_col:
        st.markdown("### Top 5 Model Drivers")
        st.plotly_chart(
            feature_importance_chart(model_outputs["importance"], limit=5),
            width="stretch",
            key="model_insights_feature_importance",
        )

    top_features = ", ".join(model_outputs["importance"].head(3)["Feature"])
    st.markdown("### Engineering Conclusions")
    st.markdown(
        f"""
        - The random forest benchmark suggests the process data contains useful defect-separation signal, but it should remain a decision-support model rather than an automated release gate.
        - The leading model drivers are **{top_features}**, which is consistent with the dashboard's broader maintenance, quality, and production-load findings.
        - Because the low-defect class is smaller, follow-up work should validate these signals with real production context such as line, machine, shift, product family, and maintenance history.
        """
    )

    with st.expander("Random Forest performance detail"):
        st.dataframe(model_outputs["classification_report"].round(3), width="stretch")


def render_data_explorer(data: pd.DataFrame) -> None:
    """Render filterable data table and CSV export."""
    section_header(
        "7. Data Explorer",
        "Audit the underlying production records and export a focused slice for follow-up analysis.",
    )

    min_quality_bound = math.floor(float(data["QualityScore"].min()) * 10) / 10
    max_quality_bound = math.ceil(float(data["QualityScore"].max()) * 10) / 10
    min_maintenance_bound = int(data["MaintenanceHours"].min())
    max_maintenance_bound = int(data["MaintenanceHours"].max())

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        status_filter = st.selectbox("Defect status", list(STATUS_OPTIONS.keys()), key="table_status")
    with col_b:
        min_quality, max_quality = st.slider(
            "Quality score range",
            min_quality_bound,
            max_quality_bound,
            (min_quality_bound, max_quality_bound),
            step=0.1,
        )
    with col_c:
        min_maintenance, max_maintenance = st.slider(
            "Maintenance hours range",
            min_maintenance_bound,
            max_maintenance_bound,
            (min_maintenance_bound, max_maintenance_bound),
            step=1,
        )

    table_data = data.copy()
    selected_status = STATUS_OPTIONS[status_filter]
    if selected_status is not None:
        table_data = table_data[table_data[TARGET_COLUMN] == selected_status]
    table_data = table_data[
        table_data["QualityScore"].between(min_quality, max_quality)
        & table_data["MaintenanceHours"].between(min_maintenance, max_maintenance)
    ]

    st.metric("Filtered Production Records", f"{len(table_data):,}")
    st.dataframe(table_data, width="stretch", hide_index=True)
    st.download_button(
        "Download filtered CSV",
        data=table_data.to_csv(index=False).encode("utf-8"),
        file_name="manufacturing_defect_filtered.csv",
        mime="text/csv",
    )


def main() -> None:
    """Run the Streamlit dashboard."""
    apply_page_styles()
    data, validation = load_validated_data()
    analysis_outputs = build_analysis_outputs(data)
    model_outputs = build_model_outputs(data)

    st.markdown('<div class="product-label">Manufacturing Quality Analytics</div>', unsafe_allow_html=True)
    st.title("Manufacturing Defect Root Cause Explorer")
    st.markdown(
        "A mechanical-engineering dashboard for diagnosing defect patterns across production load, quality performance, maintenance activity, supplier reliability, workforce conditions, inventory flow, energy use, and additive-manufacturing operations."
    )

    render_executive_summary(analysis_outputs, model_outputs)
    render_overview(data, validation, analysis_outputs)
    render_root_cause_explorer(data)
    render_feature_difference_analysis(data, analysis_outputs)
    render_statistical_evidence(analysis_outputs)
    render_correlation_explorer(data, analysis_outputs)
    render_model_insights(model_outputs)
    render_data_explorer(data)


if __name__ == "__main__":
    main()
