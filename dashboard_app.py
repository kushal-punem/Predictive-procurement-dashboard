"""
Streamlit dashboard for the
University Bulk Order & Predictive Procurement Analytics System.

Run with:
    streamlit run dashboard_app.py
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from etl_pipeline import load_feature_table
from feature_engine import compute_feature_importance_example


st.set_page_config(
    page_title="University Bulk Order & Predictive Procurement Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global styling to match the dark blue reference left filter bar
st.markdown(
    """
    <style>
    /* Overall background: darker navy gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1b2b4a 0%, #050b18 55%, #020309 100%);
    }
    /* Header bar: slightly darker blue */
    [data-testid="stHeader"] {
        background: linear-gradient(90deg, #15294b, #274a7b);
        color: #e5e7eb;
    }
    /* Left filter panel: dark vertical bar */
    .filter-panel {
        background: linear-gradient(180deg, #1b3358, #091426);
        border-radius: 12px;
        padding: 16px 12px 20px 12px;
        border: 1px solid #304c7a;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
        color: #e5e7eb;
    }
    .filter-panel h4 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: #e5e7eb;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    /* Labels inside filter panel */
    .filter-panel label {
        color: #e5e7eb !important;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    /* Streamlit select boxes / text inputs inside filter panel */
    .filter-panel [data-baseweb="select"] > div {
        background-color: #15294b;
        color: #e5e7eb;
        border-radius: 6px;
        border: 1px solid #4b6b9c;
    }
    .filter-panel [data-baseweb="select"] svg {
        fill: #e5e7eb;
    }
    .filter-panel input {
        background-color: #15294b;
        color: #e5e7eb;
        border-radius: 6px;
        border: 1px solid #4b6b9c;
    }
    /* Glassmorphism panels for the three feature charts */
    .glass-panel {
        background: rgba(15, 23, 42, 0.92);
        border-radius: 18px;
        padding: 0.6rem 0.9rem 0.9rem 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.6);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_feature_table()


def kpi_card(label: str, value: str, help_text: str | None = None, key: str | None = None):
    with st.container():
        st.markdown(
            f"""
            <div style="
                padding: 0.9rem 1.1rem;
                border-radius: 0.7rem;
                background: linear-gradient(135deg, #d6ebfb, #c0def4);
                border: 1px solid rgba(148, 163, 184, 0.4);
            ">
                <div style="font-size: 0.8rem; color: #4b5563; text-transform: uppercase; letter-spacing: 0.08em;">
                    {label}
                </div>
                <div style="font-size: 1.6rem; font-weight: 600; color: #0f172a;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if help_text:
            st.caption(help_text)


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown("#### Filters")

    term = st.selectbox("Term", options=["All"] + sorted(df["Term"].unique().tolist()), key="filter_term")
    dept = st.selectbox(
        "Department", options=["All"] + sorted(df["Dept_Code"].unique().tolist()), key="filter_dept"
    )
    publisher = st.selectbox(
        "Publisher", options=["All"] + sorted(df["Publisher"].unique().tolist()), key="filter_publisher"
    )
    student_type = st.selectbox(
        "Student Type", options=["All"] + sorted(df["Student_Type"].unique().tolist()), key="filter_student_type"
    )
    fmt = st.selectbox(
        "Format", options=["All"] + sorted(df["Format"].unique().tolist()), key="filter_format"
    )

    mask = pd.Series(True, index=df.index)
    if term != "All":
        mask &= df["Term"] == term
    if dept != "All":
        mask &= df["Dept_Code"] == dept
    if publisher != "All":
        mask &= df["Publisher"] == publisher
    if student_type != "All":
        mask &= df["Student_Type"] == student_type
    if fmt != "All":
        mask &= df["Format"] == fmt

    return df[mask].copy()


def render_header():
    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            "### University Bulk Order & Predictive Procurement Analytics",
        )
        st.caption(
            "Machine-learning driven demand forecasting and risk segmentation for university textbook procurement."
        )
    with right:
        st.markdown(
            """
            <div style="text-align: right; font-size: 0.8rem; color: #9ca3af;">
                <div>Today</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_top_kpis(df: pd.DataFrame):
    total_pred_units = df["Predicted_Demand_Units"].sum()
    total_spend = df["Projected_Spend"].sum()

    digital_share = (
        (df["Format"] == "Digital").sum() / len(df) if len(df) else 0
    )
    physical_share = 1 - digital_share

    high_risk_rate = (df["Opt_Out_Probability"] > 0.6).mean() if len(df) else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Predicted Demand", f"{int(total_pred_units):,} Units")
    with c2:
        kpi_card("Total Projected Spend", f"${total_spend:,.1f} M", "Millions equivalent")
    with c3:
        kpi_card(
            "Digital vs Physical",
            f"{digital_share*100:.0f}% / {physical_share*100:.0f}%",
            "Share of predicted units",
        )
    with c4:
        kpi_card(
            "High-Risk Opt-Out Rate",
            f"{high_risk_rate*100:.0f}%",
            "Share of records with Opt-Out Probability > 60%",
        )


def render_price_sensitivity(df: pd.DataFrame):
    st.markdown("#### Price Sensitivity & Opt-Out Threshold")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    fig = px.scatter(
        df,
        x="Rental_to_Retail_Ratio",
        y="Opt_Out_Probability",
        color="Dept_Code",
        hover_data=["Publisher", "Student_Type", "Format"],
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Rental-to-Retail Ratio",
        yaxis_title="Opt-Out Probability",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb"),
        xaxis=dict(
            tickfont=dict(color="#e5e7eb"),
            gridcolor="rgba(148,163,184,0.25)",
            zerolinecolor="rgba(148,163,184,0.4)",
        ),
        yaxis=dict(
            tickfont=dict(color="#e5e7eb"),
            gridcolor="rgba(148,163,184,0.25)",
            zerolinecolor="rgba(148,163,184,0.4)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance():
    st.markdown("#### Feature Importance (Model Explainability)")
    fi = compute_feature_importance_example()

    fig = px.bar(
        fi.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Relative Importance",
        yaxis_title="Feature",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb"),
        xaxis=dict(
            tickfont=dict(color="#e5e7eb"),
            gridcolor="rgba(148,163,184,0.25)",
            zerolinecolor="rgba(148,163,184,0.4)",
        ),
        yaxis=dict(
            tickfont=dict(color="#e5e7eb"),
            gridcolor="rgba(148,163,184,0.0)",
            zerolinecolor="rgba(148,163,184,0.0)",
        ),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_format_preference(df: pd.DataFrame):
    st.markdown("#### Format Preference by Segment")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    by_student_type = (
        df.groupby(["Student_Type", "Format"])["Predicted_Demand_Units"].sum().reset_index()
    )

    fig = px.bar(
        by_student_type,
        x="Student_Type",
        y="Predicted_Demand_Units",
        color="Format",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Student Type",
        yaxis_title="Predicted Units",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb"),
        xaxis=dict(
            tickfont=dict(color="#e5e7eb"),
            gridcolor="rgba(148,163,184,0.25)",
            zerolinecolor="rgba(148,163,184,0.4)",
        ),
        yaxis=dict(
            tickfont=dict(color="#e5e7eb"),
            gridcolor="rgba(148,163,184,0.25)",
            zerolinecolor="rgba(148,163,184,0.4)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_funding_flow(df: pd.DataFrame):
    st.markdown("#### Funding Source Planning & Strategy")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    # Simple synthetic funding sources
    sources = ["Financial Aid", "Self-Pay", "Scholarship"]
    outcomes = ["Opt-In", "Opt-Out"]

    rng = np.random.default_rng(0)
    values = rng.integers(50, 200, size=len(sources) * len(outcomes))
    values = values.astype(float)

    labels = sources + outcomes
    source_indices = []
    target_indices = []
    for i, _ in enumerate(sources):
        for j, _ in enumerate(outcomes):
            source_indices.append(i)
            target_indices.append(len(sources) + j)

    link = dict(source=source_indices, target=target_indices, value=values)

    node = dict(
        label=labels,
        pad=20,
        thickness=14,
        color=["#60a5fa", "#34d399", "#fbbf24", "#22c55e", "#f97316"],
    )

    fig = go.Figure(data=[go.Sankey(node=node, link=link)])
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color="#f9fafb"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_risk_by_department(df: pd.DataFrame):
    st.markdown("#### Procurement Risk: Top 5 High-Risk Departments")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    agg = (
        df.groupby("Dept_Code")["Opt_Out_Probability"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    fig = px.bar(
        agg,
        x="Dept_Code",
        y="Opt_Out_Probability",
        color="Opt_Out_Probability",
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Department",
        yaxis_title="Avg Opt-Out Probability",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_recommendations(df: pd.DataFrame):
    st.markdown("#### Recommended Actions")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    high_risk = df[df["Opt_Out_Probability"] > 0.6]
    risky_depts = (
        high_risk.groupby("Dept_Code")["Opt_Out_Probability"].mean().sort_values(ascending=False)
    )
    top_depts = ", ".join(risky_depts.head(3).index.tolist()) or "N/A"

    bullet_points = [
        f"Prioritize **bundle price negotiation** with key publishers in high-risk departments: {top_depts}.",
        "Increase **digital access options** for high commuter-friction and part-time student segments.",
        "Review **required vs recommended** status for titles with extreme rental-to-retail arbitrage.",
        "Share **early adoption reports** with department chairs to align on realistic order quantities.",
    ]

    st.markdown(
        "\n\n".join([f"- {bp}" for bp in bullet_points]),
    )


def main():
    df = get_data()

    render_header()
    st.markdown("---")
    
    # Two-column layout: left filter bar, right main dashboard
    filters_col, main_col = st.columns([0.9, 4.1])

    with filters_col:
        st.markdown('<div class="filter-panel"><h4>Term & Segments</h4>', unsafe_allow_html=True)
        filtered_df = render_filters(df)
        st.markdown("</div>", unsafe_allow_html=True)

    with main_col:
        render_top_kpis(filtered_df)

        st.markdown("### Feature Engineering Pipeline & Key Indicators")
        # Three glassmorphism-style blocks, each containing one visual
        upper_left, upper_mid, upper_right = st.columns(3)

        with upper_left:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            render_price_sensitivity(filtered_df)
            st.markdown('</div>', unsafe_allow_html=True)
        with upper_mid:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            render_feature_importance()
            st.markdown('</div>', unsafe_allow_html=True)
        with upper_right:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            render_format_preference(filtered_df)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Procurement Planning & Strategy")
        lower_left, lower_mid, lower_right = st.columns([1.4, 1.0, 1.0])
        with lower_left:
            render_funding_flow(filtered_df)
        with lower_mid:
            render_risk_by_department(filtered_df)
        with lower_right:
            render_recommendations(filtered_df)


if __name__ == "__main__":
    main()

