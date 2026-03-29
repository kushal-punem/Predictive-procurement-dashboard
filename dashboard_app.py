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
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from etl_pipeline import load_feature_table
# from count_student_purchase import count_student_purchases_fun 
from feature_engine import train_model, apply_predictions

st.set_page_config(
    page_title="University Bulk Order & Predictive Procurement Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .filter-panel label {
        color: #e5e7eb !important;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
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
    .filter-panel div[data-testid="stSlider"] > div > div {
        color: #e5e7eb;
    }
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
def get_raw_data() -> pd.DataFrame:
    return load_feature_table(sample_frac=0.03)

@st.cache_resource(show_spinner=False)
def get_trained_model(df: pd.DataFrame):
    return train_model(df)

def kpi_card(label: str, value: str, help_text: str | None = None):
    with st.container():
        st.markdown(
            f"""
            <div style="
                padding: 0.9rem 1.1rem;
                margin-bottom: 1rem;
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
    term = st.selectbox("Term", options=["All"] + sorted(df["Term"].unique().tolist()), key="filter_term")
    dept = st.selectbox("Department", options=["All"] + sorted(df["Dept_Code"].unique().tolist()), key="filter_dept")
    publisher = st.selectbox("Publisher", options=["All"] + sorted(df["Publisher"].unique().tolist()), key="filter_publisher")
    student_type = st.selectbox("Student Type", options=["All"] + sorted(df["Student_Type"].unique().tolist()), key="filter_student_type")
    fmt = st.selectbox("Format", options=["All"] + sorted(df["Format"].unique().tolist()), key="filter_format")

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
        st.markdown("### University Bulk Order & Predictive Procurement Analytics")
        st.caption("Machine-learning driven demand forecasting and risk segmentation with Investor ROI Modeling.")
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

    digital_share = (
        (df["Format"] == "Digital").sum() / len(df) if len(df) else 0
    )
    physical_share = 1 - digital_share

    high_risk_rate = (df["Opt_Out_Probability"] > 0.6).mean() if len(df) else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Total Predicted Demand", f"{int(total_pred_units):,} Units")
        
    with c2:
        kpi_card(
            "Digital vs Physical",
            f"{digital_share*100:.0f}% / {physical_share*100:.0f}%",
            "Share of predicted units",
        )
    with c3:
        kpi_card(
            "High-Risk Opt-Out Rate",
            f"{high_risk_rate*100:.0f}%",
            "Share of records with Opt-Out Probability > 60%",
        )

def render_book_quantities(df: pd.DataFrame):
    st.markdown("#### Quantity of Each Book")
    if df.empty:
        st.info("No data available.")
        return

    agg = df.groupby("Title")["Predicted_Demand_Units"].sum().reset_index()
    agg = agg.sort_values(by="Predicted_Demand_Units", ascending=True).tail(15)

    fig = px.bar(
        agg, x="Predicted_Demand_Units", y="Title", orientation="h",
        color="Predicted_Demand_Units", color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Predicted Units", yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb"), coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

def render_feature_importance(fi: pd.DataFrame):
    st.markdown("#### Feature Importance (Model Explainability)")
    fig = px.bar(
        fi.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Relative Importance", yaxis_title="Feature",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb"), coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

def render_publisher_risk_matrix(df: pd.DataFrame):
    st.markdown("#### Publisher Risk Matrix")
    if df.empty:
        st.info("No data.")
        return
        
    agg = df.groupby("Publisher").agg(
        Avg_Price=("Unit_Price", "mean"),
        Avg_Opt_Out=("Opt_Out_Probability", "mean"),
        Total_Units=("Predicted_Demand_Units", "sum")
    ).reset_index()
    
    agg = agg[agg["Total_Units"] >= 5]
    
    fig = px.scatter(
        agg, x="Avg_Price", y="Avg_Opt_Out", size="Total_Units",
        color="Avg_Opt_Out", hover_name="Publisher", color_continuous_scale="Reds", size_max=40
    )
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Avg Retail Price ($)", yaxis_title="Avg Opt-Out Probability",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb"), coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

def render_high_friction_titles(df: pd.DataFrame):
    st.markdown("#### Top 10 High-Friction Titles (Negotiation Targets)")
    if df.empty:
        st.info("No data.")
        return
    
    df["Lost_Revenue"] = df["Predicted_Demand_Units"] * df["Opt_Out_Probability"] * df["Unit_Price"]
    
    agg = df.groupby(["Title", "Publisher"]).agg(
        Opt_Out_Rate=("Opt_Out_Probability", "mean"),
        Lost_Revenue=("Lost_Revenue", "sum")
    ).reset_index()
    
    agg = agg.sort_values(by="Lost_Revenue", ascending=False).head(10)
    
    agg["Lost_Revenue"] = agg["Lost_Revenue"].apply(lambda x: f"${x:,.0f}")
    agg["Opt_Out_Rate"] = agg["Opt_Out_Rate"].apply(lambda x: f"{x*100:.1f}%")
    
    m_df = agg.rename(columns={"Title": "Book Title", "Opt_Out_Rate": "Opt-Out %", "Lost_Revenue": "Lost Revenue ($)"})
    
    st.dataframe(m_df, use_container_width=True, hide_index=True)

def render_format_preference(df: pd.DataFrame):
    st.markdown("#### Format Preference by Segment")
    if df.empty:
        st.info("No data.")
        return

    by_student_type = (
        df.groupby(["Student_Type", "Format"])["Predicted_Demand_Units"].sum().reset_index()
    )
    fig = px.bar(
        by_student_type, x="Student_Type", y="Predicted_Demand_Units", color="Format",
        barmode="group", color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Student Type", yaxis_title="Predicted Units",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9fafb")
    )
    st.plotly_chart(fig, use_container_width=True)

def render_word_cloud(df: pd.DataFrame):
    st.markdown("#### Book Names Word Cloud")
    if df.empty:
        st.info("No data.")
        return

    agg = df.groupby("Title")["Predicted_Demand_Units"].sum()
    freq_dict = agg.to_dict()
    
    if not freq_dict:
        return

    wc = WordCloud(
        width=1200, height=400, background_color=None, mode="RGBA", colormap="Blues",
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)

def main():
    raw_df = get_raw_data()
    clf, fi_df, features = get_trained_model(raw_df)

    render_header()
    st.markdown("---")
    
    filters_col, main_col = st.columns([0.9, 4.1])

    with filters_col:
        st.markdown('<div class="filter-panel"><h4>Term & Segments</h4>', unsafe_allow_html=True)
        filtered_df_base = render_filters(raw_df)
        
        st.markdown("<hr style='border-color: #304c7a; margin: 15px 0;'>", unsafe_allow_html=True)
        st.markdown('<h4>What-If Simulator</h4>', unsafe_allow_html=True)
        discount = st.slider("Publisher Discount (%)", min_value=0, max_value=50, value=0, step=5)
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply Simulation
    filtered_df = apply_predictions(filtered_df_base, clf, features, discount_pct=discount)

    with main_col:
        render_top_kpis(filtered_df)

        st.markdown("### Feature Engineering & Vendor Leverage")
        upper_left, upper_mid, upper_right = st.columns(3)

        with upper_left:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            render_book_quantities(filtered_df)
            st.markdown('</div>', unsafe_allow_html=True)
        with upper_mid:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            render_feature_importance(fi_df)
            st.markdown('</div>', unsafe_allow_html=True)
        with upper_right:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            render_publisher_risk_matrix(filtered_df)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Procurement Planning & Segment Preferences")
        lower_left, lower_right = st.columns(2)
        with lower_left:
            render_high_friction_titles(filtered_df)
        with lower_right:
            render_format_preference(filtered_df)
            
        st.markdown("---")
        render_word_cloud(filtered_df)

if __name__ == "__main__":
    main()
