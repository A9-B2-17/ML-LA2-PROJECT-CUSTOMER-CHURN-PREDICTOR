"""Streamlit Dashboard for Customer Churn Prediction & ML Lab.
Run: streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import ml_pipeline as mp

# ---------------------------------------------------------------- Page Config & Theme
st.set_page_config(
    page_title="Churn ML Lab · Intelligent Retention Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Font Override */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Header Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.12) 0%, rgba(147, 51, 234, 0.12) 50%, rgba(236, 72, 153, 0.12) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.15);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #3b82f6);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-top: 6px;
    margin-bottom: 14px;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.badge-purple {
    background: rgba(147, 51, 234, 0.12);
    color: #9333ea;
    border: 1px solid rgba(147, 51, 234, 0.25);
}
.badge-blue {
    background: rgba(59, 130, 246, 0.12);
    color: #2563eb;
    border: 1px solid rgba(59, 130, 246, 0.25);
}
.badge-green {
    background: rgba(16, 185, 129, 0.12);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

/* Glassmorphism Metric Cards */
.kpi-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.15);
    border-color: rgba(99, 102, 241, 0.4);
}
.kpi-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 6px;
}
.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 4px 0 2px 0;
    color: #1e293b;
}
@media (prefers-color-scheme: dark) {
    .kpi-value { color: #f8fafc; }
    .hero-subtitle { color: #94a3b8; }
}
.kpi-subtext {
    font-size: 0.75rem;
    color: #94a3b8;
}

/* Best Model Champion Card */
.champion-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.12) 100%);
    border: 1.5px solid rgba(124, 58, 237, 0.4);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 32px 0 rgba(124, 58, 237, 0.15);
}

/* Pipeline Architecture Steps */
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin: 16px 0;
}
.step-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 14px;
    padding: 16px;
    position: relative;
}
.step-number {
    position: absolute;
    top: 12px;
    right: 14px;
    font-size: 1.4rem;
    font-weight: 800;
    opacity: 0.15;
}
.step-title {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.step-desc {
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.4;
}

/* Predict Risk Card */
.risk-card-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.05) 100%);
    border: 1.5px solid rgba(239, 68, 68, 0.35);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}
.risk-card-low {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
    border: 1.5px solid rgba(16, 185, 129, 0.35);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}
.risk-card-med {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.05) 100%);
    border: 1.5px solid rgba(245, 158, 11, 0.35);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

/* Sidebar Branding */
.sidebar-brand {
    padding: 12px 6px 18px 6px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    margin-bottom: 18px;
}
.sidebar-title {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Modern Tab Enhancements */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    padding-bottom: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.12) !important;
    color: #6366f1 !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA = Path("data/churn.csv")
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
PLOTLY_TEMPLATE = "plotly_white"
COLOR_PALETTE = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#3B82F6"]

@st.cache_data
def load_df(path):
    return pd.read_csv(path)

def load_artifacts():
    f = mp.ARTIFACTS / "results.json"
    m = mp.ARTIFACTS / "models.joblib"
    if not f.exists() or not m.exists():
        return None, None, None
    d = json.loads(f.read_text())
    return d["meta"], d["results"], joblib.load(m)

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-title">⚡ Churn ML Lab</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">
            End-to-End Enterprise Retention Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📁 Dataset Source")
    up = st.file_uploader("Upload custom CSV", type="csv", help="Upload customer data to retrain and evaluate.")
    if up is not None:
        Path("data").mkdir(exist_ok=True)
        DATA = Path("data/uploaded.csv")
        DATA.write_bytes(up.getvalue())
        st.success("Uploaded dataset loaded!")

    if not DATA.exists():
        st.error("No dataset found. Run `python generate_data.py` first.")
        st.stop()

    df = load_df(DATA)
    
    st.markdown("#### ⚙️ Configuration")
    target = st.selectbox(
        "Target column", 
        df.columns, 
        index=list(df.columns).index("Churn") if "Churn" in df.columns else len(df.columns) - 1,
        help="Binary class variable to predict (e.g., Churn)"
    )
    default_ids = [c for c in ["CustomerID", "ID", "id"] if c in df.columns]
    id_cols = st.sidebar.multiselect("ID columns to drop", df.columns, default=default_ids)
    
    st.markdown("#### 🤖 Model Selection")
    all_models = list(mp.model_zoo().keys())
    chosen = st.multiselect("Models to train & tune", all_models, default=all_models)
    
    st.markdown("---")
    if st.button("🚀 Train & Tune Models", type="primary", use_container_width=True):
        bar = st.progress(0.0)
        status_text = st.empty()
        
        def update_progress(p, m):
            bar.progress(p)
            status_text.caption(f"⚡ {m}")
            
        with st.spinner("Executing leak-free cross-validation & tuning..."):
            mp.run_experiment(DATA, target, id_cols, models=chosen, progress=update_progress)
        st.success("Training & Hyperparameter tuning completed!")
        st.rerun()

meta, results, models = load_artifacts()

# ---------------------------------------------------------------- Hero Header
st.markdown("""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
        <div>
            <h1 class="hero-title">Customer Churn Intelligence Lab</h1>
            <p class="hero-subtitle">Leakage-aware ML pipeline with tuned classifiers, model comparison & customer segmentation.</p>
            <div class="hero-badges">
                <span class="badge-pill badge-purple">🛡️ Leakage-Aware Pipeline</span>
                <span class="badge-pill badge-blue">🎯 5-Fold Stratified CV</span>
                <span class="badge-pill badge-green">⚡ Live Churn Prediction</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- Tabs Navigation
tabs = st.tabs([
    "📊 1 · Data & EDA", 
    "🧹 2 · Preprocessing Pipeline", 
    "🏆 3 · Model Leaderboard",
    "🔍 4 · Model Deep Dive", 
    "🔮 5 · Live Churn Predictor", 
    "👥 6 · Customer Segmentation"
])

# ================================================================ TAB 1: EDA
with tabs[0]:
    st.markdown("### 📈 Dataset Overview & Quick Health Check")
    churn_rate = (df[target] == 1).mean() if target in df.columns else 0.0
    missing_count = int(df.isna().sum().sum())
    dup_count = int(df.duplicated().sum())

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">👥 Total Customers</div>
            <div class="kpi-value">{df.shape[0]:,}</div>
            <div class="kpi-subtext">Rows in dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📐 Attributes</div>
            <div class="kpi-value">{df.shape[1]}</div>
            <div class="kpi-subtext">Features & identifiers</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📉 Churn Rate</div>
            <div class="kpi-value">{churn_rate:.1%}</div>
            <div class="kpi-subtext">Baseline positive class</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">⚠️ Missing Cells</div>
            <div class="kpi-value">{missing_count}</div>
            <div class="kpi-subtext">Across all columns</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📑 Duplicates</div>
            <div class="kpi-value">{dup_count}</div>
            <div class="kpi-subtext">Exact duplicate rows</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 Interactive Data Preview & Summary Statistics", expanded=True):
        preview_tab, desc_tab, missing_tab = st.tabs(["📋 Data Sample", "📊 Statistical Summary", "❓ Missing Values"])
        with preview_tab:
            st.dataframe(df.head(10), use_container_width=True)
        with desc_tab:
            st.dataframe(df.describe(include="all").T.round(2), use_container_width=True)
        with missing_tab:
            miss_series = df.isna().sum()[lambda s: s > 0]
            if len(miss_series) > 0:
                miss_df = pd.DataFrame({"Missing Rows": miss_series, "Percentage": (miss_series / len(df) * 100).round(2)})
                col_m1, col_m2 = st.columns([1, 1.5])
                col_m1.dataframe(miss_df, use_container_width=True)
                fig_miss = px.bar(
                    miss_df, y="Missing Rows", text="Percentage",
                    title="Missing Value Distribution by Column",
                    color_discrete_sequence=["#EF4444"], template=PLOTLY_TEMPLATE
                )
                fig_miss.update_traces(texttemplate='%{text}%', textposition='outside')
                col_m2.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("✅ Clean dataset: No missing values detected!")

    st.markdown("---")
    num_cols = df.select_dtypes("number").columns.drop([c for c in id_cols if c in df.columns], errors="ignore").tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    col_u, col_b = st.columns(2)
    with col_u:
        st.markdown("#### 🎯 Univariate Analysis")
        selected_feature = st.selectbox("Select Feature to Inspect", num_cols + cat_cols, key="eda_uni")
        if selected_feature in num_cols:
            fig_uni = px.histogram(
                df, x=selected_feature, marginal="box", nbins=35,
                color_discrete_sequence=["#6366F1"], template=PLOTLY_TEMPLATE,
                title=f"Distribution & Outliers: {selected_feature}"
            )
            st.plotly_chart(fig_uni, use_container_width=True)
        else:
            fig_uni = px.histogram(
                df, x=selected_feature, color=selected_feature,
                color_discrete_sequence=COLOR_PALETTE, template=PLOTLY_TEMPLATE,
                title=f"Count Frequency: {selected_feature}"
            )
            st.plotly_chart(fig_uni, use_container_width=True)

    with col_b:
        st.markdown("#### 🔄 Bivariate Relationship with Churn")
        cand_features = [c for c in num_cols + cat_cols if c != target]
        selected_bivar = st.selectbox("Compare Feature vs Target", cand_features, key="eda_bi")
        if selected_bivar in num_cols:
            fig_bi = px.box(
                df, x=target, y=selected_bivar, color=target,
                color_discrete_sequence=["#10B981", "#EF4444"], template=PLOTLY_TEMPLATE,
                title=f"{selected_bivar} vs {target}"
            )
            st.plotly_chart(fig_bi, use_container_width=True)
        else:
            rate_df = df.groupby(selected_bivar)[target].agg(["mean", "count"]).reset_index()
            rate_df["Churn Rate (%)"] = (rate_df["mean"] * 100).round(1)
            fig_bi = px.bar(
                rate_df, x=selected_bivar, y="Churn Rate (%)", color="Churn Rate (%)",
                color_continuous_scale="Reds", text="Churn Rate (%)",
                title=f"Churn Rate by {selected_bivar}", template=PLOTLY_TEMPLATE
            )
            fig_bi.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig_bi, use_container_width=True)

    st.markdown("#### 🧩 Multivariate Correlation Heatmap")
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().round(2)
        fig_corr = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            template=PLOTLY_TEMPLATE, title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ================================================================ TAB 2: PREPROCESSING
with tabs[1]:
    st.markdown("### 🛡️ Preprocessing & Leakage-Aware Architecture")
    
    st.markdown("""
    <div class="pipeline-grid">
        <div class="step-card">
            <span class="step-number">01</span>
            <div class="step-title">🧹 De-duplication</div>
            <div class="step-desc">Exact duplicate records are purged before train/test splitting to prevent identical rows crossing partitions.</div>
        </div>
        <div class="step-card">
            <span class="step-number">02</span>
            <div class="step-title">✂️ Stratified Split</div>
            <div class="step-desc">80/20 train/test split preserves class ratio. No feature statistics or scalers are learned beforehand.</div>
        </div>
        <div class="step-card">
            <span class="step-number">03</span>
            <div class="step-title">📦 Outlier Capping</div>
            <div class="step-desc">Winsorization using IQR (Q1 - 1.5·IQR to Q3 + 1.5·IQR) computed strictly on training folds.</div>
        </div>
        <div class="step-card">
            <span class="step-number">04</span>
            <div class="step-title">⚙️ Impute & Scale</div>
            <div class="step-desc">Median (numeric) & mode (categorical) imputation combined with StandardScaler and OneHotEncoder.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not meta:
        st.info("💡 Train models via the sidebar to view the active preprocessing telemetry.")
    else:
        rep = meta["report"]
        st.markdown("#### 📋 Data Split & Quality Report")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Initial Rows", f"{rep['rows_before']:,}")
        c2.metric("Duplicates Dropped", rep["duplicates_removed"], delta=None if rep["duplicates_removed"] == 0 else f"-{rep['duplicates_removed']}")
        c3.metric("Training Samples (80%)", f"{rep['train_rows']:,}")
        c4.metric("Holdout Test Samples (20%)", f"{rep['test_rows']:,}")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### 🩺 Imputation Log (Missing Handled)")
            if rep["missing_before"]:
                st.dataframe(pd.Series(rep["missing_before"], name="Missing Count"), use_container_width=True)
            else:
                st.success("No missing values encountered in dataset.")
                
            st.markdown("##### 🏷️ Feature Pipeline Mapping")
            st.markdown(f"**Categorical Encoded (One-Hot):** `{', '.join(meta['cat_cols']) if meta['cat_cols'] else 'None'}`")
            st.markdown(f"**Numerical Scaled (StandardScaler):** `{', '.join(meta['num_cols']) if meta['num_cols'] else 'None'}`")

        with col_right:
            st.markdown("##### 📉 Outliers Capped in Train Partition")
            if rep["outliers_in_train"]:
                st.dataframe(pd.Series(rep["outliers_in_train"], name="Capped Outliers"), use_container_width=True)
            else:
                st.success("No severe outliers outside 1.5 IQR bounds.")

# ================================================================ TAB 3: LEADERBOARD
with tabs[2]:
    st.markdown("### 🏆 Model Performance Benchmark & Leaderboard")
    if not results or not meta:
        st.info("💡 Click **Train & Tune Models** in the sidebar to benchmark algorithms.")
    else:
        best_name = meta.get("best_model", list(results.keys())[0])
        best_res = results[best_name]

        st.markdown(f"""
        <div class="champion-box">
            <div>
                <div style="font-size:0.8rem; font-weight:700; color:#7c3aed; text-transform:uppercase; letter-spacing:0.05em;">
                    🥇 Champion Classifier Selected
                </div>
                <div style="font-size:1.8rem; font-weight:800; color:#1e293b; margin:4px 0;">
                    {best_name}
                </div>
                <div style="font-size:0.85rem; color:#64748b;">
                    Optimized via 5-Fold Stratified GridSearchCV scoring F1-Score
                </div>
            </div>
            <div style="display:flex; gap:20px; text-align:right;">
                <div>
                    <div style="font-size:0.75rem; color:#64748b; font-weight:600;">TEST F1</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#6366f1;">{best_res['f1']:.3f}</div>
                </div>
                <div>
                    <div style="font-size:0.75rem; color:#64748b; font-weight:600;">ROC-AUC</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#ec4899;">{best_res['roc_auc']:.3f}</div>
                </div>
                <div>
                    <div style="font-size:0.75rem; color:#64748b; font-weight:600;">ACCURACY</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#10b981;">{best_res['accuracy']:.1%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Leaderboard Table
        leaderboard_df = pd.DataFrame(results).T[METRICS + ["cv_f1", "fit_seconds"]].astype(float).round(3)
        leaderboard_df = leaderboard_df.sort_values(by="f1", ascending=False)
        leaderboard_df.index.name = "Model"

        st.markdown("#### 📊 Comparative Metric Matrix")
        st.dataframe(
            leaderboard_df.style
            .highlight_max(subset=METRICS + ["cv_f1"], color="#d1fae5")
            .highlight_min(subset=["fit_seconds"], color="#ede9fe"),
            use_container_width=True
        )

        # Visual Comparison Bar Chart & ROC Curves
        col_plot1, col_plot2 = st.columns([1.2, 1])
        with col_plot1:
            st.markdown("#### 📊 Multi-Metric Comparison")
            plot_df = leaderboard_df[METRICS].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig_bar = px.bar(
                plot_df, x="Model", y="Score", color="Metric", barmode="group",
                color_discrete_sequence=COLOR_PALETTE, template=PLOTLY_TEMPLATE,
                title="Model Performance Across Standard Metrics"
            )
            fig_bar.update_layout(yaxis=dict(range=[0, 1.05]))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_plot2:
            st.markdown("#### 📉 Multi-Model ROC Curves")
            fig_roc = go.Figure()
            for name, m_data in results.items():
                fig_roc.add_scatter(
                    x=m_data["roc_curve"]["fpr"], 
                    y=m_data["roc_curve"]["tpr"], 
                    name=f"{name} (AUC {m_data['roc_auc']:.2f})",
                    mode="lines",
                    line=dict(width=2.5 if name == best_name else 1.5)
                )
            fig_roc.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#94a3b8"), showlegend=False)
            fig_roc.update_layout(
                title="ROC Curve Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template=PLOTLY_TEMPLATE,
                legend=dict(font=dict(size=10))
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with st.expander("⚙️ Tuned Hyperparameters per Model (GridSearchCV)"):
            for m_name, m_res in results.items():
                st.markdown(f"**{m_name}**")
                st.json(m_res["best_params"])

# ================================================================ TAB 4: DEEP DIVE
with tabs[3]:
    st.markdown("### 🔬 Model Explainability & Error Diagnosis")
    if not results or not models:
        st.info("💡 Train models first to inspect deep-dive diagnostics.")
    else:
        chosen_inspect = st.selectbox(
            "Select Model to Inspect", 
            list(results.keys()), 
            index=list(results.keys()).index(meta.get("best_model", list(results.keys())[0]))
        )
        res_inspect = results[chosen_inspect]
        cm = np.array(res_inspect["confusion_matrix"])

        col_cm, col_fi = st.columns([1, 1.3])
        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix")
            # Labels
            tn, fp, fn, tp = cm.ravel()
            total_cases = cm.sum()
            
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=["Predicted Retain (0)", "Predicted Churn (1)"],
                y=["Actual Retain (0)", "Actual Churn (1)"],
                color_continuous_scale="Purples",
                template=PLOTLY_TEMPLATE,
                title=f"Confusion Matrix: {chosen_inspect}"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Diagnostic summary metrics
            c_s1, c_s2 = st.columns(2)
            c_s1.metric("True Positives (Caught Churn)", f"{tp}")
            c_s2.metric("False Negatives (Missed Churn)", f"{fn}", delta=f"-{fn}" if fn > 0 else "0", delta_color="inverse")

        with col_fi:
            st.markdown("#### 🧠 Feature Importance / Impact")
            est = models[chosen_inspect].named_steps.get("clf")
            prep = models[chosen_inspect].named_steps.get("prep")
            feat_names = prep.get_feature_names_out() if prep else []

            importances = getattr(est, "feature_importances_", None)
            if importances is None and hasattr(est, "coef_"):
                coefs = est.coef_
                importances = np.abs(coefs[0]) if coefs.ndim > 1 else np.abs(coefs)

            if importances is not None and len(feat_names) == len(importances):
                # Clean names (remove pipeline prefixes like cat__ or num__)
                clean_names = [f.replace("cat__", "").replace("num__", "") for f in feat_names]
                fi_series = pd.Series(importances, index=clean_names).sort_values().tail(12)
                
                fig_fi = px.bar(
                    fi_series, orientation="h",
                    color=fi_series.values, color_continuous_scale="Viridis",
                    template=PLOTLY_TEMPLATE,
                    title=f"Top 12 Drivers for {chosen_inspect}",
                    labels={"value": "Importance Magnitude", "index": "Feature"}
                )
                fig_fi.update_layout(showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info(f"{chosen_inspect} is an instance-based or kernel model without direct linear feature coefficients.")

# ================================================================ TAB 5: LIVE PREDICTOR
with tabs[4]:
    st.markdown("### 🔮 Live Customer Churn Risk Prediction")
    if not models or not meta:
        st.info("💡 Train models first to unlock live customer inference.")
    else:
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            active_model_name = st.selectbox(
                "Inference Engine (Trained Model)", 
                list(models.keys()), 
                index=list(models.keys()).index(meta.get("best_model", list(models.keys())[0]))
            )
        with col_ctrl2:
            st.markdown("<div style='padding-top:28px;'></div>", unsafe_allow_html=True)
            preset_choice = st.selectbox(
                "⚡ Quick Fill Preset Scenarios", 
                ["Manual Input", "⚠️ High Risk Customer", "💎 VIP Loyal Customer", "📊 Average Customer"],
                key="preset_sel"
            )

        # Handle Preset Defaults
        default_vals = {}
        for c in meta["num_cols"]:
            lo, hi, med = meta["num_ranges"][c]
            default_vals[c] = float(med)
        for c in meta["cat_cols"]:
            default_vals[c] = meta["cat_options"][c][0]

        if preset_choice == "⚠️ High Risk Customer":
            if "Contract" in default_vals: default_vals["Contract"] = "Month-to-month"
            if "SupportCalls" in default_vals: default_vals["SupportCalls"] = 5.0
            if "TenureMonths" in default_vals: default_vals["TenureMonths"] = 2.0
            if "MonthlyCharges" in default_vals: default_vals["MonthlyCharges"] = 95.0
        elif preset_choice == "💎 VIP Loyal Customer":
            if "Contract" in default_vals: default_vals["Contract"] = "Two year" if "Two year" in meta["cat_options"].get("Contract", []) else default_vals["Contract"]
            if "SupportCalls" in default_vals: default_vals["SupportCalls"] = 0.0
            if "TenureMonths" in default_vals: default_vals["TenureMonths"] = 65.0
            if "MonthlyCharges" in default_vals: default_vals["MonthlyCharges"] = 45.0

        st.markdown("#### 👤 Customer Profile Attributes")
        inp_data = {}
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            st.caption("Numerical Usage & Billing Metrics")
            for c in meta["num_cols"]:
                lo, hi, med = meta["num_ranges"][c]
                val = default_vals[c]
                inp_data[c] = st.number_input(
                    f"{c} (Range: {lo:.0f} - {hi:.0f})", 
                    value=float(val), 
                    min_value=float(lo), 
                    max_value=float(hi*1.5),
                    key=f"inp_{c}"
                )

        with col_form2:
            st.caption("Categorical & Contractual Attributes")
            for c in meta["cat_cols"]:
                opts = meta["cat_options"][c]
                default_idx = opts.index(default_vals[c]) if default_vals[c] in opts else 0
                inp_data[c] = st.selectbox(c, opts, index=default_idx, key=f"inp_{c}")

        st.markdown("---")
        if st.button("🚀 Calculate Churn Probability", type="primary", use_container_width=True):
            sample_df = pd.DataFrame([inp_data])
            active_model = models[active_model_name]
            churn_prob = active_model.predict_proba(sample_df)[0, 1]
            churn_pred = int(churn_prob >= 0.5)

            col_res1, col_res2 = st.columns([1, 1.2])
            with col_res1:
                # Radial Gauge Indicator
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    number={'suffix': "%", 'font': {'size': 44, 'family': "Plus Jakarta Sans"}},
                    title={'text': "Predicted Churn Probability", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#6366f1"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#cbd5e1",
                        'steps': [
                            {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.25)'},
                            {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.25)'},
                            {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_res2:
                st.markdown("<div style='padding-top:30px;'></div>", unsafe_allow_html=True)
                if churn_prob >= 0.65:
                    st.markdown("""
                    <div class="risk-card-high">
                        <div style="font-size:1.6rem; font-weight:800; color:#dc2626;">🚨 CRITICAL CHURN RISK</div>
                        <div style="font-size:0.95rem; color:#475569; margin-top:6px;">
                            This customer exhibits very high propensity to discontinue service.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("##### 💡 Recommended Retention Action Plan:")
                    st.markdown("- **Immediate Account Manager Outreach**: Trigger proactive outreach within 24 hours.")
                    st.markdown("- **Contract Upgrade Incentive**: Offer 15% discount for switching to an annual plan.")
                    st.markdown("- **Priority Support Waiver**: Address open support complaints and offer free service credit.")
                elif churn_prob >= 0.35:
                    st.markdown("""
                    <div class="risk-card-med">
                        <div style="font-size:1.6rem; font-weight:800; color:#d97706;">⚠️ MODERATE CHURN RISK</div>
                        <div style="font-size:0.95rem; color:#475569; margin-top:6px;">
                            Customer shows elevated warning signs but remains receptive to engagement.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("##### 💡 Recommended Retention Action Plan:")
                    st.markdown("- **Product Feedback Survey**: Send targeted CSAT questionnaire.")
                    st.markdown("- **Usage Engagement Campaign**: Promote high-value underutilized features.")
                else:
                    st.markdown("""
                    <div class="risk-card-low">
                        <div style="font-size:1.6rem; font-weight:800; color:#059669;">✅ HEALTHY RETENTION STATUS</div>
                        <div style="font-size:0.95rem; color:#475569; margin-top:6px;">
                            Customer profile matches high-loyalty traits with minimal churn probability.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("##### 💡 Recommended Action:")
                    st.markdown("- **Upsell / Expansion**: Candidate for premium tier upgrade or referral program.")

# ================================================================ TAB 6: CLUSTERING
with tabs[5]:
    st.markdown("### 👥 Unsupervised Customer Segmentation (K-Means & Hierarchical)")
    nums = [c for c in df.select_dtypes("number").columns if c != target and c not in id_cols]
    
    col_cl1, col_cl2 = st.columns([2, 1])
    with col_cl1:
        feats = st.multiselect("Clustering Features", nums, default=nums[:min(4, len(nums))])
    with col_cl2:
        cluster_algo = st.radio("Clustering Algorithm", ["K-Means", "Agglomerative (Hierarchical)"], horizontal=True)

    if len(feats) >= 2:
        Z = StandardScaler().fit_transform(
            df[feats].fillna(df[feats].median()).clip(
                df[feats].quantile(.01), df[feats].quantile(.99), axis=1
            )
        )
        ks = range(2, 9)
        inertia, sil = [], []
        for k_val in ks:
            km = KMeans(k_val, n_init=10, random_state=0).fit(Z)
            inertia.append(km.inertia_)
            if cluster_algo == "K-Means":
                sil.append(silhouette_score(Z, km.labels_))
            else:
                sil.append(silhouette_score(Z, AgglomerativeClustering(k_val).fit_predict(Z)))
            
        opt_k = int(ks[int(np.argmax(sil))])

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            fig_elbow = px.line(
                x=list(ks), y=inertia, markers=True, 
                title="Elbow Method (K-Means Inertia)", 
                labels={"x": "Number of Clusters (k)", "y": "Inertia"},
                template=PLOTLY_TEMPLATE, color_discrete_sequence=["#6366F1"]
            )
            col_e1.plotly_chart(fig_elbow, use_container_width=True)

        with col_e2:
            fig_sil = px.line(
                x=list(ks), y=sil, markers=True, 
                title=f"Silhouette Score ({cluster_algo})", 
                labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"},
                template=PLOTLY_TEMPLATE, color_discrete_sequence=["#EC4899"]
            )
            fig_sil.add_vline(x=opt_k, line_dash="dash", line_color="#10B981", annotation_text=f"Optimal k={opt_k}")
            col_e2.plotly_chart(fig_sil, use_container_width=True)

        k_selected = st.slider("Select k Clusters", 2, 8, value=opt_k)
        
        if cluster_algo == "K-Means":
            labels = KMeans(k_selected, n_init=10, random_state=0).fit_predict(Z)
        else:
            labels = AgglomerativeClustering(k_selected).fit_predict(Z)

        # 2D PCA Projection
        P = PCA(2).fit_transform(Z)
        pca_df = pd.DataFrame(P, columns=["PC1", "PC2"])
        pca_df["Cluster"] = [f"Segment {l}" for l in labels]
        if target in df:
            pca_df[target] = df[target].values

        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            color_discrete_sequence=COLOR_PALETTE,
            title=f"2D PCA Projection of Customer Segments ({cluster_algo}, k={k_selected})",
            template=PLOTLY_TEMPLATE,
            hover_data=[target] if target in df else None
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        # Cluster Profiles
        st.markdown("#### 📋 Segment Profile Matrix & Behavioral Traits")
        prof = df[feats].assign(Segment=[f"Segment {l}" for l in labels]).groupby("Segment").mean().round(2)
        if target in df:
            prof["Churn Rate (%)"] = (df.assign(Segment=[f"Segment {l}" for l in labels]).groupby("Segment")[target].mean() * 100).round(1)
        prof["Customer Count"] = pd.Series(labels).value_counts().sort_index().values
        st.dataframe(prof.style.highlight_max(color="#dbeafe").highlight_min(color="#fef3c7"), use_container_width=True)
    else:
        st.warning("Please select at least 2 features to compute clustering.")
