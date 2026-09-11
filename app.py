"""
AI-Powered Predictive Maintenance & Equipment Failure Prediction System
Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import FEATURE_COLUMNS, TARGET_COLUMN, generate_dataset
from ml_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    priority_window,
    score_fleet,
    train_model,
)

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🛠️",
    layout="wide",
)

RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _demo_data(n_machines: int, seed: int) -> pd.DataFrame:
    return generate_dataset(n_machines=n_machines, seed=seed)


def load_data() -> pd.DataFrame | None:
    st.sidebar.header("1. Data")
    source = st.sidebar.radio("Data source", ["Use synthetic demo fleet", "Upload my own CSV"])

    if source == "Use synthetic demo fleet":
        n_machines = st.sidebar.slider("Number of machines", 100, 2000, 500, step=100)
        seed = st.sidebar.number_input("Random seed", value=42, step=1)
        df = _demo_data(n_machines, seed)
        st.sidebar.success(f"Loaded {len(df)} synthetic machine snapshots.")
        return df
    else:
        st.sidebar.markdown(
            "CSV must include a target column plus these feature columns:\n\n"
            f"`{', '.join(NUMERIC_FEATURES + CATEGORICAL_FEATURES)}`"
        )
        file = st.sidebar.file_uploader("Upload sensor + maintenance CSV", type=["csv"])
        if file is None:
            st.info("Upload a CSV, or switch to the synthetic demo fleet in the sidebar to explore the app.")
            return None
        df = pd.read_csv(file)
        missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
        if missing:
            st.error(f"Uploaded file is missing required columns: {missing}")
            return None
        return df


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛠️ AI-Powered Predictive Maintenance System")
st.caption(
    "Analyzes historical machine sensor and maintenance data to predict failures before they happen, "
    "identify the patterns driving risk, and prioritize maintenance action."
)

df = load_data()
if df is None:
    st.stop()

target_col = TARGET_COLUMN if TARGET_COLUMN in df.columns else st.sidebar.selectbox(
    "Select failure/target column", [c for c in df.columns if df[c].nunique() == 2]
)

st.sidebar.header("2. Model")
model_type = st.sidebar.selectbox("Model type", ["Random Forest", "Gradient Boosting"])
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.25, step=0.05)
train_clicked = st.sidebar.button("🚀 Train / Retrain Model", type="primary", use_container_width=True)

if "trained" not in st.session_state:
    st.session_state.trained = None
    st.session_state.trained_key = None

current_key = (id(df), target_col, model_type, test_size)
if train_clicked or (st.session_state.trained is None):
    with st.spinner("Training model and computing SHAP explanations..."):
        st.session_state.trained = train_model(df, target_col, model_type=model_type, test_size=test_size)
        st.session_state.trained_key = current_key

trained = st.session_state.trained

tab_overview, tab_model, tab_risk, tab_explain, tab_maint = st.tabs(
    ["📊 Data Overview", "🧠 Model Performance", "⚠️ Failure Risk Dashboard", "🔍 Explainability", "📋 Maintenance Plan"]
)

# ---------------------------------------------------------------------------
# Tab 1: Data overview
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machines", f"{len(df):,}")
    c2.metric("Failure rate", f"{df[target_col].mean()*100:.1f}%")
    c3.metric("Machine types", df["machine_type"].nunique() if "machine_type" in df else "—")
    c4.metric("Avg. days since maintenance",
              f"{df['days_since_last_maintenance'].mean():.0f}" if "days_since_last_maintenance" in df else "—")

    st.subheader("Sample data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Sensor distributions by failure outcome")
    numeric_present = [c for c in NUMERIC_FEATURES if c in df.columns]
    default_feats = [f for f in ["vibration_avg", "temperature_avg", "days_since_last_maintenance"] if f in numeric_present]
    feat_choice = st.multiselect("Features to plot", numeric_present, default=default_feats or numeric_present[:3])
    for feat in feat_choice:
        fig = px.histogram(
            df, x=feat, color=df[target_col].map({0: "No failure", 1: "Failure"}),
            barmode="overlay", nbins=40, opacity=0.65,
            color_discrete_map={"No failure": "#2ecc71", "Failure": "#e74c3c"},
            title=f"{feat} — distribution by outcome",
        )
        fig.update_layout(legend_title_text="Outcome", height=350, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if "machine_type" in df.columns:
        st.subheader("Failure rate by machine type")
        rate_by_type = df.groupby("machine_type")[target_col].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(rate_by_type, x="machine_type", y=target_col, title="Failure rate by machine type",
                     labels={target_col: "Failure rate"})
        fig.update_layout(height=350, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Model performance
# ---------------------------------------------------------------------------
with tab_model:
    m = trained.metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
    c2.metric("Avg. Precision (PR-AUC)", f"{m.get('avg_precision', float('nan')):.3f}")
    c3.metric("Base failure rate (test set)", f"{trained.y_test.mean()*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        if "roc_curve" in m:
            fpr, tpr = m["roc_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color="#3498db", width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "pr_curve" in m:
            prec, rec = m["pr_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="Model", line=dict(color="#9b59b6", width=3)))
            fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=380)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion matrix (threshold = 0.5)")
    cm = m["confusion_matrix"]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["No Failure", "Failure"], y=["No Failure", "Failure"],
    )
    fig.update_layout(height=380, width=420)
    st.plotly_chart(fig, use_container_width=False)
    st.caption(
        "Note: predictive-maintenance datasets are naturally imbalanced (failures are rare). "
        "ROC-AUC and Precision-Recall are more informative than raw accuracy here — and in the "
        "risk dashboard, maintenance teams should act on the continuous risk score rather than "
        "a single 0.5 cutoff."
    )

# ---------------------------------------------------------------------------
# Tab 3: Failure risk dashboard
# ---------------------------------------------------------------------------
with tab_risk:
    scored = score_fleet(trained, df)
    scored["priority_window"] = scored["risk_level"].apply(priority_window)

    c1, c2, c3, c4 = st.columns(4)
    counts = scored["risk_level"].value_counts()
    c1.metric("🔴 Critical", int(counts.get("Critical", 0)))
    c2.metric("🟠 High", int(counts.get("High", 0)))
    c3.metric("🟡 Medium", int(counts.get("Medium", 0)))
    c4.metric("🟢 Low", int(counts.get("Low", 0)))

    fcol1, fcol2 = st.columns([1, 1])
    with fcol1:
        level_filter = st.multiselect(
            "Filter by risk level", ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
    with fcol2:
        type_options = sorted(scored["machine_type"].unique()) if "machine_type" in scored else []
        type_filter = st.multiselect("Filter by machine type", type_options, default=type_options)

    filtered = scored[scored["risk_level"].isin(level_filter)]
    if type_options:
        filtered = filtered[filtered["machine_type"].isin(type_filter)]
    filtered = filtered.sort_values("failure_risk", ascending=False)

    st.subheader(f"Prioritized maintenance queue ({len(filtered)} machines)")
    display_cols = [
        "machine_id", "machine_type", "failure_risk", "risk_level", "priority_window",
        "top_risk_factors", "recommended_action",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    def _highlight_risk(row):
        color = RISK_COLORS.get(row["risk_level"], "#ffffff")
        return [f"background-color: {color}22"] * len(row)

    styled = filtered[display_cols].style.apply(_highlight_risk, axis=1).format({"failure_risk": "{:.1%}"})
    st.dataframe(styled, use_container_width=True, height=460)

    st.download_button(
        "⬇️ Download prioritized maintenance queue (CSV)",
        filtered[display_cols].to_csv(index=False).encode(),
        file_name="maintenance_priority_queue.csv",
        mime="text/csv",
    )

    st.subheader("Risk distribution across the fleet")
    fig = px.histogram(
        scored, x="failure_risk", color="risk_level", nbins=40,
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_level": ["Low", "Medium", "High", "Critical"]},
        title="Predicted failure risk distribution",
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Explainability
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("Global feature importance (mean |SHAP value|)")
    mean_abs_shap = np.abs(trained.shap_values_test).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": trained.feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=True).tail(15)
    fig = px.bar(
        importance_df, x="mean_abs_shap", y="feature", orientation="h",
        title="Which factors drive the model's failure predictions?",
        labels={"mean_abs_shap": "Mean |SHAP value| (impact on risk)", "feature": ""},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "SHAP values quantify how much each feature pushes an individual prediction up or down "
        "relative to the average — unlike raw feature importances, they are directional and "
        "explain single predictions, not just the model overall."
    )

    st.subheader("SHAP contribution: risk-increasing vs risk-decreasing")
    signed_mean = pd.DataFrame({
        "feature": trained.feature_names,
        "mean_shap": trained.shap_values_test.mean(axis=0),
    }).sort_values("mean_shap")
    top_bottom = pd.concat([signed_mean.head(8), signed_mean.tail(8)]).drop_duplicates()
    fig = px.bar(
        top_bottom, x="mean_shap", y="feature", orientation="h",
        color=top_bottom["mean_shap"] > 0,
        color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
        title="Average effect on predicted failure risk (test set)",
        labels={"mean_shap": "Avg. SHAP value", "feature": ""},
    )
    fig.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explain an individual machine")
    scored_all = score_fleet(trained, df).sort_values("failure_risk", ascending=False)
    machine_pick = st.selectbox("Select a machine", scored_all["machine_id"].tolist())
    row = scored_all[scored_all["machine_id"] == machine_pick].iloc[0]

    st.markdown(f"**Predicted failure risk: {row['failure_risk']*100:.1f}%  ·  Level: {row['risk_level']}**")
    st.markdown(f"**Top risk factors:** {row['top_risk_factors']}")
    st.markdown(f"**Recommended action:** {row['recommended_action']}")

    machine_row_df = df[df["machine_id"] == machine_pick] if "machine_id" in df.columns else None
    if machine_row_df is not None and len(machine_row_df) == 1:
        st.dataframe(machine_row_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: Maintenance plan
# ---------------------------------------------------------------------------
with tab_maint:
    scored = score_fleet(trained, df)
    scored["priority_window"] = scored["risk_level"].apply(priority_window)

    st.subheader("Recommended maintenance schedule")
    for level in ["Critical", "High", "Medium"]:
        subset = scored[scored["risk_level"] == level].sort_values("failure_risk", ascending=False)
        if subset.empty:
            continue
        emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡"}[level]
        with st.expander(f"{emoji} {level} priority — {len(subset)} machines — act {priority_window(level)}", expanded=(level == "Critical")):
            for _, r in subset.head(15).iterrows():
                st.markdown(
                    f"**{r['machine_id']}** ({r.get('machine_type','')}) — risk **{r['failure_risk']*100:.1f}%**  \n"
                    f"Drivers: {r['top_risk_factors']}  \n"
                    f"Action: {r['recommended_action']}"
                )
                st.divider()
            if len(subset) > 15:
                st.caption(f"... and {len(subset) - 15} more. Download the full list in the Risk Dashboard tab.")

    st.subheader("Fleet-level summary")
    summary = scored.groupby("risk_level", observed=True).agg(
        machines=("machine_id", "count"),
        avg_risk=("failure_risk", "mean"),
    ).reindex(["Critical", "High", "Medium", "Low"]).dropna(how="all")
    st.table(summary.style.format({"avg_risk": "{:.1%}"}))
