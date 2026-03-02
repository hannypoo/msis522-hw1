"""
MSIS 522 HW1 — Flaredown Food & Flare Prediction
Streamlit App with 4 Tabs

Author: Hannah
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import shap
import sys

# ── Page Config ──
st.set_page_config(
    page_title="Flaredown Food & Flare Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Paths ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
FIGURES_DIR = PROJECT_DIR / "figures"


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_DIR / "processed.parquet")
    feature_cols = pd.read_csv(DATA_DIR / "feature_cols.csv", header=None)[0].tolist()
    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")["flare"]
    return df, feature_cols, X_test, y_test


@st.cache_resource
def load_models():
    models = {}
    for name, fname in [
        ("Logistic Regression", "logistic_regression.joblib"),
        ("Decision Tree", "decision_tree.joblib"),
        ("Random Forest", "random_forest.joblib"),
        ("XGBoost", "xgboost.joblib"),
        ("Neural Network (MLP)", "mlp_neural_net.joblib"),
    ]:
        path = MODELS_DIR / fname
        if path.exists():
            models[name] = joblib.load(path)
    scaler_path = MODELS_DIR / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return models, scaler


@st.cache_data
def load_comparison():
    path = DATA_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_shap_data():
    try:
        shap_values = np.load(DATA_DIR / "shap_values.npy")
        shap_sample = pd.read_parquet(DATA_DIR / "shap_sample.parquet")
        expected = np.load(DATA_DIR / "shap_expected_value.npy")
        return shap_values, shap_sample, expected[0]
    except Exception:
        return None, None, None


# ── Load Everything ──
try:
    df, feature_cols, X_test, y_test = load_data()
    models, scaler = load_models()
    comp_df = load_comparison()
    shap_values, shap_sample, shap_expected = load_shap_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Error loading data: {e}")
    st.info("Make sure you've run the notebook first to generate all data files and models.")
    st.stop()

# ── Tabs ──
# Global derived columns (used across tabs)
food_cols_only = [c for c in feature_cols if c.startswith("food_") and not c.startswith("foodcat_")]
baseline_flare = df["flare"].mean()

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🏆 Model Performance",
    "🔍 Explainability & Prediction"
])

# ═══════════════════════════════════════════════
# TAB 1: Executive Summary
# ═══════════════════════════════════════════════
with tab1:
    st.title("🔥 Flaredown Food & Flare Prediction")
    st.markdown("### Can we predict chronic illness flares from daily food tracking?")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("User-Days", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{len(feature_cols)}")
    with col3:
        st.metric("Flare Rate", f"{df['flare'].mean():.1%}")
    with col4:
        if comp_df is not None:
            best_auc = comp_df["ROC AUC"].max()
            st.metric("Best AUC", f"{best_auc:.4f}")

    st.markdown("---")

    st.markdown("""
    ### Project Overview

    As someone living with **hypermobile Ehlers-Danlos Syndrome (hEDS)** and **POTS**,
    I wanted to explore whether daily food choices can predict symptom flares using
    the [Flaredown](https://flaredown.com/) community dataset.

    **Dataset:** [Flaredown Autoimmune Symptom Tracker on Kaggle](https://www.kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker)
    — a publicly available export from the Flaredown community tracker (~8 million rows, ~686 MB).
    - Filtered to **104,447 user-days** where users tracked both food AND symptoms
    - **Target:** Binary flare (max symptom severity ≥ 3 on 0-4 scale)
    - **Features:** 50 food items, 6 food categories, 20 treatments, weather, tags, demographics

    ### Methodology
    1. **Data Preprocessing:** Pivoted long-format data into per-day feature vectors
    2. **Within-User Paired Analysis:** Compared each user's own flare rates on eating vs. non-eating days to avoid Simpson's Paradox
    3. **5 Models Trained:** Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP Neural Net
    4. **Statistical Significance:** Wilcoxon signed-rank tests on per-user flare rate differences
    5. **SHAP analysis** on the best tree-based model for feature explainability
    """)

    st.markdown("### The Key Finding: It's Not About Food")

    st.warning(
        "**Important:** This is observational, self-reported data. Associations are not causation. "
        "People may log 'stressed' on already-bad days. Results should be discussed with a "
        "healthcare provider, not used as medical advice."
    )

    # Load the all-factors analysis
    all_factors_path = DATA_DIR / "within_user_all_factors.csv"
    if all_factors_path.exists():
        all_df = pd.read_csv(all_factors_path)
        tags_df = all_df[all_df["group"] == "Tag / Lifestyle"]
        foods_df = all_df[all_df["group"] == "Food"]
        sig_tags = tags_df[tags_df["sig_corrected"] == True].sort_values("within_diff")
        sig_foods = foods_df[foods_df["sig_corrected"] == True]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Lifestyle Factors: Strong, Significant Effects")
            st.markdown(
                f"**{len(sig_tags)} of {len(tags_df)}** lifestyle tags survived strict "
                f"Bonferroni correction:"
            )
            for _, row in sig_tags.iterrows():
                direction = "lower" if row["within_diff"] < 0 else "higher"
                icon = "+" if row["within_diff"] > 0 else ""
                p_str = "p < 0.001" if row["p_value"] < 0.001 else f"p = {row['p_value']:.4f}"
                st.markdown(
                    f"- **{row['name']}** — {icon}{row['within_diff']:.1%} flare rate "
                    f"({p_str}, n={row['n_users']} users)"
                )

        with col2:
            st.markdown("#### Individual Foods: No Significant Effects")
            st.markdown(
                f"**0 of {len(foods_df)}** foods survived multiple comparison correction. "
                f"After testing 50 foods, the few uncorrected p < 0.05 results are consistent "
                f"with expected false positives (~2.5 expected by chance)."
            )
            st.markdown(
                "This doesn't mean food *never* matters — it means individual food effects "
                "are too small and variable to detect at the population level in self-reported data. "
                "Your personal triggers may differ."
            )

        st.info(
            "**Bottom line:** Stress management and sleep quality show effects 10-20x larger "
            "than any food. The ML models achieve ~77% AUC by combining many weak signals "
            "(foods + tags + treatments + weather), not from any single food."
        )
    else:
        st.info("Analysis data not yet generated. Run the notebook first.")

    st.markdown("""
    ### Takeaways
    - **Prioritize sleep and stress management** — these are the strongest, most reliable predictors of flares
    - **No single food reliably triggers or prevents flares** at the population level
    - **Track YOUR patterns** — individual triggers may exist even if they don't show up in population averages
    - **Consult your healthcare provider** before making changes based on data
    """)

# ═══════════════════════════════════════════════
# TAB 2: Descriptive Analytics
# ═══════════════════════════════════════════════
with tab2:
    st.header("📊 Descriptive Analytics")

    # Target Distribution
    st.subheader("Target Distribution")
    no_flare_count = len(df[df["flare"] == 0])
    flare_count = len(df[df["flare"] == 1])

    col1, col2 = st.columns([1, 1])
    with col1:
        fig = go.Figure(data=[
            go.Bar(x=["No Flare (0)", "Flare (1)"],
                   y=[no_flare_count, flare_count],
                   marker_color=["#2ecc71", "#e74c3c"],
                   text=[f'{no_flare_count:,} ({no_flare_count/len(df):.1%})',
                         f'{flare_count:,} ({flare_count/len(df):.1%})'],
                   textposition='inside',
                   textfont=dict(color="white", size=14))
        ])
        fig.update_layout(
            title="Flare Distribution (Binary Target)",
            yaxis_title="User-Days",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "max_symptom_severity" in df.columns:
            sev_counts = df["max_symptom_severity"].value_counts().sort_index()
            colors_sev = ["#2ecc71", "#a3d977", "#f1c40f", "#e67e22", "#e74c3c"]
            fig = go.Figure(data=[
                go.Bar(x=sev_counts.index.astype(str),
                       y=sev_counts.values,
                       marker_color=colors_sev[:len(sev_counts)])
            ])
            fig.add_vline(x=2.5, line_dash="dash", line_color="red",
                          annotation_text="Flare threshold (≥3)")
            fig.update_layout(title="Max Daily Symptom Severity", xaxis_title="Severity (0-4)",
                              yaxis_title="User-Days", height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Left chart — Flare Distribution:** The target is **imbalanced** — {flare_count/len(df):.1%} of "
        f"user-days are flare days vs. {no_flare_count/len(df):.1%} no-flare, roughly a **2:1 ratio**. "
        f"This is expected since Flaredown users are people actively managing chronic illnesses. "
        f"We handle this with `class_weight='balanced'` on all models, stratified train/test splits, "
        f"and evaluating on **F1 / ROC AUC** rather than raw accuracy."
    )
    st.markdown(
        "**Right chart — Symptom Severity:** Each user-day's maximum symptom severity is recorded on a "
        "0–4 scale (0 = none, 1 = mild, 2 = moderate, 3 = severe, 4 = unbearable). "
        "Severity **4 is the most common** single category, indicating that on many days at least one "
        "symptom peaks at the worst level. The red dashed line marks the **flare threshold (≥ 3)** — "
        "days at severity 3 or 4 are labeled as flares. This threshold captures days where symptoms "
        "meaningfully impair daily functioning."
    )

    # ── Lifestyle Factors — THE KEY FINDING (shown first) ──
    st.subheader("Lifestyle Factors & Flare Risk (Key Finding)")

    all_factors_path = DATA_DIR / "within_user_all_factors.csv"
    if all_factors_path.exists():
        all_df = pd.read_csv(all_factors_path)
        tags_all = all_df[all_df["group"] == "Tag / Lifestyle"]
        # Only show statistically significant factors (Bonferroni corrected)
        tags_sig = tags_all[tags_all["sig_corrected"] == True].sort_values("within_diff")

        if len(tags_sig) > 0:
            tag_colors = ["#e74c3c" if row["within_diff"] > 0 else "#2ecc71"
                          for _, row in tags_sig.iterrows()]

            tag_hover = [
                f"{row['name']}<br>Diff: {row['within_diff']:+.1%}<br>"
                f"p-value: {row['p_value']:.6f}<br>"
                f"Users: {row['n_users']}"
                for _, row in tags_sig.iterrows()
            ]

            tag_labels = [row["name"] for _, row in tags_sig.iterrows()]

            fig = go.Figure(data=[
                go.Bar(
                    y=tag_labels, x=tags_sig["within_diff"].values,
                    orientation="h", marker_color=tag_colors,
                    hovertext=tag_hover, hoverinfo="text"
                )
            ])
            fig.add_vline(x=0, line_color="black", line_width=2)
            fig.update_layout(
                title="Statistically Significant Lifestyle Factors (Bonferroni Corrected)<br>"
                      "<sub>Green = protective (lowers flare risk) | Red = trigger (raises flare risk)</sub>",
                xaxis_title="Within-User Flare Rate Difference",
                xaxis_tickformat="+.0%",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        n_tested = len(tags_all)
        n_sig = len(tags_sig)
        n_protective = len(tags_sig[tags_sig["within_diff"] < 0])
        n_triggers = len(tags_sig[tags_sig["within_diff"] > 0])
        st.markdown(
            f"**{n_sig} of {n_tested}** lifestyle tags survived strict Bonferroni correction "
            f"(p < 0.05/{n_tested} = {0.05/n_tested:.4f}). "
            f"**Good Sleep** is the only statistically significant **protective** factor (-12.6%), "
            f"while {n_triggers} factors significantly **increase** flare risk."
        )
        st.markdown(
            "**Why only one protective factor?** The results skew toward triggers partly because of "
            "the high baseline flare rate (68%). With flares already being the norm, there is limited "
            "statistical room for something to *further increase* the rate vs. more room for it to *decrease* it "
            "— yet the data shows the opposite pattern. This likely reflects **self-reporting bias**: users "
            "tend to tag negative states (tired, stressed, exhausted) on days they already feel bad, "
            "while positive tags like 'good sleep' are logged more deliberately. Some treatments (e.g., yoga, "
            "paracetamol) showed promising protective trends but had too few users to reach significance "
            "after correction."
        )
        st.markdown(
            "**A note on 'Poor Sleep' vs. 'Bad Sleep':** These are separate user-created tags in "
            "Flaredown — users type in their own labels, so different people chose different words for "
            "similar concepts. Both independently show a significant increase in flare risk (+6.7% and "
            "+5.5% respectively), which actually *strengthens* the finding: the sleep-flare connection "
            "replicates across two independently tagged groups of users."
        )
    else:
        st.info("Lifestyle analysis not yet available. Run the notebook first.")

    st.markdown("---")

    # ── Food Analysis (de-emphasized — null result) ──
    with st.expander("Individual Foods & Flare Risk (Null Result — click to expand)", expanded=False):
        st.markdown(
            "This chart shows the **within-user** flare rate difference for each food: comparing the **same users'** "
            "flare rates on days they ate a food vs. days they didn't. This controls for Simpson's Paradox, where "
            "differences in *who* tracks a food can make it falsely appear protective or harmful."
        )

        st.markdown(
            "**After correcting for multiple comparisons (50 tests), no food reaches statistical significance.** "
            "Faded bars had uncorrected p < 0.05 but are likely false positives. Gray bars were not significant "
            "even before correction."
        )

        wu_path = DATA_DIR / "within_user_food_analysis.csv"
        if wu_path.exists():
            wu_df = pd.read_csv(wu_path).sort_values("within_diff")

            has_correction_cols = "sig_uncorrected" in wu_df.columns

            bar_colors = []
            for _, row in wu_df.iterrows():
                if has_correction_cols:
                    if row.get("sig_corrected", False) or row.get("sig_fdr", False):
                        bar_colors.append("#e74c3c" if row["within_diff"] > 0 else "#2ecc71")
                    elif row.get("sig_uncorrected", False):
                        bar_colors.append("#f5b7b1" if row["within_diff"] > 0 else "#abebc6")
                    else:
                        bar_colors.append("#cccccc")
                else:
                    bar_colors.append("#cccccc")

            hover_text = [
                f"{row['name']}<br>Diff: {row['within_diff']:+.1%}<br>"
                f"p-value: {row['p_value']:.4f}<br>"
                f"Users: {row['n_users']}<br>"
                f"Eating days: {row['on_days']}<br>"
                f"{'Uncorrected p<0.05 (likely false positive)' if row.get('sig_uncorrected', False) and not row.get('sig_corrected', False) else 'Not significant'}"
                for _, row in wu_df.iterrows()
            ]

            fig = go.Figure(data=[
                go.Bar(
                    y=wu_df["name"].values, x=wu_df["within_diff"].values,
                    orientation="h", marker_color=bar_colors,
                    hovertext=hover_text, hoverinfo="text"
                )
            ])
            fig.add_vline(x=0, line_color="black", line_width=2)
            fig.update_layout(
                title="Within-User Flare Rate Difference by Food<br>"
                      "<sub>No food survives multiple comparison correction (Bonferroni or FDR)</sub>",
                xaxis_title="Flare Rate Difference (Eaten - Not Eaten)",
                xaxis_tickformat="+.0%",
                height=max(500, len(wu_df) * 22),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Significance table
            with st.expander("View full statistical details"):
                cols_to_show = ["name", "n_users", "on_days", "within_diff", "p_value"]
                display_df = wu_df[cols_to_show].copy()
                display_df.columns = ["Food", "Users", "Eating Days", "Flare Rate Diff", "p-value"]
                display_df["Flare Rate Diff"] = display_df["Flare Rate Diff"].apply(lambda x: f"{x:+.2%}")
                display_df["p-value"] = display_df["p-value"].apply(lambda x: f"{x:.4f}")
                bonf = 0.05 / len(wu_df)
                st.markdown(f"**Bonferroni threshold:** p < {bonf:.4f} (none pass)")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Within-user analysis not yet available. Run the notebook first.")

    # ── Top Foods Frequency (also de-emphasized) ──
    with st.expander("Top 20 Most Tracked Foods (click to expand)", expanded=False):
        st.markdown(
            "This chart shows which foods users logged most frequently. "
            "**None of these foods showed a statistically significant effect on flare risk** "
            "after correcting for multiple comparisons (Bonferroni, 50 simultaneous tests). "
            "The frequency of tracking does not imply any relationship with flares."
        )
        food_freq = df[food_cols_only].sum().sort_values(ascending=False).head(20)
        food_names = [c.replace("food_", "").replace("_", " ").title() for c in food_freq.index]
        fig = go.Figure(data=[
            go.Bar(y=food_names[::-1], x=food_freq.values[::-1],
                   orientation="h", marker_color=px.colors.sequential.YlOrRd_r[:20])
        ])
        fig.update_layout(title="Most Tracked Foods (No Significant Flare Effects Found)",
                          xaxis_title="User-Days", height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap — Top 25 Features")
    corr_flare = df[feature_cols + ["flare"]].corr()["flare"].drop("flare").abs().sort_values(ascending=False)
    top25 = corr_flare.head(25).index.tolist()
    corr_mat = df[top25 + ["flare"]].corr()
    clean_labels = [c.replace("food_", "").replace("treat_", "Tx:").replace("tag_", "Tag:").replace("weather_", "W:").replace("foodcat_", "Cat:").replace("_", " ").title() for c in corr_mat.columns]
    clean_labels[-1] = "FLARE"

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=clean_labels, yticklabels=clean_labels, ax=ax,
                vmin=-0.3, vmax=0.3, square=True, linewidths=0.5)
    ax.set_title("Correlation Heatmap — Top 25 Features vs Flare", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════
# TAB 3: Model Performance
# ═══════════════════════════════════════════════
with tab3:
    st.header("🏆 Model Performance")

    if comp_df is not None:
        st.subheader("Performance Comparison Table")
        st.dataframe(
            comp_df.style.highlight_max(axis=0, color="lightgreen").format("{:.4f}"),
            use_container_width=True
        )

        best_model_name = comp_df["F1 Score"].idxmax()
        best_f1 = comp_df.loc[best_model_name, "F1 Score"]
        st.success(f"**Best model by F1 Score:** {best_model_name} ({best_f1:.4f})")

        # Bar chart comparison
        st.subheader("Visual Comparison")
        fig = go.Figure()
        metrics = comp_df.columns.tolist()
        colors_m = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric, x=comp_df.index, y=comp_df[metric],
                marker_color=colors_m[i % len(colors_m)]
            ))
        fig.update_layout(barmode="group", title="Model Metrics Comparison",
                          yaxis_title="Score", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # ROC Curves
        st.subheader("ROC Curves")
        from sklearn.metrics import roc_curve, roc_auc_score
        fig = go.Figure()
        line_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

        for i, (name, model) in enumerate(models.items()):
            try:
                if name in ["Logistic Regression", "Neural Network (MLP)"] and scaler is not None:
                    y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
                else:
                    y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_val = roc_auc_score(y_test, y_prob)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{name} (AUC={auc_val:.4f})",
                                         line=dict(color=line_colors[i % len(line_colors)], width=2)))
            except Exception:
                pass

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 name="Random (AUC=0.5)",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(title="ROC Curves — All Models",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrix
        st.subheader("Confusion Matrix — Best Model")
        from sklearn.metrics import confusion_matrix
        if best_model_name in models:
            model = models[best_model_name]
            if best_model_name in ["Logistic Regression", "Neural Network (MLP)"] and scaler:
                y_pred = model.predict(scaler.transform(X_test))
            else:
                y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["No Flare", "Flare"], yticklabels=["No Flare", "Flare"])
            ax.set_title(f"Confusion Matrix — {best_model_name}", fontsize=14, fontweight="bold")
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("No model comparison data found. Run the notebook first.")

# ═══════════════════════════════════════════════
# TAB 4: Explainability & Interactive Prediction
# ═══════════════════════════════════════════════
with tab4:
    st.header("🔍 Explainability & Interactive Prediction")

    # SHAP Plots
    if shap_values is not None and shap_sample is not None:
        st.subheader("SHAP Feature Importance (XGBoost)")

        clean_names = [c.replace("food_", "").replace("treat_", "Tx:").replace("tag_", "Tag:").replace("weather_", "W:").replace("foodcat_", "Cat:").replace("_", " ").title() for c in feature_cols]
        shap_display = shap_sample.copy()
        shap_display.columns = clean_names

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Beeswarm Plot")
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, shap_display, max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### Mean |SHAP| Bar Plot")
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, shap_display, plot_type="bar", max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("SHAP data not yet available. Run the notebook to generate SHAP values.")

    # ── Interactive Prediction ──
    st.markdown("---")
    st.subheader("🎯 Interactive Flare Prediction")
    st.markdown("Select your foods, treatments, and conditions to get a real-time flare prediction.")

    # Use XGBoost for prediction (best tree model)
    xgb_model = models.get("XGBoost")
    if xgb_model is None:
        st.warning("XGBoost model not found. Run the notebook first.")
    else:
        # Create input vector
        input_values = np.zeros(len(feature_cols))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 😴 Lifestyle (Strongest Predictors)")
            tag_options = {
                "tag_good_sleep": "Good Sleep",
                "tag_bad_sleep": "Bad Sleep",
                "tag_poor_sleep": "Poor Sleep",
                "tag_tired": "Tired",
                "tag_exhausted": "Exhausted",
                "tag_stressed": "Stressed",
                "tag_walked": "Walked",
                "tag_ate_breakfast": "Ate Breakfast",
                "tag_period": "Period",
                "tag_dairy": "Had Dairy",
                "tag_alcohol": "Had Alcohol",
                "tag_gluten": "Had Gluten",
                "tag_had_sex": "Had Sex",
                "tag_worked": "Worked",
                "tag_went_to_work": "Went to Work",
            }
            available_tags = {k: v for k, v in tag_options.items() if k in feature_cols}
            selected_tags = st.multiselect(
                "Select lifestyle tags (these matter most!):",
                options=list(available_tags.values()),
                help="Sleep and stress tags are the strongest flare predictors"
            )
            tag_name_to_col = {v: k for k, v in available_tags.items()}

        with col2:
            st.markdown("#### 💊 Treatments Taken")
            treat_options = sorted([c.replace("treat_", "").replace("_", " ").title()
                                    for c in feature_cols if c.startswith("treat_")])
            selected_treats = st.multiselect("Select treatments:", treat_options)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### 🍽️ Foods Eaten Today")
            food_options = sorted([c.replace("food_", "").replace("_", " ").title()
                                   for c in feature_cols if c.startswith("food_") and not c.startswith("foodcat_")])
            selected_foods = st.multiselect("Select foods:", food_options)

        with col4:
            st.markdown("#### 🌤️ Demographics & Weather")
            age = st.slider("Age", 10, 80, 30)
            sex = st.selectbox("Sex", ["Female", "Male", "Other/Unknown"])

            weather_icon_opts = [c.replace("weather_icon_", "").replace("_", " ").title()
                                 for c in feature_cols if c.startswith("weather_icon_")]
            weather = st.selectbox("Weather", ["None"] + weather_icon_opts)

        # Build feature vector
        for i, col in enumerate(feature_cols):
            if col.startswith("tag_"):
                # Map tag columns to selected lifestyle tags
                if col in tag_name_to_col.values():
                    label = available_tags.get(col, "")
                    if label in selected_tags:
                        input_values[i] = 1
            elif col.startswith("food_") and not col.startswith("foodcat_"):
                name = col.replace("food_", "").replace("_", " ").title()
                if name in selected_foods:
                    input_values[i] = 1
            elif col.startswith("treat_"):
                name = col.replace("treat_", "").replace("_", " ").title()
                if name in selected_treats:
                    input_values[i] = 1
            elif col.startswith("weather_icon_"):
                name = col.replace("weather_icon_", "").replace("_", " ").title()
                if name == weather:
                    input_values[i] = 1
            elif col == "age":
                input_values[i] = age
            elif col == "sex_female":
                input_values[i] = 1.0 if sex == "Female" else (0.0 if sex == "Male" else 0.5)
            elif col == "country_us":
                input_values[i] = 1

        # Set food category rollups
        _food_cats = {
            'dairy': ['cheese', 'milk', 'yogurt', 'butter', 'cream', 'ice cream'],
            'grains': ['bread', 'rice', 'pasta', 'wheat', 'oats', 'cereal', 'crackers', 'corn'],
            'fruits': ['apple', 'banana', 'berries', 'strawberry', 'blueberry', 'orange', 'grapes', 'avocado', 'tomato'],
            'protein': ['chicken', 'beef', 'pork', 'fish', 'egg', 'eggs', 'turkey', 'bacon', 'salmon'],
            'caffeine': ['coffee', 'tea', 'caffeine', 'soda', 'cola', 'coke'],
            'sugar': ['sugar', 'chocolate', 'candy', 'cookie', 'cake', 'dessert', 'ice cream', 'honey']
        }
        for cat_name, keywords in _food_cats.items():
            cat_col = f"foodcat_{cat_name}"
            if cat_col in feature_cols:
                selected_lower = [f.lower() for f in selected_foods]
                if any(k in s for s in selected_lower for k in keywords):
                    idx = feature_cols.index(cat_col)
                    input_values[idx] = 1

        # Predict
        if st.button("🔮 Predict Flare Risk", type="primary"):
            input_df = pd.DataFrame([input_values], columns=feature_cols)
            prob = xgb_model.predict_proba(input_df)[0][1]
            pred = xgb_model.predict(input_df)[0]

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if pred == 1:
                    st.error(f"### ⚠️ Flare Predicted\nProbability: **{prob:.1%}**")
                else:
                    st.success(f"### ✅ No Flare Predicted\nProbability of flare: **{prob:.1%}**")

            with col2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Flare Risk (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#e74c3c" if prob > 0.5 else "#2ecc71"},
                        "steps": [
                            {"range": [0, 30], "color": "#d5f5e3"},
                            {"range": [30, 60], "color": "#fdebd0"},
                            {"range": [60, 100], "color": "#fadbd8"}
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # SHAP waterfall for this prediction
            if shap_values is not None:
                st.subheader("What's driving this prediction?")
                try:
                    explainer = shap.TreeExplainer(xgb_model)
                    sv = explainer.shap_values(input_df)
                    explanation = shap.Explanation(
                        values=sv[0],
                        base_values=explainer.expected_value,
                        data=input_df.iloc[0].values,
                        feature_names=clean_names
                    )
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.plots.waterfall(explanation, max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"Could not generate SHAP waterfall: {e}")

# Footer
st.markdown("---")
st.markdown(
    "*MSIS 522 HW1 — Flaredown Food & Flare Prediction | "
    "Built with Streamlit, scikit-learn, XGBoost, and SHAP*"
)
