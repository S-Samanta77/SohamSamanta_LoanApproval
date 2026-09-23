"""
Loan Approval Prediction & Risk Dashboard
==========================================
Academic Data Analytics + Machine Learning project.
Dataset: Loan Approval Classification Dataset (Kaggle – synthetic data)
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_FILE = "loan_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "loan_status"

EXPECTED_COLUMNS = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file", "loan_status",
]

NUMERICAL_FEATURES = [
    "person_age", "person_income", "person_emp_exp",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score",
]

CATEGORICAL_FEATURES = [
    "person_gender", "person_education", "person_home_ownership",
    "loan_intent", "previous_loan_defaults_on_file",
]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_data() -> pd.DataFrame:
    """Load loan_data.csv and perform initial type coercion."""
    if not os.path.exists(DATA_FILE):
        st.error(
            f"**Dataset not found:** `{DATA_FILE}` was not found in the project directory.\n\n"
            "Please download the dataset from Kaggle:\n"
            "https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data\n\n"
            "Rename the file to `loan_data.csv` if necessary and place it in the same "
            "folder as `SohamSamanta_LoanApproval.py`."
        )
        st.stop()

    df = pd.read_csv(DATA_FILE)

    # Normalise column names (strip whitespace, lowercase)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Verify critical columns exist
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        st.error(f"The following expected columns are missing from the CSV: {missing_cols}")
        st.stop()

    return df


# ─────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Cleaning data …")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data-quality fixes and return a cleaned DataFrame.
    All cleaning decisions are documented below.
    """
    df = df.copy()

    # 1. Enforce numeric types
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dups_removed = before - len(df)

    # 3. Drop rows where the target is missing
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # 4. Handle clearly invalid person_age values
    #    Kaggle documentation flags ages > 100 as erroneous.
    #    Ages below 18 are also implausible for a borrower.
    df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)]

    # 5. Drop rows with any remaining NaN in feature columns
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    before_na = len(df)
    df = df.dropna(subset=feature_cols)
    na_removed = before_na - len(df)

    # 6. Standardise categorical string values
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    # Store cleaning summary in session state for display
    st.session_state["cleaning_log"] = {
        "Duplicate rows removed": dups_removed,
        "Rows with NaN features removed": na_removed,
        "Rows with invalid age removed": before - len(df) - dups_removed - na_removed,
        "Final dataset size": len(df),
    }

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# EDA HELPERS
# ─────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    approved = int(df[TARGET_COL].sum())
    rejected = total - approved
    approval_rate = approved / total * 100
    avg_loan = df["loan_amnt"].mean()
    avg_income = df["person_income"].mean()
    avg_credit = df["credit_score"].mean()
    return dict(
        total=total,
        approved=approved,
        rejected=rejected,
        approval_rate=approval_rate,
        avg_loan=avg_loan,
        avg_income=avg_income,
        avg_credit=avg_credit,
    )


def auto_insights(df: pd.DataFrame) -> list[str]:
    """Return a short list of insights computed from the actual dataset."""
    insights = []

    # Approval rate
    rate = df[TARGET_COL].mean() * 100
    insights.append(f"Overall loan approval rate is **{rate:.1f}%**.")

    # Default impact
    def_rate = (
        df[df["previous_loan_defaults_on_file"] == "Yes"][TARGET_COL].mean() * 100
    )
    no_def_rate = (
        df[df["previous_loan_defaults_on_file"] == "No"][TARGET_COL].mean() * 100
    )
    insights.append(
        f"Applicants **with** previous defaults have a {def_rate:.1f}% approval rate "
        f"vs {no_def_rate:.1f}% for those **without** defaults."
    )

    # Credit score gap
    app_cs = df[df[TARGET_COL] == 1]["credit_score"].median()
    rej_cs = df[df[TARGET_COL] == 0]["credit_score"].median()
    insights.append(
        f"Median credit score for approved applicants is **{app_cs:.0f}** "
        f"vs **{rej_cs:.0f}** for rejected applicants."
    )

    # Income gap
    app_inc = df[df[TARGET_COL] == 1]["person_income"].median()
    rej_inc = df[df[TARGET_COL] == 0]["person_income"].median()
    insights.append(
        f"Median annual income for approved applicants is **${app_inc:,.0f}** "
        f"vs **${rej_inc:,.0f}** for rejected applicants."
    )

    # Most common loan intent
    top_intent = df["loan_intent"].value_counts().idxmax()
    insights.append(
        f"The most common loan intent in the dataset is **{top_intent}**."
    )

    return insights


# ─────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────
PLOT_STYLE = {"figure.facecolor": "white", "axes.facecolor": "#f9f9f9"}


def _fmt_thousands(x, pos=None):
    return f"{x:,.0f}"


def fig_approval_distribution(df):
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["Rejected (0)", "Approved (1)"]
    colors = ["#e07070", "#6abe6a"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Loan Approval Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Applications")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_thousands))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_income_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    approved = df[df[TARGET_COL] == 1]["person_income"]
    rejected = df[df[TARGET_COL] == 0]["person_income"]
    ax.hist(rejected, bins=50, alpha=0.6, color="#e07070", label="Rejected", density=True)
    ax.hist(approved, bins=50, alpha=0.6, color="#6abe6a", label="Approved", density=True)
    ax.set_title("Income Distribution by Approval Status", fontsize=12, fontweight="bold")
    ax.set_xlabel("Annual Income ($)")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_thousands))
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_credit_score_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    approved = df[df[TARGET_COL] == 1]["credit_score"]
    rejected = df[df[TARGET_COL] == 0]["credit_score"]
    ax.hist(rejected, bins=40, alpha=0.6, color="#e07070", label="Rejected", density=True)
    ax.hist(approved, bins=40, alpha=0.6, color="#6abe6a", label="Approved", density=True)
    ax.set_title("Credit Score Distribution by Approval Status", fontsize=12, fontweight="bold")
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_approval_rate_by(df, col, title):
    rate = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(rate.index.astype(str), rate.values, color="#3b82d4", edgecolor="white")
    for bar, val in zip(bars, rate.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Approval Rate (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_loan_intent(df):
    return fig_approval_rate_by(df, "loan_intent", "Approval Rate by Loan Intent")


def fig_scatter_credit_loan(df):
    sample = df.sample(min(3000, len(df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sample[TARGET_COL].map({1: "#6abe6a", 0: "#e07070"})
    ax.scatter(sample["credit_score"], sample["loan_amnt"], c=colors, alpha=0.35, s=15)
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Loan Amount ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_thousands))
    ax.set_title("Credit Score vs Loan Amount", fontsize=12, fontweight="bold")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6abe6a", label="Approved", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e07070", label="Rejected", markersize=8),
    ]
    ax.legend(handles=legend_elements)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_loan_to_income_dist(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    approved = df[df[TARGET_COL] == 1]["loan_percent_income"]
    rejected = df[df[TARGET_COL] == 0]["loan_percent_income"]
    ax.hist(rejected, bins=40, alpha=0.6, color="#e07070", label="Rejected", density=True)
    ax.hist(approved, bins=40, alpha=0.6, color="#6abe6a", label="Approved", density=True)
    ax.set_title("Loan-to-Income % Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Loan as % of Income")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_correlation_heatmap(df):
    corr_cols = NUMERICAL_FEATURES + [TARGET_COL]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_feature_importance(feature_names, importances):
    idx = np.argsort(importances)[-12:]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(
        [feature_names[i] for i in idx],
        [importances[i] for i in idx],
        color="#7c5cd8", edgecolor="white",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Feature Importances (Random Forest)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PREPROCESSING & MODEL TRAINING
# ─────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """Build the sklearn preprocessing + RandomForest pipeline."""
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


@st.cache_resource(show_spinner="Training model (first run only) …")
def train_model(df_hash: int):
    """
    Train the Random Forest pipeline.
    Accepts a hash of the dataframe so the cache key is stable.
    Returns: (pipeline, X_test, y_test, feature_names, metrics)
    """
    # Retrieve the cleaned dataframe from session state
    df = st.session_state["cleaned_df"]

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]),
    }

    # Feature names after OHE
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names

    importances = pipeline.named_steps["classifier"].feature_importances_

    return pipeline, X_test, y_test, all_feature_names, importances, metrics


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def make_prediction(pipeline, input_dict: dict):
    """
    Given a dict of raw user inputs, return (label, probability).
    label: 'Approved' or 'Rejected'
    probability: float 0-1 (probability of approval)
    """
    input_df = pd.DataFrame([input_dict])
    # Ensure column order matches training
    input_df = input_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]  # P(Approved)

    label = "Approved" if pred == 1 else "Rejected"
    return label, float(prob)


# ─────────────────────────────────────────────
# DASHBOARD SECTIONS
# ─────────────────────────────────────────────
def display_kpis(kpis: dict):
    st.markdown("### Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applications", f"{kpis['total']:,}")
    c2.metric("Approved", f"{kpis['approved']:,}")
    c3.metric("Rejected", f"{kpis['rejected']:,}")
    c4.metric("Approval Rate", f"{kpis['approval_rate']:.1f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("Avg Loan Amount", f"${kpis['avg_loan']:,.0f}")
    c6.metric("Avg Annual Income", f"${kpis['avg_income']:,.0f}")
    c7.metric("Avg Credit Score", f"{kpis['avg_credit']:.0f}")


def display_analytics_dashboard(df: pd.DataFrame, feature_names, importances):
    st.header("Analytics Dashboard")

    kpis = compute_kpis(df)
    display_kpis(kpis)

    st.markdown("---")

    # ── Overview ──────────────────────────────
    st.subheader("Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_approval_distribution(df))
    with col2:
        st.pyplot(fig_loan_to_income_dist(df))

    st.markdown("---")

    # ── Applicant Analysis ────────────────────
    st.subheader("Applicant Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_income_distribution(df))
    with col2:
        st.pyplot(fig_credit_score_distribution(df))

    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(fig_approval_rate_by(df, "person_education", "Approval Rate by Education"))
    with col4:
        st.pyplot(fig_approval_rate_by(df, "person_home_ownership", "Approval Rate by Home Ownership"))

    st.markdown("---")

    # ── Loan Analysis ─────────────────────────
    st.subheader("Loan Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_loan_intent(df))
    with col2:
        st.pyplot(fig_scatter_credit_loan(df))

    st.markdown("---")

    # ── Risk Analysis ─────────────────────────
    st.subheader("Risk Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_correlation_heatmap(df))
    with col2:
        st.pyplot(fig_feature_importance(feature_names, importances))

    st.markdown("---")

    # ── Insights ──────────────────────────────
    st.subheader("Dataset Insights")
    st.markdown(
        "> The following insights are computed directly from the dataset "
        "and are not manually authored."
    )
    for insight in auto_insights(df):
        st.markdown(f"- {insight}")

    # ── Data Cleaning Log ─────────────────────
    with st.expander("Data Cleaning Summary"):
        log = st.session_state.get("cleaning_log", {})
        for k, v in log.items():
            st.write(f"**{k}:** {v:,}" if isinstance(v, int) else f"**{k}:** {v}")


def display_model_evaluation(metrics: dict):
    st.header("Model Evaluation")
    st.markdown(
        "> This is an educational predictive model trained on a **synthetic dataset**. "
        "High accuracy on synthetic data does not imply real-world banking performance."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c3.metric("Recall", f"{metrics['recall']:.3f}")
    c4.metric("F1 Score", f"{metrics['f1']:.3f}")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.pyplot(fig_confusion_matrix(metrics["cm"]))
    with col2:
        st.subheader("Classification Report")
        st.code(metrics["report"], language="text")


def display_prediction_section(pipeline, feature_names, importances):
    st.header("Loan Approval Prediction")
    st.markdown(
        "Enter applicant and loan details below, then click **Predict Loan Approval** "
        "to receive a model-predicted outcome. Predictions are based on a Random Forest "
        "classifier trained on synthetic data and are intended for **educational purposes only**."
    )

    with st.form("prediction_form"):
        st.subheader("Applicant Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        with col2:
            gender = st.selectbox("Gender", ["female", "male"])
        with col3:
            education = st.selectbox(
                "Education",
                ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            income = st.number_input(
                "Annual Income ($)", min_value=5_000, max_value=2_000_000,
                value=60_000, step=1_000,
            )
        with col5:
            emp_exp = st.number_input(
                "Employment Experience (years)", min_value=0, max_value=50, value=5, step=1
            )
        with col6:
            home_ownership = st.selectbox(
                "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
            )

        st.subheader("Loan Details")
        col7, col8, col9 = st.columns(3)
        with col7:
            loan_amnt = st.number_input(
                "Loan Amount ($)", min_value=500, max_value=500_000,
                value=15_000, step=500,
            )
        with col8:
            loan_intent = st.selectbox(
                "Loan Intent",
                ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            )
        with col9:
            loan_int_rate = st.number_input(
                "Interest Rate (%)", min_value=1.0, max_value=30.0, value=12.0, step=0.1
            )

        st.subheader("Credit Profile")
        col10, col11, col12, col13 = st.columns(4)
        with col10:
            # Auto-calculate loan_percent_income and allow override
            auto_lpi = round(loan_amnt / max(income, 1), 4)
            loan_percent_income = st.number_input(
                "Loan as % of Income",
                min_value=0.0, max_value=1.0,
                value=float(min(auto_lpi, 1.0)),
                step=0.01,
                help="Loan amount divided by annual income. Auto-calculated but editable.",
            )
        with col11:
            cred_hist = st.number_input(
                "Credit History Length (years)", min_value=0, max_value=40, value=5, step=1
            )
        with col12:
            credit_score = st.number_input(
                "Credit Score", min_value=300, max_value=850, value=650, step=1
            )
        with col13:
            prev_defaults = st.selectbox("Previous Loan Defaults", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Loan Approval", use_container_width=True)

    if submitted:
        input_dict = {
            "person_age": float(age),
            "person_income": float(income),
            "person_emp_exp": float(emp_exp),
            "loan_amnt": float(loan_amnt),
            "loan_int_rate": float(loan_int_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_cred_hist_length": float(cred_hist),
            "credit_score": float(credit_score),
            "person_gender": gender,
            "person_education": education,
            "person_home_ownership": home_ownership,
            "loan_intent": loan_intent,
            "previous_loan_defaults_on_file": prev_defaults,
        }

        label, prob = make_prediction(pipeline, input_dict)

        st.markdown("---")
        st.subheader("Prediction Result")

        if label == "Approved":
            st.success(f"✅ **Loan Approved**")
        else:
            st.warning(f"⚠️ **Loan Not Approved**")

        prob_pct = prob * 100
        st.metric("Model-Predicted Approval Probability", f"{prob_pct:.1f}%")
        st.progress(min(prob, 1.0))

        st.caption(
            "This probability is a model-predicted estimate based on training data patterns. "
            "It is **not** a guaranteed approval probability and should not be used as a "
            "real financial decision."
        )

        # Feature importance explanation
        st.subheader("Key Factors Considered by the Model")
        st.markdown(
            "The following are the top features the Random Forest model found most "
            "useful across all training data. Feature importance indicates predictive "
            "utility, **not** that a feature independently causes approval or rejection."
        )
        top_n = 6
        top_idx = np.argsort(importances)[-top_n:][::-1]
        for rank, i in enumerate(top_idx, 1):
            bar_pct = importances[i] / importances[top_idx[0]] * 100
            st.markdown(f"**{rank}. {feature_names[i]}** — importance: {importances[i]:.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Loan Approval Prediction & Risk Dashboard",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("💳 Loan Approval Prediction & Risk Dashboard")
    st.caption(
        "Academic Data Analytics & Machine Learning project · "
        "Dataset: [Loan Approval Classification (Kaggle)](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data) · "
        "**Synthetic data — for educational purposes only**"
    )

    # ── Load & clean data ──────────────────────
    df_raw = load_data()
    df = clean_data(df_raw)
    st.session_state["cleaned_df"] = df

    # Stable cache key based on dataset size and first-row checksum
    df_hash = hash((len(df), tuple(df.iloc[0])))

    # ── Train model (cached) ───────────────────
    pipeline, X_test, y_test, feature_names, importances, metrics = train_model(df_hash)

    # ── Sidebar navigation ─────────────────────
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Analytics Dashboard", "Model Evaluation", "Loan Prediction"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset**")
    st.sidebar.write(f"Records: {len(df):,}")
    st.sidebar.write(f"Features: {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)}")
    st.sidebar.write(f"Target: `{TARGET_COL}`")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Disclaimer:** This project uses a synthetic dataset. "
        "Predictions are for educational purposes only and do not "
        "represent real financial decisions."
    )

    # ── Render selected page ───────────────────
    if page == "Analytics Dashboard":
        display_analytics_dashboard(df, feature_names, importances)
    elif page == "Model Evaluation":
        display_model_evaluation(metrics)
    elif page == "Loan Prediction":
        display_prediction_section(pipeline, feature_names, importances)


if __name__ == "__main__":
    main()
