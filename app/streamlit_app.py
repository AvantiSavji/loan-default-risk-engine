import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Loan Default Risk Engine",
    page_icon  = "🏦",
    layout     = "wide"
)

# ─────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('../models/xgb_model.pkl')

@st.cache_data
def load_data():
    X_test     = pd.read_csv('../data/X_test.csv')
    y_test     = pd.read_csv('../data/y_test.csv').squeeze()
    shap_vals  = pd.read_csv('../data/shap_values.csv')
    return X_test, y_test, shap_vals

model              = load_model()
X_test, y_test, shap_vals = load_data()
explainer          = shap.TreeExplainer(model)
feature_names      = X_test.columns.tolist()

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
st.sidebar.title("Loan Default Risk Engine")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Risk Predictor",
        "Model Insights",
        "SHAP Explainability",
        "Portfolio Dashboard"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown("Built by Avanti Savji")
st.sidebar.markdown("XGBoost | SHAP | Streamlit | Plotly")

# ─────────────────────────────────────────
# HELPER — RISK BADGE
# ─────────────────────────────────────────
def get_risk_badge(prob):
    if prob < 0.3:
        return "LOW RISK", "green"
    elif prob < 0.6:
        return "MEDIUM RISK", "orange"
    else:
        return "HIGH RISK", "red"

# ═════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════
if page == "Home":

    st.title("Loan Default Risk Engine")
    st.subheader("AI-Powered Credit Risk Assessment with Explainable Predictions")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Applicants",  f"{len(X_test):,}")
    with col2:
        pred_proba = model.predict_proba(X_test)[:, 1]
        high_risk  = (pred_proba >= 0.5).sum()
        st.metric("High Risk Flagged", f"{high_risk:,}")
    with col3:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, pred_proba)
        st.metric("Model AUC-ROC", f"{auc:.4f}")
    with col4:
        default_rate = y_test.mean() * 100
        st.metric("Actual Default Rate", f"{default_rate:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What This App Does")
        st.markdown("""
        This application uses a trained **XGBoost model** to assess
        the probability that a loan applicant will default.

        - Input applicant details and get an instant risk score
        - Understand **why** the model made that decision using SHAP
        - Explore model performance through interactive charts
        - Designed with **RBI compliance** in mind — every decision is explainable
        """)

    with col2:
        st.subheader("Why Explainability Matters")
        st.markdown("""
        Traditional credit models are black boxes.
        Regulators and applicants have the right to know
        **why** a loan was rejected.

        This engine uses **SHAP (SHapley Additive exPlanations)**
        to break down every prediction into individual
        feature contributions — making AI decisions transparent
        and auditable.
        """)

    st.markdown("---")
    st.info(
        "Use the sidebar to navigate. "
        "Start with Risk Predictor to score a new applicant."
    )


# ═════════════════════════════════════════
# PAGE 2 — RISK PREDICTOR
# ═════════════════════════════════════════
elif page == "Risk Predictor":

    st.title("Applicant Risk Predictor")
    st.markdown("Fill in the applicant details to get an instant risk assessment.")
    st.markdown("---")

    # Use median values as defaults
    defaults = X_test.median()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Financial Profile")
        amt_income = st.number_input(
            "Annual Income (USD)",
            min_value   = 10000,
            max_value   = 1000000,
            value       = int(defaults.get('AMT_INCOME_TOTAL', 150000)),
            step        = 5000
        )
        amt_credit = st.number_input(
            "Loan Amount Requested (USD)",
            min_value   = 10000,
            max_value   = 2000000,
            value       = int(defaults.get('AMT_CREDIT', 500000)),
            step        = 10000
        )
        amt_annuity = st.number_input(
            "Monthly Annuity / EMI (USD)",
            min_value   = 1000,
            max_value   = 100000,
            value       = int(defaults.get('AMT_ANNUITY', 25000)),
            step        = 1000
        )
        amt_goods = st.number_input(
            "Goods Price (USD)",
            min_value   = 10000,
            max_value   = 2000000,
            value       = int(defaults.get('AMT_GOODS_PRICE', 450000)),
            step        = 10000
        )

    with col2:
        st.subheader("Personal Profile")
        age_years = st.slider(
            "Applicant Age (Years)",
            min_value = 18,
            max_value = 70,
            value     = int(defaults.get('AGE_YEARS', 35))
        )
        years_employed = st.slider(
            "Years Employed",
            min_value = 0,
            max_value = 40,
            value     = int(defaults.get('YEARS_EMPLOYED', 5))
        )
        cnt_children = st.number_input(
            "Number of Children",
            min_value = 0,
            max_value = 10,
            value     = int(defaults.get('CNT_CHILDREN', 0))
        )
        cnt_fam = st.number_input(
            "Family Members",
            min_value = 1,
            max_value = 15,
            value     = int(defaults.get('CNT_FAM_MEMBERS', 2))
        )

    with col3:
        st.subheader("Credit Scores")
        ext_source_1 = st.slider(
            "External Credit Score 1",
            min_value = 0.0,
            max_value = 1.0,
            value     = float(defaults.get('EXT_SOURCE_1', 0.5)),
            step      = 0.01
        )
        ext_source_2 = st.slider(
            "External Credit Score 2",
            min_value = 0.0,
            max_value = 1.0,
            value     = float(defaults.get('EXT_SOURCE_2', 0.5)),
            step      = 0.01
        )
        ext_source_3 = st.slider(
            "External Credit Score 3",
            min_value = 0.0,
            max_value = 1.0,
            value     = float(defaults.get('EXT_SOURCE_3', 0.5)),
            step      = 0.01
        )

    st.markdown("---")

    if st.button("Calculate Risk Score", type="primary"):

        # Build input row using test set median as base
        # then override with user inputs
        input_data         = pd.DataFrame([defaults], columns=feature_names)

        # Override with user inputs
        input_data['AMT_INCOME_TOTAL']    = amt_income
        input_data['AMT_CREDIT']          = amt_credit
        input_data['AMT_ANNUITY']         = amt_annuity
        input_data['AMT_GOODS_PRICE']     = amt_goods
        input_data['CNT_CHILDREN']        = cnt_children
        input_data['CNT_FAM_MEMBERS']     = cnt_fam

        # Engineered features
        input_data['AGE_YEARS']           = age_years
        input_data['YEARS_EMPLOYED']      = years_employed
        input_data['DAYS_BIRTH']          = -age_years * 365
        input_data['DAYS_EMPLOYED']       = -years_employed * 365
        input_data['CREDIT_INCOME_RATIO'] = amt_credit / amt_income
        input_data['ANNUITY_INCOME_RATIO']= amt_annuity / amt_income
        input_data['CREDIT_GOODS_RATIO']  = amt_credit / max(amt_goods, 1)
        input_data['EMPLOYMENT_AGE_RATIO']= years_employed / max(age_years, 1)
        input_data['INCOME_PER_PERSON']   = amt_income / max(cnt_fam, 1)

        # External scores
        if 'EXT_SOURCE_1' in feature_names:
            input_data['EXT_SOURCE_1'] = ext_source_1
        if 'EXT_SOURCE_2' in feature_names:
            input_data['EXT_SOURCE_2'] = ext_source_2
        if 'EXT_SOURCE_3' in feature_names:
            input_data['EXT_SOURCE_3'] = ext_source_3

        # Predict
        prob       = model.predict_proba(input_data)[0][1]
        risk_label, risk_color = get_risk_badge(prob)

        # Results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style='text-align:center; padding:20px;
                        border-radius:10px;
                        background-color:#f0f2f6'>
                <h2 style='color:{risk_color}'>{risk_label}</h2>
                <h1 style='color:{risk_color}'>{prob*100:.1f}%</h1>
                <p>Probability of Default</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                title = {"text": "Default Risk %"},
                gauge = {
                    "axis"  : {"range": [0, 100]},
                    "bar"   : {"color": risk_color},
                    "steps" : [
                        {"range": [0,  30], "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100],"color": "#f8d7da"}
                    ],
                    "threshold": {
                        "line" : {"color": "black", "width": 4},
                        "value": 50
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=30,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            credit_income = amt_credit / amt_income
            annuity_income = amt_annuity / amt_income
            st.subheader("Key Ratios")
            st.metric("Credit / Income Ratio",
                      f"{credit_income:.2f}x",
                      delta="High Risk" if credit_income > 5 else "Acceptable",
                      delta_color="inverse")
            st.metric("Annuity / Income Ratio",
                      f"{annuity_income:.2%}",
                      delta="High Burden" if annuity_income > 0.3 else "Manageable",
                      delta_color="inverse")

        # SHAP explanation for this prediction
        st.markdown("---")
        st.subheader("Why This Risk Score? — SHAP Explanation")

        shap_input    = explainer.shap_values(input_data)
        shap_series   = pd.Series(
            shap_input[0],
            index=feature_names
        ).sort_values(key=abs, ascending=False).head(10)

        colors = ['tomato' if v > 0 else 'steelblue'
                  for v in shap_series.values]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(shap_series.index[::-1],
                       shap_series.values[::-1],
                       color=colors[::-1])
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP Value (positive = increases default risk)")
        ax.set_title("Top Factors Driving This Risk Score")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **How to read this chart:**
        - Red bars = factors **increasing** default risk
        - Blue bars = factors **decreasing** default risk
        - Longer bar = stronger influence on the decision
        """)


# ═════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═════════════════════════════════════════
elif page == "Model Insights":

    st.title("Model Performance Insights")
    st.markdown("---")

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_class = (pred_proba >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    auc = roc_auc_score(y_test, pred_proba)
    col1.metric("AUC-ROC",          f"{auc:.4f}")
    col2.metric("Test Samples",     f"{len(y_test):,}")
    col3.metric("Actual Defaulters",f"{y_test.sum():,}")
    col4.metric("Flagged as Risk",  f"{pred_class.sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, pred_proba)
        fig = px.area(
            x=fpr, y=tpr,
            title=f"ROC Curve (AUC = {auc:.4f})",
            labels={"x": "False Positive Rate",
                    "y": "True Positive Rate"},
            color_discrete_sequence=["steelblue"]
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", color="gray"),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Confusion Matrix
        cm = confusion_matrix(y_test, pred_class)
        fig = px.imshow(
            cm,
            text_auto   = True,
            color_continuous_scale = "Blues",
            labels      = dict(x="Predicted", y="Actual"),
            x           = ["Repaid", "Defaulted"],
            y           = ["Repaid", "Defaulted"],
            title       = "Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Risk Score Distribution
    st.subheader("Risk Score Distribution")
    fig = px.histogram(
        x           = pred_proba,
        color       = y_test.astype(str),
        nbins       = 50,
        barmode     = "overlay",
        opacity     = 0.7,
        title       = "Predicted Default Probability by Actual Outcome",
        labels      = {"x": "Predicted Default Probability",
                       "color": "Actual (0=Repaid, 1=Default)"},
        color_discrete_map = {"0": "steelblue", "1": "tomato"}
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINABILITY
# ═════════════════════════════════════════
elif page == "SHAP Explainability":

    st.title("SHAP Explainability Dashboard")
    st.markdown(
        "SHAP explains how each feature contributes "
        "to every individual prediction."
    )
    st.markdown("---")

    # Global feature importance
    st.subheader("Global Feature Importance")
    mean_shap = shap_vals.abs().mean().sort_values(ascending=False).head(15)

    fig = px.bar(
        x               = mean_shap.values,
        y               = mean_shap.index,
        orientation     = "h",
        title           = "Mean Absolute SHAP Values — Top 15 Features",
        labels          = {"x": "Mean |SHAP Value|", "y": "Feature"},
        color           = mean_shap.values,
        color_continuous_scale = "Reds"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Individual explanation
    st.subheader("Individual Applicant Explanation")
    applicant_idx = st.slider(
        "Select applicant from test set",
        min_value = 0,
        max_value = len(X_test) - 1,
        value     = 0
    )

    col1, col2 = st.columns(2)

    with col1:
        prob   = model.predict_proba(
            X_test.iloc[[applicant_idx]]
        )[0][1]
        actual = y_test.iloc[applicant_idx]
        label, color = get_risk_badge(prob)

        st.markdown(f"""
        <div style='padding:15px; border-radius:8px;
                    background-color:#f0f2f6; text-align:center'>
            <h3>Applicant #{applicant_idx}</h3>
            <h2 style='color:{color}'>{label}</h2>
            <h3>Risk Score: {prob*100:.1f}%</h3>
            <p>Actual Outcome:
                {'Defaulted' if actual == 1 else 'Repaid'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        shap_row = shap_vals.iloc[applicant_idx].sort_values(
            key=abs, ascending=False
        ).head(8)

        colors = ['tomato' if v > 0 else 'steelblue'
                  for v in shap_row.values]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(shap_row.index[::-1],
                shap_row.values[::-1],
                color=colors[::-1])
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP Value")
        ax.set_title("Top Factors for This Applicant")
        plt.tight_layout()
        st.pyplot(fig)


# ═════════════════════════════════════════
# PAGE 5 — PORTFOLIO DASHBOARD
# ═════════════════════════════════════════
elif page == "Portfolio Dashboard":

    st.title("Portfolio Risk Dashboard")
    st.markdown("Overview of risk across all applicants in the test set.")
    st.markdown("---")

    pred_proba = model.predict_proba(X_test)[:, 1]

    # Risk buckets
    risk_buckets = pd.cut(
        pred_proba,
        bins   = [0, 0.3, 0.6, 1.0],
        labels = ["Low Risk", "Medium Risk", "High Risk"]
    ).value_counts()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            values = risk_buckets.values,
            names  = risk_buckets.index,
            title  = "Portfolio Risk Distribution",
            color  = risk_buckets.index,
            color_discrete_map = {
                "Low Risk"   : "steelblue",
                "Medium Risk": "orange",
                "High Risk"  : "tomato"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            x     = pred_proba,
            nbins = 50,
            title = "Distribution of Risk Scores",
            labels= {"x": "Default Probability", "y": "Count"},
            color_discrete_sequence=["steelblue"]
        )
        fig.add_vline(x=0.3, line_dash="dash",
                      line_color="orange",
                      annotation_text="Medium Risk Threshold")
        fig.add_vline(x=0.6, line_dash="dash",
                      line_color="red",
                      annotation_text="High Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Risk summary table
    st.subheader("Risk Summary")
    summary = pd.DataFrame({
        "Risk Category" : ["Low Risk",   "Medium Risk", "High Risk"],
        "Threshold"     : ["0 - 30%",    "30 - 60%",    "60 - 100%"],
        "Applicants"    : [
            (pred_proba < 0.3).sum(),
            ((pred_proba >= 0.3) & (pred_proba < 0.6)).sum(),
            (pred_proba >= 0.6).sum()
        ],
        "Recommended Action": [
            "Approve",
            "Manual Review",
            "Reject / Flag"
        ]
    })
    summary["% of Portfolio"] = (
        summary["Applicants"] / len(pred_proba) * 100
    ).round(2).astype(str) + "%"

    st.dataframe(summary, use_container_width=True)