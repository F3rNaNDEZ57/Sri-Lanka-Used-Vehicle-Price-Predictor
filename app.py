import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# -----------------------------
# SHAP AGGREGATION (field-level)
# -----------------------------
CATEGORICAL_FIELDS = [
    "Brand",
    "Model_Name",
    "Location_District",
    "Condition",
    "Fuel_Type",
    "Transmission",
    "Body_Type",
    "Colour",
]

_FIELD_LABELS = {
    "Brand": "Brand",
    "Model_Name": "Model",
    "Location_District": "Location (district)",
    "Condition": "Condition",
    "Fuel_Type": "Fuel type",
    "Transmission": "Transmission",
    "Body_Type": "Body type",
    "Colour": "Colour",
}


def _pretty_num(name: str) -> str:
    mapping = {
        "Model_Year": "Model year",
        "Mileage_km": "Mileage (km)",
        "Engine_Capacity": "Engine capacity (cc)",
        "Vehicle_Age": "Vehicle age (years)",
    }
    return mapping.get(name, name.replace("_", " ").strip())


def aggregate_shap_to_fields(feature_names, shap_values):
    """Aggregate SHAP values from one-hot columns into original input fields."""
    rows = []
    for f, v in zip(feature_names, shap_values):
        short = f.split("__", 1)[-1] if "__" in f else f  # e.g. Brand_Honda, Mileage_km

        if f.startswith("cat__"):
            field = None
            for base in CATEGORICAL_FIELDS:
                if short.startswith(base + "_"):
                    field = _FIELD_LABELS.get(base, base.replace("_", " "))
                    break
            if field is None:
                field = "Categorical (other)"
            field = f"{field} (overall)"
        elif f.startswith("num__"):
            field = _pretty_num(short)
        else:
            field = _pretty_num(short)

        rows.append((field, float(v)))

    df = pd.DataFrame(rows, columns=["Field", "Value"])
    df = df.groupby("Field", as_index=False)["Value"].sum()
    df["Abs_Value"] = df["Value"].abs()
    return df


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sri Lanka Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 0rem; }
    .sub-header { font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .prediction-box { background-color: #f0f2f6; padding: 2rem; border-radius: 10px; border-left: 5px solid #1f77b4; }
    .ban-info { background-color: #fff3cd; padding: 1rem; border-radius: 5px; border-left: 5px solid #ffc107; margin-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_unique_strings(series: pd.Series):
    return sorted([str(x) for x in series.dropna().unique()])


# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
PIPELINE_FILE = "xgb_pipeline.joblib"
MODEL_FILE = "xgb_model.joblib"
PREPROCESSOR_FILE = "preprocessor_fitted.joblib"
CLEAN_DATA_FILE = "vehicles_clean.csv"
MARKET_INSIGHTS_FILE = "market_insights.json"


def _extract_pre_and_model_from_pipeline(pipe):
    """Try to locate preprocessor + model steps even if step names differ."""
    if not hasattr(pipe, "named_steps"):
        return None, None

    steps = pipe.named_steps

    pre = None
    for k, v in steps.items():
        if "preprocess" in k or "preprocessor" in k or k in ("prep", "ct"):
            pre = v
            break
    if pre is None:
        pre = list(steps.values())[0]

    mdl = None
    for k, v in steps.items():
        if k in ("model", "regressor", "estimator", "xgb"):
            mdl = v
            break
    if mdl is None:
        mdl = list(steps.values())[-1]

    return pre, mdl


@st.cache_resource
def load_artifacts():
    pipe = None
    preprocessor = None
    model = None

    if Path(PIPELINE_FILE).exists():
        pipe = joblib.load(PIPELINE_FILE)
        preprocessor, model = _extract_pre_and_model_from_pipeline(pipe)

    # Fallback if pipeline not available / step extraction failed
    if model is None and Path(MODEL_FILE).exists():
        model = joblib.load(MODEL_FILE)

    if preprocessor is None and Path(PREPROCESSOR_FILE).exists():
        preprocessor = joblib.load(PREPROCESSOR_FILE)

    if model is None or preprocessor is None:
        missing = []
        if model is None:
            missing.append(MODEL_FILE + " (or " + PIPELINE_FILE + ")")
        if preprocessor is None:
            missing.append(PREPROCESSOR_FILE + " (or " + PIPELINE_FILE + ")")
        raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing))

    return pipe, model, preprocessor


@st.cache_data
def load_clean_data():
    if not Path(CLEAN_DATA_FILE).exists():
        raise FileNotFoundError(f"Missing {CLEAN_DATA_FILE}. Run preprocess.py first.")
    return pd.read_csv(CLEAN_DATA_FILE)


@st.cache_data
def load_market_insights():
    if Path(MARKET_INSIGHTS_FILE).exists():
        try:
            with open(MARKET_INSIGHTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_resource
def get_shap_explainer(_model):
    # underscore prevents Streamlit hashing the model object
    return shap.TreeExplainer(_model)


try:
    pipeline, xgb_model, preprocessor = load_artifacts()
    df_clean = load_clean_data()
    market_insights = load_market_insights()
except Exception as e:
    st.error(
        "Error loading models or data.\n\n"
        "Make sure you ran:\n"
        "- python preprocess.py\n"
        "- python model.py\n\n"
        f"Details: {e}"
    )
    st.stop()


# -----------------------------
# MAIN UI
# -----------------------------
st.markdown('<div class="main-header">Sri Lanka Used Vehicle Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Valuation for the Post-Ban Market (2026)</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 Price Predictor", "📊 Macro Market Insights"])


# ==========================================
# TAB 1: PRICE PREDICTOR
# ==========================================
with tab1:
    st.sidebar.header("Vehicle Specifications")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        brand = st.selectbox("Brand", get_unique_strings(df_clean["Brand"]))
        model_year = st.number_input("Model Year", min_value=1980, max_value=2026, value=2018, step=1)
        engine_cap = st.number_input("Engine Capacity (cc)", min_value=500, max_value=8000, value=1500, step=100)
        fuel_type = st.selectbox("Fuel Type", get_unique_strings(df_clean["Fuel_Type"]))
        condition = st.selectbox("Condition", get_unique_strings(df_clean["Condition"]))

    with col2:
        brand_models = df_clean[df_clean["Brand"] == brand]["Model_Name"]
        available_models = get_unique_strings(brand_models)
        if not available_models:
            available_models = ["Unknown"]
        model_name = st.selectbox("Model", available_models)

        mileage = st.number_input("Mileage (km)", min_value=0, max_value=700000, value=60000, step=5000)
        transmission = st.selectbox("Transmission", get_unique_strings(df_clean["Transmission"]))
        body_type = st.selectbox("Body Type", get_unique_strings(df_clean["Body_Type"]))
        colour = st.selectbox("Colour", get_unique_strings(df_clean["Colour"]))

    location = st.sidebar.selectbox("Location (District)", get_unique_strings(df_clean["Location_District"]))

    if st.sidebar.button("Predict Market Price", type="primary", width="stretch"):
        # Input row
        vehicle_age = 2026 - model_year
        input_dict = {
            "Brand": brand,
            "Model_Name": model_name,
            "Model_Year": model_year,
            "Vehicle_Age": vehicle_age,
            "Mileage_km": mileage,
            "Engine_Capacity": engine_cap,
            "Location_District": location,
            "Condition": condition,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Body_Type": body_type,
            "Colour": colour,
        }
        input_df = pd.DataFrame([input_dict])

        try:
            # Predict (pipeline preferred)
            if pipeline is not None:
                pred_price = float(pipeline.predict(input_df)[0])
                X_proc = preprocessor.transform(input_df)
            else:
                X_proc = preprocessor.transform(input_df)
                pred_price = float(xgb_model.predict(X_proc)[0])

            # Similar cars for range
            similar_cars = df_clean[
                (df_clean["Brand"] == brand) &
                (abs(df_clean["Vehicle_Age"] - vehicle_age) <= 2)
            ]
            std_dev = similar_cars["Price_LKR"].std() if len(similar_cars) > 2 else pred_price * 0.15
            if pd.isna(std_dev) or std_dev <= 0:
                std_dev = pred_price * 0.15

            lower_bound = max(500_000, pred_price - std_dev)
            upper_bound = pred_price + std_dev

            st.markdown(
                f"""
                <div class="prediction-box">
                    <h3 style="margin-top:0;color:#1f77b4;">Estimated Market Value</h3>
                    <h1 style="color:#1f77b4; font-size:3rem; margin:0;">LKR {pred_price:,.0f}</h1>
                    <p style="color:#666; margin-bottom:0;">Expected range: LKR {lower_bound:,.0f} - LKR {upper_bound:,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Ban-period context (use computed insight if available)
            if 2 <= vehicle_age <= 5:
                premium = market_insights.get("ban_period_premium_pct", 43.2)
                st.markdown(
                    f"""
                    <div class="ban-info">
                        <strong>⚠️ Market Context (Import Ban Vehicle):</strong>
                        Vehicles aged 2–5 years often show a “ban-period” pricing effect due to supply constraints.
                        In this dataset, the estimated premium is about <strong>{premium:.1f}%</strong> vs older pre-ban vehicles (median-based).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()

            # -----------------------------
            # SHAP Explainability (Grouped)
            # -----------------------------
            st.subheader("Why is it priced this way?")
            st.write(
                "This chart shows how much (in LKR) each *original input field* increased or decreased the prediction "
                "relative to the model baseline. Categorical inputs are grouped (e.g., Brand overall, Location overall)."
            )

            # Ensure dense matrix for SHAP
            X_proc_dense = X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc

            explainer = get_shap_explainer(xgb_model)

            sv = explainer.shap_values(X_proc_dense)
            # handle shap return shape
            if isinstance(sv, list):
                sv = sv[0]
            if getattr(sv, "ndim", 1) == 2:
                shap_values = sv[0]
            else:
                shap_values = sv

            feature_names = preprocessor.get_feature_names_out()

            # raw debug table (optional)
            x_row = X_proc_dense[0]
            debug_df = pd.DataFrame(
                {"feature": feature_names, "value": x_row, "shap": shap_values}
            )
            debug_df["abs_shap"] = debug_df["shap"].abs()

            # Grouped SHAP
            shap_field_df = aggregate_shap_to_fields(feature_names, shap_values)
            shap_field_df = (
                shap_field_df.sort_values("Abs_Value", ascending=False)
                .head(10)
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#2ca02c" if x > 0 else "#d62728" for x in shap_field_df["Value"]]
            ax.barh(shap_field_df["Field"][::-1], shap_field_df["Value"][::-1], color=colors[::-1])
            ax.set_xlabel("Price Impact (LKR)")
            ax.set_ylabel("")
            ax.set_title("Top drivers (grouped by original input fields)")
            ax.axvline(x=0, color="black", linewidth=1)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("Details (optional): raw one-hot contributions behind each field"):
                st.caption(
                    "The model uses one-hot columns internally (e.g., Brand_BMW, Brand_Honda). "
                    "This table shows the top encoded columns by |SHAP| and their 0/1 values."
                )
                top_raw = debug_df.sort_values("abs_shap", ascending=False).head(30)[["feature", "value", "shap"]]
                st.dataframe(top_raw, width="stretch")

            # -----------------------------
            # Similar Market Listings
            # -----------------------------
            st.subheader("🚘 Similar Market Listings (Historical)")
            if len(similar_cars) > 0:
                display_cols = ["Brand", "Model_Name", "Model_Year", "Mileage_km", "Price_LKR"]
                st.dataframe(
                    similar_cars[display_cols].head(5).style.format({
                        "Price_LKR": "LKR {:,.0f}",
                        "Mileage_km": "{:,.0f} km",
                    }),
                    width="stretch",
                )
            else:
                st.info("No highly similar historical listings found in the database.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ==========================================
# TAB 2: MACRO MARKET INSIGHTS
# ==========================================
with tab2:
    st.header("Sri Lankan Secondary Market Dynamics")
    st.write("These insights are generated from `model_analysis.py`.")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("1. The 'Ban Premium' & Depreciation Plateau")
        try:
            st.image("regime_depreciation_analysis.png", width="stretch")
        except Exception:
            st.warning("Run `model_analysis.py` first to generate `regime_depreciation_analysis.png`.")

    with colB:
        st.subheader("2. Brand Premium Analysis")
        try:
            st.image("brand_premium_analysis.png", width="stretch")
        except Exception:
            st.warning("Run `model_analysis.py` first to generate `brand_premium_analysis.png`.")

    st.divider()
    st.subheader("Global Feature Importance (SHAP)")
    try:
        st.image("xgb_shap_summary_FIXED.png", width="stretch")
    except Exception:
        st.warning("Run `model_analysis.py` first to generate `xgb_shap_summary_FIXED.png`.")