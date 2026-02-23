import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

# Artifacts
PIPELINE_FILE = "xgb_pipeline.joblib"
CLEAN_DATA_FILE = "vehicles_clean.csv"

OUT_REGIME_PNG = "regime_depreciation_analysis.png"
OUT_BRAND_PNG = "brand_premium_analysis.png"
OUT_SHAP_SUMMARY_PNG = "xgb_shap_summary_FIXED.png"
OUT_SHAP_AGE_PNG = "shap_age_dependence_FIXED.png"
OUT_INSIGHTS_JSON = "market_insights.json"


def load_artifacts():
    if not Path(PIPELINE_FILE).exists():
        raise FileNotFoundError(
            f"{PIPELINE_FILE} not found. Run model.py first to generate the trained pipeline."
        )

    pipeline = joblib.load(PIPELINE_FILE)
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    df = pd.read_csv(CLEAN_DATA_FILE)

    # Feature names are available because the preprocessor is already fitted (Fix #1)
    feature_names = preprocessor.get_feature_names_out()

    print("Loaded artifacts:")
    print("  Records:", len(df))
    print("  Features:", len(feature_names))

    return df, model, preprocessor, feature_names


def analyze_ban_regimes(df: pd.DataFrame):
    """Quantify price anomalies by import-ban age regimes."""
    df = df.copy()

    df["Market_Regime"] = pd.cut(
        df["Vehicle_Age"],
        bins=[0, 2, 5, 40],
        labels=["Post-Ban (0-2yr)", "Ban-Period (2-5yr)", "Pre-Ban (>5yr)"],
        include_lowest=True,
    )

    regime_stats = (
        df.groupby("Market_Regime")["Price_LKR"].agg(["median", "mean", "std", "count"]).round(0)
    )

    print("\n" + "=" * 60)
    print("REGIME-DEPENDENT PRICE ANALYSIS")
    print("=" * 60)
    print(regime_stats)

    # Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    age_price = df.groupby("Vehicle_Age")["Price_LKR"].median().reset_index()
    ax1.plot(
        age_price["Vehicle_Age"],
        age_price["Price_LKR"] / 1e6,
        "b-",
        linewidth=3,
        label="Actual Market Prices",
    )
    ax1.set_xlabel("Vehicle Age (years)", fontsize=12)
    ax1.set_ylabel("Median Price (LKR Millions)", fontsize=12)
    ax1.set_title(
        "Sri Lanka Vehicle Depreciation Curve\n(Showing Ban Period Anomaly)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(2, 5, alpha=0.2, color="red", label="Import Ban Period (2020-2025)")
    ax1.legend()

    sns.boxplot(data=df, x="Market_Regime", y="Price_LKR", ax=ax2)
    ax2.set_yscale("log")
    ax2.set_ylabel("Price (LKR, log scale)", fontsize=12)
    ax2.set_xlabel("Market Regime", fontsize=12)
    ax2.set_title(
        "Price Distribution by Market Regime\n(Note: ban-period vehicles hold value)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUT_REGIME_PNG, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {OUT_REGIME_PNG}")

    # Key stats
    normal_depreciation = df[df["Vehicle_Age"] > 5]["Price_LKR"].median()
    ban_period_price = df[df["Vehicle_Age"].between(2, 5)]["Price_LKR"].median()
    post_ban_price = df[df["Vehicle_Age"] <= 2]["Price_LKR"].median()

    ban_premium_pct = ((ban_period_price / normal_depreciation - 1) * 100) if normal_depreciation else np.nan
    post_ban_change_pct = ((post_ban_price / ban_period_price - 1) * 100) if ban_period_price else np.nan

    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print(f"Pre-Ban Baseline (>5yr): LKR {normal_depreciation:,.0f}")
    print(f"Ban-Period (2-5yr): LKR {ban_period_price:,.0f}")
    print(f"Post-Ban (0-2yr): LKR {post_ban_price:,.0f}")
    print(f"\nBan Period Premium: {ban_premium_pct:.1f}% above pre-ban median")
    print(f"Post-Ban Change: {post_ban_change_pct:.1f}% vs ban-period median")
    print("=" * 60)

    insights = {
        "pre_ban_median": float(normal_depreciation) if np.isfinite(normal_depreciation) else None,
        "ban_period_median": float(ban_period_price) if np.isfinite(ban_period_price) else None,
        "post_ban_median": float(post_ban_price) if np.isfinite(post_ban_price) else None,
        "ban_premium_pct": float(ban_premium_pct) if np.isfinite(ban_premium_pct) else None,
        "post_ban_change_pct": float(post_ban_change_pct) if np.isfinite(post_ban_change_pct) else None,
    }

    return regime_stats, insights


def quantify_brand_premium(df, model, feature_names, preprocessor):
    """Quantify brand effects using SHAP (mean |SHAP| per brand feature)."""
    print("\n" + "=" * 60)
    print("BRAND PREMIUM ANALYSIS")
    print("=" * 60)

    sample_data = df.sample(n=min(500, len(df)), random_state=42)
    X_sample = sample_data.drop("Price_LKR", axis=1)
    X_processed = preprocessor.transform(X_sample)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    brand_indices = [i for i, name in enumerate(feature_names) if "Brand_" in name]
    brand_names = [feature_names[i].replace("cat__Brand_", "") for i in brand_indices]

    brand_importance = np.abs(shap_values[:, brand_indices]).mean(axis=0)

    brand_df = (
        pd.DataFrame(
            {
                "Brand": brand_names,
                "SHAP_Importance": brand_importance,
                "Model_Feature_Importance": model.feature_importances_[brand_indices],
            }
        )
        .sort_values("SHAP_Importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 10 Brands by Price Impact (SHAP values):")
    print(brand_df.head(10))

    # Pull out Toyota/Honda (if present)
    def _brand_effect(name: str):
        row = brand_df[brand_df["Brand"].str.lower() == name.lower()]
        if len(row) == 0:
            return None
        return float(row.iloc[0]["SHAP_Importance"])

    toyota_effect = _brand_effect("Toyota")
    honda_effect = _brand_effect("Honda")

    if toyota_effect is not None:
        print(f"\nToyota Effect (mean |SHAP|): {toyota_effect:,.0f} LKR")
    if honda_effect is not None:
        print(f"Honda Effect (mean |SHAP|): {honda_effect:,.0f} LKR")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    top_brands = brand_df.head(8)
    colors = [
        "#d62728" if b in ["Toyota", "Honda"] else "#1f77b4" for b in top_brands["Brand"]
    ]

    ax.barh(top_brands["Brand"], top_brands["SHAP_Importance"], color=colors)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Price)", fontsize=12)
    ax.set_title(
        "Brand Premium Analysis: Toyota/Honda Dominance\nin Sri Lankan Market",
        fontsize=14,
        fontweight="bold",
    )

    for i, row in top_brands.reset_index(drop=True).iterrows():
        ax.text(row["SHAP_Importance"], i, f" {row['SHAP_Importance']:,.0f}", va="center", fontsize=10)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#d62728", label="Premium Brands (Toyota/Honda)"),
        Patch(facecolor="#1f77b4", label="Other Brands"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_BRAND_PNG, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {OUT_BRAND_PNG}")

    insights = {
        "toyota_mean_abs_shap": toyota_effect,
        "honda_mean_abs_shap": honda_effect,
    }

    return brand_df, insights


def fix_shap_summary_plot(model, preprocessor, feature_names):
    """Regenerate SHAP summary + Vehicle_Age dependence plot with correct labels."""
    print("\nRegenerating SHAP plots with proper feature names...")

    X_test = pd.read_csv("X_test.csv")
    X_processed = preprocessor.transform(X_test)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    np.random.seed(42)
    idx = np.random.choice(X_processed.shape[0], min(200, X_processed.shape[0]), replace=False)
    X_sample = X_processed[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title("XGBoost Feature Importance (Corrected Labels)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_SHAP_SUMMARY_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_SHAP_SUMMARY_PNG}")

    age_idx = None
    for i, name in enumerate(feature_names):
        if "Vehicle_Age" in name:
            age_idx = i
            break

    if age_idx is not None:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            age_idx,
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
        )
        plt.title(
            "SHAP Dependence: Vehicle Age Effect\n(Showing Import Ban Plateau)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(OUT_SHAP_AGE_PNG, dpi=300, bbox_inches="tight")
        print(f"Saved: {OUT_SHAP_AGE_PNG}")


def main():
    df, model, preprocessor, feature_names = load_artifacts()

    # 1) Regime analysis
    _, regime_insights = analyze_ban_regimes(df)

    # 2) Brand premium analysis
    _, brand_insights = quantify_brand_premium(df, model, feature_names, preprocessor)

    # 3) SHAP plots
    fix_shap_summary_plot(model, preprocessor, feature_names)

    # 4) Save insights for the Streamlit app (Fix #4: remove hard-coded claims)
    insights = {
        "ban_regime": regime_insights,
        "brand": brand_insights,
    }
    with open(OUT_INSIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)
    print(f"\nSaved: {OUT_INSIGHTS_JSON}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - Check generated files:")
    print(f"1. {OUT_REGIME_PNG}")
    print(f"2. {OUT_BRAND_PNG}")
    print(f"3. {OUT_SHAP_SUMMARY_PNG}")
    print(f"4. {OUT_SHAP_AGE_PNG}")
    print(f"5. {OUT_INSIGHTS_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
