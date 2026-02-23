import sys
from pathlib import Path
import json

# Allow: python src/model_analysis.py from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.paths import (
    ensure_dirs,
    XGB_PIPELINE_JOBLIB,
    VEHICLES_CLEAN_CSV,
    REGIME_DEPRECIATION_PNG,
    BRAND_PREMIUM_PNG,
    XGB_SHAP_SUMMARY_FIXED_PNG,
    SHAP_AGE_DEPENDENCE_FIXED_PNG,
    MARKET_INSIGHTS_JSON,
)


def load_artifacts():
    if not XGB_PIPELINE_JOBLIB.exists():
        raise FileNotFoundError(
            f"{XGB_PIPELINE_JOBLIB} not found. Run src/model.py first."
        )

    pipeline = joblib.load(XGB_PIPELINE_JOBLIB)

    # Expected step names from src/model.py
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    df = pd.read_csv(VEHICLES_CLEAN_CSV)

    feature_names = preprocessor.get_feature_names_out()

    print("Loaded artifacts:")
    print("  Records:", len(df))
    print("  Features:", len(feature_names))

    return df, model, preprocessor, feature_names


def analyze_ban_regimes(df: pd.DataFrame):
    df = df.copy()

    df["Market_Regime"] = pd.cut(
        df["Vehicle_Age"],
        bins=[0, 2, 5, 40],
        labels=["Post-Ban (0-2yr)", "Ban-Period (2-5yr)", "Pre-Ban (>5yr)"],
        include_lowest=True,
    )

    regime_stats = (
        df.groupby("Market_Regime", observed=False)["Price_LKR"]
        .agg(["median", "mean", "std", "count"])
        .round(0)
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
        linewidth=3,
        label="Median market price",
    )
    ax1.set_xlabel("Vehicle Age (years)")
    ax1.set_ylabel("Median Price (LKR Millions)")
    ax1.set_title("Depreciation curve with ban-period anomaly")
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(2, 5, alpha=0.2, color="red", label="Import ban (2–5yr)")
    ax1.legend()

    sns.boxplot(data=df, x="Market_Regime", y="Price_LKR", ax=ax2)
    ax2.set_yscale("log")
    ax2.set_ylabel("Price (LKR, log scale)")
    ax2.set_xlabel("Market Regime")
    ax2.set_title("Price distribution by regime")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(REGIME_DEPRECIATION_PNG, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {REGIME_DEPRECIATION_PNG}")

    # Key stats (medians)
    pre_ban_median = float(df[df["Vehicle_Age"] > 5]["Price_LKR"].median())
    ban_period_median = float(df[df["Vehicle_Age"].between(2, 5)]["Price_LKR"].median())
    post_ban_median = float(df[df["Vehicle_Age"] <= 2]["Price_LKR"].median())

    ban_period_premium_pct = ((ban_period_median / pre_ban_median - 1) * 100) if pre_ban_median else np.nan
    post_ban_change_pct = ((post_ban_median / ban_period_median - 1) * 100) if ban_period_median else np.nan

    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print(f"Pre-Ban Baseline (>5yr): LKR {pre_ban_median:,.0f}")
    print(f"Ban-Period (2-5yr): LKR {ban_period_median:,.0f}")
    print(f"Post-Ban (0-2yr): LKR {post_ban_median:,.0f}")
    print(f"\nBan Period Premium: {ban_period_premium_pct:.1f}% above pre-ban median")
    print(f"Post-Ban Change: {post_ban_change_pct:.1f}% vs ban-period median")
    print("=" * 60)

    insights = {
        "pre_ban_median": pre_ban_median,
        "ban_period_median": ban_period_median,
        "post_ban_median": post_ban_median,
        "ban_period_premium_pct": float(ban_period_premium_pct) if np.isfinite(ban_period_premium_pct) else None,
        "post_ban_change_pct": float(post_ban_change_pct) if np.isfinite(post_ban_change_pct) else None,
    }

    return regime_stats, insights


def quantify_brand_premium(df, model, feature_names, preprocessor):
    print("\n" + "=" * 60)
    print("BRAND PREMIUM ANALYSIS")
    print("=" * 60)

    sample_data = df.sample(n=min(500, len(df)), random_state=42)
    X_sample = sample_data.drop("Price_LKR", axis=1)
    X_processed = preprocessor.transform(X_sample)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Prefer SHAP when available; fallback to model.feature_importances_ otherwise.
    shap_values = None
    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
    except Exception as e:
        print(f"SHAP not available ({e}); falling back to model.feature_importances_.")

    brand_indices = [i for i, f in enumerate(feature_names) if "cat__Brand_" in f]

    brand_effects = []
    if shap_values is not None:
        # SHAP-based mean absolute contribution
        for i in brand_indices:
            fname = feature_names[i]
            brand = fname.split("cat__Brand_", 1)[-1]
            mean_abs = float(np.mean(np.abs(shap_values[:, i])))
            brand_effects.append({"Brand": brand, "SHAP_Importance": mean_abs})
    else:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise RuntimeError("Neither SHAP nor model.feature_importances_ is available for brand analysis.")
        for i in brand_indices:
            fname = feature_names[i]
            brand = fname.split("cat__Brand_", 1)[-1]
            brand_effects.append({"Brand": brand, "SHAP_Importance": float(importances[i])})

    brand_df = pd.DataFrame(brand_effects).sort_values("SHAP_Importance", ascending=False)

    # Plot top brands
    top = brand_df.head(12).copy()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, y="Brand", x="SHAP_Importance")
    plt.xlabel("Mean |SHAP| (LKR)" if shap_values is not None else "Model feature importance")
    plt.ylabel("")
    plt.title("Brand impact" + (" (SHAP)" if shap_values is not None else " (feature importance)"))
    plt.tight_layout()
    plt.savefig(BRAND_PREMIUM_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {BRAND_PREMIUM_PNG}")

    def get_brand_effect(b):
        row = brand_df[brand_df["Brand"].str.lower() == b.lower()]
        return float(row["SHAP_Importance"].iloc[0]) if len(row) else None

    insights = {
        "top_brand_effects": brand_df.head(10).to_dict(orient="records"),
        "toyota_effect_mean_abs_shap": get_brand_effect("Toyota"),
        "honda_effect_mean_abs_shap": get_brand_effect("Honda"),
    }

    print("\nTop 10 Brands by impact:")
    print(brand_df.head(10))

    if insights["toyota_effect_mean_abs_shap"] is not None:
        print(f"Toyota Effect: {insights['toyota_effect_mean_abs_shap']:,.0f}")
    if insights["honda_effect_mean_abs_shap"] is not None:
        print(f"Honda Effect: {insights['honda_effect_mean_abs_shap']:,.0f}")

    return brand_df, insights

def regenerate_shap_plots(df, model, preprocessor, feature_names):
    try:
        import shap  # type: ignore
    except Exception as e:
        print(f"SHAP not available in this environment ({e}); skipping SHAP plots.")
        return
    print("\nRegenerating SHAP plots with proper feature names...")

    X = df.drop("Price_LKR", axis=1)
    X_proc = preprocessor.transform(X)

    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    np.random.seed(42)
    idx = np.random.choice(X_proc.shape[0], size=min(400, X_proc.shape[0]), replace=False)
    X_sample = X_proc[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.title("XGBoost SHAP Summary (Sample)")
    plt.savefig(XGB_SHAP_SUMMARY_FIXED_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {XGB_SHAP_SUMMARY_FIXED_PNG}")

    # Dependence for Vehicle_Age if available
    va_idx = None
    for i, n in enumerate(feature_names):
        if "Vehicle_Age" in n:
            va_idx = i
            break

    if va_idx is not None:
        plt.figure()
        shap.dependence_plot(va_idx, shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title("SHAP Dependence: Vehicle_Age")
        plt.savefig(SHAP_AGE_DEPENDENCE_FIXED_PNG, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {SHAP_AGE_DEPENDENCE_FIXED_PNG}")


def main():
    ensure_dirs()

    df, model, preprocessor, feature_names = load_artifacts()

    regime_stats, regime_insights = analyze_ban_regimes(df)
    _, brand_insights = quantify_brand_premium(df, model, feature_names, preprocessor)
    regenerate_shap_plots(df, model, preprocessor, feature_names)

    combined = {**regime_insights, **brand_insights}
    with open(MARKET_INSIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved: {MARKET_INSIGHTS_JSON}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - Check generated files:")
    print(f"1. {REGIME_DEPRECIATION_PNG}")
    print(f"2. {BRAND_PREMIUM_PNG}")
    print(f"3. {XGB_SHAP_SUMMARY_FIXED_PNG}")
    print(f"4. {SHAP_AGE_DEPENDENCE_FIXED_PNG}")
    print(f"5. {MARKET_INSIGHTS_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
