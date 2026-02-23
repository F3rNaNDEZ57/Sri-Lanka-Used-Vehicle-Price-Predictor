from __future__ import annotations
from pathlib import Path

# Project root is one level above /src
ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data files
VEHICLES_RAW_CSV = RAW_DIR / "vehicles_raw.csv"
VEHICLES_CLEAN_CSV = PROCESSED_DIR / "vehicles_clean.csv"

X_TRAIN_CSV = SPLITS_DIR / "X_train.csv"
X_VAL_CSV   = SPLITS_DIR / "X_val.csv"
X_TEST_CSV  = SPLITS_DIR / "X_test.csv"
Y_TRAIN_CSV = SPLITS_DIR / "y_train.csv"
Y_VAL_CSV   = SPLITS_DIR / "y_val.csv"
Y_TEST_CSV  = SPLITS_DIR / "y_test.csv"

# Model artifacts
PREPROCESSOR_TEMPLATE_JOBLIB = MODELS_DIR / "preprocessor_template.joblib"
PREPROCESSOR_FITTED_JOBLIB   = MODELS_DIR / "preprocessor_fitted.joblib"

RIDGE_MODEL_JOBLIB = MODELS_DIR / "ridge_model.joblib"
XGB_MODEL_JOBLIB   = MODELS_DIR / "xgb_model.joblib"

RIDGE_PIPELINE_JOBLIB = MODELS_DIR / "ridge_pipeline.joblib"
XGB_PIPELINE_JOBLIB   = MODELS_DIR / "xgb_pipeline.joblib"

# Reports / figures
XGB_PRED_VS_ACTUAL_PNG = FIGURES_DIR / "xgb_pred_vs_actual.png"
XGB_SHAP_SUMMARY_PNG   = FIGURES_DIR / "xgb_shap_summary.png"
XGB_SHAP_DEPENDENCE_PNG = FIGURES_DIR / "xgb_shap_dependence_vehicle_age.png"

REGIME_DEPRECIATION_PNG = FIGURES_DIR / "regime_depreciation_analysis.png"
BRAND_PREMIUM_PNG       = FIGURES_DIR / "brand_premium_analysis.png"
XGB_SHAP_SUMMARY_FIXED_PNG = FIGURES_DIR / "xgb_shap_summary_FIXED.png"
SHAP_AGE_DEPENDENCE_FIXED_PNG = FIGURES_DIR / "shap_age_dependence_FIXED.png"

MARKET_INSIGHTS_JSON = REPORTS_DIR / "market_insights.json"
XGB_TUNING_RESULTS_CSV = REPORTS_DIR / "xgb_tuning_results.csv"
XGB_BEST_PARAMS_JSON   = REPORTS_DIR / "xgb_best_params.json"


def ensure_dirs() -> None:
    """Create all expected output directories."""
    for d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
