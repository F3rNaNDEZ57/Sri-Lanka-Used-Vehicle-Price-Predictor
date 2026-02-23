import sys
from pathlib import Path
import json

# Allow: python src/model.py from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

from src.paths import (
    ensure_dirs,
    X_TRAIN_CSV, X_VAL_CSV, X_TEST_CSV,
    Y_TRAIN_CSV, Y_VAL_CSV, Y_TEST_CSV,
    PREPROCESSOR_TEMPLATE_JOBLIB,
    PREPROCESSOR_FITTED_JOBLIB,
    RIDGE_MODEL_JOBLIB, XGB_MODEL_JOBLIB,
    RIDGE_PIPELINE_JOBLIB, XGB_PIPELINE_JOBLIB,
    XGB_PRED_VS_ACTUAL_PNG,
    XGB_SHAP_SUMMARY_PNG,
    XGB_SHAP_DEPENDENCE_PNG,
    XGB_TUNING_RESULTS_CSV,
    XGB_BEST_PARAMS_JSON,
)

RANDOM_STATE = 42

# Tuning controls (kept small so it runs quickly)
ENABLE_TUNING = True
TUNING_ITERATIONS = 10
EARLY_STOPPING_ROUNDS = 75


def load_data():
    X_train = pd.read_csv(X_TRAIN_CSV)
    X_val = pd.read_csv(X_VAL_CSV)
    X_test = pd.read_csv(X_TEST_CSV)

    y_train = pd.read_csv(Y_TRAIN_CSV).iloc[:, 0].values
    y_val = pd.read_csv(Y_VAL_CSV).iloc[:, 0].values
    y_test = pd.read_csv(Y_TEST_CSV).iloc[:, 0].values

    print("Loaded data shapes:")
    print("  X_train:", X_train.shape, " y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, " y_val:  ", y_val.shape)
    print("  X_test: ", X_test.shape, " y_test:", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_preprocessor_template():
    if not PREPROCESSOR_TEMPLATE_JOBLIB.exists():
        raise FileNotFoundError(
            f"{PREPROCESSOR_TEMPLATE_JOBLIB} not found. Run src/preprocess.py first."
        )
    preprocessor = joblib.load(PREPROCESSOR_TEMPLATE_JOBLIB)
    print("Loaded preprocessor template from", PREPROCESSOR_TEMPLATE_JOBLIB)
    return preprocessor


def calc_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def print_metrics(name, metrics, y_true):
    mean_price = float(np.mean(y_true))
    print("\n" + "=" * 50)
    print(f"Metrics for {name}")
    print("=" * 50)
    print(
        f"RMSE : LKR {metrics['RMSE']:,.0f} ({metrics['RMSE'] / mean_price * 100:.1f}% of mean price)"
    )
    print(f"MAE  : LKR {metrics['MAE']:,.0f}")
    print(f"R²   : {metrics['R2']:.4f}")
    print(f"MAPE : {metrics['MAPE']:.2f}%")
    print("=" * 50)


def train_ridge(X_train_proc, y_train):
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE, max_iter=10000)
    ridge.fit(X_train_proc, y_train)
    return ridge


def _fit_xgb_with_early_stopping(model, X_train_proc, y_train, X_val_proc, y_val):
    """Fit with early stopping when xgboost supports it; otherwise fallback."""
    try:
        model.fit(
            X_train_proc,
            y_train,
            eval_set=[(X_val_proc, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        return model
    except TypeError:
        model.fit(X_train_proc, y_train)
        return model


def tune_xgb(X_train_proc, y_train, X_val_proc, y_val, n_iter=TUNING_ITERATIONS):
    base_params = {
        "n_estimators": 5000,
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    search_space = {
        "max_depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.85, 0.9, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [1.0, 2.0, 5.0],
    }

    sampler = ParameterSampler(search_space, n_iter=n_iter, random_state=RANDOM_STATE)

    rows = []
    best_rmse = float("inf")
    best_model = None
    best_params = None

    print("\nRunning quick hyperparameter tuning...")

    for i, params in enumerate(sampler, start=1):
        full_params = {**base_params, **params}
        model = xgb.XGBRegressor(**full_params)
        model = _fit_xgb_with_early_stopping(model, X_train_proc, y_train, X_val_proc, y_val)

        val_pred = model.predict(X_val_proc)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        best_iter = getattr(model, "best_iteration", None)
        rows.append({**params, "val_rmse": rmse, "best_iteration": best_iter})

        print(f"  [{i:02d}/{n_iter}] val RMSE = {rmse:,.0f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = full_params

    results = pd.DataFrame(rows).sort_values("val_rmse")
    results.to_csv(XGB_TUNING_RESULTS_CSV, index=False)
    print(f"Saved tuning results → {XGB_TUNING_RESULTS_CSV}")

    with open(XGB_BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params → {XGB_BEST_PARAMS_JSON}")

    print(f"Best validation RMSE from tuning: {best_rmse:,.0f}")
    return best_model


def train_xgb(X_train_proc, y_train, X_val_proc, y_val):
    if ENABLE_TUNING:
        return tune_xgb(X_train_proc, y_train, X_val_proc, y_val)

    params = {
        "n_estimators": 5000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "objective": "reg:squarederror",
    }
    model = xgb.XGBRegressor(**params)
    return _fit_xgb_with_early_stopping(model, X_train_proc, y_train, X_val_proc, y_val)


def plot_pred_vs_actual(y_true, y_pred, filepath: Path = XGB_PRED_VS_ACTUAL_PNG):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.35, s=22)

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    # ±20% band
    plt.fill_between(
        [min_val, max_val],
        [min_val * 0.8, max_val * 0.8],
        [min_val * 1.2, max_val * 1.2],
        alpha=0.12,
        label="±20% band",
    )

    plt.xlabel("Actual Price (LKR)")
    plt.ylabel("Predicted Price (LKR)")
    plt.title("XGBoost: Predicted vs Actual (Test)")
    plt.legend(loc="upper left")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f"{y/1e6:.1f}M"))

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction plot → {filepath}")


def run_shap_analysis(xgb_model, preprocessor_fitted, X_test):
    try:
        import shap  # type: ignore
    except Exception as e:
        print(f"SHAP not available in this environment ({e}); skipping SHAP plots.")
        return
    X_test_proc = preprocessor_fitted.transform(X_test)
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    feature_names = None
    try:
        feature_names = preprocessor_fitted.get_feature_names_out()
    except Exception:
        pass

    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(X_test_proc.shape[0], size=min(200, X_test_proc.shape[0]), replace=False)
    X_sample = X_test_proc[idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title("XGBoost SHAP Summary (Test Sample)")
    plt.savefig(XGB_SHAP_SUMMARY_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot → {XGB_SHAP_SUMMARY_PNG}")

    if feature_names is not None:
        va_indices = [i for i, n in enumerate(feature_names) if "Vehicle_Age" in n]
        if va_indices:
            va_index = va_indices[0]
            plt.figure()
            shap.dependence_plot(
                va_index,
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
            )
            plt.title("SHAP Dependence: Vehicle_Age")
            plt.savefig(XGB_SHAP_DEPENDENCE_PNG, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved SHAP dependence plot → {XGB_SHAP_DEPENDENCE_PNG}")


def main():
    ensure_dirs()

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    preprocessor = load_preprocessor_template()

    # Fit preprocessor on train and transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Save fitted preprocessor (deployment-safe)
    joblib.dump(preprocessor, PREPROCESSOR_FITTED_JOBLIB)
    print("Saved fitted preprocessor →", PREPROCESSOR_FITTED_JOBLIB)

    # Ridge baseline
    ridge = train_ridge(X_train_proc, y_train)
    print_metrics("Ridge - Train", calc_metrics(y_train, ridge.predict(X_train_proc)), y_train)
    print_metrics("Ridge - Val", calc_metrics(y_val, ridge.predict(X_val_proc)), y_val)
    print_metrics("Ridge - Test", calc_metrics(y_test, ridge.predict(X_test_proc)), y_test)
    joblib.dump(ridge, RIDGE_MODEL_JOBLIB)
    print("Saved Ridge model →", RIDGE_MODEL_JOBLIB)

    # XGBoost main model (tuning + early stopping)
    xgb_model = train_xgb(X_train_proc, y_train, X_val_proc, y_val)

    xgb_train_pred = xgb_model.predict(X_train_proc)
    xgb_val_pred = xgb_model.predict(X_val_proc)
    xgb_test_pred = xgb_model.predict(X_test_proc)

    print_metrics("XGBoost - Train", calc_metrics(y_train, xgb_train_pred), y_train)
    print_metrics("XGBoost - Val", calc_metrics(y_val, xgb_val_pred), y_val)
    print_metrics("XGBoost - Test", calc_metrics(y_test, xgb_test_pred), y_test)

    joblib.dump(xgb_model, XGB_MODEL_JOBLIB)
    print("Saved XGBoost model →", XGB_MODEL_JOBLIB)

    # Convenience pipelines
    ridge_pipeline = Pipeline([("preprocess", preprocessor), ("model", ridge)])
    xgb_pipeline = Pipeline([("preprocess", preprocessor), ("model", xgb_model)])

    joblib.dump(ridge_pipeline, RIDGE_PIPELINE_JOBLIB)
    joblib.dump(xgb_pipeline, XGB_PIPELINE_JOBLIB)
    print("Saved pipelines →", RIDGE_PIPELINE_JOBLIB, "and", XGB_PIPELINE_JOBLIB)

    # Plot + SHAP
    plot_pred_vs_actual(y_test, xgb_test_pred, filepath=XGB_PRED_VS_ACTUAL_PNG)

    print("\nRunning SHAP analysis (this can take a bit)...")
    run_shap_analysis(xgb_model, preprocessor, X_test)
    print("SHAP analysis complete.")


if __name__ == "__main__":
    main()
