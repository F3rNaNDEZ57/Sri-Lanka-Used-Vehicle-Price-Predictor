import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# Files produced by preprocess.py
X_TRAIN_FILE = "X_train.csv"
X_VAL_FILE = "X_val.csv"
X_TEST_FILE = "X_test.csv"
Y_TRAIN_FILE = "y_train.csv"
Y_VAL_FILE = "y_val.csv"
Y_TEST_FILE = "y_test.csv"

# preprocess.py now saves a TEMPLATE (unfitted) preprocessor
PREPROCESSOR_TEMPLATE_FILE = "preprocessor_template.joblib"

# Output artifacts
PREPROCESSOR_FITTED_FILE = "preprocessor_fitted.joblib"
RIDGE_MODEL_FILE = "ridge_model.joblib"
XGB_MODEL_FILE = "xgb_model.joblib"
RIDGE_PIPELINE_FILE = "ridge_pipeline.joblib"
XGB_PIPELINE_FILE = "xgb_pipeline.joblib"

PRED_PLOT_FILE = "xgb_pred_vs_actual.png"
SHAP_SUMMARY_FILE = "xgb_shap_summary.png"
SHAP_DEPENDENCE_FILE = "xgb_shap_dependence_vehicle_age.png"

XGB_TUNING_RESULTS_FILE = "xgb_tuning_results.csv"
XGB_BEST_PARAMS_FILE = "xgb_best_params.json"

# Tuning controls (kept small so it runs quickly)
ENABLE_TUNING = True
TUNING_ITERATIONS = 10
EARLY_STOPPING_ROUNDS = 75


def load_data():
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_val = pd.read_csv(X_VAL_FILE)
    X_test = pd.read_csv(X_TEST_FILE)

    y_train = pd.read_csv(Y_TRAIN_FILE).iloc[:, 0].values
    y_val = pd.read_csv(Y_VAL_FILE).iloc[:, 0].values
    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0].values

    print("Loaded data shapes:")
    print("  X_train:", X_train.shape, " y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, " y_val:  ", y_val.shape)
    print("  X_test: ", X_test.shape, " y_test:", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_preprocessor_template():
    if not Path(PREPROCESSOR_TEMPLATE_FILE).exists():
        raise FileNotFoundError(
            f"{PREPROCESSOR_TEMPLATE_FILE} not found. Run preprocess.py first."
        )
    preprocessor = joblib.load(PREPROCESSOR_TEMPLATE_FILE)
    print("Loaded preprocessor template from", PREPROCESSOR_TEMPLATE_FILE)
    return preprocessor


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def print_metrics(name, metrics, y_true):
    mean_price = np.mean(y_true)
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
    """Fit with early stopping when the installed xgboost supports it; else fallback."""
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
        # Older sklearn API compatibility
        model.fit(X_train_proc, y_train)
        return model


def tune_xgb(X_train_proc, y_train, X_val_proc, y_val, n_iter=TUNING_ITERATIONS):
    """A small random search tuned on validation RMSE (kept intentionally lightweight)."""

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

    sampler = ParameterSampler(
        search_space, n_iter=n_iter, random_state=RANDOM_STATE
    )

    rows = []
    best_rmse = float("inf")
    best_model = None
    best_params = None

    print("\nRunning quick hyperparameter tuning...")

    for i, params in enumerate(sampler, start=1):
        full_params = {**base_params, **params}
        model = xgb.XGBRegressor(**full_params)
        model = _fit_xgb_with_early_stopping(
            model, X_train_proc, y_train, X_val_proc, y_val
        )

        val_pred = model.predict(X_val_proc)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        # best_iteration_ exists when early stopping is supported
        best_iter = getattr(model, "best_iteration", None)
        rows.append(
            {
                **params,
                "val_rmse": rmse,
                "best_iteration": best_iter,
            }
        )

        print(f"  [{i:02d}/{n_iter}] val RMSE = {rmse:,.0f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = full_params

    results = pd.DataFrame(rows).sort_values("val_rmse")
    results.to_csv(XGB_TUNING_RESULTS_FILE, index=False)
    print(f"Saved tuning results → {XGB_TUNING_RESULTS_FILE}")

    with open(XGB_BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params → {XGB_BEST_PARAMS_FILE}")

    print(f"Best validation RMSE from tuning: {best_rmse:,.0f}")

    return best_model, best_params, results


def train_xgb(X_train_proc, y_train, X_val_proc, y_val):
    if ENABLE_TUNING:
        model, best_params, _ = tune_xgb(X_train_proc, y_train, X_val_proc, y_val)
        return model

    # No tuning: use a reasonable default + early stopping
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
    model = _fit_xgb_with_early_stopping(model, X_train_proc, y_train, X_val_proc, y_val)
    return model


def plot_pred_vs_actual(
    y_true,
    y_pred,
    filepath=PRED_PLOT_FILE,
    title="XGBoost: Predicted vs Actual (Test)",
):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, s=30, edgecolor=None)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    # ±20% band
    plt.fill_between(
        [min_val, max_val],
        [min_val * 0.8, max_val * 0.8],
        [min_val * 1.2, max_val * 1.2],
        color="green",
        alpha=0.1,
        label="±20% band",
    )

    plt.xlabel("Actual Price (LKR)")
    plt.ylabel("Predicted Price (LKR)")
    plt.title(title)
    plt.legend(loc="upper left")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M"))

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction plot to {filepath}")


def run_shap_analysis(xgb_model, preprocessor_fitted, X_test):
    """Run SHAP on a sample of the test set using the *fitted* preprocessor."""

    X_test_proc = preprocessor_fitted.transform(X_test)

    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    feature_names = None
    try:
        feature_names = preprocessor_fitted.get_feature_names_out()
    except Exception:
        pass

    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(
        X_test_proc.shape[0], size=min(200, X_test_proc.shape[0]), replace=False
    )
    X_test_sample = X_test_proc[idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_sample)

    # Global summary plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title("XGBoost SHAP Summary (Test Sample)")
    plt.savefig(SHAP_SUMMARY_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {SHAP_SUMMARY_FILE}")

    # Dependence plot for Vehicle_Age (if present)
    if feature_names is not None:
        va_indices = [i for i, n in enumerate(feature_names) if "Vehicle_Age" in n]
        if va_indices:
            va_index = va_indices[0]
            plt.figure()
            shap.dependence_plot(
                va_index,
                shap_values,
                X_test_sample,
                feature_names=feature_names,
                show=False,
            )
            plt.title("SHAP Dependence: Vehicle_Age")
            plt.savefig(SHAP_DEPENDENCE_FILE, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved SHAP dependence plot to {SHAP_DEPENDENCE_FILE}")
        else:
            print("Vehicle_Age not found in feature names; skipping dependence plot.")
    else:
        print("preprocessor.get_feature_names_out() not available; skipping dependence plot.")


def main():
    # 1) Load data + preprocessor template
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    preprocessor = load_preprocessor_template()

    # 2) Fit preprocessor on train and transform all splits
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Save the fitted preprocessor for deployment (Fix #1)
    joblib.dump(preprocessor, PREPROCESSOR_FITTED_FILE)
    print("Saved fitted preprocessor to", PREPROCESSOR_FITTED_FILE)

    # 3) Ridge baseline
    ridge = train_ridge(X_train_proc, y_train)

    ridge_train_pred = ridge.predict(X_train_proc)
    ridge_val_pred = ridge.predict(X_val_proc)
    ridge_test_pred = ridge.predict(X_test_proc)

    print_metrics("Ridge - Train", calc_metrics(y_train, ridge_train_pred), y_train)
    print_metrics("Ridge - Val", calc_metrics(y_val, ridge_val_pred), y_val)
    print_metrics("Ridge - Test", calc_metrics(y_test, ridge_test_pred), y_test)

    joblib.dump(ridge, RIDGE_MODEL_FILE)
    print("Saved Ridge model to", RIDGE_MODEL_FILE)

    # 4) XGBoost main model with early stopping + quick tuning (Fix #2)
    xgb_model = train_xgb(X_train_proc, y_train, X_val_proc, y_val)

    xgb_train_pred = xgb_model.predict(X_train_proc)
    xgb_val_pred = xgb_model.predict(X_val_proc)
    xgb_test_pred = xgb_model.predict(X_test_proc)

    print_metrics("XGBoost - Train", calc_metrics(y_train, xgb_train_pred), y_train)
    print_metrics("XGBoost - Val", calc_metrics(y_val, xgb_val_pred), y_val)
    print_metrics("XGBoost - Test", calc_metrics(y_test, xgb_test_pred), y_test)

    joblib.dump(xgb_model, XGB_MODEL_FILE)
    print("Saved XGBoost model to", XGB_MODEL_FILE)

    # 5) Convenience pipelines for deployment
    ridge_pipeline = Pipeline([("preprocess", preprocessor), ("model", ridge)])
    xgb_pipeline = Pipeline([("preprocess", preprocessor), ("model", xgb_model)])

    joblib.dump(ridge_pipeline, RIDGE_PIPELINE_FILE)
    joblib.dump(xgb_pipeline, XGB_PIPELINE_FILE)
    print("Saved pipelines →", RIDGE_PIPELINE_FILE, "and", XGB_PIPELINE_FILE)

    # 6) Plot predictions
    plot_pred_vs_actual(y_test, xgb_test_pred, filepath=PRED_PLOT_FILE)

    # 7) SHAP analysis
    print("\nRunning SHAP analysis (this can take a bit)...")
    run_shap_analysis(xgb_model, preprocessor, X_test)
    print("SHAP analysis complete.")


if __name__ == "__main__":
    main()
