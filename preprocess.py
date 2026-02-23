import pandas as pd
import numpy as np
import re
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

RANDOM_STATE = 42
CURRENT_YEAR = 2026

RAW_FILE = "vehicles_raw.csv"

# Artifacts
PREPROCESSOR_TEMPLATE_FILE = "preprocessor_template.joblib"


def load_raw_data(filepath: str = RAW_FILE) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_price(raw_price: pd.Series) -> pd.Series:
    """Convert 'Rs: 12,500,000' → 12500000 (float)."""

    def to_num(s):
        if pd.isna(s):
            return np.nan
        s = str(s)
        digits = re.sub(r"[^\d]", "", s)
        return float(digits) if digits else np.nan

    return raw_price.apply(to_num)


def extract_location(raw_meta: pd.Series) -> pd.Series:
    """Extract location name from metadata blob."""

    def parse_loc(s):
        if pd.isna(s):
            return np.nan
        s = str(s)
        parts = s.split("|", 1)
        right = parts[1].strip() if len(parts) >= 2 else s.strip()
        right = right.replace("Find Out More", "").strip()
        right = re.sub(r"\d[\d,]*\s*$", "", right).strip()
        return right or np.nan

    return raw_meta.apply(parse_loc)


def extract_body_type(vehicle_type: pd.Series) -> pd.Series:
    """From 'Car(Hatchback)' → 'Hatchback'."""

    def parse_bt(s):
        if pd.isna(s):
            return np.nan
        s = str(s)
        m = re.search(r"\((.*?)\)", s)
        if m:
            val = m.group(1).strip()
            return val if val else np.nan
        return np.nan

    return vehicle_type.apply(parse_bt)


def clean_engine_capacity(cap: pd.Series) -> pd.Series:
    """Coerce engine capacity to numeric, and drop obviously wrong outliers."""

    def to_num(x):
        if pd.isna(x):
            return np.nan
        try:
            v = float(str(x).replace(",", "").strip())
        except (TypeError, ValueError):
            # try to salvage digits from messy strings
            digits = re.sub(r"[^\d]", "", str(x))
            v = float(digits) if digits else np.nan

        # Treat zeros/negatives as missing
        if not np.isfinite(v) or v <= 0:
            return np.nan

        # Plausible range for passenger vehicles (assignment scope):
        # > 10,000 cc is almost certainly a scrape/format error in this dataset.
        if v < 500 or v > 8000:
            return np.nan

        return v

    return cap.apply(to_num)


def clean_year(year_series: pd.Series) -> pd.Series:
    """Clean Model/Register year to numeric year; drop impossible values."""

    def to_year(x):
        if pd.isna(x):
            return np.nan
        try:
            y = int(str(x).strip())
            if 1980 <= y <= CURRENT_YEAR + 1:
                return y
            return np.nan
        except (TypeError, ValueError):
            return np.nan

    return year_series.apply(to_year)


def clean_mileage(m_series: pd.Series) -> pd.Series:
    """Ensure mileage is numeric and within a reasonable range."""

    def to_num(x):
        if pd.isna(x):
            return np.nan
        try:
            v = float(str(x).replace(",", "").strip())
        except (TypeError, ValueError):
            digits = re.sub(r"[^\d]", "", str(x))
            v = float(digits) if digits else np.nan

        if not np.isfinite(v) or v < 0:
            return np.nan

        # Scrape noise / typos can create million+ mileage values.
        if v > 700_000:
            return np.nan

        return v

    return m_series.apply(to_num)


def clean_text(series: pd.Series) -> pd.Series:
    """Strip text, keep missing as NA (avoid turning NaN into literal 'nan')."""
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return s


def build_clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) Target
    df["Price_LKR"] = clean_price(df["Raw_Price"])
    df = df.dropna(subset=["Price_LKR"])

    # 2) Location
    df["Location_District"] = extract_location(df["Raw_Metadata"])

    # 3) Brand / Model
    df["Brand"] = clean_text(df["Manufacturer"])
    df["Model_Name"] = clean_text(df["Model"])

    # 4) Year / Age
    df["Model_Year"] = clean_year(df["Model Year"])
    df["Vehicle_Age"] = CURRENT_YEAR - df["Model_Year"]

    # 5) Mileage
    df["Mileage_km"] = clean_mileage(df["Mileage"])

    # 6) Engine
    df["Engine_Capacity"] = clean_engine_capacity(df["Engine/Motor Capacity"])

    # 7) Body type
    df["Body_Type"] = extract_body_type(df["Vehicle Type"])

    # 8) Other categoricals
    df["Condition"] = clean_text(df["Condition"])
    df["Fuel_Type"] = clean_text(df["Fuel Type"])
    df["Transmission"] = clean_text(df["Transmission"])
    df["Colour"] = clean_text(df["Colour"])

    # 9) Basic outlier filtering (target + age)
    df = df[(df["Price_LKR"] >= 500_000) & (df["Price_LKR"] <= 100_000_000)]
    df = df[(df["Vehicle_Age"].isna()) | ((df["Vehicle_Age"] >= 0) & (df["Vehicle_Age"] <= 40))]

    # 10) Drop obvious duplicates
    df = df.drop_duplicates(
        subset=["Brand", "Model_Name", "Model_Year", "Mileage_km", "Price_LKR", "Location_District"]
    )

    # 11) Final columns
    final_cols = [
        "Price_LKR",
        "Brand",
        "Model_Name",
        "Model_Year",
        "Vehicle_Age",
        "Mileage_km",
        "Engine_Capacity",
        "Location_District",
        "Condition",
        "Fuel_Type",
        "Transmission",
        "Body_Type",
        "Colour",
    ]

    df_final = df[final_cols].reset_index(drop=True)
    print(f"After cleaning: {df_final.shape[0]} rows, {df_final.shape[1]} columns")
    return df_final


def split_data(df_clean: pd.DataFrame):
    X = df_clean.drop("Price_LKR", axis=1)
    y = df_clean["Price_LKR"]

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    print("Train size:", X_train.shape[0])
    print("Val size:  ", X_val.shape[0])
    print("Test size: ", X_test.shape[0])

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = ["Model_Year", "Vehicle_Age", "Mileage_km", "Engine_Capacity"]
    categorical_features = [
        "Brand",
        "Model_Name",
        "Location_District",
        "Condition",
        "Fuel_Type",
        "Transmission",
        "Body_Type",
        "Colour",
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                __import__("sklearn.impute").impute.SimpleImputer(strategy="median"),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                __import__("sklearn.impute").impute.SimpleImputer(strategy="most_frequent"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def save_splits(
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor: ColumnTransformer
):
    # Save data splits
    X_train.to_csv("X_train.csv", index=False)
    X_val.to_csv("X_val.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_val.to_csv("y_val.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    # Save full cleaned dataset
    df_clean = pd.concat(
        [
            pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True),
            pd.concat([y_train, y_val, y_test], axis=0, ignore_index=True),
        ],
        axis=1,
    )
    df_clean.to_csv("vehicles_clean.csv", index=False)

    # Save a TEMPLATE preprocessor (unfitted). The fitted version is saved by model.py.
    joblib.dump(preprocessor, PREPROCESSOR_TEMPLATE_FILE)
    print(f"Saved splits + preprocessor template to disk ({PREPROCESSOR_TEMPLATE_FILE}).")


def main():
    raw_path = Path(RAW_FILE)
    if not raw_path.exists():
        raise FileNotFoundError(f"{RAW_FILE} not found in current directory.")

    df_raw = load_raw_data(RAW_FILE)
    df_clean = build_clean_dataset(df_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    preprocessor = build_preprocessor(X_train)

    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)


if __name__ == "__main__":
    main()
