# Sri Lanka Used Vehicle Price Predictor (ML Assignment)

This project scrapes used-vehicle listings, cleans/preprocesses the data, trains ML models (Ridge + XGBoost), generates explainability (SHAP), and provides a Streamlit app for predictions + market insights.

## Folder structure

- `data/raw/` — raw scraped data (`vehicles_raw.csv`)
- `data/processed/` — cleaned dataset (`vehicles_clean.csv`)
- `data/splits/` — train/val/test splits
- `models/` — trained model artifacts and pipelines (`*.joblib`)
- `reports/figures/` — plots and SHAP figures
- `reports/` — JSON insights + tuning outputs
- `src/` — pipeline scripts (scrape, preprocess, train, analysis)
- `app/` — Streamlit UI

## Run order (from project root)

```bash
python src/preprocess.py
python src/model.py
python src/model_analysis.py
streamlit run app/app.py
```

Optional: scrape fresh data (be respectful; includes delays):

```bash
python src/scrape.py
```

## Notes

- The Streamlit app loads `models/xgb_pipeline.joblib` when available, and falls back to `models/xgb_model.joblib` + `models/preprocessor_fitted.joblib`.
- SHAP explanations in the app are shown at the **field level** (Brand overall, Location overall, Fuel type overall) to avoid confusion from one-hot encoded columns.
