# AI-Powered Predictive Maintenance & Equipment Failure Prediction System

A Streamlit app that analyzes machine sensor + maintenance history to predict
equipment failures before they happen, explains *why* the model thinks a
machine is at risk, and turns that into a prioritized, actionable maintenance
plan.

## Features

- **Synthetic demo fleet generator** — no data? Instantly generate a realistic
  fleet of machines (Pumps, Compressors, Motors, Conveyors, Turbines) with
  degradation-linked sensor readings and failure history, or upload your own CSV.
- **ML failure prediction** — Random Forest or Gradient Boosting classifier
  predicting probability of failure within the next 14 days, evaluated with
  ROC-AUC, Precision-Recall (better suited than accuracy for rare-failure data),
  and a confusion matrix.
- **Failure risk dashboard** — every machine scored and bucketed into
  Low / Medium / High / Critical, filterable, downloadable as CSV.
- **Explainability (SHAP)** — global feature importance, directional
  risk-increasing vs. risk-decreasing effects, and a per-machine explanation
  of exactly which sensor readings are driving that machine's risk score.
- **Maintenance prioritization** — each machine gets a plain-language list of
  top risk factors, a recommended action, and a suggested time window
  (e.g. "Critical → within 48 hours").

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Using your own data

Switch "Data source" to **Upload my own CSV** in the sidebar. Your file needs
a binary failure/target column plus these feature columns:

```
machine_type, age_years, operational_hours, temperature_avg, temperature_std,
vibration_avg, vibration_std, pressure_avg, rotation_speed_avg, humidity_avg,
load_pct, oil_quality_index, days_since_last_maintenance, num_previous_failures
```

If your column names differ, either rename them to match, or adapt
`NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` in `ml_pipeline.py` to your schema.
Each row should represent one snapshot/observation of a machine (e.g. a daily
or weekly rollup of its sensor readings) with a label for whether it failed
within the prediction horizon you care about.

## Project structure

```
app.py              Streamlit dashboard (5 tabs: overview, model, risk, explainability, plan)
data_generator.py   Synthetic sensor/maintenance data generator
ml_pipeline.py       Training, evaluation, SHAP explainability, recommendation logic
requirements.txt
```

## Notes on the approach

- **Imbalance-aware**: failures are rare in real fleets (~8% in the demo data),
  so the app surfaces ROC-AUC / PR-AUC and a continuous risk score rather than
  leaning on raw accuracy or a single 0.5 threshold.
- **Interpretable by design**: tree ensembles + SHAP TreeExplainer give both
  global ("what matters across the fleet") and local ("why is *this* machine
  risky") explanations, which is what turns a risk score into something a
  maintenance team can actually act on.
- **Swap in your own model**: `ml_pipeline.train_model` is a clean seam if you
  want to try XGBoost/LightGBM or a different feature set — SHAP's
  `TreeExplainer` will work with any of those out of the box.
