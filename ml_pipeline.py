"""
Training, evaluation, and interpretability pipeline for the predictive
maintenance model.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "age_years",
    "operational_hours",
    "temperature_avg",
    "temperature_std",
    "vibration_avg",
    "vibration_std",
    "pressure_avg",
    "rotation_speed_avg",
    "humidity_avg",
    "load_pct",
    "oil_quality_index",
    "days_since_last_maintenance",
    "num_previous_failures",
]
CATEGORICAL_FEATURES = ["machine_type"]

# Human-readable guidance for each raw feature, used to turn SHAP
# contributions into plain-language maintenance recommendations.
FEATURE_GUIDANCE = {
    "vibration_avg": ("Elevated average vibration", "Inspect bearings, shaft alignment and mounting for wear."),
    "vibration_std": ("Unstable / erratic vibration", "Check for loosening components or intermittent faults."),
    "temperature_avg": ("Operating temperature above normal", "Check cooling, lubrication and ventilation."),
    "temperature_std": ("Temperature fluctuating unusually", "Inspect thermostat/cooling control and sensor calibration."),
    "days_since_last_maintenance": ("Overdue for scheduled maintenance", "Schedule a routine maintenance visit."),
    "oil_quality_index": ("Degraded oil/lubricant quality", "Perform an oil change or lubricant analysis."),
    "num_previous_failures": ("Repeated failure history", "Consider root-cause analysis or component replacement."),
    "age_years": ("Aging equipment", "Increase inspection frequency; evaluate replacement planning."),
    "load_pct": ("Sustained high operating load", "Review load balancing / duty cycle."),
    "pressure_avg": ("Abnormal pressure levels", "Inspect seals, valves and pressure regulation."),
    "rotation_speed_avg": ("Rotation speed drift", "Check drive/motor control and belt or coupling wear."),
    "humidity_avg": ("High ambient humidity exposure", "Check for moisture ingress / corrosion risk."),
    "operational_hours": ("High cumulative runtime", "Evaluate against manufacturer service intervals."),
}


@dataclass
class TrainedModel:
    model: object
    encoder: OneHotEncoder
    feature_names: list
    X_test: pd.DataFrame
    y_test: pd.Series
    X_test_encoded: np.ndarray
    y_proba_test: np.ndarray
    explainer: Optional[object] = None
    shap_values_test: Optional[np.ndarray] = None
    metrics: dict = field(default_factory=dict)


def _extract_positive_class_shap(explainer, X_enc) -> np.ndarray:
    """Handle the different shapes shap.TreeExplainer.shap_values can return
    across versions/models: a list [class0, class1], a single 2D array, or a
    3D array (samples, features, classes)."""
    raw = explainer.shap_values(X_enc)
    if isinstance(raw, list):
        return np.asarray(raw[1] if len(raw) > 1 else raw[0])
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def _encode(df: pd.DataFrame, encoder: OneHotEncoder, fit: bool = False):
    cat = df[CATEGORICAL_FEATURES]
    if fit:
        cat_encoded = encoder.fit_transform(cat)
    else:
        cat_encoded = encoder.transform(cat)
    cat_cols = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    num = df[NUMERIC_FEATURES].reset_index(drop=True)
    cat_df = pd.DataFrame(cat_encoded, columns=cat_cols)
    combined = pd.concat([num, cat_df], axis=1)
    return combined


def train_model(
    df: pd.DataFrame,
    target_col: str,
    model_type: str = "Random Forest",
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainedModel:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_enc = _encode(X_train, encoder, fit=True)
    X_test_enc = _encode(X_test, encoder, fit=False)

    if model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=random_state, n_estimators=250, max_depth=3, learning_rate=0.08)
    else:
        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    model.fit(X_train_enc, y_train)

    y_proba_test = model.predict_proba(X_test_enc)[:, 1]

    metrics = {}
    if y_test.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba_test)
        metrics["avg_precision"] = average_precision_score(y_test, y_proba_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        prec, rec, _ = precision_recall_curve(y_test, y_proba_test)
        metrics["roc_curve"] = (fpr, tpr)
        metrics["pr_curve"] = (prec, rec)
    metrics["confusion_matrix"] = confusion_matrix(y_test, (y_proba_test >= 0.5).astype(int))
    metrics["base_rate"] = y.mean()

    # SHAP TreeExplainer works natively/efficiently on tree ensembles.
    explainer = shap.TreeExplainer(model)
    shap_values_test = _extract_positive_class_shap(explainer, X_test_enc)

    return TrainedModel(
        model=model,
        encoder=encoder,
        feature_names=list(X_train_enc.columns),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        X_test_encoded=X_test_enc,
        y_proba_test=y_proba_test,
        explainer=explainer,
        shap_values_test=shap_values_test,
        metrics=metrics,
    )


def score_fleet(trained: TrainedModel, df: pd.DataFrame) -> pd.DataFrame:
    """Score every machine in df (not just the test split) and attach SHAP-based
    top contributing factors + a plain-language recommendation."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_enc = _encode(X, trained.encoder, fit=False)
    proba = trained.model.predict_proba(X_enc)[:, 1]

    shap_vals = _extract_positive_class_shap(trained.explainer, X_enc)

    results = df.copy().reset_index(drop=True)
    results["failure_risk"] = proba
    results["risk_level"] = pd.cut(
        results["failure_risk"],
        bins=[-0.01, 0.15, 0.35, 0.6, 1.01],
        labels=["Low", "Medium", "High", "Critical"],
    )

    top_factors = []
    recommendations = []
    for i in range(len(results)):
        row_shap = np.asarray(shap_vals[i]).ravel()
        order = np.argsort(-row_shap)  # most risk-increasing first
        top_idx = [j for j in order if row_shap[j] > 0][:3]
        factors = []
        actions = []
        seen_raw = set()
        for j in top_idx:
            fname = trained.feature_names[j]
            # map one-hot machine_type_X back to a readable factor
            if fname.startswith("machine_type_"):
                label = f"Machine type: {fname.replace('machine_type_', '')}"
                action = "Type-specific risk profile — compare against fleet baseline for this equipment class."
                key = "machine_type"
            else:
                label, action = FEATURE_GUIDANCE.get(fname, (fname, "Review this parameter against normal operating range."))
                key = fname
            if key in seen_raw:
                continue
            seen_raw.add(key)
            factors.append(label)
            actions.append(action)
        top_factors.append("; ".join(factors) if factors else "No dominant risk driver")
        recommendations.append(" | ".join(actions) if actions else "Continue routine monitoring.")

    results["top_risk_factors"] = top_factors
    results["recommended_action"] = recommendations
    return results


def priority_window(risk_level: str) -> str:
    return {
        "Critical": "Within 48 hours",
        "High": "Within 1 week",
        "Medium": "Within 1 month",
        "Low": "Routine schedule",
    }.get(str(risk_level), "Routine schedule")
