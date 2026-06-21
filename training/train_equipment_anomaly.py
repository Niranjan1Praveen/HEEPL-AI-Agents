"""
training/train_equipment_anomaly.py
-------------------------------------
Generates synthetic sensor data for pumps and blowers, then trains an
IsolationForest anomaly detection pipeline on sound, vibration, and temperature.

Designed so real sensor CSVs can replace the synthetic data later without
changing the training pipeline or the save format.

Save path: models/equipment_anomaly_model.pkl
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.equipment_thresholds import EQUIPMENT_THRESHOLDS

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = MODELS_DIR / "equipment_anomaly_model.pkl"

RANDOM_STATE = 42
TOTAL_SAMPLES = 2000
HEALTHY_FRAC  = 0.70
WARNING_FRAC  = 0.20
CRITICAL_FRAC = 0.10


def _generate_samples_for_type(eq_type: str, n_healthy: int, n_warning: int, n_critical: int) -> pd.DataFrame:
    """Generate synthetic rows for one equipment type."""
    rng = np.random.default_rng(RANDOM_STATE)
    limits = EQUIPMENT_THRESHOLDS[eq_type]

    def jitter(center, spread, n):
        return rng.normal(loc=center, scale=spread, size=n)

    # Healthy: Gaussian noise around mid-point below normal limit
    h_sound  = jitter(limits["sound"]["normal"]       * 0.80, 3.0, n_healthy)
    h_vib    = jitter(limits["vibration"]["normal"]   * 0.75, 0.3, n_healthy)
    h_temp   = jitter(limits["temperature"]["normal"] * 0.78, 3.5, n_healthy)

    # Warning: values in [normal, critical) range with correlated drift
    w_sound  = jitter((limits["sound"]["normal"]       + limits["sound"]["critical"])       / 2, 2.5, n_warning)
    w_vib    = jitter((limits["vibration"]["normal"]   + limits["vibration"]["critical"])   / 2, 0.4, n_warning)
    w_temp   = jitter((limits["temperature"]["normal"] + limits["temperature"]["critical"]) / 2, 4.0, n_warning)

    # Critical: above critical thresholds, high inter-parameter correlation
    c_sound  = jitter(limits["sound"]["critical"]       * 1.08, 2.0, n_critical)
    c_vib    = jitter(limits["vibration"]["critical"]   * 1.15, 0.5, n_critical)
    c_temp   = jitter(limits["temperature"]["critical"] * 1.12, 3.0, n_critical)

    df = pd.DataFrame({
        "sound":       np.concatenate([h_sound, w_sound, c_sound]),
        "vibration":   np.concatenate([h_vib,   w_vib,   c_vib]),
        "temperature": np.concatenate([h_temp,  w_temp,  c_temp]),
        "equipment_type": eq_type,
        "label": (["Healthy"] * n_healthy + ["Warning"] * n_warning + ["Critical"] * n_critical),
    })
    return df


def generate_synthetic_data() -> pd.DataFrame:
    """
    Build a ~2000-row synthetic sensor dataset split 70/20/10 across health states
    for both Pump and Blower equipment types.
    """
    n_healthy  = int(TOTAL_SAMPLES * HEALTHY_FRAC  / 2)
    n_warning  = int(TOTAL_SAMPLES * WARNING_FRAC  / 2)
    n_critical = int(TOTAL_SAMPLES * CRITICAL_FRAC / 2)

    pump_df   = _generate_samples_for_type("Pump",   n_healthy, n_warning, n_critical)
    blower_df = _generate_samples_for_type("Blower", n_healthy, n_warning, n_critical)
    df = pd.concat([pump_df, blower_df], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

    # Clip to physically plausible ranges
    df["sound"]       = df["sound"].clip(30, 130)
    df["vibration"]   = df["vibration"].clip(0.1, 20.0)
    df["temperature"] = df["temperature"].clip(20, 130)
    return df


def train_and_save_equipment_model(data: pd.DataFrame = None) -> dict:
    """
    Train an IsolationForest pipeline on sound / vibration / temperature.
    If `data` is None, synthetic data is generated automatically.

    Returns a summary dict with model path and basic metrics.
    """
    if data is None:
        print("Generating synthetic sensor data...")
        data = generate_synthetic_data()
        print(f"  Generated {len(data)} rows  |  "
              f"Healthy: {(data.label == 'Healthy').sum()}  "
              f"Warning: {(data.label == 'Warning').sum()}  "
              f"Critical: {(data.label == 'Critical').sum()}")

    feature_columns = ["sound", "vibration", "temperature"]
    X = data[feature_columns].copy()

    # Contamination = fraction expected to be anomalous (warning + critical)
    contamination = WARNING_FRAC + CRITICAL_FRAC

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   IsolationForest(
            contamination=contamination,
            n_estimators=200,
            max_samples="auto",
            random_state=RANDOM_STATE,
        )),
    ])

    print("Training IsolationForest pipeline...")
    pipeline.fit(X)

    # Quick evaluation: healthy samples should score > 0, critical < 0
    scores = pipeline.decision_function(X)
    data = data.copy()
    data["anomaly_score"] = scores
    data["is_anomaly"]    = pipeline.predict(X) == -1

    healthy_mean  = data.loc[data.label == "Healthy",  "anomaly_score"].mean()
    warning_mean  = data.loc[data.label == "Warning",  "anomaly_score"].mean()
    critical_mean = data.loc[data.label == "Critical", "anomaly_score"].mean()

    print(f"  Score averages — Healthy: {healthy_mean:+.3f}  Warning: {warning_mean:+.3f}  Critical: {critical_mean:+.3f}")

    model_data = {
        "pipeline":         pipeline,
        "feature_columns":  feature_columns,
        "contamination":    contamination,
        "equipment_types":  ["Pump", "Blower"],
        "score_stats": {
            "healthy_mean":  float(healthy_mean),
            "warning_mean":  float(warning_mean),
            "critical_mean": float(critical_mean),
        },
    }

    joblib.dump(model_data, OUTPUT_PATH)
    print(f"Model saved to: {OUTPUT_PATH}")

    return {
        "path":         str(OUTPUT_PATH),
        "healthy_mean": healthy_mean,
        "warning_mean": warning_mean,
        "critical_mean": critical_mean,
    }


if __name__ == "__main__":
    result = train_and_save_equipment_model()
    print("\nTraining complete.")
    print(f"  Saved to: {result['path']}")
