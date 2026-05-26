"""
SafeHer AI — Preprocessing Pipeline
Encodes, scales, and splits the safety dataset for model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path

# ── column definitions ──────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "crowd_density",
    "lighting_condition",
    "phone_motion_status",
    "network_status",
    "weather_condition",
]

ENGINEERED_FEATURES = [
    "night_risk_score",
    "isolation_score",
    "movement_risk_score",
    "location_risk_score",
    "emergency_score",
    "total_risk_score",
]

NUMERIC_COLS = [
    "latitude", "longitude", "hour", "day_type", "is_night", "is_alone",
    "crime_area_score", "walking_speed", "distance_from_home",
    "distance_from_safe_zone", "battery_level", "panic_button",
    "unusual_movement", "time_spent_at_location",
    "nearest_trusted_contact_distance",
] + ENGINEERED_FEATURES

TARGET_COL   = "risk_level"
DROP_COLS    = ["user_id"]

LABEL_ORDER  = ["Safe", "Medium Risk", "High Risk", "Emergency"]
LABEL_MAP    = {l: i for i, l in enumerate(LABEL_ORDER)}


# ── main preprocessing function ─────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    encoders: dict | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = True,
    models_dir: str = "models",
):
    """
    Clean, encode, and scale the dataset.

    Parameters
    ----------
    df        : raw DataFrame
    encoders  : pre-fitted LabelEncoders (pass when predicting)
    scaler    : pre-fitted StandardScaler (pass when predicting)
    fit       : True during training, False during inference
    models_dir: where to save/load encoders & scaler

    Returns
    -------
    X_train, X_test, y_train, y_test   (fit=True)
    X, y_encoded                        (fit=False)
    encoders, scaler
    """
    df = df.copy()

    # drop non-feature columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # ── encode categoricals ──────────────────────────────────────────────
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    # ── encode target ────────────────────────────────────────────────────
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map(LABEL_MAP)

    # ── separate X, y ────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None

    # ── scale numerics ───────────────────────────────────────────────────
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # ── persist artifacts ────────────────────────────────────────────────
    if fit:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, f"{models_dir}/encoders.pkl")
        joblib.dump(scaler,   f"{models_dir}/scaler.pkl")
        print(f"✅  Encoders & scaler saved → {models_dir}/")

    # ── train/test split (only during training) ──────────────────────────
    if fit and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test, feature_cols, encoders, scaler

    return X_scaled, y, feature_cols, encoders, scaler


def load_artifacts(models_dir: str = "models"):
    """Load encoders & scaler from disk."""
    encoders = joblib.load(f"{models_dir}/encoders.pkl")
    scaler   = joblib.load(f"{models_dir}/scaler.pkl")
    return encoders, scaler


if __name__ == "__main__":
    from data_generator import generate_dataset
    df = generate_dataset(5000, save_path=None)
    X_train, X_test, y_train, y_test, features, enc, sc = preprocess(df, fit=True)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print("Features:", features[:5], "...")
