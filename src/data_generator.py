"""
SafeHer AI — Synthetic Dataset Generator
Generates a realistic 10,000-row safety dataset for model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ─── helpers ───────────────────────────────────────────────────────────────

def _clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


def _bernoulli(p, n):
    return np.random.binomial(1, p, n).astype(int)


# ─── main generator ────────────────────────────────────────────────────────

def generate_dataset(n_rows: int = 10_000, save_path: str = "data/safety_dataset.csv") -> pd.DataFrame:
    """
    Generate a synthetic safety dataset and optionally save it to CSV.

    Parameters
    ----------
    n_rows    : number of rows to generate
    save_path : path to save the CSV file; pass None to skip saving
    """

    # ── raw features ──────────────────────────────────────────────────────
    hour                = np.random.randint(0, 24, n_rows)
    is_night            = (hour < 6) | (hour >= 20)
    is_night            = is_night.astype(int)

    # day type: 0=weekday, 1=weekend
    day_type            = np.random.choice([0, 1], n_rows, p=[0.71, 0.29])

    # geographic coordinates (India-centric defaults)
    latitude            = np.random.uniform(8.0, 35.0, n_rows).round(6)
    longitude           = np.random.uniform(68.0, 97.0, n_rows).round(6)

    # social/environmental context
    is_alone            = _bernoulli(0.45, n_rows)
    crowd_density       = np.random.choice(["low", "medium", "high"], n_rows, p=[0.35, 0.40, 0.25])
    lighting_condition  = np.where(is_night == 1,
                                   np.random.choice(["dark", "dim", "bright"], n_rows, p=[0.50, 0.35, 0.15]),
                                   np.random.choice(["dark", "dim", "bright"], n_rows, p=[0.05, 0.20, 0.75]))

    crime_area_score    = np.random.beta(2, 5, n_rows).round(3)   # 0–1, skewed low
    phone_motion_status = np.random.choice(["stationary", "walking", "running", "erratic"], n_rows,
                                           p=[0.45, 0.35, 0.12, 0.08])
    walking_speed       = np.random.uniform(0, 8, n_rows).round(2)  # km/h
    distance_from_home  = np.random.exponential(5, n_rows).clip(0, 60).round(2)  # km
    distance_from_safe_zone = np.random.exponential(3, n_rows).clip(0, 30).round(2)

    battery_level       = np.random.randint(5, 101, n_rows)
    network_status      = np.random.choice(["no_signal", "2G", "3G", "4G", "5G"], n_rows,
                                           p=[0.05, 0.10, 0.20, 0.45, 0.20])
    weather_condition   = np.random.choice(["clear", "cloudy", "rainy", "foggy", "stormy"], n_rows,
                                           p=[0.50, 0.20, 0.15, 0.10, 0.05])
    panic_button        = _bernoulli(0.05, n_rows)
    unusual_movement    = _bernoulli(0.12, n_rows)
    time_spent_at_location = np.random.exponential(20, n_rows).clip(0, 180).round(0).astype(int)  # mins
    nearest_trusted_contact_distance = np.random.exponential(4, n_rows).clip(0, 50).round(2)  # km

    # ── engineered risk scores ─────────────────────────────────────────────
    night_risk_score   = (is_night * 0.6 + (lighting_condition == "dark") * 0.4).astype(float)

    crowd_map          = {"low": 1.0, "medium": 0.5, "high": 0.1}
    isolation_score    = (is_alone * 0.6 +
                          np.vectorize(crowd_map.get)(crowd_density) * 0.4)

    motion_map         = {"stationary": 0.1, "walking": 0.2, "running": 0.6, "erratic": 1.0}
    movement_risk_score = np.vectorize(motion_map.get)(phone_motion_status).astype(float)

    location_risk_score = (crime_area_score * 0.5 +
                           (distance_from_safe_zone / 30) * 0.3 +
                           (distance_from_home / 60) * 0.2)

    emergency_score    = (panic_button * 0.5 + unusual_movement * 0.3 +
                          ((battery_level < 20).astype(int) * 0.2))

    total_risk_score   = _clamp(
        night_risk_score * 0.20 +
        isolation_score  * 0.20 +
        movement_risk_score * 0.15 +
        location_risk_score * 0.25 +
        emergency_score  * 0.20, 0, 1
    ).round(4)

    # ── label generation (rule-based + noise) ─────────────────────────────
    def assign_label(row):
        score = row["total_risk_score"]
        panic = row["panic_button"]
        if panic == 1 or score >= 0.75:
            return "Emergency"
        elif score >= 0.55:
            return "High Risk"
        elif score >= 0.35:
            return "Medium Risk"
        else:
            return "Safe"

    df = pd.DataFrame({
        "user_id":                          [f"U{str(i).zfill(5)}" for i in range(1, n_rows + 1)],
        "latitude":                         latitude,
        "longitude":                        longitude,
        "hour":                             hour,
        "day_type":                         day_type,
        "is_night":                         is_night,
        "is_alone":                         is_alone,
        "crowd_density":                    crowd_density,
        "lighting_condition":               lighting_condition,
        "crime_area_score":                 crime_area_score,
        "phone_motion_status":              phone_motion_status,
        "walking_speed":                    walking_speed,
        "distance_from_home":               distance_from_home,
        "distance_from_safe_zone":          distance_from_safe_zone,
        "battery_level":                    battery_level,
        "network_status":                   network_status,
        "weather_condition":                weather_condition,
        "panic_button":                     panic_button,
        "unusual_movement":                 unusual_movement,
        "time_spent_at_location":           time_spent_at_location,
        "nearest_trusted_contact_distance": nearest_trusted_contact_distance,
        # engineered
        "night_risk_score":                 night_risk_score,
        "isolation_score":                  isolation_score.round(4),
        "movement_risk_score":              movement_risk_score,
        "location_risk_score":              location_risk_score.round(4),
        "emergency_score":                  emergency_score.round(4),
        "total_risk_score":                 total_risk_score,
    })

    df["risk_level"] = df.apply(assign_label, axis=1)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅  Dataset saved → {save_path}  ({len(df):,} rows)")

    return df


if __name__ == "__main__":
    df = generate_dataset(10_000, "data/safety_dataset.csv")
    print(df["risk_level"].value_counts())
    print(df.head(3))
