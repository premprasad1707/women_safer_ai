"""
SafeHer AI — Prediction Engine
Loads the trained model and returns risk predictions with explanations.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

LABEL_NAMES  = ["Safe", "Medium Risk", "High Risk", "Emergency"]
LABEL_COLORS = {
    "Safe":        "#22c55e",
    "Medium Risk": "#f59e0b",
    "High Risk":   "#ef4444",
    "Emergency":   "#7c3aed",
}
LABEL_ICONS  = {
    "Safe":        "✅",
    "Medium Risk": "⚠️",
    "High Risk":   "🚨",
    "Emergency":   "🆘",
}

CATEGORICAL_COLS = [
    "crowd_density", "lighting_condition", "phone_motion_status",
    "network_status", "weather_condition",
]


# ── rule-based safety suggestions ───────────────────────────────────────────

def generate_suggestions(input_dict: dict, risk_level: str) -> list[str]:
    tips = []
    d    = input_dict

    if d.get("panic_button") == 1:
        tips.append("🆘 Panic button pressed — trigger emergency alert immediately.")

    if d.get("is_night") == 1 and d.get("is_alone") == 1:
        tips.append("🌙 You are alone at night — share live location with a trusted contact.")

    if d.get("crowd_density") == "low" and d.get("lighting_condition") == "dark":
        tips.append("💡 Area is dark and uncrowded — move toward a brighter, busier place.")

    if d.get("crime_area_score", 0) > 0.6:
        tips.append("🔴 High crime score — avoid this area and inform a trusted contact.")

    if d.get("battery_level", 100) < 20:
        tips.append("🔋 Battery low — inform a trusted contact before your phone dies.")

    if d.get("unusual_movement") == 1:
        tips.append("📳 Unusual movement detected — are you okay? Trigger alert if needed.")

    if d.get("nearest_trusted_contact_distance", 99) > 15:
        tips.append("📍 Nearest trusted contact is far — consider heading to a safe zone.")

    if d.get("phone_motion_status") == "erratic":
        tips.append("⚡ Erratic phone movement detected — check-in with a contact.")

    if d.get("walking_speed", 0) > 6:
        tips.append("🏃 High walking speed detected — stay alert and check surroundings.")

    if d.get("distance_from_safe_zone", 0) > 10:
        tips.append("🏠 You are far from a safe zone — navigate toward one now.")

    if d.get("weather_condition") in ("foggy", "stormy"):
        tips.append("🌧️ Poor weather — reduce travel and stay in a secure location.")

    # level-specific defaults
    if not tips:
        defaults = {
            "Safe":        ["✅ You are in a safe area. Stay aware of your surroundings."],
            "Medium Risk": ["⚠️ Be alert. Consider sharing your location with a trusted contact."],
            "High Risk":   ["🚨 Move to a crowded or well-lit place immediately."],
            "Emergency":   ["🆘 Emergency detected — call emergency services and alert contacts now."],
        }
        tips = defaults.get(risk_level, [])

    return tips


def recommended_action(risk_level: str) -> str:
    actions = {
        "Safe":        "Continue your journey. Stay aware.",
        "Medium Risk": "Share your live location. Keep your phone charged.",
        "High Risk":   "Move to safety immediately. Call a trusted contact.",
        "Emergency":   "Call 112 / local emergency services NOW. Alert all trusted contacts.",
    }
    return actions.get(risk_level, "Stay alert.")


# ── feature engineering (mirrors data_generator) ────────────────────────────

def _build_engineered(d: dict) -> dict:
    crowd_map  = {"low": 1.0, "medium": 0.5, "high": 0.1}
    motion_map = {"stationary": 0.1, "walking": 0.2, "running": 0.6, "erratic": 1.0}

    night_risk_score    = d["is_night"] * 0.6 + (d["lighting_condition"] == "dark") * 0.4
    isolation_score     = d["is_alone"] * 0.6 + crowd_map.get(d["crowd_density"], 0.5) * 0.4
    movement_risk_score = motion_map.get(d["phone_motion_status"], 0.2)
    location_risk_score = (d["crime_area_score"] * 0.5 +
                           (d["distance_from_safe_zone"] / 30) * 0.3 +
                           (d["distance_from_home"] / 60) * 0.2)
    emergency_score     = (d["panic_button"] * 0.5 +
                           d["unusual_movement"] * 0.3 +
                           (1 if d["battery_level"] < 20 else 0) * 0.2)
    total_risk_score    = np.clip(
        night_risk_score * 0.20 +
        isolation_score  * 0.20 +
        movement_risk_score * 0.15 +
        location_risk_score * 0.25 +
        emergency_score  * 0.20, 0, 1
    )
    return {
        "night_risk_score":    round(night_risk_score, 4),
        "isolation_score":     round(isolation_score, 4),
        "movement_risk_score": round(movement_risk_score, 4),
        "location_risk_score": round(location_risk_score, 4),
        "emergency_score":     round(emergency_score, 4),
        "total_risk_score":    round(float(total_risk_score), 4),
    }


# ── main predictor class ─────────────────────────────────────────────────────

class SafetyPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model      = None
        self.encoders   = None
        self.scaler     = None
        self.features   = None
        self._load()

    def _load(self):
        mp = Path(self.models_dir)
        self.model    = joblib.load(mp / "safety_model.pkl")
        self.encoders = joblib.load(mp / "encoders.pkl")
        self.scaler   = joblib.load(mp / "scaler.pkl")

    def predict(self, input_dict: dict) -> dict:
        """
        Parameters
        ----------
        input_dict : dict with raw feature values

        Returns
        -------
        dict with risk_level, probabilities, score, suggestions, action
        """
        d    = dict(input_dict)
        eng  = _build_engineered(d)
        d.update(eng)

        # build DataFrame
        row = {k: [v] for k, v in d.items() if k not in ("user_id",)}
        df  = pd.DataFrame(row)

        # encode categoricals
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.encoders:
                df[col] = self.encoders[col].transform(df[col].astype(str))

        # align columns to training feature order
        if hasattr(self.model, "feature_names_in_"):
            cols = list(self.model.feature_names_in_)
        else:
            cols = [c for c in df.columns]

        df = df.reindex(columns=cols, fill_value=0)

        X_scaled = self.scaler.transform(df)

        pred_idx  = self.model.predict(X_scaled)[0]
        proba     = self.model.predict_proba(X_scaled)[0]
        risk_level = LABEL_NAMES[pred_idx]
        risk_score = round(eng["total_risk_score"] * 100, 1)

        probabilities = {LABEL_NAMES[i]: round(float(p) * 100, 1) for i, p in enumerate(proba)}
        suggestions   = generate_suggestions(input_dict, risk_level)
        action        = recommended_action(risk_level)

        return {
            "risk_level":        risk_level,
            "risk_score":        risk_score,
            "color":             LABEL_COLORS[risk_level],
            "icon":              LABEL_ICONS[risk_level],
            "probabilities":     probabilities,
            "suggestions":       suggestions,
            "recommended_action": action,
            "engineered":        eng,
        }


# ── CLI quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = SafetyPredictor("models")
    sample = {
        "latitude": 28.6139, "longitude": 77.2090,
        "hour": 23, "day_type": 1, "is_night": 1,
        "is_alone": 1, "crowd_density": "low",
        "lighting_condition": "dark", "crime_area_score": 0.75,
        "phone_motion_status": "erratic", "walking_speed": 5.5,
        "distance_from_home": 12.0, "distance_from_safe_zone": 8.0,
        "battery_level": 18, "network_status": "2G",
        "weather_condition": "foggy", "panic_button": 0,
        "unusual_movement": 1, "time_spent_at_location": 45,
        "nearest_trusted_contact_distance": 20.0,
    }
    result = predictor.predict(sample)
    print(f"\nRisk Level : {result['icon']} {result['risk_level']}")
    print(f"Risk Score : {result['risk_score']} / 100")
    print(f"Probabilities: {result['probabilities']}")
    print("Suggestions:")
    for s in result["suggestions"]:
        print(f"  {s}")
    print(f"Action: {result['recommended_action']}")
