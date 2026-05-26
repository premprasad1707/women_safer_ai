"""
SafeHer AI — General Utilities
Shared helpers used across the application.
"""

import json
import math
import random
import datetime
from pathlib import Path


# ── risk display helpers ──────────────────────────────────────────────────────

RISK_PALETTE = {
    "Safe":        {"color": "#22c55e", "bg": "#052e16", "icon": "✅", "badge": "safe"},
    "Medium Risk": {"color": "#f59e0b", "bg": "#451a03", "icon": "⚠️",  "badge": "medium"},
    "High Risk":   {"color": "#ef4444", "bg": "#450a0a", "icon": "🚨", "badge": "high"},
    "Emergency":   {"color": "#a855f7", "bg": "#2e1065", "icon": "🆘", "badge": "emergency"},
}

EMERGENCY_NUMBERS = {
    "Police":        "100",
    "Ambulance":     "108",
    "Fire":          "101",
    "Women Helpline": "1091",
    "Emergency":     "112",
}


def get_risk_style(risk_level: str) -> dict:
    return RISK_PALETTE.get(risk_level, RISK_PALETTE["Safe"])


def risk_to_score_range(risk_level: str) -> tuple[int, int]:
    """Return (lo, hi) score range for a given risk level."""
    return {
        "Safe":        (0, 34),
        "Medium Risk": (35, 54),
        "High Risk":   (55, 74),
        "Emergency":   (75, 100),
    }.get(risk_level, (0, 34))


# ── distance helpers ──────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R  = 6371
    φ1 = math.radians(lat1); φ2 = math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a  = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 2)


# ── time helpers ──────────────────────────────────────────────────────────────

def is_nighttime(hour: int) -> bool:
    return hour < 6 or hour >= 20


def friendly_time() -> str:
    return datetime.datetime.now().strftime("%d %b %Y, %H:%M")


# ── model & data file checks ──────────────────────────────────────────────────

def model_exists(models_dir: str = "models") -> bool:
    return (Path(models_dir) / "safety_model.pkl").exists()


def dataset_exists(data_dir: str = "data") -> bool:
    return (Path(data_dir) / "safety_dataset.csv").exists()


def load_metrics(models_dir: str = "models") -> dict | None:
    p = Path(models_dir) / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── CSS helpers (used in app.py) ──────────────────────────────────────────────

def metric_card_html(label: str, value: str, color: str = "#6366f1", sub: str = "") -> str:
    return f"""
    <div style="background:linear-gradient(135deg,#1e1b2e,#16131f);
                border:1px solid {color}44; border-left:4px solid {color};
                border-radius:12px; padding:18px 20px; margin:6px 0;">
        <div style="color:#9ca3af; font-size:0.75rem; text-transform:uppercase;
                    letter-spacing:0.1em;">{label}</div>
        <div style="color:{color}; font-size:1.8rem; font-weight:700; margin:4px 0;">{value}</div>
        <div style="color:#6b7280; font-size:0.78rem;">{sub}</div>
    </div>
    """


def alert_banner_html(risk_level: str, message: str) -> str:
    style = get_risk_style(risk_level)
    return f"""
    <div style="background:{style['bg']}; border:1px solid {style['color']}55;
                border-left:5px solid {style['color']}; border-radius:10px;
                padding:16px 20px; margin:10px 0; color:{style['color']};">
        <span style="font-size:1.3rem;">{style['icon']}</span>&nbsp;
        <strong>{risk_level.upper()}</strong> — {message}
    </div>
    """


# ── fake live telemetry (for demo dashboard) ─────────────────────────────────

def fake_live_reading(base_score: float = 0.45) -> dict:
    """Simulate a live sensor reading with slight jitter."""
    noise = random.uniform(-0.05, 0.05)
    score = round(max(0.0, min(1.0, base_score + noise)), 3)
    hour  = datetime.datetime.now().hour
    return {
        "risk_score_raw": score,
        "risk_score_pct": round(score * 100, 1),
        "is_night":       int(is_nighttime(hour)),
        "hour":           hour,
        "timestamp":      friendly_time(),
    }
