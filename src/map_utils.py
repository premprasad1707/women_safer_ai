"""
SafeHer AI — Map Utilities
Builds Folium maps with user location, unsafe zones, safe places,
trusted contacts, hospitals, and police stations.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster
import random
import math

# ── colour palette ────────────────────────────────────────────────────────────
RISK_COLORS = {
    "Safe":        "#22c55e",
    "Medium Risk": "#f59e0b",
    "High Risk":   "#ef4444",
    "Emergency":   "#7c3aed",
}

# ── sample POI data (relative offsets from user lat/lon) ────────────────────

def _offset(lat, lon, dlat, dlon):
    return round(lat + dlat, 5), round(lon + dlon, 5)


def generate_sample_pois(center_lat: float, center_lon: float) -> dict:
    """Generate plausible nearby POIs for demonstration."""
    # rebuild properly (the destructured *_offset caused a syntax issue previously)
    def off(d1, d2):
        return round(center_lat + d1, 5), round(center_lon + d2, 5)

    safe_zones = [
        {"name": "City Hospital",        **dict(zip(["lat","lon"], off(0.012, -0.008))),  "type": "hospital"},
        {"name": "Police Station North", **dict(zip(["lat","lon"], off(-0.009, 0.015))), "type": "police"},
        {"name": "Metro Station Hub",    **dict(zip(["lat","lon"], off(0.005, 0.020))),  "type": "safe"},
        {"name": "24/7 Pharmacy",        **dict(zip(["lat","lon"], off(-0.015, -0.005))),"type": "safe"},
        {"name": "Community Centre",     **dict(zip(["lat","lon"], off(0.018, 0.012))),  "type": "safe"},
    ]
    unsafe_zones = [
        {"name": "Isolated Underpass",   **dict(zip(["lat","lon"], off(0.022, 0.030))),  "score": 0.82},
        {"name": "Abandoned Warehouse",  **dict(zip(["lat","lon"], off(-0.025, -0.018))),"score": 0.74},
        {"name": "Dimly Lit Alley",      **dict(zip(["lat","lon"], off(0.010, -0.025))), "score": 0.65},
    ]
    contacts = [
        {"name": "Priya Sharma", **dict(zip(["lat","lon"], off(0.003, -0.004))), "rel": "Friend"},
        {"name": "Ravi Kumar",   **dict(zip(["lat","lon"], off(-0.006, 0.008))), "rel": "Brother"},
        {"name": "Sunita Devi",  **dict(zip(["lat","lon"], off(0.014, 0.011))),  "rel": "Mother"},
    ]

    return {"safe_zones": safe_zones, "unsafe_zones": unsafe_zones, "contacts": contacts}


# ── icon factories ────────────────────────────────────────────────────────────

def _icon(color: str, icon: str, prefix: str = "fa") -> folium.Icon:
    return folium.Icon(color=color, icon=icon, prefix=prefix)


# ── main map builder ──────────────────────────────────────────────────────────

def build_safety_map(
    user_lat: float,
    user_lon: float,
    risk_level: str = "Safe",
    pois: dict | None = None,
    zoom: int = 14,
) -> folium.Map:
    """
    Build an interactive Folium safety map.

    Parameters
    ----------
    user_lat   : user's latitude
    user_lon   : user's longitude
    risk_level : one of Safe / Medium Risk / High Risk / Emergency
    pois       : dict returned by generate_sample_pois(); auto-generated if None
    zoom       : initial zoom level

    Returns
    -------
    folium.Map object (use st_folium to render in Streamlit)
    """
    if pois is None:
        pois = generate_sample_pois(user_lat, user_lon)

    risk_color = RISK_COLORS.get(risk_level, "#22c55e")

    m = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
    )

    # ── user marker ──────────────────────────────────────────────────────
    folium.Marker(
        location=[user_lat, user_lon],
        popup=folium.Popup(
            f"<b>📍 You are here</b><br>Risk: <b>{risk_level}</b>", max_width=200
        ),
        tooltip="Your Location",
        icon=folium.Icon(color="red" if "Risk" in risk_level or "Emergency" in risk_level else "green",
                         icon="user", prefix="fa"),
    ).add_to(m)

    # ── user accuracy circle ─────────────────────────────────────────────
    folium.Circle(
        location=[user_lat, user_lon],
        radius=300,
        color=risk_color,
        fill=True,
        fill_opacity=0.08,
        tooltip=f"Safety radius — {risk_level}",
    ).add_to(m)

    # ── safe zones ───────────────────────────────────────────────────────
    safe_group = folium.FeatureGroup("🟢 Safe Zones / Services")
    for z in pois.get("safe_zones", []):
        icon_name = "hospital-o" if z["type"] == "hospital" else ("shield" if z["type"] == "police" else "home")
        icon_col  = "blue"       if z["type"] == "hospital" else ("darkblue" if z["type"] == "police" else "green")
        folium.Marker(
            location=[z["lat"], z["lon"]],
            popup=folium.Popup(f"<b>{z['name']}</b><br>Type: {z['type'].title()}", max_width=200),
            tooltip=z["name"],
            icon=folium.Icon(color=icon_col, icon=icon_name, prefix="fa"),
        ).add_to(safe_group)
    safe_group.add_to(m)

    # ── unsafe zones ─────────────────────────────────────────────────────
    unsafe_group = folium.FeatureGroup("🔴 Unsafe Zones")
    for z in pois.get("unsafe_zones", []):
        radius = int(z["score"] * 600)
        folium.Circle(
            location=[z["lat"], z["lon"]],
            radius=radius,
            color="#ef4444",
            fill=True,
            fill_opacity=0.25,
            tooltip=f"⚠️ {z['name']}  Crime Score: {z['score']}",
        ).add_to(unsafe_group)
        folium.Marker(
            location=[z["lat"], z["lon"]],
            popup=folium.Popup(f"<b>⚠️ {z['name']}</b><br>Crime Score: {z['score']:.2f}", max_width=200),
            tooltip=z["name"],
            icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
        ).add_to(unsafe_group)
    unsafe_group.add_to(m)

    # ── trusted contacts ─────────────────────────────────────────────────
    contact_group = folium.FeatureGroup("💜 Trusted Contacts")
    for c in pois.get("contacts", []):
        folium.Marker(
            location=[c["lat"], c["lon"]],
            popup=folium.Popup(f"<b>{c['name']}</b><br>{c['rel']}", max_width=200),
            tooltip=f"👤 {c['name']}",
            icon=folium.Icon(color="purple", icon="heart", prefix="fa"),
        ).add_to(contact_group)
        # dashed line from user to contact
        folium.PolyLine(
            locations=[[user_lat, user_lon], [c["lat"], c["lon"]]],
            color="#a855f7",
            weight=1.5,
            dash_array="6 4",
            opacity=0.6,
        ).add_to(contact_group)
    contact_group.add_to(m)

    # ── heatmap overlay (unsafe zone scores) ─────────────────────────────
    heat_data = [
        [z["lat"], z["lon"], z["score"]]
        for z in pois.get("unsafe_zones", [])
    ]
    if heat_data:
        HeatMap(heat_data, radius=40, blur=30, max_zoom=13,
                gradient={"0.3": "blue", "0.6": "orange", "1.0": "red"}).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    return m


if __name__ == "__main__":
    m = build_safety_map(28.6139, 77.2090, "High Risk")
    m.save("test_map.html")
    print("✅  Map saved → test_map.html")
