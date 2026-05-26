"""
SafeHer AI — Main Streamlit Application
A professional AI-powered personal safety prediction system.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import datetime, json, time, random

# ── page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SafeHer AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── local imports ─────────────────────────────────────────────────────────────
from src.utils        import (get_risk_style, metric_card_html, alert_banner_html,
                               model_exists, dataset_exists, load_metrics,
                               EMERGENCY_NUMBERS, fake_live_reading)
from src.alert_system import (Contact, trigger_alert, get_alert_history,
                               DEFAULT_CONTACTS, build_sms_alert, build_whatsapp_alert,
                               load_contacts, save_contacts)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0914;
    color: #e2e0f0;
}
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #110e1e 0%, #0d0b18 100%) !important;
    border-right: 1px solid #2a2640;
}
section[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
section[data-testid="stSidebar"] label { font-size:0.9rem !important; color:#c4b8f0 !important; }

/* main area */
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1630, #130f24);
    border: 1px solid #2d2750;
    border-radius: 14px;
    padding: 14px 18px;
}
div[data-testid="stMetric"] label { color: #9490c0 !important; font-size:0.78rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e2e0f0 !important; }

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px #7c3aed44;
}

/* selectbox / slider */
.stSelectbox > div, .stSlider { color: #c4b8f0; }

/* dividers */
hr { border-color: #2a2640 !important; }

/* alerts */
.stAlert { border-radius: 10px; }

/* section headers */
.sh-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sh-sub { color: #7c7499; font-size: 0.88rem; margin-bottom: 1.2rem; }

/* risk badge */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
}

/* contact card */
.contact-card {
    background: linear-gradient(135deg, #1a1630, #130f24);
    border: 1px solid #2d2750;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
}

/* battery widget */
.battery-outer {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg,#1a1630,#130f24);
    border: 1px solid #2d2750;
    border-radius: 14px;
    padding: 12px 18px;
}
.battery-shell {
    position: relative;
    width: 54px;
    height: 26px;
    border: 2.5px solid #6b6488;
    border-radius: 5px;
    background: #0f0b1e;
    overflow: hidden;
}
.battery-shell::after {
    content: '';
    position: absolute;
    right: -8px;
    top: 50%;
    transform: translateY(-50%);
    width: 5px;
    height: 12px;
    background: #6b6488;
    border-radius: 0 3px 3px 0;
}
.battery-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease, background 0.4s;
}
@keyframes pulse-red {
    0%,100%{opacity:1;} 50%{opacity:0.4;}
}
.battery-low { animation: pulse-red 1s infinite; }

/* emergency pulsing panel */
@keyframes emergency-glow {
    0%,100%{box-shadow:0 0 12px #ef444460;}
    50%{box-shadow:0 0 32px #ef4444cc, 0 0 60px #ef444440;}
}
.emergency-panel {
    animation: emergency-glow 1.2s ease-in-out infinite;
}
@keyframes ping-dot {
    0%{transform:scale(1);opacity:1;}
    80%,100%{transform:scale(2.2);opacity:0;}
}
.ping-dot {
    position:relative;
    display:inline-block;
    width:12px;height:12px;
    border-radius:50%;
    background:#ef4444;
    vertical-align:middle;
}
.ping-dot::before {
    content:'';
    position:absolute;
    inset:0;
    border-radius:50%;
    background:#ef4444;
    animation:ping-dot 1.2s ease-in-out infinite;
}

/* Floating action button */
.floating-sos {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #ef4444;
    color: white !important;
    border-radius: 50%;
    width: 70px;
    height: 70px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.6);
    z-index: 99999;
    text-decoration: none;
    animation: emergency-glow 1.5s ease-in-out infinite;
    transition: transform 0.2s;
}
.floating-sos:hover {
    transform: scale(1.1);
}
</style>

<a href="?nav=Emergency+Alert+Simulation" target="_self" class="floating-sos">🆘</a>
""", unsafe_allow_html=True)


# ── Battery detection via JavaScript Battery API ────────────────────────────
import streamlit.components.v1 as components

BATTERY_COMPONENT_HTML = """
<div id="batt-widget" style="display:none;"></div>
<script>
(function() {
    function sendBattery(level, charging) {
        var pct = Math.round(level * 100);
        var msg = JSON.stringify({battery_pct: pct, charging: charging});
        // Store in sessionStorage so Streamlit can read via query param hack
        // We use a hidden iframe trick to send data to the parent
        try {
            window.parent.postMessage({type:'battery_update', battery_pct: pct, charging: charging}, '*');
        } catch(e){}
        document.getElementById('batt-widget').innerText = msg;
    }
    if (navigator.getBattery) {
        navigator.getBattery().then(function(bat){
            sendBattery(bat.level, bat.charging);
            bat.addEventListener('levelchange', function(){ sendBattery(bat.level, bat.charging); });
            bat.addEventListener('chargingchange', function(){ sendBattery(bat.level, bat.charging); });
        }).catch(function(){ sendBattery(0.99, true); });
    } else {
        sendBattery(0.99, true);
    }
})();
</script>
"""

# Initialize battery state
if "battery_pct" not in st.session_state:
    st.session_state.battery_pct = 85
if "battery_charging" not in st.session_state:
    st.session_state.battery_charging = False


def _battery_color(pct: int) -> str:
    if pct <= 15:
        return "#ef4444"
    elif pct <= 40:
        return "#f59e0b"
    return "#22c55e"


def _battery_icon(pct: int, charging: bool) -> str:
    if charging:
        return "⚡"
    if pct <= 15:
        return "🪫"
    if pct <= 40:
        return "🔋"
    return "🔋"


def render_battery_widget(pct: int, charging: bool, label: str = "") -> str:
    color  = _battery_color(pct)
    width  = max(4, pct)
    low_cls = "battery-low" if pct <= 15 and not charging else ""
    charge_indicator = " ⚡ Charging" if charging else ""
    warn = f"""<div style="color:#ef4444;font-size:0.72rem;margin-top:4px;font-weight:600;">
        ⚠️ Low Battery — Emergency call readiness reduced!</div>""" if pct <= 15 and not charging else ""
    return f"""
    <div class="battery-outer">
        <div>
            <div style="color:#9490c0;font-size:0.7rem;text-transform:uppercase;letter-spacing:.08em;">{label or 'Battery'}</div>
            <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
                <div class="battery-shell {low_cls}" style="margin-right:10px;">
                    <div class="battery-fill" style="width:{width}%;background:{color};"></div>
                </div>
                <span style="font-family:Syne,sans-serif;font-weight:700;font-size:1.15rem;color:{color};">{pct}%</span>
                <span style="font-size:0.8rem;color:#6b6488;">{charge_indicator}</span>
            </div>
            {warn}
        </div>
    </div>
    """


def render_auto_call_panel(contact_name: str, contact_phone: str, reason: str = "Emergency") -> str:
    phone_clean = contact_phone.replace("-","").replace("+","").replace(" ","")
    return f"""
    <div class="emergency-panel" style="background:linear-gradient(135deg,#1a0a0a,#2d0808);
        border:2px solid #ef4444;border-radius:16px;padding:24px 28px;margin:16px 0;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
            <span class="ping-dot"></span>
            <span style="font-family:Syne,sans-serif;font-weight:800;font-size:1.25rem;color:#ef4444;">EMERGENCY — AUTO CALLING</span>
        </div>
        <div style="color:#c4b8f0;font-size:0.9rem;margin-bottom:12px;">
            Automatically calling <strong style="color:#fff;">{contact_name}</strong> — your Priority #1 contact<br>
            <span style="color:#6b6488;font-size:0.82rem;">Reason: {reason}</span>
        </div>
        <a href="tel:+{phone_clean}" id="auto-call-link"
            style="display:inline-flex;align-items:center;gap:10px;
                   background:linear-gradient(135deg,#dc2626,#b91c1c);
                   color:#fff;text-decoration:none;border-radius:12px;
                   padding:12px 28px;font-family:Syne,sans-serif;font-weight:700;
                   font-size:1rem;box-shadow:0 4px 20px #ef444460;">
            📞 Call {contact_name} Now (+{phone_clean})
        </a>
        <script>
        // Auto-click the call link after 1.5 seconds
        setTimeout(function(){{ document.getElementById('auto-call-link').click(); }}, 1500);
        </script>
    </div>
    """


# ── session state bootstrap ───────────────────────────────────────────────────
if "contacts" not in st.session_state:
    st.session_state.contacts = load_contacts()
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []
if "live_scores" not in st.session_state:
    st.session_state.live_scores = [random.uniform(0.2, 0.6) for _ in range(20)]
if "model_ready" not in st.session_state:
    st.session_state.model_ready = model_exists()
if "battery_pct" not in st.session_state:
    st.session_state.battery_pct = 85
if "battery_charging" not in st.session_state:
    st.session_state.battery_charging = False
if "emergency_call_triggered" not in st.session_state:
    st.session_state.emergency_call_triggered = False

# Inject battery detection component (runs once per page)
components.html(BATTERY_COMPONENT_HTML, height=0)

# Handle query params for navigation if clicked via floating button
query_params = st.query_params
nav_default = "🏠  Home"
if "nav" in query_params and query_params["nav"] == "Emergency Alert Simulation":
    nav_default = "🆘  Emergency Alert Simulation"

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:2.4rem;'>🛡️</div>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800;
                    background:linear-gradient(90deg,#a78bfa,#ec4899);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            SafeHer AI
        </div>
        <div style='color:#6b6488; font-size:0.75rem; margin-top:2px;'>
            Personal Safety Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    options = [
        "🏠  Home",
        "🔮  Safety Prediction",
        "📡  Live Risk Dashboard",
        "👥  Trusted Contacts",
        "🗺️  Unsafe Zone Map",
        "📊  Model Performance",
        "🆘  Emergency Alert Simulation",
    ]
    
    index = options.index(nav_default) if nav_default in options else 0
    
    page = st.radio(
        "Navigate",
        options,
        index=index,
        label_visibility="collapsed",
    )
    
    if page != nav_default and "nav" in query_params:
        st.query_params.clear()

    st.markdown("---")

    # ── Battery indicator in sidebar ─────────────────────────────────────────
    batt_pct     = st.session_state.battery_pct
    batt_charge  = st.session_state.battery_charging
    batt_color   = _battery_color(batt_pct)
    batt_width   = max(4, batt_pct)
    batt_low_cls = "battery-low" if batt_pct <= 15 and not batt_charge else ""
    batt_icon    = "⚡" if batt_charge else ("🪫" if batt_pct <= 15 else "🔋")
    st.markdown(
        f"""<div style='padding:6px 0;'>
        <div style='color:#6b6488;font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;
                    margin-bottom:6px;'>Device Battery</div>
        <div style='display:flex;align-items:center;gap:8px;'>
            <span style='font-size:1rem;'>{batt_icon}</span>
            <div style='flex:1;height:8px;background:#1e1b2e;border-radius:4px;overflow:hidden;'>
                <div class='{batt_low_cls}' style='height:100%;width:{batt_width}%;background:{batt_color};
                            border-radius:4px;transition:width .4s;'></div>
            </div>
            <span style='color:{batt_color};font-size:0.82rem;font-weight:700;'>{batt_pct}%</span>
        </div>
        {'<div style="color:#ef4444;font-size:0.68rem;margin-top:4px;">⚠️ Low Battery!</div>' if batt_pct <= 15 and not batt_charge else ''}
        </div>""",
        unsafe_allow_html=True,
    )

    # Manual battery override (since JS postMessage can't directly update session_state)
    with st.expander("🔋 Set Battery %", expanded=False):
        manual_batt = st.slider("Battery Level", 0, 100, st.session_state.battery_pct, key="manual_batt_slider")
        if manual_batt != st.session_state.battery_pct:
            st.session_state.battery_pct = manual_batt
            st.rerun()

    st.markdown("---")

    # status indicator
    status_color = "#22c55e" if st.session_state.model_ready else "#ef4444"
    status_text  = "Model Ready" if st.session_state.model_ready else "Model Not Trained"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;color:#9490c0;font-size:0.8rem;">'
        f'<div style="width:8px;height:8px;border-radius:50%;background:{status_color};'
        f'box-shadow:0 0 6px {status_color};"></div>{status_text}</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.model_ready:
        if st.button("⚙️ Train Model Now"):
            with st.spinner("Training models … this may take 60–90 seconds."):
                try:
                    from src.data_generator import generate_dataset
                    from src.preprocessing  import preprocess
                    from src.train_model    import train_all_models

                    df = generate_dataset(10_000, "data/safety_dataset.csv")
                    X_train, X_test, y_train, y_test, features, enc, sc = preprocess(df, fit=True)
                    results, best_name, _ = train_all_models(
                        X_train, X_test, y_train, y_test, features
                    )
                    st.session_state.model_ready = True
                    st.success(f"✅ Best: {best_name}")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("---")
    st.markdown(
        '<div style="color:#4a4466; font-size:0.7rem; text-align:center;">'
        'SafeHer AI v1.0 · For demonstration only<br>'
        'Not a substitute for emergency services</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    # hero
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1230 0%,#0f0b1e 100%);
                border:1px solid #2d2750; border-radius:20px; padding:40px 48px; margin-bottom:24px;">
        <div style="font-family:Syne,sans-serif; font-size:2.8rem; font-weight:800; line-height:1.15;
                    background:linear-gradient(90deg,#a78bfa 0%,#ec4899 50%,#f97316 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Your AI-Powered Safety <br>Guardian 🛡️
        </div>
        <div style="color:#9490c0; font-size:1.05rem; margin-top:14px; max-width:620px; line-height:1.7;">
            SafeHer AI uses machine learning to assess your real-time safety risk.
            It analyses your location, time, movement behaviour, crowd density,
            and proximity to trusted contacts — then gives you instant, actionable guidance.
        </div>
        <div style="margin-top:24px; display:flex; gap:12px; flex-wrap:wrap;">
            <span style="background:#7c3aed22; border:1px solid #7c3aed; color:#a78bfa;
                         border-radius:50px; padding:6px 18px; font-size:0.82rem;">
                🤖 ML-Powered Prediction
            </span>
            <span style="background:#ec489922; border:1px solid #ec4899; color:#f472b6;
                         border-radius:50px; padding:6px 18px; font-size:0.82rem;">
                📍 Live Location Awareness
            </span>
            <span style="background:#22c55e22; border:1px solid #22c55e; color:#4ade80;
                         border-radius:50px; padding:6px 18px; font-size:0.82rem;">
                🔔 Instant Alert System
            </span>
            <span style="background:#f59e0b22; border:1px solid #f59e0b; color:#fbbf24;
                         border-radius:50px; padding:6px 18px; font-size:0.82rem;">
                🗺️ Unsafe Zone Mapping
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Battery Status Card on Home ──────────────────────────────────────────
    batt_pct    = st.session_state.battery_pct
    batt_charge = st.session_state.battery_charging
    home_b1, home_b2, home_b3 = st.columns([2, 1, 1])
    with home_b1:
        st.markdown(render_battery_widget(batt_pct, batt_charge, label="Your Device Battery"), unsafe_allow_html=True)
    with home_b2:
        batt_risk = "Critical 🔴" if batt_pct <= 15 else ("Low ⚠️" if batt_pct <= 40 else "Good ✅")
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1630,#130f24);border:1px solid #2d2750;'
            f'border-radius:14px;padding:14px 18px;height:100%;">' 
            f'<div style="color:#9490c0;font-size:0.7rem;text-transform:uppercase;letter-spacing:.08em;">Battery Status</div>'
            f'<div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:{_battery_color(batt_pct)};margin-top:6px;">{batt_risk}</div>'
            f'</div>', unsafe_allow_html=True
        )
    with home_b3:
        prem_contact = next((c for c in st.session_state.contacts if "Prem" in c.name), None)
        prem_num = prem_contact.phone if prem_contact else "+91-6370959392"
        prem_clean = prem_num.replace("-","").replace("+","").replace(" ","")
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a0a0a,#2d0808);border:1px solid #ef444466;'
            f'border-radius:14px;padding:14px 18px;text-align:center;">'
            f'<div style="color:#9490c0;font-size:0.7rem;text-transform:uppercase;letter-spacing:.08em;">Emergency Contact</div>'
            f'<a href="tel:+{prem_clean}" style="text-decoration:none;">'
            f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:#ef4444;margin-top:6px;">📞 Call Prem</div>'
            f'<div style="color:#6b6488;font-size:0.72rem;margin-top:2px;">{prem_num}</div>'
            f'</a></div>', unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # Low battery warning banner
    if batt_pct <= 20 and not batt_charge:
        st.markdown(
            f'<div class="emergency-panel" style="background:linear-gradient(135deg,#1c0a05,#2d1505);'
            f'border:2px solid #f59e0b;border-radius:14px;padding:16px 22px;margin-bottom:16px;">'
            f'<span class="ping-dot" style="background:#f59e0b;"></span>'
            f'&nbsp;&nbsp;<strong style="color:#f59e0b;font-family:Syne,sans-serif;">'
            f'⚠️ LOW BATTERY ({batt_pct}%)</strong> — '
            f'<span style="color:#c4b8f0;">Your battery is critically low. Emergency alerts may not be sent if device dies!</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── quick emergency actions ──
    st.markdown('<div class="sh-title" style="font-size:1.4rem;">🚨 Quick Action / Emergency</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh-sub" style="margin-bottom:0.8rem;">Instantly alert your top trusted contact without running a full prediction.</div>', unsafe_allow_html=True)
    
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        btn_panic = st.button("🆘 Panic / Emergency", use_container_width=True)
    with qa2:
        btn_harass = st.button("⚠️ Harassment", use_container_width=True)
    with qa3:
        btn_stalk = st.button("👣 Stalking / Followed", use_container_width=True)
    with qa4:
        btn_sus = st.button("👁️ Suspicious Activity", use_container_width=True)
        
    triggered_action = None
    if btn_panic: triggered_action = "Panic / Emergency"
    elif btn_harass: triggered_action = "Harassment"
    elif btn_stalk: triggered_action = "Stalking / Being Followed"
    elif btn_sus: triggered_action = "Suspicious Activity"
    
    if triggered_action:
        if not st.session_state.contacts:
            st.error(f"🚨 '{triggered_action}' triggered, but no trusted contacts found!")
        else:
            from src.alert_system import analyze_call_history_and_rank, trigger_alert, get_alert_history
            ranked_contacts = analyze_call_history_and_rank(st.session_state.contacts)
            top_contact = ranked_contacts[0]
            
            alert_result = trigger_alert(
                [top_contact],
                f"Emergency: {triggered_action}",
                [f"User reported {triggered_action}.", "Please contact them immediately."],
                lat=28.6139,
                lon=77.2090,
            )
            st.session_state.alert_history.extend(get_alert_history()[-3:])

            # ── Auto-call Prem (Priority #1 Emergency Contact) ───────────────
            prem = next((c for c in st.session_state.contacts if "Prem" in c.name), top_contact)
            st.markdown(render_auto_call_panel(prem.name, prem.phone, triggered_action), unsafe_allow_html=True)
            st.success(f"🚨 **{triggered_action.upper()} ALERT SENT** to **{alert_result['contact'].name}** — Open the links below to send!")

            import urllib.parse
            phone = top_contact.phone.replace("-", "").replace("+", "").replace(" ", "")
            sms_body = urllib.parse.quote(alert_result["sms"])
            wa_body  = urllib.parse.quote(alert_result["whatsapp"])

            lc1, lc2 = st.columns(2)
            with lc1:
                st.link_button("💬 Send SMS Now", f"sms:{phone}?body={sms_body}", use_container_width=True)
            with lc2:
                st.link_button("💚 Send WhatsApp Now", f"https://wa.me/{phone}?text={wa_body}", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # feature cards
    cols = st.columns(4)
    features_info = [
        ("🔮", "Safety Prediction", "Enter your current context and get an instant risk assessment powered by Random Forest, XGBoost & more.", "#7c3aed"),
        ("📡", "Live Dashboard",    "Real-time animated risk score monitor with continuous updates and historical trend chart.", "#ec4899"),
        ("👥", "Trusted Contacts",  "Add emergency contacts with priority levels. The system automatically selects the nearest one during alerts.", "#22c55e"),
        ("🗺️", "Unsafe Zone Map",   "Interactive Folium map showing your location, crime hotspots, safe places, and trusted contacts.", "#f59e0b"),
    ]
    for col, (icon, title, desc, color) in zip(cols, features_info):
        col.markdown(
            f"""<div style="background:linear-gradient(135deg,#1a1630,#130f24);
                border:1px solid {color}44; border-top:3px solid {color};
                border-radius:14px; padding:22px; height:200px;">
                <div style="font-size:1.6rem; margin-bottom:10px;">{icon}</div>
                <div style="font-family:Syne,sans-serif; font-weight:700; font-size:1rem;
                            color:{color}; margin-bottom:8px;">{title}</div>
                <div style="color:#7c7499; font-size:0.82rem; line-height:1.5;">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # emergency numbers quick reference
    st.markdown('<div class="sh-title">📞 Emergency Numbers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh-sub">Save these numbers. In an emergency, call immediately.</div>', unsafe_allow_html=True)
    ecols = st.columns(len(EMERGENCY_NUMBERS))
    colors = ["#ef4444", "#f97316", "#22c55e", "#a855f7", "#6366f1"]
    for (svc, num), col, c in zip(EMERGENCY_NUMBERS.items(), ecols, colors):
        col.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1630,#130f24);'
            f'border:1px solid {c}44;border-radius:12px;padding:14px;text-align:center;">'
            f'<div style="color:{c};font-family:Syne,sans-serif;font-weight:700;font-size:1.6rem;">{num}</div>'
            f'<div style="color:#9490c0;font-size:0.78rem;margin-top:4px;">{svc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <a href="tel:112" style="display:block; background:linear-gradient(135deg,#dc2626,#991b1b); border:none; border-radius:12px; padding:18px 24px; color:#fff; font-size:1.2rem; font-family:\'Syne\',sans-serif; font-weight:800; text-align:center; text-decoration:none; box-shadow:0 4px 15px rgba(220,38,38,0.4); margin-top:20px;">
        🚨 DIRECT CALL TO 112
    </a>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: SAFETY PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
elif "Prediction" in page:
    st.markdown('<div class="sh-title">🔮 Safety Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh-sub">Enter your current context to get an AI-powered risk assessment.</div>', unsafe_allow_html=True)

    if not st.session_state.model_ready:
        st.warning("⚠️ Model not trained yet. Go to the sidebar and click **Train Model Now**.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("**📍 Location & Emergency Trigger**")
        c1, c2 = st.columns(2)
        with c1:
            locations = {
                "Connaught Place, New Delhi": (28.6304, 77.2177),
                "Hauz Khas Village": (28.5535, 77.1936),
                "India Gate": (28.6129, 77.2295),
                "Saket": (28.5246, 77.2066),
                "Cyber Hub, Gurgaon": (28.4950, 77.0888)
            }
            location = st.selectbox("Location", list(locations.keys()))
            lat, lon = locations[location]
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            panic = st.checkbox("🆘 Panic Button / Emergency Trigger", value=True)

        submitted = st.form_submit_button("🔮 Predict Safety Level", use_container_width=True)

    if submitted:
        try:
            from src.predict import SafetyPredictor
            predictor = SafetyPredictor("models")

            input_data = {
                "latitude":  lat, "longitude": lon,
                "hour": datetime.datetime.now().hour, 
                "day_type": 0,
                "is_night": 1 if datetime.datetime.now().hour < 6 or datetime.datetime.now().hour >= 20 else 0,
                "is_alone": 1,
                "crowd_density": "low",
                "lighting_condition": "dark",
                "crime_area_score": 0.8,
                "phone_motion_status": "running",
                "walking_speed": 5.0,
                "distance_from_home": 10.0,
                "distance_from_safe_zone": 5.0,
                "battery_level": 20,
                "network_status": "4G",
                "weather_condition": "clear",
                "panic_button": int(panic),
                "unusual_movement": 1,
                "time_spent_at_location": 5,
                "nearest_trusted_contact_distance": 10.0,
            }
            result = predictor.predict(input_data)

            style = get_risk_style(result["risk_level"])
            st.markdown("<br>", unsafe_allow_html=True)

            # risk banner
            st.markdown(
                f'<div style="background:{style["bg"]};border:1px solid {style["color"]}55;'
                f'border-left:6px solid {style["color"]};border-radius:14px;'
                f'padding:24px 28px;margin:10px 0;">'
                f'<div style="font-size:2rem;">{style["icon"]}</div>'
                f'<div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;'
                f'color:{style["color"]};margin:4px 0;">{result["risk_level"]}</div>'
                f'<div style="color:#9490c0;font-size:0.9rem;">Risk Score: {result["risk_score"]} / 100</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            ra, rb = st.columns([2, 1])
            with ra:
                # probability bar chart
                prob_df = pd.DataFrame({
                    "Level": list(result["probabilities"].keys()),
                    "Probability": list(result["probabilities"].values()),
                })
                color_map = {"Safe":"#22c55e","Medium Risk":"#f59e0b","High Risk":"#ef4444","Emergency":"#a855f7"}
                fig = px.bar(prob_df, x="Level", y="Probability",
                             color="Level", color_discrete_map=color_map,
                             title="Class Probabilities (%)",
                             template="plotly_dark")
                fig.update_layout(
                    paper_bgcolor="#0f0b1e", plot_bgcolor="#0f0b1e",
                    showlegend=False, height=280,
                    font=dict(family="DM Sans"), margin=dict(t=40, b=10, l=10, r=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            with rb:
                # gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["risk_score"],
                    title={"text": "Risk Score", "font": {"size": 14, "family": "Syne"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#6b6488"},
                        "bar":  {"color": style["color"]},
                        "steps": [
                            {"range": [0, 34],  "color": "#05200a"},
                            {"range": [34, 55], "color": "#1c1005"},
                            {"range": [55, 75], "color": "#1c0505"},
                            {"range": [75, 100],"color": "#1c0828"},
                        ],
                        "threshold": {"line": {"color": style["color"], "width": 3}, "value": result["risk_score"]},
                    },
                    number={"font": {"color": style["color"], "size": 36}},
                ))
                fig_g.update_layout(
                    paper_bgcolor="#0f0b1e", font=dict(color="#9490c0", family="DM Sans"),
                    height=280, margin=dict(t=30, b=10, l=20, r=20)
                )
                st.plotly_chart(fig_g, use_container_width=True)

            # suggestions
            st.markdown("**💡 Safety Suggestions**")
            for s in result["suggestions"]:
                st.markdown(
                    f'<div style="background:#1a1630;border:1px solid #2d2750;border-radius:8px;'
                    f'padding:10px 14px;margin:4px 0;font-size:0.88rem;">{s}</div>',
                    unsafe_allow_html=True,
                )

            # recommended action
            st.markdown(
                f'<div style="background:{style["bg"]};border:1px solid {style["color"]}55;'
                f'border-radius:10px;padding:14px 18px;margin-top:10px;">'
                f'<strong style="color:{style["color"]};">Recommended Action:</strong> '
                f'<span style="color:#c4b8f0;">{result["recommended_action"]}</span></div>',
                unsafe_allow_html=True,
            )

            if result["risk_level"] == "Emergency" or panic:
                st.markdown("<br>", unsafe_allow_html=True)
                if not st.session_state.contacts:
                    st.error("🚨 Panic/Emergency detected, but no trusted contacts found to alert!")
                else:
                    from src.alert_system import analyze_call_history_and_rank
                    
                    # Analyze timeline/call history to find the most important contact
                    ranked_contacts = analyze_call_history_and_rank(st.session_state.contacts)
                    top_contact = ranked_contacts[0]
                    
                    alert_result = trigger_alert(
                        [top_contact],
                        result["risk_level"],
                        result["suggestions"],
                        lat=lat,
                        lon=lon,
                    )
                    st.session_state.alert_history.extend(get_alert_history()[-3:])

                    # ── Auto-call Prem ────────────────────────────────────────
                    prem = next((c for c in st.session_state.contacts if "Prem" in c.name), top_contact)
                    st.markdown(render_auto_call_panel(prem.name, prem.phone, result["risk_level"]), unsafe_allow_html=True)

                    # Battery warning if low
                    if st.session_state.battery_pct <= 20 and not st.session_state.battery_charging:
                        st.markdown(
                            f'<div style="background:#1c0a05;border:1px solid #f59e0b;border-radius:10px;'
                            f'padding:12px 16px;margin:8px 0;color:#f59e0b;font-size:0.88rem;">'
                            f'⚠️ <strong>Battery at {st.session_state.battery_pct}%</strong> — '
                            f'Charge your device immediately to ensure emergency alerts can be sent!</div>',
                            unsafe_allow_html=True,
                        )

                    st.success(f"🚨 **EMERGENCY ALERT SENT** to **{alert_result['contact'].name}** (Automatically identified as most important via Call History: {top_contact.recent_calls} recent calls).")
                    
                    with st.expander("View Alert Details"):
                        ta, tb = st.columns(2)
                        with ta:
                            st.markdown("**📱 Sent SMS Alert**")
                            st.code(alert_result["sms"], language=None)
                        with tb:
                            st.markdown("**💚 Sent WhatsApp Alert**")
                            st.code(alert_result["whatsapp"], language=None)


        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Make sure the model is trained first via the sidebar.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE RISK DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
elif "Live Risk" in page:
    st.markdown('<div class="sh-title">📡 Live Risk Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh-sub">Simulated real-time safety monitoring with continuous risk score updates.</div>', unsafe_allow_html=True)

    # simulate a new reading
    reading = fake_live_reading(np.mean(st.session_state.live_scores[-5:]))
    st.session_state.live_scores.append(reading["risk_score_raw"])
    if len(st.session_state.live_scores) > 60:
        st.session_state.live_scores = st.session_state.live_scores[-60:]

    score = reading["risk_score_pct"]
    if score < 35:
        level, color = "Safe", "#22c55e"
    elif score < 55:
        level, color = "Medium Risk", "#f59e0b"
    elif score < 75:
        level, color = "High Risk", "#ef4444"
    else:
        level, color = "Emergency", "#a855f7"

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Risk Score",     f"{score:.1f} / 100")
    k2.metric("Risk Level",     level)
    k3.metric("Night Mode",     "Active 🌙" if reading["is_night"] else "Inactive ☀️")
    k4.metric("Last Updated",   reading["timestamp"])

    st.markdown("<br>", unsafe_allow_html=True)

    col_gauge, col_trend = st.columns([1, 2])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": st.session_state.live_scores[-2] * 100 if len(st.session_state.live_scores) > 1 else score,
                   "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
            title={"text": f"<b>{level}</b>", "font": {"size": 16, "family": "Syne", "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a4466"},
                "bar":  {"color": color, "thickness": 0.25},
                "bgcolor": "#0f0b1e",
                "steps": [
                    {"range": [0,  34], "color": "#052e16"},
                    {"range": [34, 55], "color": "#451a03"},
                    {"range": [55, 75], "color": "#450a0a"},
                    {"range": [75, 100],"color": "#2e1065"},
                ],
                "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": score},
            },
            number={"font": {"color": color, "size": 44, "family": "Syne"}, "suffix": "%"},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0f0b1e", height=360,
            font=dict(color="#9490c0", family="DM Sans"),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_trend:
        xs = list(range(len(st.session_state.live_scores)))
        ys = [s * 100 for s in st.session_state.live_scores]
        fig_trend = go.Figure()
        # convert hex color (e.g. #f59e0b) to an rgba string for Plotly fillcolor
        def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
            h = hex_color.lstrip('#')
            try:
                r = int(h[0:2], 16)
                g = int(h[2:4], 16)
                b = int(h[4:6], 16)
            except Exception:
                return hex_color
            return f'rgba({r},{g},{b},{alpha})'
        fill_col = _hex_to_rgba(color, 0.13)
        fig_trend.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=fill_col,
            name="Risk Score",
        ))
        # zone lines
        for thresh, lab, c in [(34, "Safe", "#22c55e"), (55, "Med", "#f59e0b"), (75, "High", "#ef4444")]:
            fig_trend.add_hline(y=thresh, line_dash="dash", line_color=c, line_width=1, opacity=0.5,
                                annotation_text=lab, annotation_font_color=c,
                                annotation_position="bottom right")
        fig_trend.update_layout(
            paper_bgcolor="#0f0b1e", plot_bgcolor="#0f0b1e",
            title=dict(text="Risk Score History", font=dict(family="Syne", size=14, color="#9490c0")),
            xaxis=dict(title="Time Steps", color="#4a4466", showgrid=False),
            yaxis=dict(title="Score", range=[0, 100], color="#4a4466", gridcolor="#1e1b2e"),
            font=dict(family="DM Sans"), height=360, showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # auto-refresh
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()
    st.caption("⏱️ Click Refresh to simulate a new live reading.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: TRUSTED CONTACTS
# ════════════════════════════════════════════════════════════════════════════════
elif "Contacts" in page:
    st.markdown('<div class="sh-title">👥 Trusted Contacts</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh-sub">Manage your emergency support network. Delete contacts or change their priority anytime.</div>', unsafe_allow_html=True)

    # display existing contacts with delete + priority controls
    changed = False
    for i, c in enumerate(st.session_state.contacts):
        priority_color = "#22c55e" if c.priority == 1 else ("#f59e0b" if c.priority == 2 else "#6b7280")
        col_info, col_prio, col_del = st.columns([4, 2, 1])

        with col_info:
            st.markdown(
                f'<div class="contact-card" style="border-left:4px solid {priority_color}; margin-top:4px;">'
                f'<div style="font-family:Syne,sans-serif;font-weight:700;color:#e2e0f0;">{c.name}</div>'
                f'<div style="color:#9490c0;font-size:0.82rem;margin-top:2px;">'
                f'{c.phone} · {c.relationship} · {c.distance_km} km away</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_prio:
            new_priority = st.selectbox(
                "Priority",
                [1, 2, 3],
                index=min(c.priority - 1, 2),
                key=f"prio_{i}",
                label_visibility="collapsed",
            )
            if new_priority != c.priority:
                st.session_state.contacts[i].priority = new_priority
                changed = True

        with col_del:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"del_{i}", help=f"Delete {c.name}"):
                st.session_state.contacts.pop(i)
                save_contacts(st.session_state.contacts)
                st.success(f"🗑️ {c.name} removed from trusted contacts.")
                st.rerun()

    if changed:
        save_contacts(st.session_state.contacts)
        st.toast("✅ Priority updated and saved!", icon="✅")

    st.markdown("---")
    st.markdown("**➕ Add New Trusted Contact**")

    with st.form("add_contact"):
        fc1, fc2 = st.columns(2)
        with fc1:
            new_name  = st.text_input("Full Name")
            new_phone = st.text_input("Phone Number", placeholder="+91-XXXXX-XXXXX")
            new_rel   = st.selectbox("Relationship", ["Mother", "Father", "Sister", "Brother", "Friend", "Partner", "Other"])
        with fc2:
            new_dist  = st.slider("Distance from You (km)", 0.0, 50.0, 5.0, 0.5)
            new_prio  = st.selectbox("Priority Level", [1, 2, 3], index=2)

        if st.form_submit_button("Add Contact ➕", use_container_width=True):
            if new_name and new_phone:
                st.session_state.contacts.append(
                    Contact(new_name, new_phone, new_rel, new_dist, new_prio)
                )
                save_contacts(st.session_state.contacts)

                st.success(f"✅ {new_name} added as a trusted contact.")
                st.rerun()
            else:
                st.warning("Name and phone number are required.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: UNSAFE ZONE MAP
# ════════════════════════════════════════════════════════════════════════════════
elif "Map" in page:
    st.markdown('<div class="sh-title">🗺️ Unsafe Zone Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh-sub">Interactive safety map with crime hotspots, safe places, and trusted contacts.</div>', unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    map_lat  = mc1.number_input("Your Latitude",  value=28.6139, format="%.4f")
    map_lon  = mc2.number_input("Your Longitude", value=77.2090, format="%.4f")
    map_risk = mc3.selectbox("Current Risk Level", ["Safe", "Medium Risk", "High Risk", "Emergency"])

    try:
        from src.map_utils import build_safety_map, generate_sample_pois
        from streamlit_folium import st_folium

        pois  = generate_sample_pois(map_lat, map_lon)
        fmap  = build_safety_map(map_lat, map_lon, map_risk, pois)
        st_folium(fmap, width=None, height=560, returned_objects=[])

        # legend
        legend_items = [
            ("🔴", "#ef4444", "Unsafe / Crime Zone"),
            ("🟢", "#22c55e", "Safe Place / Hospital / Police"),
            ("💜", "#a855f7", "Trusted Contact"),
            ("📍", "#e2e0f0", "Your Location"),
        ]
        lcols = st.columns(len(legend_items))
        for (icon, c, label), col in zip(legend_items, lcols):
            col.markdown(
                f'<div style="background:#1a1630;border:1px solid {c}44;border-radius:8px;'
                f'padding:8px 12px;text-align:center;font-size:0.82rem;color:{c};">'
                f'{icon} {label}</div>', unsafe_allow_html=True
            )
    except ImportError:
        st.warning("Install `streamlit-folium` to enable the interactive map: `pip install streamlit-folium`")
        st.info("Map will appear once `streamlit-folium` is installed.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown('<div class="sh-title">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh-sub">Evaluation metrics for all trained classifiers.</div>', unsafe_allow_html=True)

    metrics = load_metrics("models")

    if not metrics:
        st.warning("No metrics found. Please train the model first.")
        st.stop()

    results   = metrics["results"]
    best_name = metrics.get("best_model", "")

    # summary comparison
    st.markdown(f"**🏆 Best Model: `{best_name}`**")
    st.markdown("<br>", unsafe_allow_html=True)

    model_names = list(results.keys())
    accs  = [results[m]["accuracy"]    for m in model_names]
    f1s   = [results[m]["f1_weighted"] for m in model_names]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="Accuracy", x=model_names, y=accs,
                              marker_color="#6366f1", text=[f"{v:.3f}" for v in accs], textposition="outside"))
    fig_cmp.add_trace(go.Bar(name="F1 (Weighted)", x=model_names, y=f1s,
                              marker_color="#ec4899", text=[f"{v:.3f}" for v in f1s], textposition="outside"))
    fig_cmp.update_layout(
        barmode="group", paper_bgcolor="#0f0b1e", plot_bgcolor="#0f0b1e",
        title="Model Comparison — Accuracy & F1 Score",
        font=dict(family="DM Sans", color="#9490c0"),
        yaxis=dict(range=[0, 1.15], gridcolor="#1e1b2e"),
        xaxis=dict(color="#6b6488"),
        legend=dict(bgcolor="#1a1630"),
        height=350, margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # per-model detailed view
    selected_model = st.selectbox("View Detailed Metrics For:", model_names,
                                   index=model_names.index(best_name) if best_name in model_names else 0)
    sel = results[selected_model]
    mc1, mc2 = st.columns(2)

    with mc1:
        # confusion matrix
        cm_data = sel.get("confusion_matrix")
        if cm_data:
            labels = ["Safe", "Med Risk", "High Risk", "Emergency"]
            fig_cm = px.imshow(
                cm_data, x=labels, y=labels,
                color_continuous_scale=[[0, "#0f0b1e"], [0.5, "#4c1d95"], [1, "#a855f7"]],
                title=f"Confusion Matrix — {selected_model}",
                text_auto=True,
            )
            fig_cm.update_layout(
                paper_bgcolor="#0f0b1e", font=dict(family="DM Sans", color="#9490c0"),
                height=350, margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with mc2:
        # feature importance
        fi = sel.get("feature_importances")
        if fi:
            fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:12]
            feat_names = [f[0] for f in fi_sorted]
            feat_vals  = [f[1] for f in fi_sorted]
            fig_fi = go.Figure(go.Bar(
                x=feat_vals[::-1], y=feat_names[::-1],
                orientation="h",
                marker=dict(color=feat_vals[::-1], colorscale=[[0,"#1e1b2e"],[0.5,"#6d28d9"],[1,"#ec4899"]]),
            ))
            fig_fi.update_layout(
                title=f"Feature Importance — {selected_model}",
                paper_bgcolor="#0f0b1e", plot_bgcolor="#0f0b1e",
                font=dict(family="DM Sans", color="#9490c0"),
                xaxis=dict(title="Importance", gridcolor="#1e1b2e"),
                yaxis=dict(color="#9490c0"),
                height=350, margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # classification report table
    report = sel.get("report", {})
    if report:
        st.markdown("**Classification Report**")
        label_order = ["Safe", "Medium Risk", "High Risk", "Emergency"]
        rows = []
        for label in label_order:
            if label in report:
                r = report[label]
                rows.append({
                    "Class":     label,
                    "Precision": round(r["precision"], 3),
                    "Recall":    round(r["recall"], 3),
                    "F1-Score":  round(r["f1-score"], 3),
                    "Support":   int(r["support"]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: EMERGENCY ALERT SIMULATION
# ════════════════════════════════════════════════════════════════════════════════
elif "Emergency" in page:
    st.markdown('<div class="sh-title">🆘 Emergency Alert Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh-sub">Click any button below — the alert goes directly to your <strong>Priority #1</strong> contact via SMS, WhatsApp & <strong style="color:#ef4444;">Auto-Call</strong>.</div>', unsafe_allow_html=True)

    sim_lat = 28.6139
    sim_lon = 77.2090

    sim_suggestions = [
        "Move to a crowded and well-lit area immediately.",
        "Share your live location with all trusted contacts.",
        "Call emergency services: 112",
    ]

    # ── Battery status on Emergency page ────────────────────────────────────
    em_batt_pct    = st.session_state.battery_pct
    em_batt_charge = st.session_state.battery_charging
    em_b1, em_b2 = st.columns([1, 2])
    with em_b1:
        st.markdown(render_battery_widget(em_batt_pct, em_batt_charge, label="Device Battery"), unsafe_allow_html=True)
    with em_b2:
        if em_batt_pct <= 15 and not em_batt_charge:
            st.markdown(
                f'<div class="emergency-panel" style="background:#1c0a05;border:2px solid #ef4444;'
                f'border-radius:12px;padding:14px 18px;">'
                f'<strong style="color:#ef4444;">🪫 CRITICAL: {em_batt_pct}% Battery!</strong><br>'
                f'<span style="color:#c4b8f0;font-size:0.85rem;">'
                f'Emergency calls may fail if battery dies. Charge immediately!</span></div>',
                unsafe_allow_html=True,
            )
        elif em_batt_pct <= 40 and not em_batt_charge:
            st.markdown(
                f'<div style="background:#1c1005;border:1px solid #f59e0b44;border-radius:12px;padding:14px 18px;">'
                f'<strong style="color:#f59e0b;">⚠️ Low Battery: {em_batt_pct}%</strong><br>'
                f'<span style="color:#c4b8f0;font-size:0.85rem;">'
                f'Consider charging to ensure emergency features work reliably.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:#052e16;border:1px solid #22c55e44;border-radius:12px;padding:14px 18px;">'
                f'<strong style="color:#22c55e;">✅ Battery OK: {em_batt_pct}%</strong><br>'
                f'<span style="color:#c4b8f0;font-size:0.85rem;">'
                f'Device is ready to send emergency alerts and calls.</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

    # Find priority 1 contact (Prem first)
    prem_contact = next((c for c in st.session_state.contacts if "Prem" in c.name), None)
    priority1    = prem_contact or (sorted(st.session_state.contacts, key=lambda c: c.priority)[0] if st.session_state.contacts else None)

    if priority1:
        prem_clean_phone = priority1.phone.replace("-","").replace("+","").replace(" ","")
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a1230,#0f0b1e);border:1px solid #ef444455;'
            f'border-left:5px solid #ef4444;border-radius:12px;padding:16px 20px;margin-bottom:16px;'
            f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">'
            f'<div><div style="color:#9490c0;font-size:0.8rem;margin-bottom:4px;">Alert will be sent to</div>'
            f'<div style="font-family:Syne,sans-serif;font-weight:800;font-size:1.2rem;color:#ef4444;">🎯 {priority1.name}</div>'
            f'<div style="color:#9490c0;font-size:0.82rem;margin-top:4px;">{priority1.phone} · {priority1.relationship} · Priority #1</div></div>'
            f'<a href="tel:+{prem_clean_phone}" style="background:linear-gradient(135deg,#dc2626,#b91c1c);'
            f'color:#fff;text-decoration:none;border-radius:10px;padding:10px 20px;font-family:Syne,sans-serif;'
            f'font-weight:700;font-size:0.9rem;">📞 Call {priority1.name}</a>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error("No trusted contacts found. Please add contacts first from the 👥 Trusted Contacts page.")
        st.stop()

    # Action buttons
    import urllib.parse
    
    alert_scenarios = [
        ("🆘", "PANIC / EMERGENCY",       "#ef4444", "Panic / Emergency"),
        ("⚠️", "HARASSMENT",               "#f97316", "Harassment"),
        ("👣", "STALKING / BEING FOLLOWED", "#a855f7", "Stalking / Being Followed"),
        ("👁️", "SUSPICIOUS ACTIVITY",      "#f59e0b", "Suspicious Activity"),
    ]

    b1, b2, b3, b4 = st.columns(4)
    buttons = [
        b1.button("🆘 Panic / Emergency",    use_container_width=True, key="sim_panic"),
        b2.button("⚠️ Harassment",            use_container_width=True, key="sim_harass"),
        b3.button("👣 Stalking / Followed",   use_container_width=True, key="sim_stalk"),
        b4.button("👁️ Suspicious Activity",  use_container_width=True, key="sim_sus"),
    ]

    triggered_idx = next((i for i, b in enumerate(buttons) if b), None)

    if triggered_idx is not None:
        icon, label, color, risk_desc = alert_scenarios[triggered_idx]
        risk_level = f"Emergency: {risk_desc}"

        result = trigger_alert(
            [priority1],
            risk_level,
            sim_suggestions,
            lat=sim_lat,
            lon=sim_lon,
        )
        st.session_state.alert_history.extend(get_alert_history()[-3:])

        # ── Pulsing emergency banner ─────────────────────────────────────────
        st.markdown(
            f'<div class="emergency-panel" style="background:{color}18;border:2px solid {color};'
            f'border-radius:16px;padding:20px 24px;margin:12px 0;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
            f'<span class="ping-dot" style="background:{color};"></span>'
            f'<span style="font-family:Syne,sans-serif;font-weight:800;font-size:1.2rem;color:{color};">'
            f'{icon} {label} ALERT SENT!</span></div>'
            f'<div style="color:#c4b8f0;font-size:0.88rem;">'
            f'Sent to <strong>{priority1.name}</strong> ({priority1.phone}) via SMS, WhatsApp & Auto-Call</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Auto-call Prem panel ─────────────────────────────────────────────
        st.markdown(render_auto_call_panel(priority1.name, priority1.phone, risk_desc), unsafe_allow_html=True)

        phone_clean = priority1.phone.replace("-", "").replace("+", "").replace(" ", "")
        sms_body = urllib.parse.quote(result["sms"])
        wa_body  = urllib.parse.quote(result["whatsapp"])

        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            st.link_button("💬 Send SMS Now", f"sms:{phone_clean}?body={sms_body}", use_container_width=True)
        with lc2:
            st.link_button("💚 Send WhatsApp Now", f"https://wa.me/{phone_clean}?text={wa_body}", use_container_width=True)
        with lc3:
            st.link_button(f"📞 Call {priority1.name}", f"tel:+{phone_clean}", use_container_width=True)

        ta, tb = st.columns(2)
        with ta:
            st.markdown("**📱 SMS Message Sent**")
            st.code(result["sms"], language=None)
        with tb:
            st.markdown("**💚 WhatsApp Message Sent**")
            st.code(result["whatsapp"], language=None)


    # alert history
    st.markdown("---")
    st.markdown("**📋 Alert History**")

    history = get_alert_history()
    if not history:
        st.caption("No alerts triggered yet in this session.")
    else:
        hist_rows = [
            {
                "Timestamp": r.timestamp,
                "Risk Level": r.risk_level,
                "Channel":    r.channel,
                "Contact":    r.contact,
                "Status":     r.status,
            }
            for r in history[:15]
        ]
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    # extra ideas box
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh-title" style="font-size:1.2rem;">🚀 Future Integrations</div>', unsafe_allow_html=True)
    ideas = [
        ("📱", "Twilio SMS API",    "Send real SMS/WhatsApp alerts with Twilio's messaging API."),
        ("🎤", "Voice Trigger",     "Say 'Help!' to auto-trigger an emergency alert via speech recognition."),
        ("📍", "Live GPS",          "Integrate with Google Maps SDK or device GPS for real-time tracking."),
        ("📷", "YOLO Camera AI",    "Detect suspicious activity via webcam using YOLOv8 + OpenCV."),
        ("📲", "Flutter Mobile App","Build a cross-platform mobile app with Flutter for iOS & Android."),
    ]
    icols = st.columns(len(ideas))
    for (icon, title, desc), col in zip(ideas, icols):
        col.markdown(
            f'<div style="background:#1a1630;border:1px solid #2d2750;border-radius:10px;'
            f'padding:14px;font-size:0.8rem;">'
            f'<div style="font-size:1.3rem;">{icon}</div>'
            f'<div style="color:#a78bfa;font-weight:600;margin:6px 0 4px;">{title}</div>'
            f'<div style="color:#6b6488;line-height:1.5;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
