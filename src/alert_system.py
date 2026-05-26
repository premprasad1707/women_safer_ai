"""
SafeHer AI — Alert System
Simulates SMS, WhatsApp, and emergency alerts with alert history tracking.
"""

import random
import datetime
from dataclasses import dataclass, field

# ── alert history (in-memory store) ─────────────────────────────────────────
_ALERT_HISTORY: list = []


@dataclass
class Contact:
    name:         str
    phone:        str
    relationship: str
    distance_km:  float
    priority:     int   # 1 = highest
    recent_calls: int = 0


@dataclass
class AlertRecord:
    timestamp:   str
    risk_level:  str
    message:     str
    contact:     str
    channel:     str
    status:      str = "Sent"


# ── contact selector ─────────────────────────────────────────────────────────

def analyze_call_history_and_rank(contacts: list[Contact]) -> list[Contact]:
    """Simulates reading phone timeline/call history to find important contacts."""
    # Rank by recent calls (highest first) and update priority
    ranked = sorted(contacts, key=lambda c: c.recent_calls, reverse=True)
    for i, c in enumerate(ranked):
        c.priority = i + 1
    return ranked

def select_best_contact(contacts: list[Contact]) -> Contact | None:
    """Pick contact with highest priority; ties broken by proximity."""
    if not contacts:
        return None
    return sorted(contacts, key=lambda c: (c.priority, c.distance_km))[0]


# ── message builders ─────────────────────────────────────────────────────────

def _location_link(lat: float = 28.6139, lon: float = 77.2090) -> str:
    return f"https://maps.google.com/?q={lat},{lon}"


def build_sms_alert(
    contact: Contact,
    risk_level: str,
    lat: float = 28.6139,
    lon: float = 77.2090,
) -> str:
    ts   = datetime.datetime.now().strftime("%d %b %Y %H:%M")
    link = _location_link(lat, lon)
    return (
        f"🚨 SAFEHER ALERT [{ts}]\n"
        f"Hi {contact.name}, your trusted contact needs help!\n"
        f"Risk Level : {risk_level}\n"
        f"Live Location : {link}\n"
        f"Please respond or call emergency services.\n"
        f"— SafeHer AI"
    )


def build_whatsapp_alert(
    contact: Contact,
    risk_level: str,
    lat: float = 28.6139,
    lon: float = 77.2090,
) -> str:
    link = _location_link(lat, lon)
    return (
        f"*SafeHer Emergency Alert* 🆘\n\n"
        f"Dear *{contact.name}*,\n"
        f"A safety alert has been triggered for a person who listed you "
        f"as a trusted contact.\n\n"
        f"*Risk Level:* {risk_level}\n"
        f"*Live Location:* {link}\n\n"
        f"Please check on them immediately or contact emergency services.\n\n"
        f"_This message was sent automatically by SafeHer AI._"
    )


def build_email_subject(risk_level: str) -> str:
    return f"⚠️ SafeHer AI — {risk_level} Alert | Immediate Action Required"


def build_email_body(
    contact: Contact,
    risk_level: str,
    suggestions: list[str],
    lat: float = 28.6139,
    lon: float = 77.2090,
) -> str:
    link = _location_link(lat, lon)
    suggestion_block = "\n".join(f"  • {s}" for s in suggestions[:3])
    return (
        f"Dear {contact.name},\n\n"
        f"SafeHer AI has detected a {risk_level} situation for your trusted contact.\n\n"
        f"📍 Live Location: {link}\n"
        f"⚠️  Risk Level: {risk_level}\n\n"
        f"Safety Notes:\n{suggestion_block}\n\n"
        f"Please call them or contact emergency services immediately.\n\n"
        f"— SafeHer AI Safety System\n"
        f"   This is an automated safety alert. Do not reply to this email."
    )


# ── trigger alerts ───────────────────────────────────────────────────────────

def trigger_alert(
    contacts: list[Contact],
    risk_level: str,
    suggestions: list[str],
    lat: float = 28.6139,
    lon: float = 77.2090,
) -> dict:
    """
    Simulate sending SMS + WhatsApp + email alerts.

    Returns a dict with all generated messages and the selected contact.
    """
    contact = select_best_contact(contacts)
    if not contact:
        return {"error": "No trusted contacts available."}

    sms       = build_sms_alert(contact, risk_level, lat, lon)
    whatsapp  = build_whatsapp_alert(contact, risk_level, lat, lon)
    email_sub = build_email_subject(risk_level)
    email_bod = build_email_body(contact, risk_level, suggestions, lat, lon)

    ts = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    # log to history
    for channel, msg in [("SMS", sms), ("WhatsApp", whatsapp), ("Email", email_sub)]:
        _ALERT_HISTORY.append(AlertRecord(
            timestamp  = ts,
            risk_level = risk_level,
            message    = msg,
            contact    = contact.name,
            channel    = channel,
            status     = "Sent ✅",
        ))

    return {
        "contact":    contact,
        "sms":        sms,
        "whatsapp":   whatsapp,
        "email_sub":  email_sub,
        "email_body": email_bod,
        "timestamp":  ts,
    }


def get_alert_history() -> list[AlertRecord]:
    return list(reversed(_ALERT_HISTORY))


def clear_alert_history():
    _ALERT_HISTORY.clear()


import json
import os

# ── default demo contacts ────────────────────────────────────────────────────

DEFAULT_CONTACTS = [
    Contact("Prem",          "+91-6370959392", "Emergency Contact", 1.0, 1, recent_calls=999),
    Contact("Priya Sharma",  "+91-98765-43210", "Friend",  2.3, 1, recent_calls=12),
    Contact("Ravi Kumar",    "+91-87654-32109", "Brother", 8.1, 2, recent_calls=5),
    Contact("Sunita Devi",   "+91-76543-21098", "Mother",  15.0, 3, recent_calls=24),
]

CONTACTS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "contacts.json")

def load_contacts() -> list[Contact]:
    if not os.path.exists(CONTACTS_FILE):
        return list(DEFAULT_CONTACTS)
    try:
        with open(CONTACTS_FILE, "r") as f:
            data = json.load(f)
            return [Contact(**c) for c in data]
    except Exception:
        return list(DEFAULT_CONTACTS)

def save_contacts(contacts: list[Contact]):
    os.makedirs(os.path.dirname(CONTACTS_FILE), exist_ok=True)
    with open(CONTACTS_FILE, "w") as f:
        json.dump([c.__dict__ for c in contacts], f, indent=4)



if __name__ == "__main__":
    result = trigger_alert(
        DEFAULT_CONTACTS,
        "High Risk",
        ["Move to a safe place.", "Share location."],
        lat=28.6139,
        lon=77.2090,
    )
    print("── SMS ──────────────────────────────────")
    print(result["sms"])
    print("\n── WhatsApp ─────────────────────────────")
    print(result["whatsapp"])
    print("\n── Alert History ────────────────────────")
    for rec in get_alert_history():
        print(f"  [{rec.timestamp}] {rec.channel} → {rec.contact} | {rec.risk_level}")
