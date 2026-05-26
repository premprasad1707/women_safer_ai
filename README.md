# 🛡️ SafeHer AI — Personal Safety Prediction System

<div align="center">

![SafeHer AI](https://img.shields.io/badge/SafeHer_AI-v1.0-7c3aed?style=for-the-badge&logo=shield&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ML](https://img.shields.io/badge/ML-Random_Forest_|_XGBoost-22c55e?style=for-the-badge)

**An AI-powered safety prediction system that analyses real-time context to assess personal safety risk and trigger emergency alerts.**

</div>

---

## 📋 Project Overview

SafeHer AI is a complete, production-grade data science and machine learning application designed to predict personal safety risk levels — Safe, Medium Risk, High Risk, or Emergency — based on 21 contextual input features including location, time of day, crowd density, lighting conditions, movement behaviour, and proximity to trusted contacts.

> ⚠️ **Disclaimer:** This is a portfolio demonstration project. It does not guarantee safety, replace emergency services, or provide real GPS tracking. In a genuine emergency, call your local emergency number (112 / 911).

---

## 🎯 Features at a Glance

| Feature | Description |
|---|---|
| 🔮 **Safety Prediction** | 4-class ML classification using RF, XGBoost, LR, Decision Tree |
| 📡 **Live Risk Dashboard** | Animated real-time gauge + historical trend chart |
| 👥 **Trusted Contacts** | Priority-based contact management with auto-selection |
| 🗺️ **Unsafe Zone Map** | Folium map with crime hotspots, safe places, heatmap |
| 📊 **Model Performance** | Accuracy, F1, confusion matrix, feature importance charts |
| 🆘 **Emergency Alerts** | Simulated SMS, WhatsApp, and email alert generation |
| 💡 **Rule-Based Suggestions** | Context-aware safety tips alongside ML prediction |

---

## 📁 Project Structure

```
safeher_ai/
├── app.py                          # Main Streamlit application (7 pages)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   └── safety_dataset.csv          # Generated synthetic dataset (10,000 rows)
│
├── models/
│   ├── safety_model.pkl            # Best trained ML model (joblib)
│   ├── encoders.pkl                # Fitted LabelEncoders for categorical columns
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── metrics.json                # Model evaluation results
│
├── notebooks/
│   └── safety_model_training.ipynb # Jupyter notebook for EDA + training
│
└── src/
    ├── data_generator.py           # Synthetic dataset generator (10,000 rows)
    ├── preprocessing.py            # Encoding, scaling, train/test split
    ├── train_model.py              # Multi-model training pipeline
    ├── predict.py                  # Prediction engine + rule-based suggestions
    ├── alert_system.py             # SMS / WhatsApp / email alert simulation
    ├── map_utils.py                # Folium map builder
    └── utils.py                    # Shared helpers and styling utilities
```

---

## 🤖 Machine Learning Details

### Input Features (21 total)

| Category | Features |
|---|---|
| **Location** | latitude, longitude |
| **Time** | hour, day_type, is_night |
| **Social** | is_alone, crowd_density |
| **Environment** | lighting_condition, crime_area_score, weather_condition |
| **Movement** | phone_motion_status, walking_speed |
| **Distance** | distance_from_home, distance_from_safe_zone, nearest_trusted_contact_distance |
| **Device** | battery_level, network_status |
| **Events** | panic_button, unusual_movement, time_spent_at_location |

### Engineered Features (6 additional)

| Feature | Formula |
|---|---|
| `night_risk_score` | `is_night × 0.6 + (lighting=dark) × 0.4` |
| `isolation_score` | `is_alone × 0.6 + crowd_risk × 0.4` |
| `movement_risk_score` | Motion type → {stationary: 0.1, walking: 0.2, running: 0.6, erratic: 1.0} |
| `location_risk_score` | `crime × 0.5 + dist_safe/30 × 0.3 + dist_home/60 × 0.2` |
| `emergency_score` | `panic × 0.5 + unusual × 0.3 + (battery<20) × 0.2` |
| `total_risk_score` | Weighted composite of all five above |

### Target Classes

| Class | Score Range |
|---|---|
| ✅ Safe | 0 – 34 |
| ⚠️ Medium Risk | 35 – 54 |
| 🚨 High Risk | 55 – 74 |
| 🆘 Emergency | 75 – 100 |

### Models Trained

- Random Forest (200 estimators, depth 15)
- XGBoost (200 estimators, lr=0.1)
- Logistic Regression (multinomial, 500 iter)
- Decision Tree (depth 12)

---

## 🚀 Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/safeher-ai.git
cd safeher-ai
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Train the Model

```bash
# Option A: From the command line
python -m src.train_model

# Option B: From the Streamlit app sidebar
# → Click "Train Model Now" button
```

### Step 5 — Launch the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🔬 Jupyter Notebook Walkthrough

```bash
pip install jupyter
cd notebooks
jupyter notebook safety_model_training.ipynb
```

The notebook covers:
1. Dataset generation and inspection
2. EDA with Plotly charts
3. Correlation heatmap
4. Preprocessing pipeline
5. Model training comparison
6. Confusion matrix visualisation
7. Feature importance chart
8. Sample prediction test

---

## 🧪 Testing a Prediction (CLI)

```bash
python -m src.predict
```

This runs a sample high-risk scenario and prints the prediction, probabilities, and suggestions.

---

## 🔮 Safety Prediction — Rule-Based Logic

Alongside the ML model, rule-based suggestions fire automatically:

```python
# Examples from predict.py

if is_night == 1 and is_alone == 1:
    → "Share your live location with a trusted contact."

if crowd_density == "low" and lighting_condition == "dark":
    → "Move toward a brighter and crowded place."

if panic_button == 1:
    → "Emergency alert should be sent immediately."

if battery_level < 20:
    → "Battery low. Inform trusted contact now."

if crime_area_score > 0.6:
    → "High crime score — avoid this area."
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this project to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in
3. Click **New app** → select your repository
4. Set **Main file path** to `app.py`
5. Add `requirements.txt` (already included)
6. Click **Deploy**

> Note: Train the model locally first and commit `models/` to your repo, since Streamlit Cloud has no GPU and training may be slow.

---

## 🚀 Future Improvements

### Short Term
- [ ] Real Twilio SMS + WhatsApp API integration
- [ ] Google Maps JavaScript SDK for live GPS
- [ ] SOS shake gesture detection
- [ ] Dark mode toggle

### Medium Term
- [ ] User authentication with OAuth
- [ ] Cloud database for alert history (Supabase / Firebase)
- [ ] Real crime dataset integration (police open data APIs)
- [ ] Periodic model retraining pipeline

### Advanced
- [ ] **Flutter mobile app** — cross-platform iOS/Android
- [ ] **Voice trigger** — "Help!" keyword detection via SpeechRecognition
- [ ] **YOLOv8 + OpenCV** — suspicious activity detection via camera
- [ ] **FastAPI backend** — REST API for mobile client
- [ ] **BERT NLP** — analyse distress signals in typed/voice messages

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, Plotly, Folium |
| **ML Models** | Scikit-learn, XGBoost |
| **Data** | Pandas, NumPy |
| **Persistence** | Joblib |
| **Maps** | Folium + streamlit-folium |

---

## 👨‍💻 Author

Built by **Prem** as a portfolio data science project demonstrating:
- End-to-end ML pipeline design
- Feature engineering for safety analytics
- Multi-model training and evaluation
- Professional Streamlit UI development
- Modular, production-grade Python code structure

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">
<strong>⭐ Star this repo if you found it useful!</strong><br><br>
<em>SafeHer AI — Intelligent Safety, Always On.</em>
</div>
