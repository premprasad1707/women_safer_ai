"""
SafeHer AI — Model Training Pipeline
Trains Random Forest, XGBoost, Logistic Regression, Decision Tree.
Saves the best model to disk.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed — skipping XGBClassifier.")

warnings.filterwarnings("ignore")

LABEL_NAMES = ["Safe", "Medium Risk", "High Risk", "Emergency"]


# ── model definitions ───────────────────────────────────────────────────────

def _get_models():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1
        )
    return models


# ── evaluation helper ────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred).tolist()

    result = {
        "model":        model_name,
        "accuracy":     round(acc, 4),
        "f1_weighted":  round(f1, 4),
        "report":       report,
        "confusion_matrix": cm,
    }
    print(f"  {model_name:<25} Acc={acc:.4f}  F1={f1:.4f}")
    return result


# ── main training function ───────────────────────────────────────────────────

def train_all_models(
    X_train, X_test, y_train, y_test,
    feature_names: list,
    models_dir: str = "models",
) -> dict:
    """
    Train all classifiers, evaluate them, and save the best one.

    Returns
    -------
    dict  — metrics for each model
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    models      = _get_models()
    results     = {}
    best_f1     = -1
    best_name   = None
    best_model  = None

    print("\n🔧  Training models...")
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_test, y_test, name)
        results[name] = metrics
        results[name]["model_obj"] = clf          # store temporarily

        # feature importances (if available)
        if hasattr(clf, "feature_importances_"):
            fi = dict(zip(feature_names, clf.feature_importances_.tolist()))
            results[name]["feature_importances"] = fi

        if metrics["f1_weighted"] > best_f1:
            best_f1   = metrics["f1_weighted"]
            best_name = name
            best_model = clf

    # ── persist best model ───────────────────────────────────────────────
    model_path = f"{models_dir}/safety_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n🏆  Best model → {best_name}  (F1={best_f1:.4f})")
    print(f"✅  Saved → {model_path}")

    # ── save metrics JSON (without unpicklable objects) ──────────────────
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: vv for kk, vv in v.items() if kk != "model_obj"}

    with open(f"{models_dir}/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": serializable}, f, indent=2)
    print(f"✅  Metrics saved → {models_dir}/metrics.json")

    # clean up non-serialisable key
    for k in results:
        results[k].pop("model_obj", None)

    return results, best_name, best_model


# ── standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_generator import generate_dataset
    from src.preprocessing  import preprocess

    print("📊  Generating dataset …")
    df = generate_dataset(10_000, "data/safety_dataset.csv")

    print("🔄  Preprocessing …")
    X_train, X_test, y_train, y_test, features, enc, sc = preprocess(df, fit=True)

    results, best_name, best_model = train_all_models(
        X_train, X_test, y_train, y_test, features, models_dir="models"
    )

    print("\n── Summary ─────────────────────────────────")
    for name, m in results.items():
        print(f"  {name:<25} Acc={m['accuracy']}  F1={m['f1_weighted']}")
