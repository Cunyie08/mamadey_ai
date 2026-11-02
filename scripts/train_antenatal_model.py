"""
Train the MamaDey AI antenatal screening model for the CHW prototype.

Why a new model: the dissertation model used delivery and postnatal features
(skilled birth, postnatal check, delivery place, caesarean section). A community
health worker screening a pregnant woman cannot know these yet, so this script
retrains using only information available DURING pregnancy.

It also fixes the two known limitations noted in the README:
  1. SMOTE now runs inside each cross-validation fold (imblearn Pipeline),
     so cross-validation scores are no longer inflated.
  2. The alert threshold is chosen from out-of-fold training predictions,
     never from the test set. The test set is used once, for the final report.

Run from the repository root, after the Week 4-5 notebook has produced
data/processed/X_train.csv, X_test.csv, y_train.csv and y_test.csv:

    python scripts/train_antenatal_model.py

Outputs:
    models/antenatal_rf.pkl
    models/antenatal_rf_meta.json
    outputs/antenatal_model_report.md
"""

import json
import os
import pickle
import sys
from datetime import date

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
RANDOM_STATE = 42
TARGET_SENSITIVITY = 0.70  # catch at least 70% of high-risk women on training folds

# Features a CHW can collect during pregnancy, in the order the app sends them.
ANTENATAL_FEATURES = [
    "residence",               # 1 urban, 2 rural (DHS coding)
    "age",
    "education_level",         # 0 none, 1 primary, 2 secondary, 3 higher
    "wealth_index",            # 1 poorest ... 5 richest
    "total_children_ever_born",
    "married",
    "anc_visits",
    "anc_timing_months",
    "anc_any",
    "anc_4plus",
    "anc_dropout",
    "anc_late_entry",
    "anc_provider_doc",
    "anc_provider_nurse",
    "anc_provider_tba",
    "anc_blood_pressure",
    "anc_urine_sample",
    "anc_iron_tablets",
    "parity_wealth_risk",
    "late_and_dropout",
    "adolescent",
    "advanced_maternal_age",
]

# Excluded because they are only known at or after delivery.
POST_DELIVERY_FEATURES = [
    "delivery_csection", "tba_any", "skilled_birth", "delivery_alone",
    "postnatal_check", "delivery_location_enc", "no_skilled_care",
    # DHS counts living children at interview, which already reflects whether
    # the index child survived; that leaks the outcome, so it is excluded too.
    "living_children",
]


def load():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze("columns")
    missing = [c for c in ANTENATAL_FEATURES if c not in X_train.columns]
    if missing:
        sys.exit(f"Missing columns in X_train.csv: {missing}")
    return X_train[ANTENATAL_FEATURES], X_test[ANTENATAL_FEATURES], y_train, y_test


def build_pipeline():
    return Pipeline([
        ("smote", SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            max_features="sqrt", class_weight=None, random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def pick_threshold(y, proba, target=TARGET_SENSITIVITY):
    """Highest threshold whose sensitivity on out-of-fold predictions meets the target."""
    best = 0.05
    for t in np.round(np.arange(0.05, 0.95, 0.01), 2):
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) else 0
        if sens >= target:
            best = float(t)
    return best


def metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc_roc": round(float(roc_auc_score(y, proba)), 4),
        "sensitivity": round(float(tp / (tp + fn)), 4) if (tp + fn) else None,
        "specificity": round(float(tn / (tn + fp)), 4) if (tn + fp) else None,
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load()
    print(f"Train {X_train.shape}, test {X_test.shape}, features {len(ANTENATAL_FEATURES)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(build_pipeline(), X_train, y_train, cv=cv,
                            method="predict_proba")[:, 1]
    threshold = pick_threshold(y_train.values, oof)
    cv_metrics = metrics(y_train.values, oof, threshold)
    print(f"Threshold from training folds: {threshold}  CV: {cv_metrics}")

    model = build_pipeline().fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = metrics(y_test.values, test_proba, threshold)
    print(f"Held-out test: {test_metrics}")

    rf = model.named_steps["rf"]
    importance = sorted(zip(ANTENATAL_FEATURES, rf.feature_importances_),
                        key=lambda x: -x[1])

    with open(os.path.join(MODELS_DIR, "antenatal_rf.pkl"), "wb") as f:
        pickle.dump(model, f)
    meta = {
        "model": "RandomForest (SMOTE inside pipeline)",
        "trained_on": str(date.today()),
        "features": ANTENATAL_FEATURES,
        "excluded_post_delivery_features": POST_DELIVERY_FEATURES,
        "threshold": threshold,
        "threshold_rule": f"highest threshold with >= {TARGET_SENSITIVITY:.0%} sensitivity on out-of-fold training predictions",
        "cv_out_of_fold": cv_metrics,
        "held_out_test": test_metrics,
        "feature_importance": [[f, round(float(v), 4)] for f, v in importance],
    }
    with open(os.path.join(MODELS_DIR, "antenatal_rf_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    lines = [
        "# MamaDey AI antenatal screening model",
        "",
        f"Trained {date.today()} on {len(X_train):,} records; tested once on {len(X_test):,} held-out records.",
        "Uses only information available during pregnancy; delivery and postnatal features are excluded.",
        "",
        "| Metric | Cross-validation (out-of-fold) | Held-out test |",
        "| --- | --- | --- |",
    ]
    for k in ["auc_roc", "sensitivity", "specificity", "brier"]:
        lines.append(f"| {k} | {cv_metrics[k]} | {test_metrics[k]} |")
    lines += ["", f"Alert threshold: {threshold} ({meta['threshold_rule']}).",
              f"Test confusion matrix: TP {test_metrics['TP']}, FP {test_metrics['FP']}, "
              f"TN {test_metrics['TN']}, FN {test_metrics['FN']}.", "",
              "## Top features", ""]
    lines += [f"{i + 1}. {f} ({v:.3f})" for i, (f, v) in enumerate(importance[:10])]
    with open(os.path.join(OUTPUTS_DIR, "antenatal_model_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved models/antenatal_rf.pkl, models/antenatal_rf_meta.json, outputs/antenatal_model_report.md")


if __name__ == "__main__":
    main()
