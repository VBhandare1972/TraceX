import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def train_tracex_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    dataset_path = os.path.join(project_root, "TraceX_project_dataset.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print(f"[*] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"[*] Dataset shape: {df.shape}")

    feature_cols = [
        "length_url",
        "length_hostname",
        "nb_www",
        "ratio_digits_url",
        "length_words_raw",
        "longest_word_path",
        "avg_word_path",
        "phish_hints",
        "nb_hyperlinks",
        "ratio_intHyperlinks",
        "ratio_extHyperlinks",
        "safe_anchor",
        "domain_registration_length",
        "domain_age",
        "web_traffic",
        "google_index",
        "page_rank",
        "https",
        "ssl_valid",
        "redirect_count",
    ]

    target_col = "label"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Fill any null values if present
    X = X.fillna(0)

    # Train-test split (80-20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    print("[*] Training Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # Predictions & Probabilities
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    # Metrics Computation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    print("\n================ EVALUATION METRICS ================")
    print(f"Accuracy:              {acc * 100:.2f}%")
    print(f"Precision:             {prec * 100:.2f}%")
    print(f"Recall:                {rec * 100:.2f}%")
    print(f"F1-Score:              {f1 * 100:.2f}%")
    print(f"False Positive Rate:   {fpr * 100:.2f}%")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"Confusion Matrix:      TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("====================================================\n")

    # Feature Importances
    importances = clf.feature_importances_
    feature_importance_list = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(feature_cols, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )

    print("Top 10 Most Important Features:")
    for idx, item in enumerate(feature_importance_list[:10], 1):
        print(f"  {idx}. {item['feature']:<26} : {item['importance'] * 100:.2f}%")

    # Save artifacts
    model_path = os.path.join(base_dir, "model.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    features_path = os.path.join(base_dir, "feature_names.json")
    metrics_path = os.path.join(base_dir, "metrics.json")

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    metrics_dict = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "feature_importances": feature_importance_list
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\n[+] Saved artifacts to:\n - {model_path}\n - {scaler_path}\n - {features_path}\n - {metrics_path}")

if __name__ == "__main__":
    train_tracex_model()
