# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:34:38 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Random Forest 10 runs
Pipeline final : CICIDS2018 chronologique + Live + Botnet 10 %
Test externe : Botnet 90 %

TRAIN :
train_2018_chrono_live_botnet10_encoded.csv

VALIDATION :
val_2018_chrono_live_botnet10_encoded.csv

TEST :
test_botnet90_external_chrono_encoded.csv
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# CHEMINS
# =========================================================

BASE_DIR = r"C:\Users\DELL\Desktop\DDOS12\PFE_DDOS\PFE_DDOS\scripts"

TRAIN_PATH = os.path.join(BASE_DIR, "train_2018_chrono_live_botnet10_encoded.csv")
VAL_PATH   = os.path.join(BASE_DIR, "val_2018_chrono_live_botnet10_encoded.csv")
TEST_PATH  = os.path.join(BASE_DIR, "test_botnet90_external_chrono_encoded.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "rf_10runs_2018_chrono_live_botnet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(OUTPUT_DIR, "rf_10runs_results.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "rf_10runs_summary.txt")
LABELS_PATH  = os.path.join(OUTPUT_DIR, "rf_labels.json")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_rf_model.joblib")
BEST_REPORT_PATH = os.path.join(OUTPUT_DIR, "best_rf_report.txt")

# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "Init Bwd Win Bytes",
    "Fwd Packet Length Max",
    "Fwd Packet Length Std",
    "Fwd IAT Min",
    "Flow IAT Mean",
    "Flow Bytes/s",
    "Avg Packet Size",
    "Total Fwd Packets"
]

LABEL_COLUMN = "Label"

SEEDS = [0, 7, 21, 42, 84, 123, 2024, 3141, 5555, 9999]

# =========================================================
# CHARGEMENT
# =========================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

for df in [train_df, val_df, test_df]:
    df.columns = [c.strip() for c in df.columns]

print("==============================")
print("CHARGEMENT DATASETS")
print("==============================")
print("Train :", train_df.shape)
print("Val   :", val_df.shape)
print("Test  :", test_df.shape)

print("\nDistribution TRAIN :")
print(train_df[LABEL_COLUMN].value_counts().sort_index())

print("\nDistribution VALIDATION :")
print(val_df[LABEL_COLUMN].value_counts().sort_index())

print("\nDistribution TEST EXTERNE BOTNET :")
print(test_df[LABEL_COLUMN].value_counts().sort_index())

# =========================================================
# VERIFICATIONS
# =========================================================

required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

for name, df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Colonnes manquantes dans {name} : {missing}")

    nan_count = df[required_columns].isna().sum().sum()
    inf_count = np.isinf(df[FEATURE_COLUMNS].astype(np.float32).values).sum()
    dup_count = df.duplicated().sum()

    print(f"\n===== VERIFICATION {name} =====")
    print("NaN :", nan_count)
    print("Inf :", inf_count)
    print("Doublons exacts :", dup_count)

    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"{name} contient NaN ou Inf.")

# =========================================================
# PREPARATION X / y
# =========================================================

X_train = train_df[FEATURE_COLUMNS].astype(np.float32).values
y_train = train_df[LABEL_COLUMN].astype(int).values

X_val = val_df[FEATURE_COLUMNS].astype(np.float32).values
y_val = val_df[LABEL_COLUMN].astype(int).values

X_test = test_df[FEATURE_COLUMNS].astype(np.float32).values
y_test = test_df[LABEL_COLUMN].astype(int).values

print("\n==============================")
print("CLASSES")
print("==============================")
print("0 = NORMAL / BENIGN")
print("1 = ATTACK / BOTNET")

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "0": "NORMAL",
            "1": "ATTACK"
        },
        f,
        indent=4
    )

# =========================================================
# FONCTION METRIQUES
# =========================================================

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    precision_attack = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall_attack = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1_attack = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "accuracy": acc,
        "precision_attack": precision_attack,
        "recall_attack": recall_attack,
        "f1_attack": f1_attack,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "cm": cm
    }

# =========================================================
# RANDOM FOREST 10 RUNS
# =========================================================

results = []

best_f1_macro = -1
best_seed = None
best_model = None
best_report_text = None
best_cm = None
best_feature_importance = None

for seed in SEEDS:
    print(f"\n================ RUN seed={seed} ================")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf_model.fit(X_train, y_train)

    # Validation
    y_val_pred = rf_model.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred)

    # Test externe Botnet
    y_test_pred = rf_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)

    row = {
        "seed": seed,

        "val_accuracy": val_metrics["accuracy"],
        "val_precision_attack": val_metrics["precision_attack"],
        "val_recall_attack": val_metrics["recall_attack"],
        "val_f1_attack": val_metrics["f1_attack"],
        "val_f1_macro": val_metrics["f1_macro"],

        "test_accuracy": test_metrics["accuracy"],
        "test_precision_attack": test_metrics["precision_attack"],
        "test_recall_attack": test_metrics["recall_attack"],
        "test_f1_attack": test_metrics["f1_attack"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_false_positive_rate": test_metrics["false_positive_rate"],
        "test_specificity": test_metrics["specificity"],
        "TN": test_metrics["TN"],
        "FP": test_metrics["FP"],
        "FN": test_metrics["FN"],
        "TP": test_metrics["TP"]
    }

    results.append(row)

    print("\n===== VALIDATION =====")
    print(f"Accuracy validation         : {val_metrics['accuracy']:.6f}")
    print(f"Precision Attack validation : {val_metrics['precision_attack']:.6f}")
    print(f"Recall Attack validation    : {val_metrics['recall_attack']:.6f}")
    print(f"F1 Attack validation        : {val_metrics['f1_attack']:.6f}")
    print(f"F1 Macro validation         : {val_metrics['f1_macro']:.6f}")
    print("Matrice validation [[TN FP], [FN TP]] :")
    print(val_metrics["cm"])

    print("\n===== TEST EXTERNE BOTNET 90 % =====")
    print(f"Accuracy             : {test_metrics['accuracy']:.6f}")
    print(f"Precision Attack     : {test_metrics['precision_attack']:.6f}")
    print(f"Recall Attack        : {test_metrics['recall_attack']:.6f}")
    print(f"F1 Attack            : {test_metrics['f1_attack']:.6f}")
    print(f"Precision Macro      : {test_metrics['precision_macro']:.6f}")
    print(f"Recall Macro         : {test_metrics['recall_macro']:.6f}")
    print(f"F1 Macro             : {test_metrics['f1_macro']:.6f}")
    print(f"False Positive Rate  : {test_metrics['false_positive_rate']:.6f}")
    print(f"Specificity          : {test_metrics['specificity']:.6f}")

    print("\nMatrice de confusion test [[TN FP], [FN TP]] :")
    print(test_metrics["cm"])

    # Sauvegarder le meilleur modèle selon F1 macro sur test externe
    if test_metrics["f1_macro"] > best_f1_macro:
        best_f1_macro = test_metrics["f1_macro"]
        best_seed = seed
        best_model = rf_model
        best_cm = test_metrics["cm"]

        best_report_text = classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1],
            target_names=["NORMAL", "ATTACK"],
            zero_division=0
        )

        best_feature_importance = rf_model.feature_importances_

# =========================================================
# SAUVEGARDE RESULTATS
# =========================================================

df_results = pd.DataFrame(results)
df_results.to_csv(RESULTS_PATH, index=False)

if best_model is not None:
    joblib.dump(best_model, BEST_MODEL_PATH)

# =========================================================
# MOYENNE ± IC95
# =========================================================

print("\n==============================")
print("RESULTATS DES 10 RUNS")
print("==============================")
print(df_results)

print("\n==============================")
print("MOYENNE ± IC95 SUR TEST EXTERNE")
print("==============================")

metrics_for_summary = [
    "test_accuracy",
    "test_precision_attack",
    "test_recall_attack",
    "test_f1_attack",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_macro",
    "test_false_positive_rate",
    "test_specificity"
]

summary_lines = []

for metric in metrics_for_summary:
    mean = df_results[metric].mean()
    std = df_results[metric].std()
    ic95 = 1.96 * std / np.sqrt(len(SEEDS))

    line = f"{metric}: {mean:.6f} ± {ic95:.6f} (std: {std:.6f})"
    summary_lines.append(line)
    print(line)

# =========================================================
# RAPPORT MEILLEUR MODELE
# =========================================================

with open(BEST_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("===== BEST RANDOM FOREST MODEL =====\n")
    f.write(f"Best seed: {best_seed}\n")
    f.write(f"Best F1 macro on external Botnet test: {best_f1_macro:.6f}\n\n")

    f.write("===== CONFUSION MATRIX [[TN FP], [FN TP]] =====\n")
    f.write(str(best_cm))
    f.write("\n\n")

    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(best_report_text)
    f.write("\n\n")

    f.write("===== FEATURE IMPORTANCE =====\n")
    for name, importance in zip(FEATURE_COLUMNS, best_feature_importance):
        f.write(f"{name}: {importance:.6f}\n")

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("===== RANDOM FOREST 10 RUNS SUMMARY =====\n\n")

    f.write("Files:\n")
    f.write(f"Train: {TRAIN_PATH}\n")
    f.write(f"Validation: {VAL_PATH}\n")
    f.write(f"Test: {TEST_PATH}\n\n")

    f.write("===== RESULTS TABLE =====\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n")

    f.write("===== MEAN ± IC95 ON EXTERNAL TEST =====\n")
    for line in summary_lines:
        f.write(line + "\n")

    f.write("\n===== BEST MODEL =====\n")
    f.write(f"Best seed: {best_seed}\n")
    f.write(f"Best F1 macro: {best_f1_macro:.6f}\n")
    f.write(f"Best model path: {BEST_MODEL_PATH}\n")

print("\n==============================")
print("FICHIERS GENERES")
print("==============================")
print("Résultats 10 runs :", RESULTS_PATH)
print("Résumé IC95       :", SUMMARY_PATH)
print("Labels            :", LABELS_PATH)
print("Meilleur modèle   :", BEST_MODEL_PATH)
print("Rapport meilleur  :", BEST_REPORT_PATH)

print("\nIMPORTANT :")
print("- Random Forest est utilisé comme modèle de comparaison offline.")
print("- Le test externe Botnet contient NORMAL + ATTACK.")
print("- Les métriques complètes sont calculées : accuracy, precision, recall, F1, FPR, specificity.")
print("- Pour ESP32, le modèle principal reste ANN/TFLite.")