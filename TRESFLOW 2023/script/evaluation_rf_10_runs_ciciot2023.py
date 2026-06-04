# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:29:46 2026

@author: DELL
"""

# -*- coding: utf-8 -*-

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# CHEMINS
# =========================================================

TRAIN_PATH = r"C:\Users\DELL\Desktop\pfe 2023\result\train_clean.csv"
VAL_PATH   = r"C:\Users\DELL\Desktop\pfe 2023\result\val_clean.csv"
TEST_PATH  = r"C:\Users\DELL\Desktop\pfe 2023\result\test_clean_no_similarity.csv"

OUTPUT_DIR = r"C:\Users\DELL\Desktop\pfe 2023\result\rf_10runs_clean_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(OUTPUT_DIR, "rf_10runs_results.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "rf_10runs_summary.txt")
LABELS_PATH  = os.path.join(OUTPUT_DIR, "rf_labels.json")

# =========================================================
# CONFIGURATION
# =========================================================

FEATURE_COLUMNS = [
    "IAT",
    "Header_Length",
    "Protocol Type",
    "flow_duration",
    "rst_count",
    "Tot size",
    "Magnitue",
    "AVG"
]

LABEL_COLUMN = "label"

SEEDS = [0, 7, 21, 42, 84, 123, 2024, 3141, 5555, 9999]

N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1

# =========================================================
# CHARGER TRAIN / VAL / TEST
# =========================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

for df in [train_df, val_df, test_df]:
    df.columns = [c.strip() for c in df.columns]

print("===== DATASETS =====")
print("Train :", train_df.shape)
print("Val   :", val_df.shape)
print("Test  :", test_df.shape)

print("\n===== DISTRIBUTION TRAIN =====")
print(train_df[LABEL_COLUMN].value_counts())

print("\n===== DISTRIBUTION TEST =====")
print(test_df[LABEL_COLUMN].value_counts())

# =========================================================
# PREPARATION X / y
# =========================================================

X_train = train_df[FEATURE_COLUMNS].astype(np.float32).values
X_val   = val_df[FEATURE_COLUMNS].astype(np.float32).values
X_test  = test_df[FEATURE_COLUMNS].astype(np.float32).values

y_train_text = train_df[LABEL_COLUMN].astype(str).values
y_val_text   = val_df[LABEL_COLUMN].astype(str).values
y_test_text  = test_df[LABEL_COLUMN].astype(str).values

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_text)
y_val   = label_encoder.transform(y_val_text)
y_test  = label_encoder.transform(y_test_text)

class_names = list(label_encoder.classes_)

expected_classes = {
    "BenignTraffic",
    "DDoS-ICMP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-SYN_Flood"
}

if set(class_names) != expected_classes:
    raise ValueError(f"Classes incorrectes : {class_names}")

print("\n===== CLASSES =====")
for i, cls in enumerate(class_names):
    print(f"{i} -> {cls}")

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump({str(i): cls for i, cls in enumerate(class_names)}, f, indent=4)

# =========================================================
# 10 RUNS RANDOM FOREST
# =========================================================

results = []

for seed in SEEDS:
    print(f"\n================ RUN seed={seed} ================")

    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append({
        "seed": seed,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    })

    print(f"Accuracy        : {acc:.6f}")
    print(f"Precision Macro : {precision_macro:.6f}")
    print(f"Recall Macro    : {recall_macro:.6f}")
    print(f"F1 Macro        : {f1_macro:.6f}")

# =========================================================
# RESULTATS
# =========================================================

df_results = pd.DataFrame(results)
df_results.to_csv(RESULTS_PATH, index=False)

print("\n===== RESULTATS DES 10 RUNS =====")
print(df_results)

print("\n===== MOYENNE ± IC95 =====")

summary_lines = []

for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
    mean = df_results[metric].mean()
    std = df_results[metric].std()
    ic95 = 1.96 * std / np.sqrt(len(SEEDS))

    line = f"{metric}: {mean:.6f} ± {ic95:.6f} (std: {std:.6f})"
    summary_lines.append(line)
    print(line)

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("===== RESULTATS DES 10 RUNS =====\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n===== MOYENNE ± IC95 =====\n")
    for line in summary_lines:
        f.write(line + "\n")

print("\n===== FICHIERS GENERES =====")
print("Résultats 10 runs :", RESULTS_PATH)
print("Résumé IC95       :", SUMMARY_PATH)
print("Labels            :", LABELS_PATH)