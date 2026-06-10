# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:34:53 2026

@author: DELL
"""

# -*- coding: utf-8 -*-

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# CHEMINS
# =========================================================

TRAIN_PATH = r"C:\Users\DELL\Desktop\pfe 2023\result\train_clean.csv"
VAL_PATH   = r"C:\Users\DELL\Desktop\pfe 2023\result\val_clean.csv"
TEST_PATH  = r"C:\Users\DELL\Desktop\pfe 2023\result\test_clean_no_similarity.csv"

OUTPUT_DIR = r"C:\Users\DELL\Desktop\pfe 2023\result\rf_final_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_final_4classes.pkl")
LABELS_JSON_PATH = os.path.join(OUTPUT_DIR, "rf_labels.json")
METRICS_PATH = os.path.join(OUTPUT_DIR, "rf_final_metrics.txt")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(OUTPUT_DIR, "rf_confusion_matrix.png")
FEATURE_IMPORTANCE_PLOT_PATH = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")

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

BEST_SEED = 0

N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1

# =========================================================
# 1) CHARGER TRAIN / VALIDATION / TEST
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
# 2) PREPARATION X / y
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

with open(LABELS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump({str(i): cls for i, cls in enumerate(class_names)}, f, indent=4)

# =========================================================
# 3) MODELE RANDOM FOREST FINAL
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=BEST_SEED,
    n_jobs=-1,
    class_weight="balanced"
)

print("\n===== ENTRAINEMENT RANDOM FOREST FINAL =====")
rf_model.fit(X_train, y_train)

joblib.dump(rf_model, MODEL_PATH)

# =========================================================
# 4) EVALUATION TEST
# =========================================================

y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\n===== RESULTATS TEST FINAL RF =====")
print(f"Best seed          : {BEST_SEED}")
print(f"Accuracy Score     : {acc:.6f}")
print(f"Precision Macro    : {precision_macro:.6f}")
print(f"Recall Macro       : {recall_macro:.6f}")
print(f"F1 Macro           : {f1_macro:.6f}")
print(f"Precision Weighted : {precision_weighted:.6f}")
print(f"Recall Weighted    : {recall_weighted:.6f}")
print(f"F1 Weighted        : {f1_weighted:.6f}")

print("\n===== CLASSIFICATION REPORT =====")
print(report)

print("\n===== CONFUSION MATRIX =====")
print(cm)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("===== RESULTATS TEST FINAL RF =====\n")
    f.write(f"Best seed          : {BEST_SEED}\n")
    f.write(f"Accuracy Score     : {acc:.6f}\n")
    f.write(f"Precision Macro    : {precision_macro:.6f}\n")
    f.write(f"Recall Macro       : {recall_macro:.6f}\n")
    f.write(f"F1 Macro           : {f1_macro:.6f}\n")
    f.write(f"Precision Weighted : {precision_weighted:.6f}\n")
    f.write(f"Recall Weighted    : {recall_weighted:.6f}\n")
    f.write(f"F1 Weighted        : {f1_weighted:.6f}\n\n")
    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(report)
    f.write("\n===== CONFUSION MATRIX =====\n")
    f.write(np.array2string(cm))
    f.write("\n")

# =========================================================
# 5) MATRICE DE CONFUSION
# =========================================================

plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("Random Forest 4 Classes - Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=300)
plt.show()

# =========================================================
# 6) FEATURE IMPORTANCE
# =========================================================

importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(
    range(len(importances)),
    [FEATURE_COLUMNS[i] for i in sorted_idx],
    rotation=45,
    ha="right"
)
plt.title("Random Forest - Feature Importance")
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=300)
plt.show()

print("\n===== FEATURE IMPORTANCE =====")
for i in sorted_idx:
    print(f"{FEATURE_COLUMNS[i]} : {importances[i]:.6f}")

# =========================================================
# 7) FIN
# =========================================================

print("\n===== FICHIERS GENERES =====")
print("Modele RF              :", MODEL_PATH)
print("Metrics txt            :", METRICS_PATH)
print("Labels json            :", LABELS_JSON_PATH)
print("Confusion matrix       :", CONFUSION_MATRIX_PLOT_PATH)
print("Feature importance     :", FEATURE_IMPORTANCE_PLOT_PATH)