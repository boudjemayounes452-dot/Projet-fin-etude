# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:55:10 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Random Forest final - 7 FEATURES
CICIDS2018 + Botnet 10 % SANS LIVE

TRAIN :
train_2018_9features_botnet10_encoded.csv

VALIDATION :
val_2018_9features_botnet10_encoded.csv

TEST EXTERNE :
test_botnet90_9features_external_encoded.csv

Remarque :
Les fichiers contiennent 9 features, mais ce modèle utilise seulement 7 features.
Paramètres RF inchangés.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

BASE_DIR = r"C:\Users\DELL\Desktop\DDOS12\PFE_DDOS\PFE_DDOS\scripts"

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "train_2018_9features_botnet10_encoded.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR,
    "val_2018_9features_botnet10_encoded.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "test_botnet90_9features_external_encoded.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "rf_final_2018_7features_botnet10_sans_live"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_final_2018_7features_botnet10_sans_live.joblib"
)

LABELS_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_7features_labels.json"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_7features_metrics.txt"
)

CONFUSION_MATRIX_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_7features_confusion_matrix.png"
)

FEATURE_IMPORTANCE_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_7features_feature_importance.png"
)

# =========================================================
# CONFIGURATION 7 FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "Init Bwd Win Bytes",
    "Fwd Packet Length Max",
    "Fwd Packet Length Std",
    "Fwd IAT Min",
    "Flow IAT Mean",
    "Avg Packet Size",
    "Total Fwd Packets"
]

LABEL_COLUMN = "Label"

BEST_SEED = 0

# mêmes paramètres que le premier modèle RF
N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1

CLASS_NAMES = ["NORMAL", "ATTACK"]

# =========================================================
# 1) CHARGER TRAIN / VALIDATION / TEST
# =========================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

for df in [train_df, val_df, test_df]:
    df.columns = [c.strip() for c in df.columns]

print("===== DATASETS =====")
print("Train :", train_df.shape)
print("Val   :", val_df.shape)
print("Test  :", test_df.shape)

print("\n===== DISTRIBUTION TRAIN =====")
print(train_df[LABEL_COLUMN].value_counts().sort_index())

print("\n===== DISTRIBUTION VALIDATION =====")
print(val_df[LABEL_COLUMN].value_counts().sort_index())

print("\n===== DISTRIBUTION TEST EXTERNE BOTNET 90 % =====")
print(test_df[LABEL_COLUMN].value_counts().sort_index())

# =========================================================
# 2) VERIFICATION DES COLONNES / NaN / INF
# =========================================================

required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

for name, df in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:
    missing_cols = [c for c in required_columns if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans {name} : {missing_cols}")

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
# 3) PREPARATION X / y
# =========================================================

X_train = train_df[FEATURE_COLUMNS].astype(np.float32).values
X_val = val_df[FEATURE_COLUMNS].astype(np.float32).values
X_test = test_df[FEATURE_COLUMNS].astype(np.float32).values

y_train = train_df[LABEL_COLUMN].astype(int).values
y_val = val_df[LABEL_COLUMN].astype(int).values
y_test = test_df[LABEL_COLUMN].astype(int).values

for name, y in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test)
]:
    unique_labels = sorted(np.unique(y).tolist())
    print(f"\nLabels dans {name} :", unique_labels)

    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels incorrects dans {name} : {unique_labels}")

print("\n===== CLASSES =====")
print("0 -> NORMAL / BENIGN")
print("1 -> ATTACK / BOTNET")

with open(LABELS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "0": "NORMAL",
            "1": "ATTACK"
        },
        f,
        indent=4
    )

# =========================================================
# 4) MODELE RANDOM FOREST FINAL
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

print("\n===== ENTRAINEMENT RANDOM FOREST FINAL 7 FEATURES =====")
print("Best seed :", BEST_SEED)
print("N estimators :", N_ESTIMATORS)
print("Max depth :", MAX_DEPTH)
print("Min samples split :", MIN_SAMPLES_SPLIT)
print("Min samples leaf :", MIN_SAMPLES_LEAF)
print("Class weight : balanced")
print("Nombre de features :", len(FEATURE_COLUMNS))

rf_model.fit(X_train, y_train)

joblib.dump(rf_model, MODEL_PATH)

# =========================================================
# 5) EVALUATION VALIDATION
# =========================================================

y_val_pred = rf_model.predict(X_val)

val_acc = accuracy_score(y_val, y_val_pred)

val_precision_attack = precision_score(
    y_val,
    y_val_pred,
    pos_label=1,
    zero_division=0
)

val_recall_attack = recall_score(
    y_val,
    y_val_pred,
    pos_label=1,
    zero_division=0
)

val_f1_attack = f1_score(
    y_val,
    y_val_pred,
    pos_label=1,
    zero_division=0
)

val_f1_macro = f1_score(
    y_val,
    y_val_pred,
    average="macro",
    zero_division=0
)

val_cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])

print("\n===== RESULTATS VALIDATION =====")
print(f"Accuracy validation         : {val_acc:.6f}")
print(f"Precision Attack validation : {val_precision_attack:.6f}")
print(f"Recall Attack validation    : {val_recall_attack:.6f}")
print(f"F1 Attack validation        : {val_f1_attack:.6f}")
print(f"F1 Macro validation         : {val_f1_macro:.6f}")
print("\nMatrice validation [[TN FP], [FN TP]] :")
print(val_cm)

# =========================================================
# 6) EVALUATION TEST EXTERNE BOTNET 90 %
# =========================================================

y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

precision_attack = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall_attack = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1_attack = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

normal_total = tn + fp
attack_total = fn + tp

normal_correct = tn
attack_correct = tp

normal_detection_rate = normal_correct / normal_total if normal_total > 0 else 0
attack_detection_rate = attack_correct / attack_total if attack_total > 0 else 0

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\n===== RESULTATS TEST EXTERNE BOTNET 90 % =====")
print(f"Best seed          : {BEST_SEED}")
print(f"Accuracy           : {acc:.6f}")
print(f"Precision Attack   : {precision_attack:.6f}")
print(f"Recall Attack      : {recall_attack:.6f}")
print(f"F1 Attack          : {f1_attack:.6f}")
print(f"Precision Macro    : {precision_macro:.6f}")
print(f"Recall Macro       : {recall_macro:.6f}")
print(f"F1 Macro           : {f1_macro:.6f}")
print(f"Precision Weighted : {precision_weighted:.6f}")
print(f"Recall Weighted    : {recall_weighted:.6f}")
print(f"F1 Weighted        : {f1_weighted:.6f}")
print(f"False Positive Rate: {false_positive_rate:.6f}")
print(f"Specificity        : {specificity:.6f}")

print("\n===== MATRICE DE CONFUSION TEST [[TN FP], [FN TP]] =====")
print(cm)

print("\n===== DETAILS TEST =====")
print("Total NORMAL test       :", normal_total)
print("NORMAL bien classés     :", normal_correct)
print("NORMAL classés ATTACK   :", fp)
print("Taux NORMAL bien classés:", normal_detection_rate)

print("\nTotal ATTACK test       :", attack_total)
print("ATTACK bien détectées   :", attack_correct)
print("ATTACK ratées NORMAL    :", fn)
print("Taux détection ATTACK   :", attack_detection_rate)

print("\n===== CLASSIFICATION REPORT =====")
print(report)

# =========================================================
# 7) SAUVEGARDE METRIQUES
# =========================================================

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("===== RANDOM FOREST FINAL - 2018 7 FEATURES + BOTNET 10 % SANS LIVE =====\n\n")

    f.write("===== FICHIERS UTILISES =====\n")
    f.write(f"Train      : {TRAIN_PATH}\n")
    f.write(f"Validation : {VAL_PATH}\n")
    f.write(f"Test       : {TEST_PATH}\n\n")

    f.write("===== CONFIGURATION =====\n")
    f.write(f"Best seed         : {BEST_SEED}\n")
    f.write(f"N estimators      : {N_ESTIMATORS}\n")
    f.write(f"Max depth         : {MAX_DEPTH}\n")
    f.write(f"Min samples split : {MIN_SAMPLES_SPLIT}\n")
    f.write(f"Min samples leaf  : {MIN_SAMPLES_LEAF}\n")
    f.write(f"Class weight      : balanced\n")
    f.write(f"Nombre features   : {len(FEATURE_COLUMNS)}\n\n")

    f.write("===== FEATURES =====\n")
    for col in FEATURE_COLUMNS:
        f.write(f"- {col}\n")
    f.write("\n")

    f.write("===== DISTRIBUTIONS =====\n")
    f.write("Train:\n")
    f.write(str(train_df[LABEL_COLUMN].value_counts().sort_index()))
    f.write("\n\nValidation:\n")
    f.write(str(val_df[LABEL_COLUMN].value_counts().sort_index()))
    f.write("\n\nTest:\n")
    f.write(str(test_df[LABEL_COLUMN].value_counts().sort_index()))
    f.write("\n\n")

    f.write("===== RESULTATS VALIDATION =====\n")
    f.write(f"Accuracy validation         : {val_acc:.6f}\n")
    f.write(f"Precision Attack validation : {val_precision_attack:.6f}\n")
    f.write(f"Recall Attack validation    : {val_recall_attack:.6f}\n")
    f.write(f"F1 Attack validation        : {val_f1_attack:.6f}\n")
    f.write(f"F1 Macro validation         : {val_f1_macro:.6f}\n")
    f.write("Matrice validation [[TN FP], [FN TP]] :\n")
    f.write(str(val_cm))
    f.write("\n\n")

    f.write("===== RESULTATS TEST EXTERNE BOTNET 90 % =====\n")
    f.write(f"Accuracy           : {acc:.6f}\n")
    f.write(f"Precision Attack   : {precision_attack:.6f}\n")
    f.write(f"Recall Attack      : {recall_attack:.6f}\n")
    f.write(f"F1 Attack          : {f1_attack:.6f}\n")
    f.write(f"Precision Macro    : {precision_macro:.6f}\n")
    f.write(f"Recall Macro       : {recall_macro:.6f}\n")
    f.write(f"F1 Macro           : {f1_macro:.6f}\n")
    f.write(f"Precision Weighted : {precision_weighted:.6f}\n")
    f.write(f"Recall Weighted    : {recall_weighted:.6f}\n")
    f.write(f"F1 Weighted        : {f1_weighted:.6f}\n")
    f.write(f"False Positive Rate: {false_positive_rate:.6f}\n")
    f.write(f"Specificity        : {specificity:.6f}\n\n")

    f.write("===== MATRICE DE CONFUSION TEST [[TN FP], [FN TP]] =====\n")
    f.write(np.array2string(cm))
    f.write("\n\n")

    f.write("===== DETAILS TEST =====\n")
    f.write(f"Total NORMAL test        : {normal_total}\n")
    f.write(f"NORMAL bien classés      : {normal_correct}\n")
    f.write(f"NORMAL classés ATTACK    : {fp}\n")
    f.write(f"Taux NORMAL bien classés : {normal_detection_rate:.6f}\n\n")

    f.write(f"Total ATTACK test        : {attack_total}\n")
    f.write(f"ATTACK bien détectées    : {attack_correct}\n")
    f.write(f"ATTACK ratées NORMAL     : {fn}\n")
    f.write(f"Taux détection ATTACK    : {attack_detection_rate:.6f}\n\n")

    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(report)
    f.write("\n")

# =========================================================
# 8) MATRICE DE CONFUSION EN IMAGE
# =========================================================

plt.figure(figsize=(7, 7))
plt.imshow(cm, interpolation="nearest")
plt.title("Random Forest 7 features - External Botnet 90% Test")
plt.colorbar()

tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES)
plt.yticks(tick_marks, CLASS_NAMES)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=300)
plt.show()

# =========================================================
# 9) FEATURE IMPORTANCE
# =========================================================

importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(9, 5))
plt.bar(range(len(importances)), importances[sorted_idx])
plt.xticks(
    range(len(importances)),
    [FEATURE_COLUMNS[i] for i in sorted_idx],
    rotation=45,
    ha="right"
)
plt.title("Random Forest 7 features - Feature Importance")
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=300)
plt.show()

print("\n===== FEATURE IMPORTANCE =====")
for i in sorted_idx:
    print(f"{FEATURE_COLUMNS[i]} : {importances[i]:.6f}")

with open(METRICS_PATH, "a", encoding="utf-8") as f:
    f.write("\n===== FEATURE IMPORTANCE =====\n")
    for i in sorted_idx:
        f.write(f"{FEATURE_COLUMNS[i]} : {importances[i]:.6f}\n")

# =========================================================
# 10) FIN
# =========================================================

print("\n===== FICHIERS GENERES =====")
print("Modèle RF              :", MODEL_PATH)
print("Metrics txt            :", METRICS_PATH)
print("Labels json            :", LABELS_JSON_PATH)
print("Confusion matrix       :", CONFUSION_MATRIX_PLOT_PATH)
print("Feature importance     :", FEATURE_IMPORTANCE_PLOT_PATH)

print("\nIMPORTANT :")
print("- Paramètres RF inchangés par rapport au premier modèle.")
print("- Version 7 features : Flow Bytes/s et Avg Fwd Segment Size ne sont pas utilisées.")
print("- Version sans live.")
print("- Train/validation : CICIDS2018 + début 10 % Botnet.")
print("- Test externe : fin 90 % Botnet.")
print("- Le test externe Botnet contient NORMAL + ATTACK.")