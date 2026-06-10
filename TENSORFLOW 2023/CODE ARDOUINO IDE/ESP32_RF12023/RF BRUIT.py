# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:48:49 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Random Forest final - CICIDS2018 + Robustesse bruit + adversarial simplifie

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
    "train_2018_chrono_live_botnet10_encoded.csv"
)

VAL_PATH = os.path.join(
    BASE_DIR,
    "val_2018_chrono_live_botnet10_encoded.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "test_botnet90_external_chrono_encoded.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "rf_2018_robustesse_bruit_adversarial"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_final_2018_robustesse.joblib"
)

LABELS_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_labels.json"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_metrics_robustesse.txt"
)

CONFUSION_MATRIX_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_confusion_matrix.png"
)

FEATURE_IMPORTANCE_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_feature_importance.png"
)

ROBUSTNESS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_robustesse_bruit_adversarial_cicids2018.csv"
)

ROBUSTNESS_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_robustesse_bruit_adversarial_cicids2018.png"
)

# =========================================================
# CONFIGURATION
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

CLASS_NAMES = ["NORMAL", "ATTACK"]

BEST_SEED = 0

N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1

SAMPLE_SIZE_ROBUSTNESS = 5000

NOISE_LEVELS = [0.01, 0.03, 0.05, 0.10]
ADVERSARIAL_LEVELS = [0.01, 0.03, 0.05, 0.10]

np.random.seed(BEST_SEED)

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

print("\n===== DISTRIBUTION TEST =====")
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
print("0 -> NORMAL")
print("1 -> ATTACK")

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
# 4) ENTRAINEMENT RANDOM FOREST
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

print("\n===== ENTRAINEMENT RANDOM FOREST =====")
print("Best seed :", BEST_SEED)
print("N estimators :", N_ESTIMATORS)

rf_model.fit(X_train, y_train)

joblib.dump(rf_model, MODEL_PATH)

print("Modele RF sauvegarde :", MODEL_PATH)

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
# 6) EVALUATION TEST SANS PERTURBATION
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

print("\n===== RESULTATS TEST SANS PERTURBATION - RF =====")
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

# =========================================================
# 7) MATRICE DE CONFUSION EN IMAGE
# =========================================================

plt.figure(figsize=(7, 7))
plt.imshow(cm, interpolation="nearest")
plt.title("Random Forest - Test sans perturbation")
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

print("Figure matrice de confusion sauvegardee :", CONFUSION_MATRIX_PLOT_PATH)

# =========================================================
# 8) FEATURE IMPORTANCE
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
plt.title("Random Forest - Feature Importance")
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=300)
plt.show()

print("\n===== FEATURE IMPORTANCE =====")
for i in sorted_idx:
    print(f"{FEATURE_COLUMNS[i]} : {importances[i]:.6f}")

# =========================================================
# 9) ANALYSE DE ROBUSTESSE : BRUIT + ADVERSARIAL SIMPLIFIE
# =========================================================

print("\n===== ANALYSE DE ROBUSTESSE - RANDOM FOREST CICIDS2018 =====")

if X_test.shape[0] > SAMPLE_SIZE_ROBUSTNESS:
    idx = np.random.choice(
        X_test.shape[0],
        SAMPLE_SIZE_ROBUSTNESS,
        replace=False
    )
    X_test_robust = X_test[idx].astype(np.float32)
    y_test_robust = y_test[idx]
else:
    X_test_robust = X_test.astype(np.float32)
    y_test_robust = y_test

print("Nombre total test :", X_test.shape[0])
print("Nombre utilise pour robustesse :", X_test_robust.shape[0])

# Résultat sans perturbation sur le même échantillon
y_base_pred = rf_model.predict(X_test_robust)

baseline_acc = accuracy_score(y_test_robust, y_base_pred)

baseline_f1_macro = f1_score(
    y_test_robust,
    y_base_pred,
    average="macro",
    zero_division=0
)

robustness_results = []

robustness_results.append({
    "Dataset": "CICIDS2018",
    "Modele": "Random Forest",
    "Perturbation": "Aucune",
    "Niveau": "0%",
    "Accuracy": baseline_acc,
    "F1_macro": baseline_f1_macro,
    "Perte_F1_macro": 0.0
})

print("\n--- Sans perturbation ---")
print(f"Accuracy : {baseline_acc:.6f}")
print(f"F1 Macro : {baseline_f1_macro:.6f}")

# Ecart-type des features pour bruit proportionnel
feature_std = np.std(X_test_robust, axis=0)
feature_std[feature_std == 0] = 1.0

# =========================================================
# 9-A) BRUIT ALEATOIRE
# =========================================================

for noise_level in NOISE_LEVELS:

    noise = np.random.normal(
        loc=0.0,
        scale=noise_level,
        size=X_test_robust.shape
    ).astype(np.float32)

    X_test_noisy = X_test_robust + noise * feature_std

    y_noisy_pred = rf_model.predict(X_test_noisy)

    noisy_acc = accuracy_score(y_test_robust, y_noisy_pred)

    noisy_f1_macro = f1_score(
        y_test_robust,
        y_noisy_pred,
        average="macro",
        zero_division=0
    )

    perte_f1_macro = baseline_f1_macro - noisy_f1_macro

    robustness_results.append({
        "Dataset": "CICIDS2018",
        "Modele": "Random Forest",
        "Perturbation": "Bruit aleatoire",
        "Niveau": f"{int(noise_level * 100)}%",
        "Accuracy": noisy_acc,
        "F1_macro": noisy_f1_macro,
        "Perte_F1_macro": perte_f1_macro
    })

    print(f"\n--- Bruit {int(noise_level * 100)}% ---")
    print(f"Accuracy bruitee  : {noisy_acc:.6f}")
    print(f"F1 Macro bruite   : {noisy_f1_macro:.6f}")
    print(f"Perte F1 Macro    : {perte_f1_macro:.6f}")

    del noise
    del X_test_noisy
    del y_noisy_pred

# =========================================================
# 9-B) ADVERSARIAL SIMPLIFIE
# =========================================================
# Random Forest n'est pas différentiable, donc FGSM n'est pas applicable.
# On teste +epsilon et -epsilon, puis on garde la direction qui baisse le plus le F1-score.

print("\n===== ADVERSARIAL SIMPLIFIE - RANDOM FOREST =====")

for adv_level in ADVERSARIAL_LEVELS:

    X_adv_plus = X_test_robust + adv_level * feature_std
    X_adv_minus = X_test_robust - adv_level * feature_std

    y_adv_plus = rf_model.predict(X_adv_plus)
    y_adv_minus = rf_model.predict(X_adv_minus)

    f1_plus = f1_score(
        y_test_robust,
        y_adv_plus,
        average="macro",
        zero_division=0
    )

    f1_minus = f1_score(
        y_test_robust,
        y_adv_minus,
        average="macro",
        zero_division=0
    )

    if f1_plus <= f1_minus:
        y_adv_pred = y_adv_plus
        adv_direction = "+epsilon"
        adv_f1_macro = f1_plus
    else:
        y_adv_pred = y_adv_minus
        adv_direction = "-epsilon"
        adv_f1_macro = f1_minus

    adv_acc = accuracy_score(y_test_robust, y_adv_pred)

    perte_f1_macro = baseline_f1_macro - adv_f1_macro

    robustness_results.append({
        "Dataset": "CICIDS2018",
        "Modele": "Random Forest",
        "Perturbation": "Adversarial simplifie",
        "Niveau": f"{int(adv_level * 100)}% ({adv_direction})",
        "Accuracy": adv_acc,
        "F1_macro": adv_f1_macro,
        "Perte_F1_macro": perte_f1_macro
    })

    print(f"\n--- Adversarial simplifie {int(adv_level * 100)}% ---")
    print("Direction choisie       :", adv_direction)
    print(f"Accuracy adversarial    : {adv_acc:.6f}")
    print(f"F1 Macro adversarial    : {adv_f1_macro:.6f}")
    print(f"Perte F1 Macro          : {perte_f1_macro:.6f}")

    del X_adv_plus
    del X_adv_minus
    del y_adv_plus
    del y_adv_minus
    del y_adv_pred

# =========================================================
# 10) SAUVEGARDE RESULTATS ROBUSTESSE
# =========================================================

robustness_df = pd.DataFrame(robustness_results)

print("\n===== TABLEAU ROBUSTESSE RF =====")
print(robustness_df)

robustness_df.to_csv(ROBUSTNESS_CSV_PATH, index=False)

print("\nFichier robustesse sauvegarde :", ROBUSTNESS_CSV_PATH)

# =========================================================
# 11) FIGURE ROBUSTESSE
# =========================================================

plt.figure(figsize=(10, 5))

for perturbation in robustness_df["Perturbation"].unique():
    subset = robustness_df[robustness_df["Perturbation"] == perturbation]
    plt.plot(
        subset["Niveau"],
        subset["F1_macro"],
        marker="o",
        label=perturbation
    )

plt.title("Robustesse bruit et adversarial - Random Forest CICIDS2018")
plt.xlabel("Niveau de perturbation")
plt.ylabel("F1-score macro")
plt.grid(True)
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(ROBUSTNESS_PLOT_PATH, dpi=300)
plt.show()

print("Figure robustesse sauvegardee :", ROBUSTNESS_PLOT_PATH)

# =========================================================
# 12) SAUVEGARDE METRIQUES
# =========================================================

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("===== RANDOM FOREST FINAL + ROBUSTESSE - CICIDS2018 =====\n\n")

    f.write("===== FICHIERS UTILISES =====\n")
    f.write(f"Train      : {TRAIN_PATH}\n")
    f.write(f"Validation : {VAL_PATH}\n")
    f.write(f"Test       : {TEST_PATH}\n\n")

    f.write("===== CONFIGURATION =====\n")
    f.write(f"Best seed         : {BEST_SEED}\n")
    f.write(f"N estimators      : {N_ESTIMATORS}\n")
    f.write(f"Max depth         : {MAX_DEPTH}\n")
    f.write(f"Class weight      : balanced\n")
    f.write(f"Robustesse sample : {X_test_robust.shape[0]}\n")
    f.write(f"Niveaux bruit     : {NOISE_LEVELS}\n")
    f.write(f"Niveaux adversarial : {ADVERSARIAL_LEVELS}\n\n")

    f.write("===== FEATURES =====\n")
    for col in FEATURE_COLUMNS:
        f.write(f"- {col}\n")
    f.write("\n")

    f.write("===== RESULTATS VALIDATION =====\n")
    f.write(f"Accuracy validation         : {val_acc:.6f}\n")
    f.write(f"Precision Attack validation : {val_precision_attack:.6f}\n")
    f.write(f"Recall Attack validation    : {val_recall_attack:.6f}\n")
    f.write(f"F1 Attack validation        : {val_f1_attack:.6f}\n")
    f.write(f"F1 Macro validation         : {val_f1_macro:.6f}\n")
    f.write("Matrice validation [[TN FP], [FN TP]] :\n")
    f.write(str(val_cm))
    f.write("\n\n")

    f.write("===== RESULTATS TEST SANS PERTURBATION =====\n")
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
    f.write(f"NORMAL bien classes      : {normal_correct}\n")
    f.write(f"NORMAL classes ATTACK    : {fp}\n")
    f.write(f"Taux NORMAL bien classes : {normal_detection_rate:.6f}\n\n")

    f.write(f"Total ATTACK test        : {attack_total}\n")
    f.write(f"ATTACK bien detectees    : {attack_correct}\n")
    f.write(f"ATTACK ratees NORMAL     : {fn}\n")
    f.write(f"Taux detection ATTACK    : {attack_detection_rate:.6f}\n\n")

    f.write("===== FEATURE IMPORTANCE =====\n")
    for i in sorted_idx:
        f.write(f"{FEATURE_COLUMNS[i]} : {importances[i]:.6f}\n")
    f.write("\n")

    f.write("===== ROBUSTESSE BRUIT + ADVERSARIAL SIMPLIFIE =====\n")
    f.write(robustness_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(report)
    f.write("\n\n")

    f.write("===== FICHIERS GENERES =====\n")
    f.write(f"Modele RF          : {MODEL_PATH}\n")
    f.write(f"Labels json        : {LABELS_JSON_PATH}\n")
    f.write(f"Matrice confusion  : {CONFUSION_MATRIX_PLOT_PATH}\n")
    f.write(f"Feature importance : {FEATURE_IMPORTANCE_PLOT_PATH}\n")
    f.write(f"CSV robustesse     : {ROBUSTNESS_CSV_PATH}\n")
    f.write(f"Figure robustesse  : {ROBUSTNESS_PLOT_PATH}\n")

# =========================================================
# 13) FIN
# =========================================================

print("\n===== FICHIERS GENERES =====")
print("Modele RF              :", MODEL_PATH)
print("Metrics txt            :", METRICS_PATH)
print("Labels json            :", LABELS_JSON_PATH)
print("Confusion matrix       :", CONFUSION_MATRIX_PLOT_PATH)
print("Feature importance     :", FEATURE_IMPORTANCE_PLOT_PATH)
print("CSV robustesse         :", ROBUSTNESS_CSV_PATH)
print("Figure robustesse      :", ROBUSTNESS_PLOT_PATH)

print("\nIMPORTANT :")
print("- Ce script entraîne Random Forest sur CICIDS2018.")
print("- Le test normal est réalisé sur tout le jeu de test.")
print("- La robustesse est évaluée sur un échantillon de 5000 lignes.")
print("- Random Forest n'utilise pas FGSM car il n'est pas différentiable.")
print("- L'adversarial utilisé ici est une perturbation simplifiée (+epsilon / -epsilon).")
print("- Le fichier CSV contient les résultats à utiliser dans le mémoire.")