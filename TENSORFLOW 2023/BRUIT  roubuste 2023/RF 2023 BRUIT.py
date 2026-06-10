# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:32:42 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
RANDOM FOREST FINAL + ROBUSTESSE BRUIT REPETE + ADVERSARIAL SIMPLIFIE - CICIoT2023

Classes :
BenignTraffic
DDoS-ICMP_Flood
DDoS-UDP_Flood
DDoS-SYN_Flood

Important :
- Random Forest n'utilise pas FGSM car il n'est pas differentiable.
- Le bruit aleatoire est repete 5 fois pour chaque niveau.
- L'adversarial RF est simplifie : +epsilon et -epsilon, puis on garde le cas le plus degradant.
"""

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

OUTPUT_DIR = r"C:\Users\DELL\Desktop\pfe 2023\result\rf_2023_robustesse_bruit_adversarial_moyenne"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_final_4classes.pkl")
LABELS_JSON_PATH = os.path.join(OUTPUT_DIR, "rf_labels.json")
METRICS_PATH = os.path.join(OUTPUT_DIR, "rf_final_metrics.txt")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(OUTPUT_DIR, "rf_confusion_matrix.png")
FEATURE_IMPORTANCE_PLOT_PATH = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")

ROBUSTNESS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_robustesse_bruit_adversarial_ciciot2023.csv"
)

ROBUSTNESS_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_robustesse_bruit_adversarial_ciciot2023.png"
)

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

SAMPLE_SIZE_ROBUSTNESS = 5000

NOISE_LEVELS = [0.01, 0.03, 0.05, 0.10]
ADVERSARIAL_LEVELS = [0.01, 0.03, 0.05, 0.10]

N_REPEATS_NOISE = 5

np.random.seed(BEST_SEED)

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

print("\n===== DISTRIBUTION VALIDATION =====")
print(val_df[LABEL_COLUMN].value_counts())

print("\n===== DISTRIBUTION TEST =====")
print(test_df[LABEL_COLUMN].value_counts())

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

print("\n===== ENTRAINEMENT RANDOM FOREST FINAL =====")
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
val_precision_macro = precision_score(y_val, y_val_pred, average="macro", zero_division=0)
val_recall_macro = recall_score(y_val, y_val_pred, average="macro", zero_division=0)
val_f1_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

val_cm = confusion_matrix(y_val, y_val_pred)

print("\n===== RESULTATS VALIDATION RF =====")
print(f"Accuracy validation      : {val_acc:.6f}")
print(f"Precision Macro val      : {val_precision_macro:.6f}")
print(f"Recall Macro val         : {val_recall_macro:.6f}")
print(f"F1 Macro val             : {val_f1_macro:.6f}")
print("\nMatrice validation :")
print(val_cm)

# =========================================================
# 6) EVALUATION TEST SANS PERTURBATION
# =========================================================

y_pred = rf_model.predict(X_test)

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

print("\n===== RESULTATS TEST SANS PERTURBATION RF =====")
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

# =========================================================
# 7) MATRICE DE CONFUSION
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

print("Figure matrice de confusion sauvegardee :", CONFUSION_MATRIX_PLOT_PATH)

# =========================================================
# 8) FEATURE IMPORTANCE
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
# 9) ANALYSE DE ROBUSTESSE : BRUIT + ADVERSARIAL SIMPLIFIE
# =========================================================

print("\n===== ANALYSE DE ROBUSTESSE - RANDOM FOREST CICIoT2023 =====")

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

# Résultat sans perturbation sur le même ensemble
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
    "Dataset": "CICIoT2023",
    "Modele": "Random Forest",
    "Perturbation": "Aucune",
    "Niveau": "0%",
    "Accuracy": baseline_acc,
    "Accuracy_std": 0.0,
    "F1_macro": baseline_f1_macro,
    "F1_macro_std": 0.0,
    "Perte_F1_macro": 0.0
})

print("\n--- Sans perturbation ---")
print(f"Accuracy : {baseline_acc:.6f}")
print(f"F1 Macro : {baseline_f1_macro:.6f}")

# Ecart-type des features pour bruit proportionnel
feature_std = np.std(X_test_robust, axis=0)
feature_std[feature_std == 0] = 1.0

# =========================================================
# 9-A) BRUIT ALEATOIRE AVEC REPETITIONS
# =========================================================

for noise_level in NOISE_LEVELS:

    acc_values = []
    f1_values = []

    for repeat in range(N_REPEATS_NOISE):

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

        acc_values.append(noisy_acc)
        f1_values.append(noisy_f1_macro)

        del noise
        del X_test_noisy
        del y_noisy_pred

    mean_acc = np.mean(acc_values)
    std_acc = np.std(acc_values)

    mean_f1_macro = np.mean(f1_values)
    std_f1_macro = np.std(f1_values)

    perte_f1_macro = baseline_f1_macro - mean_f1_macro

    robustness_results.append({
        "Dataset": "CICIoT2023",
        "Modele": "Random Forest",
        "Perturbation": "Bruit aleatoire",
        "Niveau": f"{int(noise_level * 100)}%",
        "Accuracy": mean_acc,
        "Accuracy_std": std_acc,
        "F1_macro": mean_f1_macro,
        "F1_macro_std": std_f1_macro,
        "Perte_F1_macro": perte_f1_macro
    })

    print(f"\n--- Bruit {int(noise_level * 100)}% ---")
    print(f"Accuracy moyenne  : {mean_acc:.6f} ± {std_acc:.6f}")
    print(f"F1 Macro moyen    : {mean_f1_macro:.6f} ± {std_f1_macro:.6f}")
    print(f"Perte F1 Macro    : {perte_f1_macro:.6f}")

# =========================================================
# 9-B) ADVERSARIAL SIMPLIFIE
# =========================================================
# Random Forest n'est pas differentiable.
# FGSM n'est donc pas applicable.
# On teste +epsilon et -epsilon puis on garde la direction qui degrade le plus le F1-score.

print("\n===== ADVERSARIAL SIMPLIFIE - RANDOM FOREST CICIoT2023 =====")

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
        "Dataset": "CICIoT2023",
        "Modele": "Random Forest",
        "Perturbation": "Adversarial simplifie",
        "Niveau": f"{int(adv_level * 100)}% ({adv_direction})",
        "Accuracy": adv_acc,
        "Accuracy_std": 0.0,
        "F1_macro": adv_f1_macro,
        "F1_macro_std": 0.0,
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

print("\n===== TABLEAU ROBUSTESSE RF 2023 =====")
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

plt.title("Robustesse bruit et adversarial - Random Forest CICIoT2023")
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
    f.write("===== RANDOM FOREST FINAL + ROBUSTESSE - CICIoT2023 =====\n\n")

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
    f.write(f"Repetitions bruit : {N_REPEATS_NOISE}\n")
    f.write(f"Niveaux adversarial : {ADVERSARIAL_LEVELS}\n\n")

    f.write("===== FEATURES =====\n")
    for col in FEATURE_COLUMNS:
        f.write(f"- {col}\n")
    f.write("\n")

    f.write("===== CLASSES =====\n")
    for i, cls in enumerate(class_names):
        f.write(f"{i} -> {cls}\n")
    f.write("\n")

    f.write("===== RESULTATS VALIDATION =====\n")
    f.write(f"Accuracy validation      : {val_acc:.6f}\n")
    f.write(f"Precision Macro val      : {val_precision_macro:.6f}\n")
    f.write(f"Recall Macro val         : {val_recall_macro:.6f}\n")
    f.write(f"F1 Macro val             : {val_f1_macro:.6f}\n")
    f.write("Matrice validation :\n")
    f.write(np.array2string(val_cm))
    f.write("\n\n")

    f.write("===== RESULTATS TEST SANS PERTURBATION =====\n")
    f.write(f"Accuracy Score     : {acc:.6f}\n")
    f.write(f"Precision Macro    : {precision_macro:.6f}\n")
    f.write(f"Recall Macro       : {recall_macro:.6f}\n")
    f.write(f"F1 Macro           : {f1_macro:.6f}\n")
    f.write(f"Precision Weighted : {precision_weighted:.6f}\n")
    f.write(f"Recall Weighted    : {recall_weighted:.6f}\n")
    f.write(f"F1 Weighted        : {f1_weighted:.6f}\n\n")

    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(report)
    f.write("\n")

    f.write("===== CONFUSION MATRIX =====\n")
    f.write(np.array2string(cm))
    f.write("\n\n")

    f.write("===== FEATURE IMPORTANCE =====\n")
    for i in sorted_idx:
        f.write(f"{FEATURE_COLUMNS[i]} : {importances[i]:.6f}\n")
    f.write("\n")

    f.write("===== ROBUSTESSE BRUIT + ADVERSARIAL SIMPLIFIE =====\n")
    f.write(robustness_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== FICHIERS GENERES =====\n")
    f.write(f"Modele RF              : {MODEL_PATH}\n")
    f.write(f"Labels json            : {LABELS_JSON_PATH}\n")
    f.write(f"Confusion matrix       : {CONFUSION_MATRIX_PLOT_PATH}\n")
    f.write(f"Feature importance     : {FEATURE_IMPORTANCE_PLOT_PATH}\n")
    f.write(f"CSV robustesse         : {ROBUSTNESS_CSV_PATH}\n")
    f.write(f"Figure robustesse      : {ROBUSTNESS_PLOT_PATH}\n")

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
print("- Ce script entraîne Random Forest sur CICIoT2023 avec 4 classes.")
print("- Le test normal est réalisé sur tout le jeu de test.")
print("- La robustesse est évaluée sur le test complet si le test contient moins de 5000 lignes.")
print("- Le bruit aleatoire est repete 5 fois par niveau.")
print("- Les resultats du bruit sont presentes avec moyenne et ecart-type.")
print("- Random Forest n'utilise pas FGSM car il n'est pas differentiable.")
print("- L'adversarial utilise ici est une perturbation simplifiee (+epsilon / -epsilon).")
print("- Le fichier CSV contient les résultats à utiliser dans le mémoire.")