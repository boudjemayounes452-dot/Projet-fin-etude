# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:15:37 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
ANN 10 RUNS - CICIDS2018 + LIVE + BOTNET

TRAIN :
train_2018_chrono_live_botnet10_encoded.csv

VALIDATION :
val_2018_chrono_live_botnet10_encoded.csv

TEST EXTERNE :
test_botnet90_external_chrono_encoded.csv

Objectif :
- Entraîner un modèle ANN binaire
- 0 = NORMAL / BENIGN
- 1 = ATTACK / BOTNET
- Répéter l'entraînement sur 10 seeds
- Tester chaque run sur le test externe Botnet 90 %
- Calculer moyenne, écart-type et IC95
- Sauvegarder métriques, normalisation, labels et meilleur modèle
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


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
    "ann_10runs_2018_chrono_live_botnet"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RUNS_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_10runs_results.csv"
)

RUNS_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_10runs_summary.txt"
)

NORMALIZATION_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_normalization.json"
)

SCALER_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_scaler.joblib"
)

LABELS_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_labels.json"
)

BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_best_model.keras"
)

BEST_METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_best_metrics.txt"
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

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

SEEDS = [0, 7, 21, 42, 84, 123, 2024, 3141, 5555, 9999]


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

print("\n===== DISTRIBUTION TEST EXTERNE BOTNET =====")
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
# 4) NORMALISATION
# fit sur TRAIN seulement
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

normalization_data = {
    "feature_names": FEATURE_COLUMNS,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}

with open(NORMALIZATION_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(normalization_data, f, indent=4)

joblib.dump(scaler, SCALER_PATH)


# =========================================================
# 5) CLASS WEIGHTS
# =========================================================

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(np.unique(y_train), class_weights_values)
}

print("\n===== CLASS WEIGHTS =====")
print(class_weights)


# =========================================================
# 6) FONCTION MODELE ANN BINAIRE
# =========================================================

def build_ann(seed):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = Sequential([
        Input(shape=(len(FEATURE_COLUMNS),)),

        Dense(32, activation="relu"),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dropout(0.2),

        Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================================================
# 7) 10 RUNS ANN
# =========================================================

results = []

best_seed = None
best_f1_attack = -1
best_model = None
best_cm = None
best_report = None
best_details = None

for seed in SEEDS:
    print(f"\n================ RUN seed={seed} ================")

    model = build_ann(seed)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

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

    normal_detection_rate = tn / normal_total if normal_total > 0 else 0
    attack_detection_rate = tp / attack_total if attack_total > 0 else 0

    results.append({
        "seed": seed,
        "accuracy": acc,
        "precision_attack": precision_attack,
        "recall_attack": recall_attack,
        "f1_attack": f1_attack,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "normal_detection_rate": normal_detection_rate,
        "attack_detection_rate": attack_detection_rate,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    print(f"Accuracy           : {acc:.6f}")
    print(f"Precision Attack   : {precision_attack:.6f}")
    print(f"Recall Attack      : {recall_attack:.6f}")
    print(f"F1 Attack          : {f1_attack:.6f}")
    print(f"F1 Macro           : {f1_macro:.6f}")
    print(f"False Positive Rate: {false_positive_rate:.6f}")
    print(f"Specificity        : {specificity:.6f}")
    print("Matrice confusion [[TN FP], [FN TP]] :")
    print(cm)

    if f1_attack > best_f1_attack:
        best_f1_attack = f1_attack
        best_seed = seed
        best_model = model
        best_cm = cm
        best_report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0
        )
        best_details = results[-1]


# =========================================================
# 8) SAUVEGARDE RESULTATS DES 10 RUNS
# =========================================================

df_results = pd.DataFrame(results)
df_results.to_csv(RUNS_RESULTS_PATH, index=False)

print("\n===== RESULTATS DES 10 RUNS =====")
print(df_results)


# =========================================================
# 9) MOYENNE ± IC95
# =========================================================

summary_lines = []

metrics_to_summarize = [
    "accuracy",
    "precision_attack",
    "recall_attack",
    "f1_attack",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "false_positive_rate",
    "specificity",
    "normal_detection_rate",
    "attack_detection_rate"
]

print("\n===== MOYENNE ± IC95 =====")

for metric in metrics_to_summarize:
    mean = df_results[metric].mean()
    std = df_results[metric].std()
    ic95 = 1.96 * std / np.sqrt(len(SEEDS))

    line = f"{metric}: {mean:.6f} ± {ic95:.6f} (std: {std:.6f})"
    summary_lines.append(line)
    print(line)

with open(RUNS_SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("===== ANN 10 RUNS - 2018 CHRONO + LIVE + BOTNET =====\n\n")

    f.write("===== FICHIERS UTILISES =====\n")
    f.write(f"Train      : {TRAIN_PATH}\n")
    f.write(f"Validation : {VAL_PATH}\n")
    f.write(f"Test       : {TEST_PATH}\n\n")

    f.write("===== CONFIGURATION =====\n")
    f.write(f"Epochs        : {EPOCHS}\n")
    f.write(f"Batch size    : {BATCH_SIZE}\n")
    f.write(f"Learning rate : {LEARNING_RATE}\n")
    f.write(f"Seeds         : {SEEDS}\n")
    f.write(f"Class weights : {class_weights}\n\n")

    f.write("===== RESULTATS DES 10 RUNS =====\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n")

    f.write("===== MOYENNE ± IC95 =====\n")
    for line in summary_lines:
        f.write(line + "\n")

    f.write("\n===== MEILLEUR RUN =====\n")
    f.write(f"Best seed : {best_seed}\n")
    f.write(f"Best F1 Attack : {best_f1_attack:.6f}\n")


# =========================================================
# 10) SAUVEGARDE DU MEILLEUR MODELE
# =========================================================

best_model.save(BEST_MODEL_PATH)

with open(BEST_METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("===== MEILLEUR MODELE ANN =====\n\n")
    f.write(f"Best seed : {best_seed}\n")
    f.write(f"Best F1 Attack : {best_f1_attack:.6f}\n\n")

    f.write("===== METRIQUES MEILLEUR RUN =====\n")
    for key, value in best_details.items():
        f.write(f"{key}: {value}\n")

    f.write("\n===== MATRICE DE CONFUSION [[TN FP], [FN TP]] =====\n")
    f.write(np.array2string(best_cm))
    f.write("\n\n")

    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(best_report)
    f.write("\n")


# =========================================================
# 11) FIN
# =========================================================

print("\n===== FICHIERS GENERES =====")
print("Résultats 10 runs      :", RUNS_RESULTS_PATH)
print("Résumé 10 runs IC95    :", RUNS_SUMMARY_PATH)
print("Meilleur modèle ANN    :", BEST_MODEL_PATH)
print("Métriques meilleur run :", BEST_METRICS_PATH)
print("Labels json            :", LABELS_JSON_PATH)
print("Normalisation json     :", NORMALIZATION_JSON_PATH)
print("Scaler joblib          :", SCALER_PATH)

print("\nIMPORTANT :")
print("- Ce modèle ANN est entraîné avec les mêmes fichiers que le modèle RF.")
print("- La normalisation est apprise uniquement sur TRAIN.")
print("- Le test externe Botnet 90 % reste indépendant.")
print("- Le meilleur modèle est choisi selon F1 Attack.")
print("- Pour ESP32/TinyML, il faudra convertir le meilleur modèle ANN vers TFLite.")