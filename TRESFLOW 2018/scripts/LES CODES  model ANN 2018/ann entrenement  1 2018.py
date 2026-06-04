# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:39:07 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
ANN FINAL STABLE + TFLITE - CICIDS2018 + LIVE + BOTNET

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
- Utiliser le meilleur seed trouvé dans les 10 runs : 84
- Stabiliser l'entraînement avec :
    * learning_rate plus faible
    * Dropout réduit
    * EarlyStopping plus patient
    * ReduceLROnPlateau
- Sauvegarder le modèle ANN final au format Keras
- Convertir le modèle ANN vers TFLite pour ESP32 / TinyML
- Sauvegarder les métriques, la normalisation, les labels
- Sauvegarder la matrice de confusion en image
- Sauvegarder la courbe loss train / validation
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
    "ann_final_stable_2018_chrono_live_botnet_seed84_tflite"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_final_stable_seed84.keras"
)

TFLITE_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_final_stable_seed84.tflite"
)

SCALER_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_scaler_stable_seed84.joblib"
)

NORMALIZATION_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_normalization_stable_seed84.json"
)

LABELS_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_labels_stable_seed84.json"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_final_stable_metrics_seed84.txt"
)

CONFUSION_MATRIX_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_confusion_matrix_stable_seed84.png"
)

LOSS_CURVE_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_loss_curve_stable_seed84.png"
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

BEST_SEED = 84

EPOCHS = 50
BATCH_SIZE = 64

# Learning rate réduit pour rendre l'entraînement plus stable
LEARNING_RATE = 0.0005

THRESHOLD = 0.5


# =========================================================
# FIXER LE SEED
# =========================================================

tf.keras.backend.clear_session()
tf.random.set_seed(BEST_SEED)
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

print("\n===== NORMALISATION =====")
print("Normalisation apprise sur TRAIN seulement.")
print("Normalisation json :", NORMALIZATION_JSON_PATH)
print("Scaler joblib      :", SCALER_PATH)


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
# 6) MODELE ANN FINAL STABLE
# =========================================================

model = Sequential([
    Input(shape=(len(FEATURE_COLUMNS),)),

    Dense(32, activation="relu"),
    Dropout(0.1),

    Dense(16, activation="relu"),
    Dropout(0.1),

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

print("\n===== ARCHITECTURE ANN =====")
model.summary()


# =========================================================
# 7) CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)


# =========================================================
# 8) ENTRAINEMENT ANN FINAL
# =========================================================

print("\n===== ENTRAINEMENT ANN FINAL STABLE =====")
print("Best seed :", BEST_SEED)
print("Learning rate initial :", LEARNING_RATE)
print("Dropout :", 0.1)
print("EarlyStopping patience :", 10)
print("ReduceLROnPlateau activé")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

epochs_done = len(history.history["loss"])

print("\n===== ENTRAINEMENT TERMINE =====")
print("Epochs max :", EPOCHS)
print("Epochs réellement exécutées :", epochs_done)
print("Remarque : si epochs exécutées < 50, c'est à cause de EarlyStopping.")


# =========================================================
# 9) FIGURE LOSS TRAIN / VALIDATION
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("ANN stable - Loss en fonction des epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_CURVE_PLOT_PATH, dpi=300)
plt.show()

print("Figure loss sauvegardée :", LOSS_CURVE_PLOT_PATH)


# =========================================================
# 10) EVALUATION VALIDATION
# =========================================================

y_val_pred_prob = model.predict(X_val, verbose=0).ravel()
y_val_pred = (y_val_pred_prob >= THRESHOLD).astype(int)

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
# 11) EVALUATION TEST EXTERNE BOTNET
# =========================================================

y_pred_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_prob >= THRESHOLD).astype(int)

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

print("\n===== RESULTATS TEST EXTERNE BOTNET =====")
print(f"Best seed          : {BEST_SEED}")
print(f"Threshold          : {THRESHOLD}")
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
# 12) FIGURE MATRICE DE CONFUSION TEST
# =========================================================

plt.figure(figsize=(7, 7))
plt.imshow(cm, interpolation="nearest")
plt.title("ANN stable - External Botnet Test")
plt.colorbar()

tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES)
plt.yticks(tick_marks, CLASS_NAMES)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=300)
plt.show()

print("Figure matrice de confusion sauvegardée :", CONFUSION_MATRIX_PLOT_PATH)


# =========================================================
# 13) DETAILS TEST
# =========================================================

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
# 14) SAUVEGARDE MODELE KERAS
# =========================================================

model.save(MODEL_PATH)

print("\n===== MODELE KERAS SAUVEGARDE =====")
print("Modèle ANN final stable :", MODEL_PATH)


# =========================================================
# 15) CONVERSION DU MODELE ANN VERS TFLITE
# =========================================================

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimisation simple pour réduire la taille du modèle.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print("\n===== CONVERSION TFLITE TERMINEE =====")
print("Modèle TFLite :", TFLITE_MODEL_PATH)


# =========================================================
# 16) SAUVEGARDE METRIQUES
# =========================================================

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("===== ANN FINAL STABLE + TFLITE - 2018 CHRONO + LIVE + BOTNET =====\n\n")

    f.write("===== FICHIERS UTILISES =====\n")
    f.write(f"Train      : {TRAIN_PATH}\n")
    f.write(f"Validation : {VAL_PATH}\n")
    f.write(f"Test       : {TEST_PATH}\n\n")

    f.write("===== CONFIGURATION =====\n")
    f.write(f"Best seed        : {BEST_SEED}\n")
    f.write(f"Epochs max       : {EPOCHS}\n")
    f.write(f"Epochs executes  : {epochs_done}\n")
    f.write(f"Batch size       : {BATCH_SIZE}\n")
    f.write(f"Learning rate    : {LEARNING_RATE}\n")
    f.write(f"Threshold        : {THRESHOLD}\n")
    f.write(f"Class weights    : {class_weights}\n")
    f.write("Dropout          : 0.1\n")
    f.write("EarlyStopping    : monitor=val_loss, patience=10, restore_best_weights=True\n")
    f.write("ReduceLROnPlateau: monitor=val_loss, factor=0.5, patience=3, min_lr=0.00001\n\n")

    f.write("===== FEATURES =====\n")
    for col in FEATURE_COLUMNS:
        f.write(f"- {col}\n")
    f.write("\n")

    f.write("===== ARCHITECTURE ANN =====\n")
    f.write("Input : 8 features normalisees\n")
    f.write("Dense 32 ReLU\n")
    f.write("Dropout 0.1\n")
    f.write("Dense 16 ReLU\n")
    f.write("Dropout 0.1\n")
    f.write("Dense 1 Sigmoid\n")
    f.write("Loss : binary_crossentropy\n")
    f.write("Optimizer : Adam\n\n")

    f.write("===== LABELS =====\n")
    f.write("0 = NORMAL / BENIGN\n")
    f.write("1 = ATTACK / BOTNET\n\n")

    f.write("===== RESULTATS VALIDATION =====\n")
    f.write(f"Accuracy validation         : {val_acc:.6f}\n")
    f.write(f"Precision Attack validation : {val_precision_attack:.6f}\n")
    f.write(f"Recall Attack validation    : {val_recall_attack:.6f}\n")
    f.write(f"F1 Attack validation        : {val_f1_attack:.6f}\n")
    f.write(f"F1 Macro validation         : {val_f1_macro:.6f}\n")
    f.write("Matrice validation [[TN FP], [FN TP]] :\n")
    f.write(np.array2string(val_cm))
    f.write("\n\n")

    f.write("===== RESULTATS TEST EXTERNE BOTNET =====\n")
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

    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(report)
    f.write("\n\n")

    f.write("===== FIGURES =====\n")
    f.write(f"Matrice confusion  : {CONFUSION_MATRIX_PLOT_PATH}\n")
    f.write(f"Courbe loss        : {LOSS_CURVE_PLOT_PATH}\n\n")

    f.write("===== FICHIERS GENERES =====\n")
    f.write(f"Modele Keras       : {MODEL_PATH}\n")
    f.write(f"Modele TFLite      : {TFLITE_MODEL_PATH}\n")
    f.write(f"Scaler joblib      : {SCALER_PATH}\n")
    f.write(f"Normalisation json : {NORMALIZATION_JSON_PATH}\n")
    f.write(f"Labels json        : {LABELS_JSON_PATH}\n")
    f.write(f"Matrice confusion  : {CONFUSION_MATRIX_PLOT_PATH}\n")
    f.write(f"Courbe loss        : {LOSS_CURVE_PLOT_PATH}\n")


# =========================================================
# 17) FIN
# =========================================================

print("\n===== FICHIERS GENERES =====")
print("Modèle ANN final Keras :", MODEL_PATH)
print("Modèle ANN TFLite      :", TFLITE_MODEL_PATH)
print("Metrics txt            :", METRICS_PATH)
print("Labels json            :", LABELS_JSON_PATH)
print("Normalisation json     :", NORMALIZATION_JSON_PATH)
print("Scaler joblib          :", SCALER_PATH)
print("Matrice confusion      :", CONFUSION_MATRIX_PLOT_PATH)
print("Courbe loss            :", LOSS_CURVE_PLOT_PATH)

print("\nIMPORTANT :")
print("- Ce modèle ANN utilise le meilleur seed trouvé dans les 10 runs : 84.")
print("- Version stabilisée : learning_rate=0.0005, Dropout=0.1, patience=10.")
print("- ReduceLROnPlateau est activé pour réduire le learning rate si val_loss stagne.")
print("- La normalisation est apprise uniquement sur TRAIN.")
print("- Le test externe Botnet 90 % reste indépendant.")
print("- Le fichier .tflite est généré pour préparer le déploiement ESP32.")
print("- Sur ESP32, il faut appliquer la même normalisation : x_norm = (x - mean) / scale.")
print("- Les valeurs mean et scale sont dans ann_normalization_stable_seed84.json.")