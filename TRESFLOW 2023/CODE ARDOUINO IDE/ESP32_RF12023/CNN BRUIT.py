# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:34:01 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
CNN FINAL STABLE + ROBUSTESSE BRUIT + ADVERSARIAL - CICIDS2018
Version légère pour éviter kernel died

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
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, Input
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
    "cnn_2018_robustesse_bruit_adversarial_light"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_final_stable_seed42.keras"
)

TFLITE_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_final_stable_seed42.tflite"
)

SCALER_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_scaler_stable_seed42.joblib"
)

NORMALIZATION_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_normalization_stable_seed42.json"
)

LABELS_JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_labels_stable_seed42.json"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_final_metrics_robustesse_seed42.txt"
)

CONFUSION_MATRIX_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_confusion_matrix_seed42.png"
)

LOSS_CURVE_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_loss_curve_seed42.png"
)

ROBUSTNESS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_robustesse_bruit_adversarial_cicids2018.csv"
)

ROBUSTNESS_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "cnn_robustesse_bruit_adversarial_cicids2018.png"
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

BEST_SEED = 42

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
THRESHOLD = 0.5

# IMPORTANT :
# 5000 lignes seulement pour éviter kernel died avec FGSM
SAMPLE_SIZE_ROBUSTNESS = 5000

NOISE_LEVELS = [0.01, 0.03, 0.05, 0.10]
ADVERSARIAL_LEVELS = [0.01, 0.03, 0.05, 0.10]


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
# 4) NORMALISATION
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
# 5) RESHAPE POUR CNN
# CNN attend : (samples, features, channels)
# =========================================================

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("\n===== SHAPE CNN =====")
print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)


# =========================================================
# 6) CLASS WEIGHTS
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
# 7) MODELE CNN FINAL STABLE
# =========================================================

model = Sequential([
    Input(shape=(len(FEATURE_COLUMNS), 1)),

    Conv1D(
        filters=8,
        kernel_size=2,
        activation="relu",
        padding="same"
    ),

    Conv1D(
        filters=8,
        kernel_size=2,
        activation="relu",
        padding="same"
    ),

    Flatten(),

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

print("\n===== ARCHITECTURE CNN STABLE =====")
model.summary()


# =========================================================
# 8) CALLBACKS
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
# 9) ENTRAINEMENT CNN FINAL STABLE
# =========================================================

print("\n===== ENTRAINEMENT CNN FINAL STABLE =====")
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
# 10) FIGURE LOSS TRAIN / VALIDATION
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("CNN stable - Loss en fonction des epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_CURVE_PLOT_PATH, dpi=300)
plt.show()

print("Figure loss sauvegardée :", LOSS_CURVE_PLOT_PATH)


# =========================================================
# 11) EVALUATION VALIDATION
# =========================================================

y_val_pred_prob = model.predict(X_val, batch_size=1024, verbose=0).ravel()
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
# 12) EVALUATION TEST SANS PERTURBATION
# =========================================================

y_pred_prob = model.predict(X_test, batch_size=1024, verbose=0).ravel()
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

print("\n===== RESULTATS TEST SANS PERTURBATION =====")
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
# 13) FIGURE MATRICE DE CONFUSION TEST
# =========================================================

plt.figure(figsize=(7, 7))
plt.imshow(cm, interpolation="nearest")
plt.title("CNN stable - Test sans perturbation")
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
# 14) ANALYSE DE ROBUSTESSE : BRUIT + ADVERSARIAL
# =========================================================

print("\n===== ANALYSE DE ROBUSTESSE - CNN CICIDS2018 =====")

np.random.seed(BEST_SEED)

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
print("Nombre utilisé pour robustesse :", X_test_robust.shape[0])
print("Shape robustesse CNN :", X_test_robust.shape)

# Résultat sans perturbation sur le même échantillon
y_base_prob = model.predict(
    X_test_robust,
    batch_size=512,
    verbose=0
).ravel()

y_base_pred = (y_base_prob >= THRESHOLD).astype(int)

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
    "Modele": "CNN 1D",
    "Perturbation": "Aucune",
    "Niveau": "0%",
    "Accuracy": baseline_acc,
    "F1_macro": baseline_f1_macro,
    "Perte_F1_macro": 0.0
})

print("\n--- Sans perturbation ---")
print(f"Accuracy : {baseline_acc:.6f}")
print(f"F1 Macro : {baseline_f1_macro:.6f}")


# =========================================================
# 14-A) BRUIT ALEATOIRE
# =========================================================

for noise_level in NOISE_LEVELS:

    noise = np.random.normal(
        loc=0.0,
        scale=noise_level,
        size=X_test_robust.shape
    ).astype(np.float32)

    X_test_noisy = X_test_robust + noise

    y_noisy_prob = model.predict(
        X_test_noisy,
        batch_size=512,
        verbose=0
    ).ravel()

    y_noisy_pred = (y_noisy_prob >= THRESHOLD).astype(int)

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
        "Modele": "CNN 1D",
        "Perturbation": "Bruit aleatoire",
        "Niveau": f"{int(noise_level * 100)}%",
        "Accuracy": noisy_acc,
        "F1_macro": noisy_f1_macro,
        "Perte_F1_macro": perte_f1_macro
    })

    print(f"\n--- Bruit {int(noise_level * 100)}% ---")
    print(f"Accuracy bruitée  : {noisy_acc:.6f}")
    print(f"F1 Macro bruité   : {noisy_f1_macro:.6f}")
    print(f"Perte F1 Macro    : {perte_f1_macro:.6f}")

    del noise
    del X_test_noisy
    del y_noisy_prob
    del y_noisy_pred


# =========================================================
# 14-B) ADVERSARIAL FGSM
# =========================================================

print("\n===== GENERATION ADVERSARIAL FGSM - CNN =====")

x_tensor = tf.convert_to_tensor(X_test_robust, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_test_robust.reshape(-1, 1), dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x_tensor)
    predictions = model(x_tensor, training=False)
    loss = tf.keras.losses.binary_crossentropy(y_tensor, predictions)

gradient = tape.gradient(loss, x_tensor)
signed_gradient = tf.sign(gradient).numpy().astype(np.float32)

del x_tensor
del y_tensor
del predictions
del gradient

for adv_level in ADVERSARIAL_LEVELS:

    X_test_adv = X_test_robust + adv_level * signed_gradient

    y_adv_prob = model.predict(
        X_test_adv,
        batch_size=512,
        verbose=0
    ).ravel()

    y_adv_pred = (y_adv_prob >= THRESHOLD).astype(int)

    adv_acc = accuracy_score(y_test_robust, y_adv_pred)

    adv_f1_macro = f1_score(
        y_test_robust,
        y_adv_pred,
        average="macro",
        zero_division=0
    )

    perte_f1_macro = baseline_f1_macro - adv_f1_macro

    robustness_results.append({
        "Dataset": "CICIDS2018",
        "Modele": "CNN 1D",
        "Perturbation": "Adversarial FGSM",
        "Niveau": f"{int(adv_level * 100)}%",
        "Accuracy": adv_acc,
        "F1_macro": adv_f1_macro,
        "Perte_F1_macro": perte_f1_macro
    })

    print(f"\n--- Adversarial FGSM {int(adv_level * 100)}% ---")
    print(f"Accuracy adversarial : {adv_acc:.6f}")
    print(f"F1 Macro adversarial : {adv_f1_macro:.6f}")
    print(f"Perte F1 Macro       : {perte_f1_macro:.6f}")

    del X_test_adv
    del y_adv_prob
    del y_adv_pred

del signed_gradient


# =========================================================
# 15) SAUVEGARDE RESULTATS ROBUSTESSE
# =========================================================

robustness_df = pd.DataFrame(robustness_results)

print("\n===== TABLEAU ROBUSTESSE CNN =====")
print(robustness_df)

robustness_df.to_csv(ROBUSTNESS_CSV_PATH, index=False)

print("\nFichier robustesse sauvegardé :", ROBUSTNESS_CSV_PATH)


# =========================================================
# 16) FIGURE ROBUSTESSE
# =========================================================

plt.figure(figsize=(9, 5))

for perturbation in robustness_df["Perturbation"].unique():
    subset = robustness_df[robustness_df["Perturbation"] == perturbation]
    plt.plot(
        subset["Niveau"],
        subset["F1_macro"],
        marker="o",
        label=perturbation
    )

plt.title("Robustesse bruit et adversarial - CNN 1D CICIDS2018")
plt.xlabel("Niveau de perturbation")
plt.ylabel("F1-score macro")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(ROBUSTNESS_PLOT_PATH, dpi=300)
plt.show()

print("Figure robustesse sauvegardée :", ROBUSTNESS_PLOT_PATH)


# =========================================================
# 17) DETAILS TEST
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
# 18) SAUVEGARDE MODELE KERAS
# =========================================================

model.save(MODEL_PATH)

print("\n===== MODELE KERAS SAUVEGARDE =====")
print("Modèle CNN final stable :", MODEL_PATH)


# =========================================================
# 19) CONVERSION DU MODELE CNN VERS TFLITE SANS OPTIMISATION
# =========================================================

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# IMPORTANT ESP32 :
# On ne met PAS converter.optimizations = [tf.lite.Optimize.DEFAULT]
# car la version optimisée peut donner PROB_ATTACK:nan sur ESP32.
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print("\n===== CONVERSION TFLITE TERMINEE SANS OPTIMISATION =====")
print("Modèle TFLite :", TFLITE_MODEL_PATH)


# =========================================================
# 20) SAUVEGARDE METRIQUES
# =========================================================

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("===== CNN FINAL STABLE + ROBUSTESSE BRUIT + ADVERSARIAL - CICIDS2018 =====\n\n")

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
    f.write("ReduceLROnPlateau: monitor=val_loss, factor=0.5, patience=3, min_lr=0.00001\n")
    f.write("TFLite           : sans optimisation\n")
    f.write(f"Robustesse sample size : {X_test_robust.shape[0]}\n")
    f.write(f"Niveaux bruit : {NOISE_LEVELS}\n")
    f.write(f"Niveaux adversarial : {ADVERSARIAL_LEVELS}\n\n")

    f.write("===== FEATURES =====\n")
    for col in FEATURE_COLUMNS:
        f.write(f"- {col}\n")
    f.write("\n")

    f.write("===== ARCHITECTURE CNN =====\n")
    f.write("Input : 8 features normalisees reshapees en (8, 1)\n")
    f.write("Conv1D 8 filtres, kernel_size=2, ReLU, padding=same\n")
    f.write("Conv1D 8 filtres, kernel_size=2, ReLU, padding=same\n")
    f.write("Flatten\n")
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

    f.write("===== ROBUSTESSE BRUIT + ADVERSARIAL =====\n")
    f.write(robustness_df.to_string(index=False))
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
    f.write(f"Courbe loss        : {LOSS_CURVE_PLOT_PATH}\n")
    f.write(f"Figure robustesse  : {ROBUSTNESS_PLOT_PATH}\n\n")

    f.write("===== FICHIERS GENERES =====\n")
    f.write(f"Modele Keras       : {MODEL_PATH}\n")
    f.write(f"Modele TFLite      : {TFLITE_MODEL_PATH}\n")
    f.write(f"Scaler joblib      : {SCALER_PATH}\n")
    f.write(f"Normalisation json : {NORMALIZATION_JSON_PATH}\n")
    f.write(f"Labels json        : {LABELS_JSON_PATH}\n")
    f.write(f"Matrice confusion  : {CONFUSION_MATRIX_PLOT_PATH}\n")
    f.write(f"Courbe loss        : {LOSS_CURVE_PLOT_PATH}\n")
    f.write(f"CSV robustesse     : {ROBUSTNESS_CSV_PATH}\n")
    f.write(f"Figure robustesse  : {ROBUSTNESS_PLOT_PATH}\n")


# =========================================================
# 21) FIN
# =========================================================

print("\n===== FICHIERS GENERES =====")
print("Modèle CNN final Keras :", MODEL_PATH)
print("Modèle CNN TFLite      :", TFLITE_MODEL_PATH)
print("Metrics txt            :", METRICS_PATH)
print("Labels json            :", LABELS_JSON_PATH)
print("Normalisation json     :", NORMALIZATION_JSON_PATH)
print("Scaler joblib          :", SCALER_PATH)
print("Matrice confusion      :", CONFUSION_MATRIX_PLOT_PATH)
print("Courbe loss            :", LOSS_CURVE_PLOT_PATH)
print("CSV robustesse         :", ROBUSTNESS_CSV_PATH)
print("Figure robustesse      :", ROBUSTNESS_PLOT_PATH)

print("\nIMPORTANT :")
print("- Ce modèle CNN utilise le meilleur seed trouvé dans les 10 runs : 42.")
print("- Les données sont reshapees en (8, 1) pour Conv1D.")
print("- Le test normal est réalisé sur tout le jeu de test.")
print("- La robustesse est évaluée sur un échantillon de 5000 lignes pour éviter kernel died.")
print("- Deux perturbations sont testées : bruit aléatoire et adversarial FGSM.")
print("- Les niveaux testés sont : 1%, 3%, 5%, 10%.")
print("- Le fichier .tflite est généré SANS optimisation.")
print("- Le fichier CSV contient les résultats à utiliser dans le mémoire.")