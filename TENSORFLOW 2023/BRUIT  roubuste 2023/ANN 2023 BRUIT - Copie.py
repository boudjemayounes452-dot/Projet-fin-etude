# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:13:24 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
ANN FINAL + ROBUSTESSE BRUIT REPETE + ADVERSARIAL FGSM - CICIoT2023

Classes :
BenignTraffic
DDoS-ICMP_Flood
DDoS-UDP_Flood
DDoS-SYN_Flood

Correction :
- Le bruit aleatoire est repete 5 fois pour chaque niveau.
- Les resultats du bruit sont presentes avec moyenne et ecart-type.
- FGSM reste applique une fois par niveau.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_PATH = r"C:\Users\DELL\Desktop\pfe 2023\result\train_clean.csv"
VAL_PATH   = r"C:\Users\DELL\Desktop\pfe 2023\result\val_clean.csv"
TEST_PATH  = r"C:\Users\DELL\Desktop\pfe 2023\result\test_clean_no_similarity.csv"

OUTPUT_DIR = r"C:\Users\DELL\Desktop\pfe 2023\result\ann_2023_robustesse_bruit_adversarial_moyenne"

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

BEST_SEED = 21
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

SAMPLE_SIZE_ROBUSTNESS = 5000

NOISE_LEVELS = [0.01, 0.03, 0.05, 0.10]
ADVERSARIAL_LEVELS = [0.01, 0.03, 0.05, 0.10]

# Nombre de repetitions pour stabiliser le bruit
N_REPEATS_NOISE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

KERAS_MODEL_PATH = os.path.join(OUTPUT_DIR, "ann_final_4classes.keras")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "ann_final_4classes_best.keras")
TFLITE_MODEL_PATH = os.path.join(OUTPUT_DIR, "ann_final_4classes.tflite")
HEADER_FILE_PATH = os.path.join(OUTPUT_DIR, "ann_model_data.h")

METRICS_PATH = os.path.join(OUTPUT_DIR, "ann_final_metrics.txt")
LABELS_JSON_PATH = os.path.join(OUTPUT_DIR, "ann_labels.json")
NORMALIZATION_JSON_PATH = os.path.join(OUTPUT_DIR, "ann_normalization.json")

LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "ann_loss_curve.png")
ACCURACY_PLOT_PATH = os.path.join(OUTPUT_DIR, "ann_accuracy_curve.png")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(OUTPUT_DIR, "ann_confusion_matrix.png")

ROBUSTNESS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_robustesse_bruit_adversarial_ciciot2023.csv"
)

ROBUSTNESS_PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "ann_robustesse_bruit_adversarial_ciciot2023.png"
)

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

y_train_int = label_encoder.fit_transform(y_train_text)
y_val_int   = label_encoder.transform(y_val_text)
y_test_int  = label_encoder.transform(y_test_text)

class_names = list(label_encoder.classes_)
num_classes = len(class_names)

expected_classes = {
    "BenignTraffic",
    "DDoS-ICMP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-SYN_Flood"
}

if set(class_names) != expected_classes:
    raise ValueError(f"Classes incorrectes : {class_names}")

y_train_cat = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
y_val_cat   = tf.keras.utils.to_categorical(y_val_int, num_classes=num_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test_int, num_classes=num_classes)

with open(LABELS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump({str(i): cls for i, cls in enumerate(class_names)}, f, indent=4)

print("\n===== CLASSES =====")
for i, cls in enumerate(class_names):
    print(f"{i} -> {cls}")

# =========================================================
# 4) NORMALISATION
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

normalization_data = {
    "feature_names": FEATURE_COLUMNS,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}

with open(NORMALIZATION_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(normalization_data, f, indent=4)

print("\n===== NORMALISATION =====")
print("Normalisation apprise sur TRAIN seulement.")
print("Normalisation json :", NORMALIZATION_JSON_PATH)

# =========================================================
# 5) CLASS WEIGHTS
# =========================================================

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_int),
    y=y_train_int
)

class_weights = {
    int(i): float(w)
    for i, w in zip(np.unique(y_train_int), class_weights_values)
}

print("\n===== CLASS WEIGHTS =====")
print(class_weights)

# =========================================================
# 6) MODELE ANN FINAL
# =========================================================

model = Sequential([
    Input(shape=(len(FEATURE_COLUMNS),)),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n===== ARCHITECTURE ANN 2023 =====")
model.summary()

# =========================================================
# 7) CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =========================================================
# 8) ENTRAINEMENT FINAL
# =========================================================

print("\n===== ENTRAINEMENT ANN 2023 =====")

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

model.save(KERAS_MODEL_PATH)

epochs_done = len(history.history["loss"])

print("\n===== ENTRAINEMENT TERMINE =====")
print("Epochs exécutées :", epochs_done)

# =========================================================
# 9) EVALUATION TEST SANS PERTURBATION
# =========================================================

test_loss, test_acc = model.evaluate(
    X_test,
    y_test_cat,
    batch_size=1024,
    verbose=0
)

y_pred_prob = model.predict(
    X_test,
    batch_size=1024,
    verbose=0
)

y_pred_int = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test_int, y_pred_int)
precision_macro = precision_score(y_test_int, y_pred_int, average="macro", zero_division=0)
recall_macro = recall_score(y_test_int, y_pred_int, average="macro", zero_division=0)
f1_macro = f1_score(y_test_int, y_pred_int, average="macro", zero_division=0)

precision_weighted = precision_score(y_test_int, y_pred_int, average="weighted", zero_division=0)
recall_weighted = recall_score(y_test_int, y_pred_int, average="weighted", zero_division=0)
f1_weighted = f1_score(y_test_int, y_pred_int, average="weighted", zero_division=0)

cm = confusion_matrix(y_test_int, y_pred_int)

report = classification_report(
    y_test_int,
    y_pred_int,
    target_names=class_names,
    zero_division=0
)

print("\n===== RESULTATS TEST SANS PERTURBATION =====")
print(f"Best seed          : {BEST_SEED}")
print(f"Test Loss          : {test_loss:.6f}")
print(f"Test Accuracy      : {test_acc:.6f}")
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
# 10) ANALYSE DE ROBUSTESSE : BRUIT REPETE + ADVERSARIAL FGSM
# =========================================================

print("\n===== ANALYSE DE ROBUSTESSE - ANN CICIoT2023 =====")

np.random.seed(BEST_SEED)

if X_test.shape[0] > SAMPLE_SIZE_ROBUSTNESS:
    idx = np.random.choice(
        X_test.shape[0],
        SAMPLE_SIZE_ROBUSTNESS,
        replace=False
    )
    X_test_robust = X_test[idx].astype(np.float32)
    y_test_robust_int = y_test_int[idx]
    y_test_robust_cat = y_test_cat[idx].astype(np.float32)
else:
    X_test_robust = X_test.astype(np.float32)
    y_test_robust_int = y_test_int
    y_test_robust_cat = y_test_cat.astype(np.float32)

print("Nombre total test :", X_test.shape[0])
print("Nombre utilisé pour robustesse :", X_test_robust.shape[0])

# Résultat sans perturbation sur le même ensemble
y_base_prob = model.predict(
    X_test_robust,
    batch_size=512,
    verbose=0
)

y_base_pred = np.argmax(y_base_prob, axis=1)

baseline_acc = accuracy_score(y_test_robust_int, y_base_pred)

baseline_f1_macro = f1_score(
    y_test_robust_int,
    y_base_pred,
    average="macro",
    zero_division=0
)

robustness_results = []

robustness_results.append({
    "Dataset": "CICIoT2023",
    "Modele": "ANN",
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

# =========================================================
# 10-A) BRUIT ALEATOIRE AVEC REPETITIONS
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

        X_test_noisy = X_test_robust + noise

        y_noisy_prob = model.predict(
            X_test_noisy,
            batch_size=512,
            verbose=0
        )

        y_noisy_pred = np.argmax(y_noisy_prob, axis=1)

        noisy_acc = accuracy_score(y_test_robust_int, y_noisy_pred)

        noisy_f1_macro = f1_score(
            y_test_robust_int,
            y_noisy_pred,
            average="macro",
            zero_division=0
        )

        acc_values.append(noisy_acc)
        f1_values.append(noisy_f1_macro)

        del noise
        del X_test_noisy
        del y_noisy_prob
        del y_noisy_pred

    mean_acc = np.mean(acc_values)
    std_acc = np.std(acc_values)

    mean_f1_macro = np.mean(f1_values)
    std_f1_macro = np.std(f1_values)

    perte_f1_macro = baseline_f1_macro - mean_f1_macro

    robustness_results.append({
        "Dataset": "CICIoT2023",
        "Modele": "ANN",
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
# 10-B) ADVERSARIAL FGSM MULTI-CLASSES
# =========================================================

print("\n===== GENERATION ADVERSARIAL FGSM - ANN CICIoT2023 =====")

x_tensor = tf.convert_to_tensor(X_test_robust, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_test_robust_cat, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x_tensor)
    predictions = model(x_tensor, training=False)
    loss = tf.keras.losses.categorical_crossentropy(y_tensor, predictions)

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
    )

    y_adv_pred = np.argmax(y_adv_prob, axis=1)

    adv_acc = accuracy_score(y_test_robust_int, y_adv_pred)

    adv_f1_macro = f1_score(
        y_test_robust_int,
        y_adv_pred,
        average="macro",
        zero_division=0
    )

    perte_f1_macro = baseline_f1_macro - adv_f1_macro

    robustness_results.append({
        "Dataset": "CICIoT2023",
        "Modele": "ANN",
        "Perturbation": "Adversarial FGSM",
        "Niveau": f"{int(adv_level * 100)}%",
        "Accuracy": adv_acc,
        "Accuracy_std": 0.0,
        "F1_macro": adv_f1_macro,
        "F1_macro_std": 0.0,
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
# 11) SAUVEGARDE RESULTATS ROBUSTESSE
# =========================================================

robustness_df = pd.DataFrame(robustness_results)

print("\n===== TABLEAU ROBUSTESSE ANN 2023 =====")
print(robustness_df)

robustness_df.to_csv(ROBUSTNESS_CSV_PATH, index=False)

print("\nFichier robustesse sauvegardé :", ROBUSTNESS_CSV_PATH)

# =========================================================
# 12) FIGURE ROBUSTESSE
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

plt.title("Robustesse bruit et adversarial - ANN CICIoT2023")
plt.xlabel("Niveau de perturbation")
plt.ylabel("F1-score macro")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(ROBUSTNESS_PLOT_PATH, dpi=300)
plt.show()

print("Figure robustesse sauvegardée :", ROBUSTNESS_PLOT_PATH)

# =========================================================
# 13) COURBES TRAINING
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ANN 4 Classes - Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH, dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ANN 4 Classes - Accuracy Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(ACCURACY_PLOT_PATH, dpi=300)
plt.show()

# =========================================================
# 14) MATRICE DE CONFUSION
# =========================================================

plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("ANN 4 Classes - Confusion Matrix")
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
# 15) SAUVEGARDE METRIQUES
# =========================================================

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("===== ANN FINAL + ROBUSTESSE BRUIT REPETE + ADVERSARIAL - CICIoT2023 =====\n\n")

    f.write("===== FICHIERS UTILISES =====\n")
    f.write(f"Train      : {TRAIN_PATH}\n")
    f.write(f"Validation : {VAL_PATH}\n")
    f.write(f"Test       : {TEST_PATH}\n\n")

    f.write("===== CONFIGURATION =====\n")
    f.write(f"Best seed       : {BEST_SEED}\n")
    f.write(f"Epochs max      : {EPOCHS}\n")
    f.write(f"Epochs executes : {epochs_done}\n")
    f.write(f"Batch size      : {BATCH_SIZE}\n")
    f.write(f"Learning rate   : {LEARNING_RATE}\n")
    f.write(f"Class weights   : {class_weights}\n")
    f.write(f"Robustesse sample size : {X_test_robust.shape[0]}\n")
    f.write(f"Niveaux bruit : {NOISE_LEVELS}\n")
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

    f.write("===== RESULTATS TEST SANS PERTURBATION =====\n")
    f.write(f"Test Loss          : {test_loss:.6f}\n")
    f.write(f"Test Accuracy      : {test_acc:.6f}\n")
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

    f.write("===== ROBUSTESSE BRUIT + ADVERSARIAL =====\n")
    f.write(robustness_df.to_string(index=False))
    f.write("\n\n")

    f.write("===== FICHIERS GENERES =====\n")
    f.write(f"Modele Keras final      : {KERAS_MODEL_PATH}\n")
    f.write(f"Modele Keras best       : {BEST_MODEL_PATH}\n")
    f.write(f"Modele TFLite           : {TFLITE_MODEL_PATH}\n")
    f.write(f"Header ESP32 .h         : {HEADER_FILE_PATH}\n")
    f.write(f"Labels json             : {LABELS_JSON_PATH}\n")
    f.write(f"Normalisation json      : {NORMALIZATION_JSON_PATH}\n")
    f.write(f"Loss curve              : {LOSS_PLOT_PATH}\n")
    f.write(f"Accuracy curve          : {ACCURACY_PLOT_PATH}\n")
    f.write(f"Confusion matrix figure : {CONFUSION_MATRIX_PLOT_PATH}\n")
    f.write(f"CSV robustesse          : {ROBUSTNESS_CSV_PATH}\n")
    f.write(f"Figure robustesse       : {ROBUSTNESS_PLOT_PATH}\n")

# =========================================================
# 16) CONVERSION TFLITE
# =========================================================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print("\n===== CONVERSION TFLITE TERMINEE =====")
print("Modele TFLite :", TFLITE_MODEL_PATH)

# =========================================================
# 17) EXPORT .h POUR ESP32
# =========================================================

with open(TFLITE_MODEL_PATH, "rb") as f:
    tflite_bytes = f.read()

with open(HEADER_FILE_PATH, "w", encoding="utf-8") as f:
    f.write("#ifndef ANN_MODEL_DATA_H\n")
    f.write("#define ANN_MODEL_DATA_H\n\n")
    f.write("const unsigned char ann_model_tflite[] = {\n")

    for i, b in enumerate(tflite_bytes):
        if i % 12 == 0:
            f.write("    ")
        f.write(f"0x{b:02x}")
        if i != len(tflite_bytes) - 1:
            f.write(", ")
        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n\n")
    f.write(f"const unsigned int ann_model_tflite_len = {len(tflite_bytes)};\n\n")
    f.write("#endif\n")

# =========================================================
# 18) FIN
# =========================================================

print("\n===== FICHIERS GENERES =====")
print("Modele Keras final      :", KERAS_MODEL_PATH)
print("Modele Keras best       :", BEST_MODEL_PATH)
print("Modele TFLite           :", TFLITE_MODEL_PATH)
print("Header ESP32 .h         :", HEADER_FILE_PATH)
print("Metrics txt             :", METRICS_PATH)
print("Labels json             :", LABELS_JSON_PATH)
print("Normalisation json      :", NORMALIZATION_JSON_PATH)
print("Loss curve              :", LOSS_PLOT_PATH)
print("Accuracy curve          :", ACCURACY_PLOT_PATH)
print("Confusion matrix figure :", CONFUSION_MATRIX_PLOT_PATH)
print("CSV robustesse          :", ROBUSTNESS_CSV_PATH)
print("Figure robustesse       :", ROBUSTNESS_PLOT_PATH)

print("\nIMPORTANT :")
print("- Ce script entraîne ANN sur CICIoT2023 avec 4 classes.")
print("- Le test normal est réalisé sur tout le jeu de test.")
print("- La robustesse est évaluée sur le test complet si le test contient moins de 5000 lignes.")
print("- Le bruit aleatoire est repete 5 fois par niveau.")
print("- Les resultats du bruit sont presentes avec moyenne et ecart-type.")
print("- FGSM multi-classes est applique avec categorical_crossentropy.")
print("- Le fichier CSV contient les résultats à utiliser dans le mémoire.")