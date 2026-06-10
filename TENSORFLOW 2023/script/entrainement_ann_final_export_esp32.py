# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:11:41 2026

@author: DELL
"""

# -*- coding: utf-8 -*-

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

OUTPUT_DIR = r"C:\Users\DELL\Desktop\pfe 2023\result\ann_final_clean_esp322222"

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
# 2) PREPARATION X / y
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
# 3) NORMALISATION
# fit sur TRAIN uniquement
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

# =========================================================
# 4) CLASS WEIGHTS
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

# =========================================================
# 5) MODELE ANN FINAL
# =========================================================

tf.keras.backend.clear_session()
tf.random.set_seed(BEST_SEED)
np.random.seed(BEST_SEED)

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

model.summary()

# =========================================================
# 6) CALLBACKS
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
# 7) ENTRAINEMENT FINAL
# =========================================================

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

# =========================================================
# 8) EVALUATION TEST
# =========================================================

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

y_pred_prob = model.predict(X_test, verbose=0)
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

print("\n===== RESULTATS TEST FINAL =====")
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

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("===== RESULTATS TEST FINAL =====\n")
    f.write(f"Best seed          : {BEST_SEED}\n")
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
    f.write("\n===== CONFUSION MATRIX =====\n")
    f.write(np.array2string(cm))
    f.write("\n")

# =========================================================
# 9) COURBES
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
# 10) CONVERSION TFLITE
# =========================================================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

# =========================================================
# 11) EXPORT .h POUR ESP32
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