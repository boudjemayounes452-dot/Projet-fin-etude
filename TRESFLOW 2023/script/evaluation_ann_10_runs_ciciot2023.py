# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:55:48 2026

@author: DELL
"""

# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_PATH = r"C:\Users\DELL\Desktop\pfe 2023\result\train_clean.csv"
VAL_PATH   = r"C:\Users\DELL\Desktop\pfe 2023\result\val_clean.csv"
TEST_PATH  = r"C:\Users\DELL\Desktop\pfe 2023\result\test_clean_no_similarity.csv"

OUTPUT_DIR = r"C:\Users\DELL\Desktop\pfe 2023\result\ann_10runs_clean_final"

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

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

SEEDS = [0, 7, 21, 42, 84, 123, 2024, 3141, 5555, 9999]

os.makedirs(OUTPUT_DIR, exist_ok=True)

RUNS_RESULTS_PATH = os.path.join(OUTPUT_DIR, "ann_10runs_results.csv")
RUNS_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "ann_10runs_summary.txt")
NORMALIZATION_JSON_PATH = os.path.join(OUTPUT_DIR, "ann_normalization.json")
LABELS_JSON_PATH = os.path.join(OUTPUT_DIR, "ann_labels.json")

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

print("\n===== CLASSES =====")
for i, cls in enumerate(class_names):
    print(f"{i} -> {cls}")

with open(LABELS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump({str(i): cls for i, cls in enumerate(class_names)}, f, indent=4)

# =========================================================
# 3) NORMALISATION
# fit sur TRAIN seulement
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
# 5) FONCTION MODELE ANN
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
        Dense(num_classes, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# =========================================================
# 6) 10 RUNS
# =========================================================

results = []

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
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred_int = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test_int, y_pred_int)
    precision_macro = precision_score(y_test_int, y_pred_int, average="macro", zero_division=0)
    recall_macro = recall_score(y_test_int, y_pred_int, average="macro", zero_division=0)
    f1_macro = f1_score(y_test_int, y_pred_int, average="macro", zero_division=0)

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
# 7) RESULTATS FINAUX
# =========================================================

df_results = pd.DataFrame(results)
df_results.to_csv(RUNS_RESULTS_PATH, index=False)

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

with open(RUNS_SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("===== RESULTATS DES 10 RUNS =====\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n===== MOYENNE ± IC95 =====\n")
    for line in summary_lines:
        f.write(line + "\n")

print("\n===== FICHIERS GENERES =====")
print("Résultats 10 runs :", RUNS_RESULTS_PATH)
print("Résumé IC95       :", RUNS_SUMMARY_PATH)
print("Labels            :", LABELS_JSON_PATH)
print("Normalisation     :", NORMALIZATION_JSON_PATH)