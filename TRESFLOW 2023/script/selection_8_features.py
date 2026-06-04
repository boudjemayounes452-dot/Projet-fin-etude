# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:16:37 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Reduction des features avec Random Forest
- Charge le dataset original CICIoT2023
- Entraine un Random Forest
- Calcule l'importance des features
- Garde les 8 features les plus importantes
- Sauvegarde un nouveau CSV reduit
- Sauvegarde une figure de l'importance des features
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================================================
# CHEMINS
# =========================================================

DATASET_PATH = r"C:\Users\DELL\Desktop\pfe 2023\data\cic_iot_2023_part_00.csv"
OUTPUT_DIR = r"C:\Users\DELL\Desktop\pfe 2023\result\rf_feature_selection"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_REDUCED_CSV = os.path.join(OUTPUT_DIR, "cic_iot_2023_top8_features.csv")
OUTPUT_IMPORTANCE_CSV = os.path.join(OUTPUT_DIR, "feature_importance_rf.csv")
OUTPUT_FIGURE = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")

# =========================================================
# PARAMETRES
# =========================================================

LABEL_COLUMN = "label"
TOP_K = 8
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =========================================================
# CHARGEMENT DU DATASET
# =========================================================

df = pd.read_csv(DATASET_PATH)
df.columns = [c.strip() for c in df.columns]

# Si la colonne s'appelle Label au lieu de label
if "Label" in df.columns and LABEL_COLUMN not in df.columns:
    df = df.rename(columns={"Label": LABEL_COLUMN})

if LABEL_COLUMN not in df.columns:
    raise ValueError(f"Colonne cible introuvable : {LABEL_COLUMN}")

print("===== DATASET CHARGE =====")
print("Shape :", df.shape)
print("Colonnes :", len(df.columns))

# =========================================================
# NETTOYAGE
# =========================================================

# Supprimer colonnes inutiles si presentes
drop_candidates = ["ts", "Timestamp", "timestamp"]
for col in drop_candidates:
    if col in df.columns:
        df = df.drop(columns=[col])

# Re-nettoyer les noms de colonnes
df.columns = [c.strip() for c in df.columns]

# Supprimer lignes avec valeurs manquantes
df = df.dropna()

print("\n===== APRES NETTOYAGE =====")
print("Shape :", df.shape)

# =========================================================
# SEPARATION X / y
# =========================================================

feature_columns = [c for c in df.columns if c != LABEL_COLUMN]

X = df[feature_columns].copy()
y_text = df[LABEL_COLUMN].astype(str).copy()

# Encoder les labels texte
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

print("\n===== CLASSES =====")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} -> {cls}")

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n===== SPLIT =====")
print("Train :", X_train.shape)
print("Test  :", X_test.shape)

# =========================================================
# MODELE RANDOM FOREST
# =========================================================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)

print("\n===== ENTRAINEMENT RANDOM FOREST =====")
rf.fit(X_train, y_train)

# =========================================================
# IMPORTANCE DES FEATURES
# =========================================================

importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importances
}).sort_values(by="importance", ascending=False).reset_index(drop=True)

print("\n===== IMPORTANCE DES FEATURES =====")
print(importance_df)

# Sauvegarder importance dans CSV
importance_df.to_csv(OUTPUT_IMPORTANCE_CSV, index=False, encoding="utf-8")

# =========================================================
# TOP 8 FEATURES
# =========================================================

top_features = importance_df["feature"].head(TOP_K).tolist()

print(f"\n===== TOP {TOP_K} FEATURES =====")
for i, feat in enumerate(top_features, start=1):
    print(f"{i}. {feat}")

# Creer dataset reduit
reduced_df = df[top_features + [LABEL_COLUMN]].copy()
reduced_df.to_csv(OUTPUT_REDUCED_CSV, index=False, encoding="utf-8")

print("\n===== DATASET REDUIT SAUVEGARDE =====")
print("Shape :", reduced_df.shape)
print("Fichier :", OUTPUT_REDUCED_CSV)

# =========================================================
# FIGURE IMPORTANCE DES FEATURES
# =========================================================

plt.figure(figsize=(10, 6))
plt.bar(importance_df["feature"], importance_df["importance"])
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.close()

print("\n===== FIGURE SAUVEGARDEE =====")
print("Figure :", OUTPUT_FIGURE)

# =========================================================
# RESUME FINAL
# =========================================================

print("\n===== RESUME FINAL =====")
print("Dataset original :", DATASET_PATH)
print("CSV importance   :", OUTPUT_IMPORTANCE_CSV)
print("Figure importance:", OUTPUT_FIGURE)
print("Dataset top 8    :", OUTPUT_REDUCED_CSV)