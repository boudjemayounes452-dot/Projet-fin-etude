# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:51:16 2026

@author: DELL
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# =========================================================
# CHEMINS
# =========================================================

TRAIN_PATH = r"C:\Users\DELL\Desktop\pfe 2023\result\train_clean.csv"
VAL_PATH   = r"C:\Users\DELL\Desktop\pfe 2023\result\val_clean.csv"
TEST_PATH  = r"C:\Users\DELL\Desktop\pfe 2023\result\test_clean.csv"

OUTPUT_TEST_PATH = r"C:\Users\DELL\Desktop\pfe 2023\result\test_clean_no_similarity.csv"

# =========================================================
# PARAMETRES
# =========================================================

THRESHOLD = 0.01

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

# =========================================================
# CHARGEMENT
# =========================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns = [c.strip() for c in test_df.columns]

X_train = train_df[FEATURE_COLUMNS].astype(np.float32).values
X_test = test_df[FEATURE_COLUMNS].astype(np.float32).values

# =========================================================
# NORMALISATION FIT SUR TRAIN SEULEMENT
# =========================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# PLUS PROCHE VOISIN TEST -> TRAIN
# =========================================================

nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(X_train_scaled)

distances, indices = nn.kneighbors(X_test_scaled)
distances = distances.flatten()

# =========================================================
# GARDER SEULEMENT LES LIGNES TEST ASSEZ DIFFERENTES
# =========================================================

mask_keep = distances >= THRESHOLD

test_before = len(test_df)
test_after = int(mask_keep.sum())

test_clean_no_similarity = test_df[mask_keep].copy().reset_index(drop=True)

# =========================================================
# SAUVEGARDE
# =========================================================

test_clean_no_similarity.to_csv(OUTPUT_TEST_PATH, index=False, encoding="utf-8")

# =========================================================
# AFFICHAGE
# =========================================================

print("===== NETTOYAGE TEST PAR SIMILARITE =====")
print("Seuil distance :", THRESHOLD)
print("Test avant :", test_before)
print("Test apres :", test_after)
print("Lignes supprimees :", test_before - test_after)
print("Fichier sauvegarde :", OUTPUT_TEST_PATH)

print("\n===== REPARTITION TEST NETTOYE =====")
print(test_clean_no_similarity[LABEL_COLUMN].value_counts())

print("\n===== VERIFICATION APRES NETTOYAGE =====")

X_test_clean = test_clean_no_similarity[FEATURE_COLUMNS].astype(np.float32).values
X_test_clean_scaled = scaler.transform(X_test_clean)

distances_after, _ = nn.kneighbors(X_test_clean_scaled)
distances_after = distances_after.flatten()

print("Min distance   :", distances_after.min())
print("Mean distance  :", distances_after.mean())
print("Median distance:", np.median(distances_after))
print("Max distance   :", distances_after.max())

for threshold in [0.0001, 0.001, 0.01, 0.05, 0.1]:
    count = np.sum(distances_after < threshold)
    percent = 100 * count / len(distances_after)
    print(f"Distances < {threshold}: {count} lignes ({percent:.2f}%)")