# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:33:15 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Réduction dataset Botnet CICIDS2018 vers les 8 features utilisées dans le PFE

Objectif :
- Charger la dataset Botnet CICIDS2018
- Garder seulement les 8 features utilisées dans le modèle 2018
- Convertir le label en format binaire :
    0 = NORMAL / BENIGN
    1 = ATTACK / BOTNET
- Supprimer NaN / inf
- Remplacer Init Bwd Win Bytes = -1 par 0
- Sauvegarder un nouveau fichier CSV propre
"""

import os
import numpy as np
import pandas as pd

# =========================================================
# CHEMINS
# =========================================================

BOTNET_PATH = r"C:\Users\DELL\Desktop\DDOS12\PFE_DDOS\PFE_DDOS\scripts\Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"

OUTPUT_PATH = r"C:\Users\DELL\Desktop\DDOS12\PFE_DDOS\PFE_DDOS\scripts\botnet_2018_8features.csv"

# =========================================================
# FEATURES A GARDER
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

FINAL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

# =========================================================
# CHARGEMENT
# =========================================================

df = pd.read_csv(BOTNET_PATH, low_memory=False)

# Nettoyer les noms de colonnes
df.columns = [c.strip() for c in df.columns]

print("==============================")
print("DATASET BOTNET ORIGINAL")
print("==============================")
print("Shape :", df.shape)
print("Colonnes disponibles :")
print(df.columns.tolist())

# =========================================================
# VERIFICATION DES COLONNES
# =========================================================

missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]

if missing_features:
    raise ValueError(
        "Colonnes features manquantes dans la dataset Botnet :\n"
        + str(missing_features)
    )

if LABEL_COLUMN not in df.columns:
    raise ValueError(
        "Colonne Label introuvable dans la dataset Botnet.\n"
        "Colonnes trouvées :\n"
        + str(df.columns.tolist())
    )

# =========================================================
# GARDER LES 8 FEATURES + LABEL
# =========================================================

df_reduced = df[FINAL_COLUMNS].copy()

print("\n==============================")
print("DATASET BOTNET REDUIT")
print("==============================")
print("Shape avant nettoyage :", df_reduced.shape)

# =========================================================
# CONVERSION NUMERIQUE DES FEATURES
# =========================================================

for col in FEATURE_COLUMNS:
    df_reduced[col] = pd.to_numeric(df_reduced[col], errors="coerce")

# =========================================================
# CONVERSION DU LABEL
# =========================================================

print("\nLabels originaux :")
print(df_reduced[LABEL_COLUMN].value_counts())

def convert_label(x):
    """
    Conversion binaire :
    - BENIGN / Benign / Normal -> 0
    - Tout le reste -> 1
    """
    x = str(x).strip().lower()

    if x in ["benign", "normal", "0"]:
        return 0
    else:
        return 1

df_reduced[LABEL_COLUMN] = df_reduced[LABEL_COLUMN].apply(convert_label)

print("\nLabels après conversion binaire :")
print(df_reduced[LABEL_COLUMN].value_counts().sort_index())

# =========================================================
# NETTOYAGE NaN / INF
# =========================================================

df_reduced = df_reduced.replace([np.inf, -np.inf], np.nan)

before_nan = len(df_reduced)
df_reduced = df_reduced.dropna().reset_index(drop=True)
after_nan = len(df_reduced)

print("\nLignes supprimées NaN/inf :", before_nan - after_nan)

# =========================================================
# REMPLACER Init Bwd Win Bytes = -1 PAR 0
# =========================================================

minus_count = (df_reduced["Init Bwd Win Bytes"] == -1).sum()
print("Init Bwd Win Bytes = -1 avant remplacement :", minus_count)

df_reduced["Init Bwd Win Bytes"] = df_reduced["Init Bwd Win Bytes"].replace(-1, 0)

minus_count_after = (df_reduced["Init Bwd Win Bytes"] == -1).sum()
print("Init Bwd Win Bytes = -1 après remplacement :", minus_count_after)

# =========================================================
# SUPPRESSION DES DOUBLONS EXACTS
# =========================================================

before_dup = len(df_reduced)
df_reduced = df_reduced.drop_duplicates().reset_index(drop=True)
after_dup = len(df_reduced)

print("Doublons supprimés :", before_dup - after_dup)

# =========================================================
# VERIFICATION FINALE
# =========================================================

print("\n==============================")
print("VERIFICATION FINALE")
print("==============================")

print("Shape finale :", df_reduced.shape)

print("\nDistribution finale Label :")
print(df_reduced[LABEL_COLUMN].value_counts().sort_index())

print("\nNaN total :", df_reduced.isna().sum().sum())
print("Inf total :", np.isinf(df_reduced[FEATURE_COLUMNS].astype(np.float32).values).sum())
print("Doublons exacts :", df_reduced.duplicated().sum())

print("\nColonnes finales :")
print(df_reduced.columns.tolist())

# =========================================================
# SAUVEGARDE
# =========================================================

df_reduced.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("\n==============================")
print("FICHIER CREE")
print("==============================")
print(OUTPUT_PATH)

print("\nIMPORTANT :")
print("- Le fichier Botnet est maintenant réduit aux mêmes 8 features que le modèle CICIDS2018.")
print("- Label 0 = NORMAL / BENIGN.")
print("- Label 1 = ATTACK / BOTNET.")
print("- Ce fichier peut être utilisé pour test externe ou pour enrichir train/validation.")