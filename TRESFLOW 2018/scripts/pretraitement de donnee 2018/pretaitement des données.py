# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:58:50 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
PIPELINE FINAL CORRIGE - CICIDS2018 + LIVE + BOTNET

Objectif :
1. Nettoyer CICIDS2018
2. Faire un split chronologique séquentiel global avec gap
3. Ajouter les captures live seulement dans train/validation
4. Charger Botnet avec encodage correct :
      0 = NORMAL / BENIGN
      1 = ATTACK / BOTNET
5. Séparer Botnet chronologiquement :
      début 10 % -> train/validation
      fin 90 %   -> test externe
6. Sauvegarder les fichiers finaux
"""

import os
import numpy as np
import pandas as pd

# =========================================================
# CHEMINS
# =========================================================

BASE_DIR = r"C:\Users\DELL\Desktop\DDOS12\PFE_DDOS\PFE_DDOS\scripts"

MINI_2018_PATH = os.path.join(BASE_DIR, "minidataset2018.csv")

LIVE_NORMAL_PATH = os.path.join(
    BASE_DIR,
    "live_features_cicids2018_auto noraml.csv"
)

LIVE_ATTACK_PATH = os.path.join(
    BASE_DIR,
    "live_features_cicids2018_auto attaque.csv"
)

BOTNET_PATH = os.path.join(
    BASE_DIR,
    "botnet_2018_8features.csv"
)

OUT_TRAIN = os.path.join(
    BASE_DIR,
    "train_2018_chrono_live_botnet10_encoded.csv"
)

OUT_VAL = os.path.join(
    BASE_DIR,
    "val_2018_chrono_live_botnet10_encoded.csv"
)

OUT_TEST_2018 = os.path.join(
    BASE_DIR,
    "test_2018_chrono_secondary_encoded.csv"
)

OUT_BOTNET_TEST = os.path.join(
    BASE_DIR,
    "test_botnet90_external_chrono_encoded.csv"
)

OUT_BOTNET_10 = os.path.join(
    BASE_DIR,
    "botnet10_for_train_val_chrono_encoded.csv"
)

# =========================================================
# PARAMETRES
# =========================================================

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
GAP_RATIO = 0.02

LIVE_TRAIN_RATIO = 0.75

BOTNET_TRAINVAL_RATIO = 0.10
BOTNET_TRAIN_RATIO = 0.75

RANDOM_STATE = 42

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
ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

# =========================================================
# LECTURE CSV ROBUSTE
# =========================================================

def read_csv_auto(path):
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(path, sep=",", low_memory=False)
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


# =========================================================
# ENCODAGE LABEL
# =========================================================

def encode_label_binary(x):
    """
    Encodage binaire :
    0 = NORMAL / BENIGN
    1 = ATTACK / BOTNET / autre attaque
    """

    value = str(x).strip().lower()

    if value in ["0", "benign", "normal", "normal traffic"]:
        return 0

    if value in ["1", "attack", "botnet", "ddos", "malicious"]:
        return 1

    # Si le label est autre chose que normal, on le considère comme attaque
    return 1


# =========================================================
# PREPARATION GENERALE
# =========================================================

def prepare_df(path, name):
    df = read_csv_auto(path)

    df.columns = [c.strip() for c in df.columns]

    if "label" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={"label": "Label"})

    missing = [c for c in ALL_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"\nErreur dans {name}\n"
            f"Colonnes manquantes : {missing}\n"
            f"Colonnes trouvées : {list(df.columns)}"
        )

    df = df[ALL_COLUMNS].copy()

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(encode_label_binary)

    df = df.replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna().reset_index(drop=True)

    print(f"\n{name} : lignes supprimées NaN/inf = {before - len(df)}")

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df = df[df[LABEL_COLUMN].isin([0, 1])].copy().reset_index(drop=True)

    print("\n==============================")
    print(name)
    print("==============================")
    print("Shape :", df.shape)
    print("Distribution :")
    print(df[LABEL_COLUMN].value_counts().sort_index())

    return df


# =========================================================
# NETTOYAGE CICIDS2018
# =========================================================

def clean_cicids2018(df):
    print("\n==============================")
    print("NETTOYAGE MINI-DATASET CICIDS2018")
    print("==============================")

    print("Avant nettoyage :", df.shape)

    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print("Doublons supprimés :", before_dup - len(df))

    suspect = (
        (df["Init Bwd Win Bytes"] == -1) &
        (df["Flow Bytes/s"] == 0) &
        (df["Avg Packet Size"] == 0) &
        (df["Total Fwd Packets"] > 0)
    )

    print("Lignes suspectes trouvées :", suspect.sum())
    print("Distribution des lignes suspectes :")
    print(df[suspect][LABEL_COLUMN].value_counts().sort_index())

    df = df[~suspect].copy().reset_index(drop=True)

    minus_count = (df["Init Bwd Win Bytes"] == -1).sum()
    print("Init Bwd Win Bytes = -1 restants avant remplacement :", minus_count)

    df["Init Bwd Win Bytes"] = df["Init Bwd Win Bytes"].replace(-1, 0)

    print("Après nettoyage CICIDS2018 :", df.shape)
    print("Distribution finale :")
    print(df[LABEL_COLUMN].value_counts().sort_index())

    return df


# =========================================================
# NETTOYAGE LIVE
# =========================================================

def clean_live(df, name):
    print("\n==============================")
    print("NETTOYAGE", name)
    print("==============================")

    print("Avant :", df.shape)

    df["Init Bwd Win Bytes"] = df["Init Bwd Win Bytes"].replace(-1, 0)

    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    print("Doublons supprimés :", before_dup - len(df))
    print("Après :", df.shape)
    print("Distribution :")
    print(df[LABEL_COLUMN].value_counts().sort_index())

    return df


# =========================================================
# NETTOYAGE BOTNET CORRIGE
# =========================================================

def clean_botnet(df):
    print("\n==============================")
    print("NETTOYAGE BOTNET")
    print("==============================")

    print("Avant nettoyage Botnet :", df.shape)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encodage correct :
    # 0 = NORMAL / BENIGN
    # 1 = ATTACK / BOTNET
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(encode_label_binary)

    df = df.replace([np.inf, -np.inf], np.nan)

    before_nan = len(df)
    df = df.dropna().reset_index(drop=True)
    print("Lignes supprimées NaN/inf :", before_nan - len(df))

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    df["Init Bwd Win Bytes"] = df["Init Bwd Win Bytes"].replace(-1, 0)

    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print("Doublons supprimés :", before_dup - len(df))

    print("Après nettoyage Botnet :", df.shape)
    print("Distribution Botnet corrigée :")
    print(df[LABEL_COLUMN].value_counts().sort_index())

    return df


# =========================================================
# SPLIT CHRONOLOGIQUE GLOBAL CICIDS2018
# =========================================================

def split_chrono_global_with_gap(df):
    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    gap_size = int(n * GAP_RATIO)

    val_start = train_end + gap_size
    val_end = val_start + int(n * VAL_RATIO)

    test_start = val_end + gap_size

    train = df.iloc[:train_end].copy()
    gap1 = df.iloc[train_end:val_start].copy()
    val = df.iloc[val_start:val_end].copy()
    gap2 = df.iloc[val_end:test_start].copy()
    test = df.iloc[test_start:].copy()

    print("\n==============================")
    print("SPLIT CHRONOLOGIQUE GLOBAL CICIDS2018 AVEC GAP")
    print("==============================")
    print("Total :", n)
    print("Train :", train.shape)
    print("Gap 1 :", gap1.shape)
    print("Val   :", val.shape)
    print("Gap 2 :", gap2.shape)
    print("Test  :", test.shape)

    print("\nDistribution train CICIDS2018 :")
    print(train[LABEL_COLUMN].value_counts().sort_index())

    print("\nDistribution validation CICIDS2018 :")
    print(val[LABEL_COLUMN].value_counts().sort_index())

    print("\nDistribution test CICIDS2018 :")
    print(test[LABEL_COLUMN].value_counts().sort_index())

    return train, val, test


# =========================================================
# SPLIT LIVE
# =========================================================

def split_live_train_val(df):
    n = len(df)
    train_end = int(n * LIVE_TRAIN_RATIO)

    live_train = df.iloc[:train_end].copy()
    live_val = df.iloc[train_end:].copy()

    return live_train, live_val


# =========================================================
# SPLIT CHRONOLOGIQUE BOTNET CORRIGE
# =========================================================

def split_botnet_chrono(df):
    """
    Botnet dans l'ordre original :
    - début 10 % pour train/validation
    - fin 90 % pour test externe

    Les labels sont conservés :
    0 = NORMAL
    1 = ATTACK
    """

    n = len(df)

    cut_10 = int(n * BOTNET_TRAINVAL_RATIO)

    botnet_10 = df.iloc[:cut_10].copy().reset_index(drop=True)
    botnet_90 = df.iloc[cut_10:].copy().reset_index(drop=True)

    n_10 = len(botnet_10)
    cut_train = int(n_10 * BOTNET_TRAIN_RATIO)

    botnet_train = botnet_10.iloc[:cut_train].copy().reset_index(drop=True)
    botnet_val = botnet_10.iloc[cut_train:].copy().reset_index(drop=True)

    print("\n==============================")
    print("SPLIT CHRONOLOGIQUE BOTNET")
    print("==============================")
    print("Botnet total :", n)
    print("Botnet 10 % train/val :", botnet_10.shape)
    print("Botnet 90 % test externe :", botnet_90.shape)
    print("Botnet train :", botnet_train.shape)
    print("Botnet val   :", botnet_val.shape)

    print("\nDistribution Botnet 10 % train/val :")
    print(botnet_10[LABEL_COLUMN].value_counts().sort_index())

    print("\nDistribution Botnet train :")
    print(botnet_train[LABEL_COLUMN].value_counts().sort_index())

    print("\nDistribution Botnet validation :")
    print(botnet_val[LABEL_COLUMN].value_counts().sort_index())

    print("\nDistribution Botnet 90 % test externe :")
    print(botnet_90[LABEL_COLUMN].value_counts().sort_index())

    return botnet_train, botnet_val, botnet_10, botnet_90


# =========================================================
# VERIFICATIONS
# =========================================================

def check_df(df, name):
    print("\n==============================")
    print(name)
    print("==============================")
    print("Shape :", df.shape)
    print("NaN :", df.isna().sum().sum())
    print("Inf :", np.isinf(df[FEATURE_COLUMNS].astype(np.float32).values).sum())
    print("Init Bwd Win Bytes = -1 :", (df["Init Bwd Win Bytes"] == -1).sum())
    print("Doublons exacts :", df.duplicated().sum())
    print("Distribution :")
    print(df[LABEL_COLUMN].value_counts().sort_index())


def check_botnet_leakage(botnet_10, botnet_90):
    merged = pd.merge(
        botnet_10,
        botnet_90,
        on=ALL_COLUMNS,
        how="inner"
    )

    print("\n==============================")
    print("VERIFICATION ANTI-LEAKAGE BOTNET")
    print("==============================")
    print("Doublons exacts entre Botnet train/val et Botnet test :", len(merged))

    if len(merged) > 0:
        raise ValueError(
            "Attention : leakage Botnet détecté entre train/val et test externe."
        )


# =========================================================
# EXECUTION
# =========================================================

mini_2018 = prepare_df(MINI_2018_PATH, "MINI-DATASET CICIDS2018")
live_normal = prepare_df(LIVE_NORMAL_PATH, "LIVE NORMAL 2018")
live_attack = prepare_df(LIVE_ATTACK_PATH, "LIVE ATTACK 2018")
botnet = prepare_df(BOTNET_PATH, "BOTNET 2018 8 FEATURES")

mini_2018 = clean_cicids2018(mini_2018)
live_normal = clean_live(live_normal, "LIVE NORMAL 2018")
live_attack = clean_live(live_attack, "LIVE ATTACK 2018")
botnet = clean_botnet(botnet)

train_2018, val_2018, test_2018 = split_chrono_global_with_gap(mini_2018)

live_normal_train, live_normal_val = split_live_train_val(live_normal)
live_attack_train, live_attack_val = split_live_train_val(live_attack)

print("\n==============================")
print("SPLIT LIVE")
print("==============================")
print("Live normal train :", live_normal_train.shape)
print("Live normal val   :", live_normal_val.shape)
print("Live attack train :", live_attack_train.shape)
print("Live attack val   :", live_attack_val.shape)

botnet_train, botnet_val, botnet_10, botnet_90 = split_botnet_chrono(botnet)

check_botnet_leakage(botnet_10, botnet_90)

train_final = pd.concat(
    [
        train_2018,
        live_normal_train,
        live_attack_train,
        botnet_train
    ],
    ignore_index=True
)

val_final = pd.concat(
    [
        val_2018,
        live_normal_val,
        live_attack_val,
        botnet_val
    ],
    ignore_index=True
)

test_2018_final = test_2018.copy()
botnet_test_final = botnet_90.copy()

train_final = train_final.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
val_final = val_final.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

test_2018_final = test_2018_final.reset_index(drop=True)
botnet_test_final = botnet_test_final.reset_index(drop=True)

before_train = len(train_final)
before_val = len(val_final)

train_final = train_final.drop_duplicates().reset_index(drop=True)
val_final = val_final.drop_duplicates().reset_index(drop=True)

print("\n==============================")
print("DOUBLONS APRES FUSION")
print("==============================")
print("Train supprimés :", before_train - len(train_final))
print("Val supprimés   :", before_val - len(val_final))

check_df(train_final, "TRAIN FINAL = 2018 CHRONO + LIVE + BOTNET 10 %")
check_df(val_final, "VALIDATION FINAL = 2018 CHRONO + LIVE + BOTNET 10 %")
check_df(test_2018_final, "TEST SECONDAIRE CICIDS2018 CHRONO")
check_df(botnet_test_final, "TEST EXTERNE BOTNET 90 %")

train_final.to_csv(OUT_TRAIN, index=False, encoding="utf-8")
val_final.to_csv(OUT_VAL, index=False, encoding="utf-8")
test_2018_final.to_csv(OUT_TEST_2018, index=False, encoding="utf-8")
botnet_test_final.to_csv(OUT_BOTNET_TEST, index=False, encoding="utf-8")
botnet_10.to_csv(OUT_BOTNET_10, index=False, encoding="utf-8")

print("\n==============================")
print("FICHIERS CREES")
print("==============================")
print("Train final :", OUT_TRAIN)
print("Validation final :", OUT_VAL)
print("Test secondaire 2018 :", OUT_TEST_2018)
print("Test externe Botnet 90 % :", OUT_BOTNET_TEST)
print("Botnet 10 % train/val :", OUT_BOTNET_10)

print("\nIMPORTANT :")
print("- Encodage corrigé : 0 = NORMAL / BENIGN, 1 = ATTACK / BOTNET.")
print("- Split CICIDS2018 : chronologique séquentiel global avec gap.")
print("- Pas de split par classe.")
print("- Les captures live sont ajoutées seulement dans train et validation.")
print("- Split Botnet : début 10 % dans train/validation, fin 90 % en test externe.")
print("- Les labels Botnet sont conservés : 0 pour normal, 1 pour attaque.")