# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:37:26 2026

@author: DELL
"""

# -*- coding: utf-8 -*-

import pandas as pd

# =========================================================
# CHEMINS
# =========================================================

base_path = r"C:\Users\DELL\Desktop\pfe 2023\result\mini_dataset2023_best_classes.csv"

normal_path = r"C:\Users\DELL\Desktop\pfe 2023\result\live_features_2023 noraml2.csv"
icmp_path   = r"C:\Users\DELL\Desktop\pfe 2023\result\live_features_2023 icmp2.csv"
syn_path    = r"C:\Users\DELL\Desktop\pfe 2023\result\live_features_2023 SYN1.csv"
udp_path    = r"C:\Users\DELL\Desktop\pfe 2023\result\live_features_2023 udp1.csv"

output_dir = r"C:\Users\DELL\Desktop\pfe 2023\result"

train_path = output_dir + r"\train_clean.csv"
val_path   = output_dir + r"\val_clean.csv"
test_path  = output_dir + r"\test_clean.csv"

# =========================================================
# PARAMETRES
# =========================================================

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
GAP_RATIO = 0.02

expected_columns = [
    "IAT",
    "Header_Length",
    "Protocol Type",
    "flow_duration",
    "rst_count",
    "Tot size",
    "Magnitue",
    "AVG",
    "label"
]

feature_columns = expected_columns[:-1]

keep_labels = [
    "BenignTraffic",
    "DDoS-ICMP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-SYN_Flood"
]

# =========================================================
# FONCTIONS
# =========================================================

def prepare_df(path, name):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Label" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"Label": "label"})

    df = df[expected_columns].copy()

    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label"] = df["label"].astype(str).str.strip()
    df = df.dropna()
    df = df[df["label"].isin(keep_labels)].copy()

    print(f"\n{name} :", df.shape)
    print(df["label"].value_counts())

    return df


def split_base_with_gap(df):
    n = len(df)
    gap = int(n * GAP_RATIO)

    train_end = int(n * TRAIN_RATIO)

    val_start = train_end + gap
    val_end = val_start + int(n * VAL_RATIO)

    test_start = val_end + gap

    train = df.iloc[:train_end].copy()
    val   = df.iloc[val_start:val_end].copy()
    test  = df.iloc[test_start:].copy()

    gap1 = df.iloc[train_end:val_start].copy()
    gap2 = df.iloc[val_end:test_start].copy()

    print("\n===== SPLIT BASE AVEC GAP =====")
    print("Total :", n)
    print("Train :", train.shape)
    print("Gap 1 :", gap1.shape)
    print("Val   :", val.shape)
    print("Gap 2 :", gap2.shape)
    print("Test  :", test.shape)

    return train, val, test


def split_live_train_val_only(df):
    n = len(df)
    train_end = int(0.75 * n)

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:].copy()

    return train, val

# =========================================================
# CHARGEMENT
# =========================================================

base_df = prepare_df(base_path, "BASE")

normal_df = prepare_df(normal_path, "LIVE NORMAL")
icmp_df   = prepare_df(icmp_path, "LIVE ICMP")
syn_df    = prepare_df(syn_path, "LIVE SYN")
udp_df    = prepare_df(udp_path, "LIVE UDP")

# =========================================================
# SPLIT BASE AVEC GAP
# =========================================================

base_train, base_val, base_test = split_base_with_gap(base_df)

# =========================================================
# LIVE SEULEMENT TRAIN + VALIDATION
# =========================================================

normal_train, normal_val = split_live_train_val_only(normal_df)
icmp_train, icmp_val     = split_live_train_val_only(icmp_df)
syn_train, syn_val       = split_live_train_val_only(syn_df)
udp_train, udp_val       = split_live_train_val_only(udp_df)

# =========================================================
# CREATION DES 3 FICHIERS
# =========================================================

train_df = pd.concat(
    [base_train, normal_train, icmp_train, syn_train, udp_train],
    ignore_index=True
)

val_df = pd.concat(
    [base_val, normal_val, icmp_val, syn_val, udp_val],
    ignore_index=True
)

test_df = base_test.copy()

# Mélanger seulement train et validation
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df   = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# Supprimer doublons exacts
train_df = train_df.drop_duplicates().reset_index(drop=True)
val_df   = val_df.drop_duplicates().reset_index(drop=True)
test_df  = test_df.drop_duplicates().reset_index(drop=True)

# =========================================================
# SAUVEGARDE
# =========================================================

train_df.to_csv(train_path, index=False, encoding="utf-8")
val_df.to_csv(val_path, index=False, encoding="utf-8")
test_df.to_csv(test_path, index=False, encoding="utf-8")

# =========================================================
# AFFICHAGE
# =========================================================

print("\n===== FICHIERS CREES =====")
print("Train :", train_path, train_df.shape)
print("Val   :", val_path, val_df.shape)
print("Test  :", test_path, test_df.shape)

print("\n===== TRAIN =====")
print(train_df["label"].value_counts())

print("\n===== VALIDATION =====")
print(val_df["label"].value_counts())

print("\n===== TEST =====")
print(test_df["label"].value_counts())

print("\nIMPORTANT :")
print("Le gap est ignoré.")
print("Les lives sont seulement dans train et validation.")
print("Le test contient seulement le mini dataset de base.")