# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:38:02 2026

@author: DELL
"""

# -*- coding: utf-8 -*-

import os
import joblib
import emlearn

# =========================================================
# CHEMINS DU MODELE FINAL RF
# =========================================================

MODEL_PATH = r"C:\Users\DELL\Desktop\pfe 2023\result\rf_final_clean\rf_final_4classes.pkl"
OUT_HEADER = r"C:\Users\DELL\Desktop\pfe 2023\result\rf_final_clean\rf_model_esp32.h"

# =========================================================
# VERIFICATION
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}")

# =========================================================
# CHARGER LE MODELE
# =========================================================

rf = joblib.load(MODEL_PATH)
print("Modele charge :", MODEL_PATH)

# =========================================================
# CONVERSION EN .H POUR ESP32
# =========================================================

cmodel = emlearn.convert(rf, method="inline")

# =========================================================
# SAUVEGARDE HEADER
# =========================================================

cmodel.save(file=OUT_HEADER, name="rf_model")

print("Header genere :", OUT_HEADER)
print("Export termine avec succes")