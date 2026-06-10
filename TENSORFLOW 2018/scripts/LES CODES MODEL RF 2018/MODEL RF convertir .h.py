# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:34:19 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Conversion Random Forest final 2018 + Live + Botnet vers header ESP32

Modele :
rf_final_2018_chrono_live_botnet.joblib

Sortie :
rf_model_esp32.h

Classes :
0 = NORMAL
1 = ATTACK
"""

import os
import joblib
import emlearn

# =========================================================
# CHEMINS
# =========================================================

BASE_DIR = r"C:\Users\DELL\Desktop\DDOS12\PFE_DDOS\PFE_DDOS\scripts"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "rf_final_2018_chrono_live_botnet",
    "rf_final_2018_chrono_live_botnet.joblib"
)

OUT_HEADER = os.path.join(
    BASE_DIR,
    "rf_model_esp32.h"
)

# =========================================================
# VERIFICATION
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}")

# =========================================================
# CHARGEMENT MODELE
# =========================================================

rf = joblib.load(MODEL_PATH)

print("Modele charge :", MODEL_PATH)
print("Nombre arbres :", len(rf.estimators_))
print("Classes :", rf.classes_)

# =========================================================
# CONVERSION EMLEARN
# =========================================================

cmodel = emlearn.convert(
    rf,
    method="inline"
)

# =========================================================
# SAUVEGARDE HEADER
# =========================================================

cmodel.save(
    file=OUT_HEADER,
    name="rf_model"
)

print("Header genere :", OUT_HEADER)
print("Export termine avec succes")