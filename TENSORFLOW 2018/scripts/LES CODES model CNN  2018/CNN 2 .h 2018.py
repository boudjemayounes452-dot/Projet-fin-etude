# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:19:09 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Conversion CNN TFLite vers header C pour ESP32

Entrée :
cnn_final_stable_seed42.tflite

Sortie :
cnn_model_data.h

Objectif :
- Convertir le modèle CNN .tflite en tableau C
- Générer un fichier .h utilisable dans Arduino IDE / PlatformIO
- Le fichier généré contient :
    const unsigned char cnn_model_tflite[]
    const unsigned int cnn_model_tflite_len
"""

import os


# =========================================================
# CHEMINS
# =========================================================

BASE_DIR = r"C:\Users\DELL\Desktop\DDOS12\PFE_DDOS\PFE_DDOS\scripts\cnn_final_stable_2018_chrono_live_botnet_seed42_tflite_no_opt"

TFLITE_PATH = os.path.join(
    BASE_DIR,
    "cnn_final_stable_seed42.tflite"
)

HEADER_PATH = os.path.join(
    BASE_DIR,
    "cnn_model_data.h"
)


# =========================================================
# VERIFICATION DU FICHIER TFLITE
# =========================================================

if not os.path.exists(TFLITE_PATH):
    print("ERREUR : fichier TFLite introuvable.")
    print("Chemin recherché :", TFLITE_PATH)
    print("\nVérifie que le fichier existe bien dans ce dossier :")
    print(BASE_DIR)
    raise FileNotFoundError(TFLITE_PATH)

print("Fichier TFLite trouvé :", TFLITE_PATH)


# =========================================================
# LECTURE DU MODELE TFLITE
# =========================================================

with open(TFLITE_PATH, "rb") as f:
    tflite_data = f.read()

model_size = len(tflite_data)

print("Taille du modèle TFLite :", model_size, "octets")


# =========================================================
# CONVERSION EN TABLEAU C
# =========================================================

with open(HEADER_PATH, "w", encoding="utf-8") as f:
    f.write("#ifndef CNN_MODEL_DATA_H\n")
    f.write("#define CNN_MODEL_DATA_H\n\n")

    f.write("// =========================================================\n")
    f.write("// Modele CNN TFLite converti en header C pour ESP32\n")
    f.write("// Genere automatiquement depuis cnn_final_stable_seed42.tflite\n")
    f.write("// =========================================================\n\n")

    f.write("#include <cstdint>\n\n")

    f.write("const unsigned char cnn_model_tflite[] = {\n")

    for i, byte in enumerate(tflite_data):
        if i % 12 == 0:
            f.write("  ")

        f.write(f"0x{byte:02x}")

        if i != model_size - 1:
            f.write(", ")

        if i % 12 == 11:
            f.write("\n")

    if model_size % 12 != 0:
        f.write("\n")

    f.write("};\n\n")

    f.write(f"const unsigned int cnn_model_tflite_len = {model_size};\n\n")

    f.write("#endif // CNN_MODEL_DATA_H\n")


# =========================================================
# FIN
# =========================================================

print("\n===== CONVERSION TERMINEE =====")
print("Header généré :", HEADER_PATH)
print("Nom du tableau :", "cnn_model_tflite")
print("Taille modèle  :", model_size, "octets")

print("\nIMPORTANT :")
print("- Copie cnn_model_data.h dans le dossier du projet Arduino.")
print("- Dans Arduino, utilise : #include \"cnn_model_data.h\"")
print("- Dans modelInit, utilise : cnn_model_tflite")