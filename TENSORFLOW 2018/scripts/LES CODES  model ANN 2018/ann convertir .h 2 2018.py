# -*- coding: utf-8 -*-
"""
Conversion TFLite vers header C pour ESP32

Entrée :
ann_final_stable_seed84.tflite

Sortie :
ann_model_data.h
"""

import os


# =========================================================
# CHEMINS
# =========================================================

BASE_DIR = r"C:\Users\DELL\Desktop\DDOS12\PFE_DDOS\PFE_DDOS\scripts\ann_final_stable_2018_chrono_live_botnet_seed84_tflite"

TFLITE_PATH = os.path.join(
    BASE_DIR,
    "ann_final_stable_seed84.tflite"
)

HEADER_PATH = os.path.join(
    BASE_DIR,
    "ann_model_data.h"
)


# =========================================================
# VERIFICATION
# =========================================================

if not os.path.exists(TFLITE_PATH):
    print("ERREUR : fichier TFLite introuvable.")
    print("Chemin recherché :", TFLITE_PATH)
    print("\nVérifie le nom exact du fichier dans ce dossier :")
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
    f.write("#ifndef ANN_MODEL_DATA_H\n")
    f.write("#define ANN_MODEL_DATA_H\n\n")

    f.write("// =========================================================\n")
    f.write("// Modele ANN TFLite converti en header C pour ESP32\n")
    f.write("// Genere automatiquement depuis ann_final_stable_seed84.tflite\n")
    f.write("// =========================================================\n\n")

    f.write("#include <cstdint>\n\n")

    f.write("const unsigned char ann_model_tflite[] = {\n")

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

    f.write(f"const unsigned int ann_model_tflite_len = {model_size};\n\n")

    f.write("#endif // ANN_MODEL_DATA_H\n")


# =========================================================
# FIN
# =========================================================

print("\n===== CONVERSION TERMINEE =====")
print("Header généré :", HEADER_PATH)
print("Nom du tableau :", "ann_model_tflite")
print("Taille modèle  :", model_size, "octets")

print("\nIMPORTANT :")
print("- Copie ann_model_data.h dans le dossier du projet Arduino.")
print("- Dans Arduino, garde : #include \"ann_model_data.h\"")
print("- Le nom du tableau utilise dans Arduino est : ann_model_tflite")