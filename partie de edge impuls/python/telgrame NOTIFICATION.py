# -*- coding: utf-8 -*-
"""
Script d'alerte UDP Flood - ESP32 -> Telegram
Copier-coller et exécuter.
"""

import serial
import requests
import time

# ========== CONFIGURATION ==========
PORT = 'COM4'                     # Port série de l'ESP32
BAUDRATE = 115200                 # Doit correspondre à celui de l'ESP32
TELEGRAM_TOKEN = "8994671335:AAGoWBoDvigJ-5Zpj0gpJrEosHRlFMKab7A"  # À révoquer et remplacer si exposé
CHAT_ID = "5425221365"            # Votre identifiant Telegram (ou groupe)
# ===================================

predictions = []
last_alert = 0

def send_telegram(msg):
    """Envoie un message via le bot Telegram, avec vérification d'erreur."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.get(url, params={"chat_id": CHAT_ID, "text": msg}, timeout=5)
        if resp.status_code == 200 and resp.json().get("ok"):
            print("✅ Alerte envoyée")
        else:
            print(f"❌ Échec Telegram: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Erreur réseau: {e}")

# Ouverture du port série
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"📡 Connexion série établie sur {PORT} à {BAUDRATE} bauds")
    print("📡 En attente des prédictions (PRED:...) ...")
except Exception as e:
    print(f"❌ Impossible d'ouvrir le port {PORT}: {e}")
    exit(1)

try:
    while True:
        line = ser.readline().decode().strip()
        if not line:
            continue

        if line.startswith("PRED:"):
            # Format attendu: "PRED:udp_flood" ou "PRED:normal"
            parts = line.split(",")
            label = parts[0].split(":")[1]   # extrait "udp_flood" ou "normal"
            predictions.append(label)

            # Garder seulement les 10 dernières prédictions
            if len(predictions) > 10:
                predictions.pop(0)

            flood = sum(1 for p in predictions if p == "udp_flood")
            print(f"PRED: {label} | floodCount={flood}/10")

            # Alerte si au moins 6 "udp_flood" sur les 10 dernières, et pas plus d'une par minute
            if flood >= 6 and (time.time() - last_alert) > 60:
                send_telegram("⚠️ ALERTE : UDP Flood détecté par ESP32 !")
                last_alert = time.time()

except KeyboardInterrupt:
    print("\n🛑 Script arrêté manuellement")
finally:
    ser.close()
    print("🔌 Port série fermé")