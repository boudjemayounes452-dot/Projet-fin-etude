import serial
import csv
import time
import os

PORT = 'COM4'
BAUDRATE = 115200
MAX_ROWS = 1000

# Change le nom selon normal ou attack
FILENAME = r"..\data\attack_data.csv"

headers = ["total_pkts", "target_pkts", "pkt_ratio", "avg_pkt_size", 
           "unique_MACs", "mean_IAT_us", "IAT_variance"]

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ Connecté à {PORT}")
except:
    print(f"❌ Impossible de se connecter à {PORT}")
    exit()

with open(FILENAME, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    row_count = 0
    
    print(f"📊 Collecte vers {FILENAME}")
    print("Appuyez sur CTRL+C pour arrêter\n")
    
    try:
        while row_count < MAX_ROWS:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                parts = line.split(',')
                if len(parts) == 7:
                    writer.writerow(parts)
                    csvfile.flush()
                    row_count += 1
                    print(f"✅ Ligne {row_count}/{MAX_ROWS}")
    except KeyboardInterrupt:
        print(f"\n🛑 Arrêt - {row_count} lignes")
    finally:
        print(f"\n📁 Fichier : {os.path.abspath(FILENAME)}")
        print(f"📊 Total : {row_count} lignes")
        ser.close()
