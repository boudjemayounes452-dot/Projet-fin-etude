import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

NORMAL_FILE = os.path.join(DATA_DIR, "normal_data.csv")
ATTACK_FILE = os.path.join(DATA_DIR, "attack_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset_final7_features.csv")

print("=" * 50)
print("📊 CRÉATION DU DATASET")
print("=" * 50)

if not os.path.exists(NORMAL_FILE):
    print(f"❌ normal_data.csv introuvable")
    exit()
if not os.path.exists(ATTACK_FILE):
    print(f"❌ attack_data.csv introuvable")
    exit()

normal = pd.read_csv(NORMAL_FILE)
attack = pd.read_csv(ATTACK_FILE)

print(f"📁 Normal : {len(normal)} lignes")
print(f"📁 Attack : {len(attack)} lignes")

normal = normal.dropna()
attack = attack.dropna()

NB = 500
normal_sample = normal.sample(n=min(NB, len(normal)), random_state=42)
attack_sample = attack.sample(n=min(NB, len(attack)), random_state=42)

normal_sample['label'] = 'normal'
attack_sample['label'] = 'udp_flood'

dataset = pd.concat([normal_sample, attack_sample], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n📊 Dataset final : {len(dataset)} lignes")
print(f"   normal : {len(dataset[dataset['label']=='normal'])}")
print(f"   udp_flood : {len(dataset[dataset['label']=='udp_flood'])}")

dataset.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Sauvegardé : {OUTPUT_FILE}")

print("\n✅ PRET POUR EDGE IMPULSE !")