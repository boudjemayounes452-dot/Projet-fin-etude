# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

# Chemin absolu vers votre dossier de données
base = r"C:\Users\Z-REPAIR\Desktop\PFE_ESP32_IDS"
data_dir = os.path.join(base, "data")

# Charger les fichiers collectés
normal = pd.read_csv(os.path.join(data_dir, "normal_data.csv"))
attack = pd.read_csv(os.path.join(data_dir, "attack_data.csv"))

# Ajouter la colonne label
normal['label'] = 'normal'
attack['label'] = 'udp_flood'

# Concaténer et mélanger
df = pd.concat([normal, attack], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Liste des 7 caractéristiques
features = ["total_pkts", "target_pkts", "pkt_ratio", "avg_pkt_size", 
            "unique_MACs", "mean_IAT_us", "IAT_variance"]

X = df[features]
y = df['label']

# Random Forest (200 arbres, seed fixe pour reproductibilité)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Récupérer l'importance
importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Sauvegarder le tableau CSV
importance.to_csv(os.path.join(data_dir, "feature_importance.csv"), index=False)

# Tracer le graphique à barres horizontales
plt.figure(figsize=(8, 4))
plt.barh(importance['Feature'], importance['Importance'], color='steelblue')
plt.gca().invert_yaxis()  # pour que la plus importante soit en haut
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest (ESP32 data)")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "feature_importance.png"), dpi=150)
plt.show()

print("\n✅ Fichiers générés :")
print(f"   - {os.path.join(data_dir, 'feature_importance.csv')}")
print(f"   - {os.path.join(data_dir, 'feature_importance.png')}")