# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:42:10 2026

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger le dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\projet\DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv")

# 2. Nettoyage
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# 3. Transformer le label
df["Label"] = df["Label"].apply(
    lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
)

# 4. Séparer X et y
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=[np.number])  # garder seulement les colonnes numériques
y = df["Label"]

# 5. Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Random Forest pour importance
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

# 7. Évaluation
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report :")
print(classification_report(y_test, y_pred))

# 8. Importance des features
importances = rf.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 features :")
print(feature_importance_df.head(10))

# 9. Top 4 features
top_features = feature_importance_df["Feature"].head(4).tolist()
print("\nTop 4 features :", top_features)

# 10. Construire mini-dataset
mini_df = df[top_features + ["Label"]]

# 11. Sauvegarder
mini_df.to_csv("mini_dataset_4features.csv", index=False)
print("\nMini-dataset sauvegardé : mini_dataset_4features.csv")