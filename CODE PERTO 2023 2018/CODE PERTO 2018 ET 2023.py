# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:21:51 2026

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Analyse Pareto stricte et visualisation 3D
Critères :
- F1-score macro : à maximiser
- Temps d'inférence : à minimiser
- Mémoire Flash : à minimiser
"""

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Données expérimentales
# ============================================================

data = pd.DataFrame({
    "Dataset": [
        "CICIDS2018", "CICIDS2018", "CICIDS2018",
        "CICIoT2023", "CICIoT2023", "CICIoT2023"
    ],
    "Modele": [
        "ANN", "CNN 1D", "Random Forest",
        "ANN", "CNN 1D", "Random Forest"
    ],
    "F1_macro": [
        0.933132, 0.946957, 0.989859,
        0.963170, 0.965112, 0.987170
    ],
    "Temps_us": [
        165.5, 903, 1877,
        149, 366.25, 790.25
    ],
    "Flash_bytes": [
        597268, 603208, 481048,
        596136, 598132, 391588
    ]
})


# ============================================================
# 2. Fonction de dominance Pareto
# ============================================================

def dominates(a, b):
    """
    Le modèle a domine le modèle b si :
    - a a un F1-score supérieur ou égal à b ;
    - a a un temps d'inférence inférieur ou égal à b ;
    - a a une mémoire Flash inférieure ou égale à b ;
    - et a est strictement meilleur dans au moins un critère.
    """
    return (
        a["F1_macro"] >= b["F1_macro"] and
        a["Temps_us"] <= b["Temps_us"] and
        a["Flash_bytes"] <= b["Flash_bytes"] and
        (
            a["F1_macro"] > b["F1_macro"] or
            a["Temps_us"] < b["Temps_us"] or
            a["Flash_bytes"] < b["Flash_bytes"]
        )
    )


# ============================================================
# 3. Calcul du front de Pareto + figure 3D
# ============================================================

for dataset in data["Dataset"].unique():

    subset = data[data["Dataset"] == dataset].reset_index(drop=True)
    pareto_models = []

    print("\n==============================")
    print("Dataset :", dataset)
    print("==============================")

    # Calcul des modèles non dominés
    for i, model_i in subset.iterrows():
        dominated = False

        for j, model_j in subset.iterrows():
            if i != j and dominates(model_j, model_i):
                dominated = True
                break

        if not dominated:
            pareto_models.append(model_i["Modele"])

    print("Front de Pareto :", pareto_models)

    # Afficher si chaque modèle est dominé ou non
    for i, model_i in subset.iterrows():
        status = "Non dominé (front de Pareto)" if model_i["Modele"] in pareto_models else "Dominé"
        print(f"{model_i['Modele']} : {status}")

    # ========================================================
    # Figure 3D
    # ========================================================

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    for i, row in subset.iterrows():

        if row["Modele"] in pareto_models:
            ax.scatter(
                row["F1_macro"],
                row["Temps_us"],
                row["Flash_bytes"],
                s=120,
                color="red",
                label="Front de Pareto" if i == 0 else ""
            )
        else:
            ax.scatter(
                row["F1_macro"],
                row["Temps_us"],
                row["Flash_bytes"],
                s=90,
                color="blue",
                label="Modèle dominé" if i == 0 else ""
            )

        ax.text(
            row["F1_macro"],
            row["Temps_us"],
            row["Flash_bytes"],
            row["Modele"]
        )

    ax.set_xlabel("F1-score macro")
    ax.set_ylabel("Temps d'inférence (µs)")
    ax.set_zlabel("Mémoire Flash utilisée (bytes)")

    ax.set_title(
        "Analyse Pareto 3D du compromis performance / temps / Flash - "
        + dataset
    )

    ax.legend()
    plt.tight_layout()
    plt.show()