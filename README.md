# Classification de Drones RF — Tableau de Bord Interactif

Projet d'apprentissage automatique et deep learning pour la **classification de types et modes de vols de drones** à partir de signaux RF.

---

## 🚀 Démarrage Rapide

### Installation
```bash
pip install tkinter pillow numpy matplotlib seaborn xgboost scikit-learn joblib
```

### Lancer le Tableau de Bord
```bash
cd Scripts/
python dashboard.py
```

**C'est tout !** Le tableau de bord affichera tous les résultats interactivement.

---

## 📁 Structure du Projet

```
PPP/
├── Scripts/
│   ├── dashboard.py                    ← FICHIER PRINCIPAL À EXÉCUTER
│   ├── ML/                             # Modèles classiques
│   └── DL/                             # Deep Learning (CNN)
│
├── ml_trained_models_type_only/        # Résultats ML (4 classes)
├── ml_trained_models_mode_included/    # Résultats ML (10 classes)
├── tsne/                               # Visualisations t-SNE
└── results/                            # Résultats Deep Learning
```

---

## 📊 Résultats Clés

### Expérience A : Classification du Type (4 classes)
**Drone : Background | Bebop | AR_Drone | Phantom**

| Modèle | Précision |
|--------|-----------|
| **Forêt Aléatoire** | **89.3%** |
| XGBoost | 87.5% |
| KNN (k=52) | 85.2% |

### Expérience B : Type + Mode de Vol (10 classes)
**Challenge :** Les modes de vol intra-marque sont difficiles à distinguer avec les descripteurs scalaires

| Modèle | Précision |
|--------|-----------|
| **Forêt Aléatoire** | **62.1%** |
| XGBoost | 58.7% |
| KNN (k=5) | 54.3% |

### Deep Learning : Experts CNN par SNR

| Niveau SNR | Précision |
|-----------|-----------|
| 30dB (Signal propre) | 77.82% |
| **10dB (Meilleur)** | **79.67%** ⭐ |
| 0dB (Bruité) | 73.69% |
| -10dB (Très bruité) | 71.00% |

**Gain CNN :** +17.5 points vs meilleur ML sur 10 classes

---

## 📈 Fonctionnalités du Tableau de Bord

| Onglet | Contenu |
|--------|---------|
| **1. ML — Exp. A (Type seul)** | Comparaison RF/KNN/XGBoost, matrices de confusion, t-SNE |
| **2. ML — Exp. B (Type + Mode)** | Même analyse sur tâche plus difficile (10 classes) |
| **3. Comparaison ML (A vs B)** | Performance côte à côte, insights |
| **4. Deep Learning** | Modèles experts CNN par SNR + approche fusionnée |
| **5. ML vs DL** | Comparaison finale, courbes radar, métriques |

---

## 🧠 Points Clés Techniques

### Pourquoi ML échoue sur les modes
Les descripteurs scalaires (moyenne, variance, etc.) perdent la structure temps-fréquence. Les variations subtiles entre modes de vol d'un même drone sont invisibles.

### Pourquoi DL réussit
Les CNN sur spectrogrammes préservent la structure 2D → capturent les micro-différences temps-fréquence → robustes au bruit.

### Innovation : Modèles Experts par SNR
Au lieu d'un modèle unique, nous entraînons des experts spécialisés pour chaque niveau de bruit. Résultat : meilleure compréhension à chaque régime de bruit, identification du point optimal (10dB).

---

## 💡 Pour Votre Professeur

**À présenter :**
1. Ouvrir le tableau de bord
2. Naviguer vers l'onglet 5 (Comparaison ML vs DL)
3. Montrer les matrices de confusion et courbes

**Message clé :**
> *« Bien que les modèles ML classifient efficacement les types de drones, le deep learning sur spectrogrammes atteint une précision supérieure et une robustesse remarquable au bruit, avec un gain de +17.5 points de pourcentage sur 10 classes. »*

---

## 📝 Détails Techniques

- **ML :** Random Forest, KNN, XGBoost avec 8-12 descripteurs spectraux
- **DL :** CNN sur spectrogrammes avec détection d'activité signal (SAD)
- **Validation :** Split 80/20 stratifié, matrices de confusion, reports de classification
- **Robustesse :** Pondération SNR-aware, dropout (0.5), learning rate scheduler

---

## ✅ Statut

Projet complet et prêt pour évaluation.

**Créateur :** Islem Briki | **Dataset :** DroneRF | **Dernière mise à jour :** Juin 2026
