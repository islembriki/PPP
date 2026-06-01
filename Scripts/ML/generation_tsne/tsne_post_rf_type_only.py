import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. CONFIGURATION et CARTE DES COULEURS

# Définition des couleurs spécifiques pour chaque type de drone pour la visualisation
COLOR_MAP = {
    "Background": "#26a5d8", 
    "Bebop":      "#f039e1", 
    "AR_Drone":   "#c8d243", 
    "Phantom":    "#18C15E"  
}

# CHEMINS RELATIFS 
# Chemin vers le fichier CSV contenant les caractéristiques extraites
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"
# Chemin vers le modèle Random Forest sauvegardé
MODEL_PATH = "./PPP/ml_trained_models_type_only/rf_model_final.pkl"

# NOMBRE D'ÉCHANTILLONS : 30 000 permet une haute résolution sans saturer la RAM
N_SAMPLES = 30000 

def generate_rf_tsne():
    """Génère une visualisation t-SNE de l'espace statistique appris par le modèle"""
    print(f" Chargement des données pour {N_SAMPLES} échantillons...")
    start_time = time.time()
    
    # Vérification de la présence du modèle pkl
    if not os.path.exists(MODEL_PATH):
        print(f" Attention : Modèle pkl non trouvé à {MODEL_PATH}, mais continuation de la visualisation de l'espace des caractéristiques.")

    # Lecture du dataset et nettoyage des valeurs infinies ou manquantes
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    #  ESPACE DES CARACTÉRISTIQUES "APPRISES" 
    print(" Application de l'ingénierie des caractéristiques apprise (Log_Var, PAPR_Mean, Kurt_Skew)...")
    # Reconstruction des colonnes mathématiques utilisées par le Random Forest
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

    # Liste des 8 colonnes de caractéristiques finales
    feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Log_Var', 'PAPR_Mean', 'Kurt_Skew']
    
    #  SOUS-ÉCHANTILLONNAGE STRATIFIÉ 
    print(f" Échantillonnage de {N_SAMPLES} points...")
    # On sélectionne les points tout en respectant les proportions réelles de chaque drone (stratify)
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Label'], random_state=42)

    X = df_sample[feature_cols]
    y = df_sample['Label']
    
    #  MISE À L'ÉCHELLE (SCALING) 
    # Le t-SNE est basé sur les distances, on normalise donc les 8 caractéristiques pour un meilleur regroupement
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Mapping des identifiants numériques vers les noms de drones pour la légende
    id_to_name = {0: "Background", 1: "Bebop", 2: "AR_Drone", 3: "Phantom"}
    y_names = [id_to_name[val] for val in y]

    # CALCUL DU T-SNE 
    print(f" Calcul du t-SNE... (Cela prendra environ 10 minutes pour {N_SAMPLES} points)")
    # Réduction des 8 dimensions vers 2 dimensions (X et Y)
    # Perplexity=50 est adapté aux jeux de données plus volumineux
    tsne = TSNE(
        n_components=2, 
        perplexity=50,      
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    
    X_embedded = tsne.fit_transform(X_scaled)

    # CRÉATION DU GRAPHIQUE 
    plt.figure(figsize=(16, 11))
    sns.set_style("whitegrid") # Utilisation d'un fond avec grille
    
    # Création du nuage de points avec Seaborn
    sns.scatterplot(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        hue=y_names,
        palette=COLOR_MAP,
        s=45,               # Points plus petits pour gérer la haute densité
        alpha=0.6,          # Transparence pour voir les superpositions
        edgecolor=None      # Pas de bordure pour plus de clarté visuelle
    )

    # Calcul du temps total écoulé
    total_time = (time.time() - start_time) / 60
    
    # Configuration du titre et des étiquettes
    plt.title(f"Post-Training t-SNE: Espace Statistique Appris (N={N_SAMPLES})\nVisualisation de la séparation du modèle à 82.7%", 
              fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
    
    plt.xlabel("Dimension t-SNE 1", fontsize=12)
    plt.ylabel("Dimension t-SNE 2", fontsize=12)
    # Ajout de la légende stylisée
    plt.legend(title="Type de Drone", title_fontsize=14, fontsize=12, loc='best', shadow=True)
    
    # Ajout d'une boîte d'information avec les métriques en bas à gauche
    plt.text(0.01, 0.01, f"Précision Modèle: 82.71%\nNb Échantillons: {N_SAMPLES}\nTemps calcul: {total_time:.2f} min", 
             transform=plt.gca().transAxes, fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Optimisation de l'affichage
    plt.tight_layout()
    
    # Sauvegarde de l'image finale en haute résolution
    save_name = f"tsne_rf_post_training_{N_SAMPLES}.png"
    plt.savefig(save_name, dpi=300)
    print(f" Succès ! t-SNE sauvegardé sous : {save_name}")
    
    # Affichage du graphique
    plt.show()

# Point d'entrée principal du script
if __name__ == "__main__":
    generate_rf_tsne()