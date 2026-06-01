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

# 1. CONFIGURATION & CHEMINS RELATIFS

# Chemin vers le modèle Random Forest déjà entraîné
MODEL_PATH = "./PPP/ml_trained_models_mode_included/rf_mode.pkl"
# Chemin vers le fichier CSV contenant les données globales
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"

# Nombre d'échantillons à traiter 
N_SAMPLES = 20000 

def generate_post_training_model_tsne():
    """Génère une carte t-SNE basée sur la logique interne (probabilités) du Random Forest"""
    print(" Chargement du modèle entraîné et du dataset...")
    start_time = time.time()
    
    # Vérification de l'existence du fichier modèle avant chargement
    if not os.path.exists(MODEL_PATH):
        print(f" ERREUR: Fichier pkl introuvable à {MODEL_PATH}")
        return

    # Chargement du modèle binaire avec joblib
    model = joblib.load(MODEL_PATH)
    
    # Chargement du dataset et nettoyage des données infinies ou manquantes
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    #  1. INGÉNIERIE DES CARACTÉRISTIQUES (Identique à l'entraînement) 
    print(" Reconstruction des caractéristiques comportementales...")
    # Recréation des colonnes mathématiques pour que le modèle puisse analyser les données
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)
    df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + 1e-9)
    df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
    df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])
    df['Mean_Cube'] = df['Mean']**3

    # Liste des colonnes de caractéristiques attendues par le Random Forest
    feature_cols = [
        'Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 
        'Log_Var', 'PAPR_Mean', 'Kurt_Skew', 
        'PAPR_Var_Ratio', 'Signal_Energy', 'Shape_Factor', 'Mean_Cube'
    ]
    
    # Création de l'étiquette combinée Type_Mode 
    df['Target'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)

    #  2. ÉCHANTILLONNAGE 
    print(f" Sélection de {N_SAMPLES} points au hasard (stratifié)...")
    # On sélectionne les points tout en respectant les proportions de chaque classe
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Target'], random_state=42)

    X_eval = df_sample[feature_cols]
    y_true = df_sample['Target']

    # 3. EXTRACTION DES "PENSÉES" DU MODÈLE (Espace de probabilité) 
    print("Extraction de la confiance du modèle (predict_proba)...")
    # Au lieu d'utiliser les données brutes, on utilise les probabilités de sortie (10 dimensions)
    # Cela permet de visualiser comment le modèle regroupe les classes selon sa propre logique interne.
    model_embeddings = model.predict_proba(X_eval)

    #  4. CALCUL DU T-SNE 
    print(" Calcul du t-SNE sur la logique du modèle...")
    # t-SNE réduit l'espace des probabilités (10D) vers un plan 2D pour l'affichage
    tsne = TSNE(
        n_components=2, 
        perplexity=50, 
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    
    X_embedded = tsne.fit_transform(model_embeddings)

    #  5. CRÉATION DU GRAPHIQUE 
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid") # Style avec grille pour la lisibilité
    
    # Récupération de la liste des classes triées
    unique_targets = sorted(y_true.unique())
    
    # Création du nuage de points avec Seaborn
    # On utilise la palette 'turbo' pour avoir un contraste maximal entre les 10 classes
    scatter = sns.scatterplot(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        hue=y_true,
        hue_order=unique_targets,
        palette="turbo", 
        s=60, 
        alpha=0.7, 
        edgecolor="w", 
        linewidth=0.3
    )

    total_time = (time.time() - start_time) / 60
    
    # Configuration des titres et légendes
    plt.title(f"Post-Training t-SNE: Espace Logique du Random Forest (N={N_SAMPLES})\n"
              f"Visualisation de la façon dont le modèle regroupe la confiance Type+Mode", 
              fontsize=20, fontweight='bold', pad=20, color='#1a1a1a')
    
    plt.xlabel("Dimension de décision t-SNE 1", fontsize=12)
    plt.ylabel("Dimension de décision t-SNE 2", fontsize=12)
    # Placement de la légende à l'extérieur à droite
    plt.legend(title="Classe (Label_Mode)", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Boîte d'information récapitulative
    plt.text(0.01, 0.01, 
             f"Précision du Modèle: 70.08%\n"
             f"Base: Espace predict_proba()\n"
             f"Temps de calcul: {total_time:.2f} min", 
             transform=plt.gca().transAxes, fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Optimisation de la mise en page pour éviter les chevauchements
    plt.tight_layout()
    
    # Sauvegarde de l'image finale
    save_name = "tsne_rf_MODEL_LOGIC_MODES.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f" Succès ! Carte de logique post-entraînement sauvegardée sous : {save_name}")
    plt.show()

# Point d'entrée principal du script
if __name__ == "__main__":
    generate_post_training_model_tsne()