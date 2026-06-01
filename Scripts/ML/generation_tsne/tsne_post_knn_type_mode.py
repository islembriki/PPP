import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# 1. CONFIGURATION DES CHEMINS RELATIFS

# Dossier contenant le modèle k-NN et le Scaler déjà entraînés
MODEL_DIR = "./PPP/ml_trained_models_mode_included"
# Chemin vers le fichier de données global
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"

# Construction des chemins complets vers les fichiers .pkl
MODEL_PATH = os.path.join(MODEL_DIR, "knn_mode.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "knn_scaler_mode.pkl")

# Nombre de points à afficher (15 000 permet une bonne visualisation de la densité)
N_SAMPLES = 15000 

def generate_true_knn_tsne():
    """Génère une carte t-SNE montrant comment le modèle k-NN 'voit' les données"""
    print(" Chargement du modèle et de la logique k-NN...")
    start_time = time.time()
    
    # Chargement du modèle k-NN et du scaler (normalisation) sauvegardés
    knn = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Lecture du dataset et nettoyage (suppression des valeurs infinies et NaN)
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 1. RE-CALCUL DES CARACTÉRISTIQUES (Les 11 features utilisées à l'entraînement) 
    print(" Reconstruction des caractéristiques mathématiques comportementales...")
    eps = 1e-9
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + eps)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + eps)
    df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + eps)
    df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
    df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])

    # Récupération de la liste exacte des colonnes attendues par le scaler
    feature_cols = list(scaler.feature_names_in_)
    # Création de l'étiquette combinée Drone_Mode (ex: Bebop_M1)
    df['Target_Mode'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)

    #  2. ÉCHANTILLONNAGE STRATIFIÉ 
    print(f" Sélection de {N_SAMPLES} points au hasard...")
    # On prend un échantillon tout en gardant les mêmes proportions de chaque classe
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Target_Mode'], random_state=42)

    # 3. VISION DU MODÈLE (ESPACE LATENT) 
    print(" Normalisation des caractéristiques (tel que le k-NN analyse le monde)...")
    X_raw = df_sample[feature_cols]
    # Très important : le t-SNE doit être fait sur les données normalisées pour être fidèle au k-NN
    X_scaled = scaler.transform(X_raw)

    #  4. CALCUL DU t-SNE SUR LES DONNÉES NORMALISÉES 
    print(" Calcul du t-SNE (Cartographie de l'espace de décision)...")
    tsne = TSNE(
        n_components=2,      # Réduction à 2 dimensions pour l'affichage
        perplexity=40,       # Paramètre de densité des clusters
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    # Transformation des 11 dimensions vers seulement 2 coordonnées (X, Y)
    X_embedded = tsne.fit_transform(X_scaled)

    #  5. CRÉATION DU GRAPHIQUE 
    plt.figure(figsize=(18, 12))
    sns.set_style("white") # Fond blanc pour plus de clarté
    
    unique_modes = sorted(df_sample['Target_Mode'].unique())
    
    # Création du nuage de points
    # Alpha=0.5 permet de voir la transparence là où les classes se chevauchent (confusion)
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1], 
        hue=df_sample['Target_Mode'], 
        hue_order=unique_modes,
        palette="turbo", 
        s=45, alpha=0.5, edgecolor=None 
    )

    total_time = (time.time() - start_time) / 60
    
    # Configuration des titres et labels
    plt.title("POST-TRAINING t-SNE : LA RÉALITÉ DE LA CONFUSION DU k-NN\n"
              "Visualisation du chevauchement des signatures dans l'espace des caractéristiques", 
              loc='left', fontsize=22, fontweight='bold', color='#2c3e50', pad=35)
    
    # Ajout d'une barre de style rouge en haut du titre
    plt.gca().add_artist(plt.Line2D((0, 0.25), (1.08, 1.08), transform=plt.gca().transAxes, color='#db2b39', linewidth=6))
    
    # Placement de la légende à l'extérieur du graphique
    plt.legend(title="Label_Mode Réel", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

    # Boîte d'information sur les résultats
    info_box = (
        f"Approche: k-NN Traditionnel (Type+Mode)\n"
        f"Précision: 56.29%\n"
        f"Problème: Fort chevauchement statistique\n"
        f"Points affichés: {N_SAMPLES}"
    )
    plt.text(0.98, 0.02, info_box, transform=plt.gca().transAxes, ha='right', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='#fdfdfd', alpha=0.9, edgecolor='#db2b39', boxstyle='round,pad=1'))

    # Suppression des axes (non nécessaires pour une visualisation de clusters)
    plt.axis('off')
    plt.tight_layout()
    
    # Sauvegarde de l'image en haute résolution (300 DPI)
    save_path = "tsne_knn_MODES_THE_REALITY.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Succès ! Carte de logique sauvegardée sous : {save_path}")
    plt.show()

# Point d'entrée du script
if __name__ == "__main__":
    generate_true_knn_tsne()