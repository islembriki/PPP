import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# 1. CONFIGURATION & CARTE DES COULEURS

# Définition des couleurs spécifiques pour chaque type de drone
COLOR_MAP = {
    "Background": "#26a5d8", 
    "Bebop":      "#f039e1", 
    "AR_Drone":   "#c8d243", 
    "Phantom":    "#18C15E"  
}

# --- CHEMINS RELATIFS ---
# Chemin vers le fichier CSV global
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"
# Chemins vers le scaler et le modèle KNN dans le dossier des modèles de type seul
SCALER_PATH  = "./PPP/ml_trained_models_type_only/knn_scaler.pkl"
MODEL_PATH   = "./PPP/ml_trained_models_type_only/knn_model.pkl"

# Nombre de points à afficher pour la visualisation
N_SAMPLES = 5000

def generate_knn_tsne():
    """Génère une visualisation t-SNE basée sur l'espace latent du modèle KNN"""
    print("Chargement des données...")
    # Lecture du fichier CSV
    df = pd.read_csv(DATASET_PATH)
    # Nettoyage : suppression des valeurs infinies et des lignes vides
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Ingénierie des caractéristiques (Feature engineering) 
    print(" Application de l'ingénierie des caractéristiques...")
    # Recréation des colonnes mathématiques utilisées lors de l'entraînement
    df['Log_Var']   = np.log10(np.abs(df['Variance']) + 1e-9)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

    # Liste des 8 caractéristiques finales
    feature_cols = ['Mean','Variance','Kurtosis','Skewness','PAPR',
                    'Log_Var','PAPR_Mean','Kurt_Skew']

    #  Sous-échantillonnage 
    print(f" Échantillonnage de {N_SAMPLES} points...")
    # On sélectionne un échantillon stratifié pour représenter équitablement chaque drone
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES,
                                    stratify=df['Label'], random_state=42)

    X = df_sample[feature_cols]
    y_true = df_sample['Label']

    #  Chargement du scaler et du KNN 
    # On charge les fichiers binaires sauvegardés précédemment
    scaler = joblib.load(SCALER_PATH)
    knn_model = joblib.load(MODEL_PATH)

    # Normalisation des données de l'échantillon
    X_scaled = scaler.transform(X)
    # Prédiction des étiquettes par le modèle KNN
    y_pred = knn_model.predict(X_scaled)

    # Mapping des IDs numériques vers les noms textuels
    id_to_name = {0:"Background", 1:"Bebop", 2:"AR_Drone", 3:"Phantom"}
    y_names = [id_to_name[label] for label in y_pred]

    #  Calcul du t-SNE 
    print("Calcul du t-SNE (Réduction de dimension)...")
    # Réduction des 8 caractéristiques vers 2 dimensions (X et Y)
    tsne = TSNE(n_components=2, perplexity=40, max_iter=500,
                learning_rate='auto', init='pca', random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)

    #  Création du graphique 
    plt.figure(figsize=(14,10))
    # On trace chaque classe séparément pour appliquer la COLOR_MAP
    for name, color in COLOR_MAP.items():
        # Masque pour filtrer les points appartenant à la classe actuelle
        mask = [yn == name for yn in y_names]
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                    c=color, label=name, s=80, alpha=0.7,
                    edgecolor="w", linewidth=0.5)

    # Configuration esthétique du graphique
    plt.title("Post-Training t-SNE : Espace Latent du KNN (Type Only)",
              fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    plt.xlabel("Dimension t-SNE 1", fontsize=12)
    plt.ylabel("Dimension t-SNE 2", fontsize=12)
    # Ajout de la légende avec ombre
    plt.legend(title="Signatures Drones", fontsize=11, loc='best', shadow=True)

    # Boîte d'information sur les paramètres en bas à gauche
    plt.text(0.02, 0.02, "Modèle: KNN\nCaractéristiques: 8 (Standards + Ingénierie)",
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Optimisation de la mise en page
    plt.tight_layout()
    # Sauvegarde de l'image en haute qualité
    plt.savefig("tsne_knn_post_training.png", dpi=300)
    print("Succès ! Image sauvegardée sous : tsne_knn_post_training.png")
    # Affichage du graphique final
    plt.show()

# Point d'entrée principal du script
if __name__ == "__main__":
    generate_knn_tsne()