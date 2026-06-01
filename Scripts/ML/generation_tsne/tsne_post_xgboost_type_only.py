import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# 1. CONFIGURATION ET COULEURS FIXES
# Définition des couleurs spécifiques pour chaque drone pour une identité visuelle cohérente
COLOR_MAP = {
    "Background": "#26a5d8", 
    "Bebop":      "#f039e1", 
    "AR_Drone":   "#c8d243", 
    "Phantom":    "#18C15E"  
}

#  CHEMINS RELATIFS 
# Chemin vers le fichier CSV global des données
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"
# Chemin vers le modèle XGBoost sauvegardé lors de l'entraînement
MODEL_PATH = "./PPP/ml_trained_models_type_only/xgboost_type_model.pkl"

# 30,000 samples pour obtenir une haute résolution de l'espace de décision
N_SAMPLES = 30000 

def generate_xgboost_tsne():
    """Génère une carte t-SNE basée sur les probabilités de décision du modèle XGBoost"""
    print(f" Chargement du modèle et des données ({N_SAMPLES} points)...")
    start_time = time.time()
    
    # Vérification que le modèle existe bien à l'endroit indiqué
    if not os.path.exists(MODEL_PATH):
        print(f" ERREUR : Modèle introuvable à {MODEL_PATH}")
        return

    # Chargement du modèle XGBoost binaire (.pkl)
    model = joblib.load(MODEL_PATH)
    
    # Chargement du dataset complet et nettoyage (NaN et Infinis)
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    #  ÉTAPE 1 : FEATURE ENGINEERING (Doit être identique à l'entraînement XGB) 
    print(" Reconstruction de l'espace de caractéristiques XGBoost...")
    # Constante de sécurité pour les calculs logarithmiques
    eps = 1e-9
    # Recréation des colonnes mathématiques "Hardware Signatures"
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + eps)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + eps)
    df['Signal_Power'] = (df['Mean']**2) + df['Variance']
    df['Var_PAPR_Ratio'] = df['Variance'] / (df['PAPR'] + eps)

    # Liste ordonnée des colonnes attendues par le modèle XGBoost
    feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 
                    'Log_Var', 'PAPR_Mean', 'Kurt_Skew', 'Signal_Power', 'Var_PAPR_Ratio']
    
    #  ÉTAPE 2 : ÉCHANTILLONNAGE STRATIFIÉ 
    # On sélectionne N points en conservant les proportions de chaque drone (stratify)
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Label'], random_state=42)

    X_eval = df_sample[feature_cols]
    y_true = df_sample['Label']
    
    # Traduction des IDs numériques (0,1,2,3) en noms lisibles
    id_to_name = {0: "Background", 1: "Bebop", 2: "AR_Drone", 3: "Phantom"}
    y_names = [id_to_name[val] for val in y_true]

    #  ÉTAPE 3 : EXTRACTION DE LA LOGIQUE DU MODÈLE (Probabilités) 
    print(" Calcul des probabilités de décision (Decision Space)...")
    # On utilise predict_proba : t-SNE va travailler sur la confiance du modèle (vecteur de probabilités)
    # plutôt que sur les données brutes, ce qui montre comment le modèle "voit" les clusters
    model_probs = model.predict_proba(X_eval)

    # ÉTAPE 4 : CALCUL T-SNE 
    print(f" Calcul du t-SNE... (Prévu ~10-12 minutes)")
    # Réduction de l'espace de probabilité multiclasse vers un plan 2D
    tsne = TSNE(
        n_components=2, 
        perplexity=50, 
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    
    X_embedded = tsne.fit_transform(model_probs)

    #  ÉTAPE 5 : VISUALISATION 
    plt.figure(figsize=(16, 11))
    sns.set_style("white") # Style pur sans grille pour mettre en valeur les amas (clusters)
    
    # Création du nuage de points
    sns.scatterplot(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        hue=y_names,
        hue_order=["Background", "Bebop", "AR_Drone", "Phantom"],
        palette=COLOR_MAP,
        s=50, 
        alpha=0.6, 
        edgecolor="w", 
        linewidth=0.2
    )

    total_time = (time.time() - start_time) / 60
    
    # Titre 
    plt.title("POST-TRAINING t-SNE : XGBOOST DECISION SPACE (TYPE ONLY)\n"
              "Visualisation de la séparation des signatures après apprentissage Gradient Boosting", 
              fontsize=20, fontweight='bold', pad=30, loc='left', color='#1a1a1a')
    
    # Ajout d'une barre de soulignement stylisée sous le titre
    plt.gca().add_artist(plt.Line2D((0, 0.2), (1.06, 1.08), transform=plt.gca().transAxes, color='#26a5d8', linewidth=5))

    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    # Légende avec ombre pour la profondeur
    plt.legend(title="Drone Type", title_fontsize=13, loc='upper right', frameon=True, shadow=True)
    
    # Boîte récapitulative des statistiques de l'expérience
    plt.text(0.98, 0.02, 
             f"Modèle: XGBoost\nPrécision: 79.86%\nÉchantillons: {N_SAMPLES}\nTemps: {total_time:.2f} min", 
             transform=plt.gca().transAxes, ha='right', fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#26a5d8', boxstyle='round,pad=1'))

    # Désactivation des axes car les valeurs X/Y de t-SNE n'ont pas de signification physique
    plt.axis('off')
    plt.tight_layout()
    
    # Sauvegarde de l'image finale en haute résolution
    save_name = f"tsne_xgboost_post_training_{N_SAMPLES}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Terminé ! Image sauvegardée : {save_name}")
    plt.show()

# Exécution du script
if __name__ == "__main__":
    generate_xgboost_tsne()