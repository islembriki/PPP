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

# Chemin relatif vers le fichier de données global
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"
# Chemin relatif vers le modèle XGBoost (Type + Mode) sauvegardé
MODEL_PATH   = "./PPP/ml_trained_models_mode_included/xgboost_mode.pkl"
# Le LabelEncoder est indispensable pour décoder les IDs numériques des classes (ex: 0 -> Bebop_M1)
ENCODER_PATH = "./PPP/ml_trained_models_mode_included/label_encoder.pkl"

# Nombre d'échantillons pour la visualisation (30 000 permet une excellente précision des clusters)
N_SAMPLES = 30000 

def generate_xgboost_mode_tsne():
    """Génère une carte t-SNE basée sur la confiance de décision du modèle XGBoost"""
    print(" Chargement du modèle XGBoost et du Dataset...")
    start_time = time.time()
    
    # Vérification de l'existence du modèle et de l'encodeur avant chargement
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        print(" Erreur : Modèle ou LabelEncoder introuvable.")
        return

    # Chargement binaire du modèle et de son encodeur de labels associé
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    
    # Chargement du dataset et nettoyage (suppression des valeurs infinies et NaN)
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # ÉTAPE 1 : FEATURE ENGINEERING 
    print(" Reconstruction des caractéristiques comportementales...")
    # Constante pour éviter les erreurs mathématiques (division par zéro, log de zéro)
    eps = 1e-9
    # Recréation des caractéristiques complexes calculées lors de la phase d'entraînement
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + eps)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + eps)
    df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + eps)
    df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
    df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])
    df['Mean_Cube'] = df['Mean']**3

    # Liste ordonnée des 12 colonnes de caractéristiques utilisées par XGBoost
    feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 
                    'Log_Var', 'PAPR_Mean', 'Kurt_Skew', 
                    'PAPR_Var_Ratio', 'Signal_Energy', 'Shape_Factor', 'Mean_Cube']
    
    # Création de l'étiquette combinée Drone_Mode (ex: "1_M2") pour la légende du graphique
    df['Target_Mode'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)

    #  ÉTAPE 2 : ÉCHANTILLONNAGE STRATIFIÉ (10 CLASSES) 
    print(f" Échantillonnage de {N_SAMPLES} points...")
    # On sélectionne N points tout en respectant l'équilibre des 10 classes
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Target_Mode'], random_state=42)

    X_eval = df_sample[feature_cols]
    y_true = df_sample['Target_Mode']

    #  ÉTAPE 3 : EXTRACTION DE LA LOGIQUE DU MODÈLE 
    print(" Analyse de l'espace de décision (predict_proba)...")
    # On extrait les probabilités pour les 10 classes (chaque point devient un vecteur de 10 probabilités)
    # Le t-SNE travaillera sur ces probabilités pour montrer comment le modèle sépare les modes
    model_probs = model.predict_proba(X_eval)

    #  ÉTAPE 4 : CALCUL T-SNE 
    print(" Calcul du t-SNE (Decision Space - 10 Classes)...")
    # Réduction de l'espace des probabilités (10D) vers un plan 2D pour visualisation
    tsne = TSNE(
        n_components=2, 
        perplexity=50,       # Densité des clusters
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    X_embedded = tsne.fit_transform(model_probs)

    #  ÉTAPE 5 : VISUALISATION 
    plt.figure(figsize=(18, 12))
    sns.set_style("white") # Style pur sans grille
    
    # Liste triée des classes pour une légende ordonnée
    unique_modes = sorted(y_true.unique())
    
    # Affichage du nuage de points
    # Palette 'turbo' pour assurer un contraste maximal entre les 10 classes
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1], 
        hue=y_true, 
        hue_order=unique_modes,
        palette="turbo", 
        s=55, alpha=0.7, edgecolor="w", linewidth=0.2
    )

    # Calcul du temps total écoulé
    total_time = (time.time() - start_time) / 60
    
    # Titre 
    plt.title("POST-TRAINING t-SNE : LOGIQUE LATENTE XGBOOST (TYPE + MODE)\n"
              "Distribution de la confiance du modèle sur 10 classes comportementales", 
              loc='left', fontsize=22, fontweight='bold', color='#1a1a1a', pad=35)
    
    # Ajout d'une barre décorative rouge sous le titre
    plt.gca().add_artist(plt.Line2D((0, 0.25), (1.08, 1.08), transform=plt.gca().transAxes, color='#db2b39', linewidth=6))

    # Configuration des labels des axes et de la légende
    plt.xlabel("Dimension de décision t-SNE 1", fontsize=12)
    plt.ylabel("Dimension de décision t-SNE 2", fontsize=12)
    plt.legend(title="Classe (Label_Mode)", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    
    # Boîte d'information récapitulative en bas à droite
    info_box = (
        f"Modèle: XGBoost (Gradient Boosting)\n"
        f"Cible: Classification (Type + Mode)\n"
        f"État: Entraînement Terminé\n"
        f"Points affichés: {N_SAMPLES}\n"
        f"Temps de calcul: {total_time:.2f} min"
    )
    plt.text(0.98, 0.02, info_box, transform=plt.gca().transAxes, ha='right', fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#db2b39', boxstyle='round,pad=1'))

    # Suppression des axes numériques (les coordonnées t-SNE sont abstraites)
    plt.axis('off')
    plt.tight_layout()
    
    # Sauvegarde de l'image finale
    save_path = "tsne_xgboost_MODES_post_training.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Succès ! Graphique des 10 modes généré : {save_path}")
    # Affichage du graphique
    plt.show()

# Point d'entrée principal du script
if __name__ == "__main__":
    generate_xgboost_mode_tsne()
    