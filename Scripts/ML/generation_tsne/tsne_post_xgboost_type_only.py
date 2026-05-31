import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION ET COULEURS FIXES
# ==========================================
COLOR_MAP = {
    "Background": "#26a5d8", 
    "Bebop":      "#f039e1", 
    "AR_Drone":   "#c8d243", 
    "Phantom":    "#18C15E"  
}

DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"
MODEL_PATH = r"C:\Users\HP\Desktop\PPP\ml_trained_models_type_only\xgboost_type_model.pkl"

# 30,000 samples pour une haute résolution
N_SAMPLES = 30000 

def generate_xgboost_tsne():
    print(f"📂 Chargement du modèle et des données ({N_SAMPLES} points)...")
    start_time = time.time()
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERREUR : Modèle introuvable à {MODEL_PATH}")
        return

    # Chargement du modèle
    model = joblib.load(MODEL_PATH)
    
    # Chargement des données
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # --- ÉTAPE 1 : FEATURE ENGINEERING (Doit être identique à l'entraînement XGB) ---
    print("🧪 Reconstruction de l'espace de caractéristiques XGBoost...")
    eps = 1e-9
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + eps)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + eps)
    df['Signal_Power'] = (df['Mean']**2) + df['Variance']
    df['Var_PAPR_Ratio'] = df['Variance'] / (df['PAPR'] + eps)

    # Liste des colonnes attendues par ton modèle XGBoost
    feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 
                    'Log_Var', 'PAPR_Mean', 'Kurt_Skew', 'Signal_Power', 'Var_PAPR_Ratio']
    
    # --- ÉTAPE 2 : ÉCHANTILLONNAGE STRATIFIÉ ---
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Label'], random_state=42)

    X_eval = df_sample[feature_cols]
    y_true = df_sample['Label']
    
    # Mapping des noms
    id_to_name = {0: "Background", 1: "Bebop", 2: "AR_Drone", 3: "Phantom"}
    y_names = [id_to_name[val] for val in y_true]

    # --- ÉTAPE 3 : EXTRACTION DE LA LOGIQUE DU MODÈLE (Probabilités) ---
    print("🧠 Calcul des probabilités de décision (Decision Space)...")
    # On utilise predict_proba pour voir comment le modèle sépare les classes
    model_probs = model.predict_proba(X_eval)

    # --- ÉTAPE 4 : CALCUL T-SNE ---
    print(f"⏳ Calcul du t-SNE... (Prévu ~10-12 minutes)")
    tsne = TSNE(
        n_components=2, 
        perplexity=50, 
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    
    X_embedded = tsne.fit_transform(model_probs)

    # --- ÉTAPE 5 : VISUALISATION ---
    plt.figure(figsize=(16, 11))
    sns.set_style("white") # Style propre sans grille pour faire ressortir les clusters
    
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
    
    # Titre Pro
    plt.title("POST-TRAINING t-SNE : XGBOOST DECISION SPACE (TYPE ONLY)\n"
              "Visualisation de la séparation des signatures après apprentissage Gradient Boosting", 
              fontsize=20, fontweight='bold', pad=30, loc='left', color='#1a1a1a')
    
    # Barre de style
    plt.gca().add_artist(plt.Line2D((0, 0.2), (1.06, 1.08), transform=plt.gca().transAxes, color='#26a5d8', linewidth=5))

    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Drone Type", title_fontsize=13, loc='upper right', frameon=True, shadow=True)
    
    # Box de stats
    plt.text(0.98, 0.02, 
             f"Model: XGBoost\nAccuracy: 79.86%\nSamples: {N_SAMPLES}\nTime: {total_time:.2f} min", 
             transform=plt.gca().transAxes, ha='right', fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#26a5d8', boxstyle='round,pad=1'))

    plt.axis('off')
    plt.tight_layout()
    
    # Sauvegarde
    save_name = f"tsne_xgboost_post_training_{N_SAMPLES}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ Terminé ! Image sauvegardée : {save_name}")
    plt.show()

if __name__ == "__main__":
    generate_xgboost_tsne()