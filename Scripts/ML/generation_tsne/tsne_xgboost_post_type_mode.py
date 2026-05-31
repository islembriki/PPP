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
# 1. CONFIGURATION
# ==========================================
DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"
MODEL_PATH   = r"C:\Users\HP\Desktop\PPP\ml_trained_models_mode_included\xgboost_mode.pkl"
# Le LabelEncoder est nécessaire pour retrouver les noms des classes (0_M0, etc.)
ENCODER_PATH = r"C:\Users\HP\Desktop\PPP\ml_trained_models_mode_included\label_encoder.pkl"

N_SAMPLES = 30000 

def generate_xgboost_mode_tsne():
    print("📂 Chargement du modèle XGBoost et du Dataset...")
    start_time = time.time()
    
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        print("❌ Erreur : Modèle ou LabelEncoder introuvable.")
        return

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # --- ÉTAPE 1 : FEATURE ENGINEERING (Exactement comme à l'entraînement) ---
    print("🧪 Reconstruction des caractéristiques comportementales...")
    eps = 1e-9
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + eps)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + eps)
    df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + eps)
    df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
    df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])
    df['Mean_Cube'] = df['Mean']**3

    feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 
                    'Log_Var', 'PAPR_Mean', 'Kurt_Skew', 
                    'PAPR_Var_Ratio', 'Signal_Energy', 'Shape_Factor', 'Mean_Cube']
    
    # Création de la cible combinée pour la légende
    df['Target_Mode'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)

    # --- ÉTAPE 2 : ÉCHANTILLONNAGE STRATIFIÉ (10 CLASSES) ---
    print(f"✂️ Subsampling {N_SAMPLES} points...")
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Target_Mode'], random_state=42)

    X_eval = df_sample[feature_cols]
    y_true = df_sample['Target_Mode']

    # --- ÉTAPE 3 : EXTRACTION DE LA LOGIQUE DU MODÈLE ---
    print("🧠 Analyse de l'espace de décision (predict_proba)...")
    # On récupère les probabilités pour les 10 classes
    # C'est ici que l'on voit comment le modèle "hésite" entre les modes
    model_probs = model.predict_proba(X_eval)

    # --- ÉTAPE 4 : CALCUL T-SNE ---
    print("⏳ Calcul du t-SNE (Decision Space - 10 Classes)...")
    tsne = TSNE(
        n_components=2, 
        perplexity=50, 
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    X_embedded = tsne.fit_transform(model_probs)

    # --- ÉTAPE 5 : VISUALISATION ---
    plt.figure(figsize=(18, 12))
    sns.set_style("white")
    
    unique_modes = sorted(y_true.unique())
    
    # Utilisation d'une palette à fort contraste pour 10 classes
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1], 
        hue=y_true, 
        hue_order=unique_modes,
        palette="turbo", 
        s=55, alpha=0.7, edgecolor="w", linewidth=0.2
    )

    total_time = (time.time() - start_time) / 60
    
    # Titre Style Expert
    plt.title("POST-TRAINING t-SNE : XGBOOST LATENT LOGIC (TYPE + MODE)\n"
              "Distribution de la confiance du modèle sur 10 classes comportementales", 
              loc='left', fontsize=22, fontweight='bold', color='#1a1a1a', pad=35)
    
    # Barre décorative style DL
    plt.gca().add_artist(plt.Line2D((0, 0.25), (1.08, 1.08), transform=plt.gca().transAxes, color='#db2b39', linewidth=6))

    plt.xlabel("t-SNE Decision Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Decision Dimension 2", fontsize=12)
    plt.legend(title="Class (Label_Mode)", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    
    # Box d'information
    info_box = (
        f"Approach: XGBoost (Gradient Boosting)\n"
        f"Goal: Classification (Type + Mode)\n"
        f"Accuracy: {model.get_booster().attr('best_score') if hasattr(model, 'get_booster') else 'Done'} \n"
        f"Points: {N_SAMPLES}\n"
        f"Time: {total_time:.2f} min"
    )
    plt.text(0.98, 0.02, info_box, transform=plt.gca().transAxes, ha='right', fontsize=11, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#db2b39', boxstyle='round,pad=1'))

    plt.axis('off')
    plt.tight_layout()
    
    # Sauvegarde
    save_path = f"tsne_xgboost_MODES_post_training.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Succès ! Graphique des 10 modes généré : {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_xgboost_mode_tsne()