import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION & COLOR MAP
# ==========================================
COLOR_MAP = {
    "Background": "#26a5d8", 
    "Bebop":      "#f039e1", 
    "AR_Drone":   "#c8d243", 
    "Phantom":    "#18C15E"  
}

DATASET_PATH = r"C:\Users\user\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\9F909D7F23F9044D7744BC499E08F6F178990617\transfers\2026-22\FINAL_GLOBAL_DRONE_DATASET.csv"
SCALER_PATH  = "knn_scaler.pkl"
MODEL_PATH   = "knn_model.pkl"

N_SAMPLES = 5000

def generate_knn_tsne():
    print("📂 Loading data...")
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # --- Feature engineering ---
    print("🧪 Applying feature engineering...")
    df['Log_Var']   = np.log10(np.abs(df['Variance']) + 1e-9)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

    feature_cols = ['Mean','Variance','Kurtosis','Skewness','PAPR',
                    'Log_Var','PAPR_Mean','Kurt_Skew']

    # --- Subsampling ---
    print(f"✂️ Subsampling {N_SAMPLES} points...")
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES,
                                    stratify=df['Label'], random_state=42)

    X = df_sample[feature_cols]
    y_true = df_sample['Label']

    # --- Load scaler + KNN ---
    scaler = joblib.load(SCALER_PATH)
    knn_model = joblib.load(MODEL_PATH)

    X_scaled = scaler.transform(X)
    y_pred = knn_model.predict(X_scaled)

    id_to_name = {0:"Background",1:"Bebop",2:"AR_Drone",3:"Phantom"}
    y_names = [id_to_name[label] for label in y_pred]

    # --- t-SNE ---
    print("⏳ Calculating t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=500,
                learning_rate='auto', init='pca', random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)

    # --- Plotting ---
    plt.figure(figsize=(14,10))
    for name, color in COLOR_MAP.items():
        mask = [yn == name for yn in y_names]
        plt.scatter(X_embedded[mask,0], X_embedded[mask,1],
                    c=color, label=name, s=80, alpha=0.7,
                    edgecolor="w", linewidth=0.5)

    plt.title("Post-Training t-SNE : KNN Latent Space",
              fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Drone Signatures", fontsize=11, loc='best', shadow=True)

    plt.text(0.02,0.02,"Model: KNN\nFeatures: 8 (Standard + Engineered)",
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig("tsne_knn_post_training.png", dpi=300)
    print("✅ Success! Saved as tsne_knn_post_training.png")
    plt.show()

if __name__ == "__main__":
    generate_knn_tsne()