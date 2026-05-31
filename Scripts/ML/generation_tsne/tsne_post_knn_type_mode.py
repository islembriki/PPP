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
# Use the folder where your k-NN model and Scaler are stored
MODEL_DIR = r"C:\Users\HP\Desktop\PPP\ml_trained_models_mode_included"
DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"

MODEL_PATH = os.path.join(MODEL_DIR, "knn_mode.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "knn_scaler_mode.pkl")

# We take 15,000 points to show the density clearly
N_SAMPLES = 15000 

def generate_true_knn_tsne():
    print("📂 Loading data and k-NN logic...")
    start_time = time.time()
    
    # Load model and scaler
    knn = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # --- 1. RE-ENGINEER FEATURES (The 11 features from your training) ---
    print("🧪 Reconstructing behavioral math features...")
    eps = 1e-9
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + eps)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + eps)
    df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + eps)
    df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
    df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])

    # Get column names used in scaler
    feature_cols = list(scaler.feature_names_in_)
    df['Target_Mode'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)

    # --- 2. STRATIFIED SAMPLING ---
    print(f"✂️ Sampling {N_SAMPLES} points...")
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Target_Mode'], random_state=42)

    # --- 3. THE "MODEL VIEW" (POST-TRAINING LATENT SPACE) ---
    print("⚖️ Scaling features (this is how k-NN sees the world)...")
    X_raw = df_sample[feature_cols]
    X_scaled = scaler.transform(X_raw)

    # --- 4. T-SNE ON SCALED FEATURES ---
    # We do NOT use predict_proba. We use the scaled features to show the true overlap.
    print("⏳ Calculating t-SNE (The True Decision Space)...")
    tsne = TSNE(
        n_components=2, 
        perplexity=40, # Standard perplexity to show clusters vs. noise
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    X_embedded = tsne.fit_transform(X_scaled)

    # --- 5. PLOTTING ---
    plt.figure(figsize=(18, 12))
    sns.set_style("white") 
    
    unique_modes = sorted(df_sample['Target_Mode'].unique())
    
    # We use a lower alpha (0.5) so you can see where colors "bleed" together
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1], 
        hue=df_sample['Target_Mode'], 
        hue_order=unique_modes,
        palette="turbo", 
        s=45, alpha=0.5, edgecolor=None # Remove borders to see density better
    )

    total_time = (time.time() - start_time) / 60
    
    plt.title("POST-TRAINING t-SNE : THE REALITY OF k-NN CONFUSION\n"
              "Visualizing overlapping signatures in the Feature Space", 
              loc='left', fontsize=22, fontweight='bold', color='#2c3e50', pad=35)
    
    # Header bar
    plt.gca().add_artist(plt.Line2D((0, 0.25), (1.08, 1.08), transform=plt.gca().transAxes, color='#db2b39', linewidth=6))
    
    plt.legend(title="True Label_Mode", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

    # ACCURACY ALERT BOX
    info_box = (
        f"Approach: Traditional k-NN (Type+Mode)\n"
        f"Accuracy: 56.29%\n"
        f"Problem: High Statistical Overlap\n"
        f"Points: {N_SAMPLES}"
    )
    plt.text(0.98, 0.02, info_box, transform=plt.gca().transAxes, ha='right', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='#fdfdfd', alpha=0.9, edgecolor='#db2b39', boxstyle='round,pad=1'))

    plt.axis('off')
    plt.tight_layout()
    
    save_path = "tsne_knn_MODES_THE_REALITY.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! True logic map saved as: {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_true_knn_tsne()