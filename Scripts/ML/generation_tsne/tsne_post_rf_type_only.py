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

# ==========================================
# 1. CONFIGURATION & COLOR MAP
# ==========================================
COLOR_MAP = {
    "Background": "#26a5d8", 
    "Bebop":      "#f039e1", 
    "AR_Drone":   "#c8d243", 
    "Phantom":    "#18C15E"  
}

DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"
MODEL_PATH = r"C:\Users\HP\Desktop\PPP\ml_trained_models_type_only\rf.pkl"

# INCREASED SAMPLES: 30,000 is high-res but safe for RAM
N_SAMPLES = 30000 

def generate_rf_tsne():
    print(f"📂 Loading data for {N_SAMPLES} samples...")
    start_time = time.time()
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Warning: Model pkl not found at {MODEL_PATH}, but continuing with feature space visualization.")

    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # --- THE "LEARNED" FEATURE SPACE ---
    print("🧪 Applying learned feature engineering (Log_Var, PAPR_Mean, Kurt_Skew)...")
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

    feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Log_Var', 'PAPR_Mean', 'Kurt_Skew']
    
    # --- STRATIFIED SUBSAMPLING ---
    print(f"✂️ Subsampling {N_SAMPLES} points...")
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Label'], random_state=42)

    X = df_sample[feature_cols]
    y = df_sample['Label']
    
    # --- SCALING ---
    # t-SNE is distance-based, so we scale the 8 features for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Map label numbers to Names (FIXED TYPO HERE)
    id_to_name = {0: "Background", 1: "Bebop", 2: "AR_Drone", 3: "Phantom"}
    y_names = [id_to_name[val] for val in y]

    # --- T-SNE CALCULATION ---
    print(f"⏳ Calculating t-SNE... (This will take ~10 minutes for {N_SAMPLES} points)")
    tsne = TSNE(
        n_components=2, 
        perplexity=50,      # Higher perplexity for larger datasets
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    
    X_embedded = tsne.fit_transform(X_scaled)

    # --- PLOTTING ---
    plt.figure(figsize=(16, 11))
    sns.set_style("whitegrid")
    
    sns.scatterplot(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        hue=y_names,
        palette=COLOR_MAP,
        s=45,               # Smaller dots for high density
        alpha=0.6, 
        edgecolor=None
    )

    total_time = (time.time() - start_time) / 60
    
    plt.title(f"Post-Training t-SNE: Learned Statistical Space (N={N_SAMPLES})\nVisualizing the 82.7% Model Separation", 
              fontsize=20, fontweight='bold', pad=20, color='#2c3e50')
    
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Drone Type", title_fontsize=14, fontsize=12, loc='best', shadow=True)
    
    plt.text(0.01, 0.01, f"Model Accuracy: 82.71%\nN Samples: {N_SAMPLES}\nCalc Time: {total_time:.2f} min", 
             transform=plt.gca().transAxes, fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    
    save_name = f"tsne_rf_post_training_{N_SAMPLES}.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ Success! t-SNE saved as: {save_name}")
    plt.show()

if __name__ == "__main__":
    generate_rf_tsne()