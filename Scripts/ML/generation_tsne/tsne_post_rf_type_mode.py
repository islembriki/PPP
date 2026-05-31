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
# 1. CONFIGURATION & PATHS
# ==========================================
# Update this to your ACTUAL model path
MODEL_PATH = r"C:\Users\HP\Desktop\PPP\ml_trained_models_mode_included\rf_mode.pkl"
DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"

# Number of samples (20,000 is perfect for 10 classes)
N_SAMPLES = 20000 

def generate_post_training_model_tsne():
    print("📂 Loading trained model and dataset...")
    start_time = time.time()
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model pkl not found at {MODEL_PATH}")
        return

    # Load Model
    model = joblib.load(MODEL_PATH)
    
    # Load Data
    df = pd.read_csv(DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # --- 1. FEATURE ENGINEERING (Exactly as used in training) ---
    print("🧪 Reconstructing behavioral features...")
    df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
    df['PAPR_Mean'] = df['PAPR'] * df['Mean']
    df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)
    df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + 1e-9)
    df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
    df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])
    df['Mean_Cube'] = df['Mean']**3

    feature_cols = [
        'Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 
        'Log_Var', 'PAPR_Mean', 'Kurt_Skew', 
        'PAPR_Var_Ratio', 'Signal_Energy', 'Shape_Factor', 'Mean_Cube'
    ]
    
    df['Target'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)

    # --- 2. SAMPLING ---
    print(f"✂️ Subsampling {N_SAMPLES} points...")
    df_sample, _ = train_test_split(df, train_size=N_SAMPLES, stratify=df['Target'], random_state=42)

    X_eval = df_sample[feature_cols]
    y_true = df_sample['Target']

    # --- 3. EXTRACT MODEL "THOUGHTS" (Probability Space) ---
    print("🧠 Extracting Model Confidence (predict_proba)...")
    # Instead of raw features, we run t-SNE on the 10-dimensional probabilities
    # This shows how the model clusters classes internally.
    model_embeddings = model.predict_proba(X_eval)

    # --- 4. T-SNE CALCULATION ---
    print("⏳ Calculating t-SNE on Model Logic...")
    tsne = TSNE(
        n_components=2, 
        perplexity=50, 
        learning_rate='auto', 
        init='pca', 
        random_state=42
    )
    
    X_embedded = tsne.fit_transform(model_embeddings)

    # --- 5. PLOTTING ---
    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")
    
    # We use a custom palette to group Drone families by color
    # Bebop = Pinks, AR = Yellows/Oranges, Phantom = Greens, BG = Blue
    unique_targets = sorted(y_true.unique())
    
    scatter = sns.scatterplot(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        hue=y_true,
        hue_order=unique_targets,
        palette="turbo", # turbo provides high contrast for 10 classes
        s=60, 
        alpha=0.7, 
        edgecolor="w", 
        linewidth=0.3
    )

    total_time = (time.time() - start_time) / 60
    
    plt.title(f"Post-Training t-SNE: Random Forest Logic Space (N={N_SAMPLES})\n"
              f"Visualizing how the model clusters Type+Mode Confidence", 
              fontsize=20, fontweight='bold', pad=20, color='#1a1a1a')
    
    plt.xlabel("t-SNE Decision Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Decision Dimension 2", fontsize=12)
    plt.legend(title="Class (Label_Mode)", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.text(0.01, 0.01, 
             f"Model Accuracy: 70.08%\n"
             f"Basis: predict_proba() Space\n"
             f"Time: {total_time:.2f} min", 
             transform=plt.gca().transAxes, fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    save_name = "tsne_rf_MODEL_LOGIC_MODES.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Post-training logic map saved as: {save_name}")
    plt.show()

if __name__ == "__main__":
    generate_post_training_model_tsne()