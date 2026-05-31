import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
CSV_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\GLOBAL_DRONE_DATASET.csv"
OUTPUT_DIR = r"C:\Users\HP\Desktop\PPP\tsne"

# ⚡ SET THIS TO True TO SKIP RECOMPUTING t-SNE NEXT TIME
REUSE_TSNE = False  # <-- False for first run, True after that forever

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("RAW t-SNE VISUALIZATION - DRONE RF DETECTION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("\n[STEP 1] Loading data...")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✓ CSV loaded successfully")
    print(f"  - Total rows: {df.shape[0]}")
    print(f"  - Total columns: {df.shape[1]}")
except Exception as e:
    print(f"✗ ERROR loading CSV: {e}")
    exit()

# ============================================
# STEP 2: INSPECT DATA
# ============================================
print("\n[STEP 2] Inspecting data structure...")
print(f"Column names: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nLabel distribution:")
print(df['Label'].value_counts().sort_index())
print(f"\nMode distribution:")
print(df['Mode'].value_counts().sort_index())
print(f"\nLabel + Mode combinations:")
print(pd.crosstab(df['Label'], df['Mode']))

# ============================================
# STEP 3: EXTRACT FEATURES
# ============================================
print("\n[STEP 3] Extracting features...")
feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR']

try:
    X = df[feature_cols].values
    label = df['Label'].values
    mode = df['Mode'].values
    print(f"✓ Features extracted")
    print(f"  - Feature matrix shape: {X.shape}")
except Exception as e:
    print(f"✗ ERROR extracting features: {e}")
    exit()

# ============================================
# STEP 4: CLEAN DATA
# ============================================
print("\n[STEP 4] Cleaning data...")
X = np.where(np.isinf(X), np.nan, X)
valid_rows = ~np.isnan(X).any(axis=1)
X_clean = X[valid_rows]
label_clean = label[valid_rows]
mode_clean = mode[valid_rows]
print(f"  - Rows after cleaning: {len(X_clean)}")
print(f"  - Rows removed: {len(X) - len(X_clean)}")

# ============================================
# STEP 5: STANDARDIZE FEATURES
# ============================================
print("\n[STEP 5] Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
print(f"✓ Features standardized")

# ============================================
# STEP 5.5: SUBSAMPLE ⚡ (~10 min instead of 2 hours)
# ============================================
tsne_path  = os.path.join(OUTPUT_DIR, 'X_tsne.npy')
label_path = os.path.join(OUTPUT_DIR, 'label_clean.npy')
mode_path  = os.path.join(OUTPUT_DIR, 'mode_clean.npy')

if not REUSE_TSNE:
    print("\n[STEP 5.5] Subsampling for t-SNE...")
    SAMPLE_SIZE = 50000
    np.random.seed(42)
    sample_idx  = np.random.choice(len(X_scaled), size=SAMPLE_SIZE, replace=False)
    X_scaled    = X_scaled[sample_idx]
    label_clean = label_clean[sample_idx]
    mode_clean  = mode_clean[sample_idx]
    print(f"✓ Subsampled to {SAMPLE_SIZE} rows")

# ============================================
# STEP 6: APPLY t-SNE (or load saved result)
# ============================================
if REUSE_TSNE and os.path.exists(tsne_path):
    print("\n[STEP 6] Loading saved t-SNE results (skipping recomputation)...")
    X_tsne      = np.load(tsne_path)
    label_clean = np.load(label_path)
    mode_clean  = np.load(mode_path)
    print(f"✓ Loaded saved t-SNE — shape: {X_tsne.shape}")
else:
    print("\n[STEP 6] Applying t-SNE (grab a coffee, ~10 mins)...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        max_iter=1000,
        verbose=1
    )
    X_tsne = tsne.fit_transform(X_scaled)
    print(f"✓ t-SNE completed — shape: {X_tsne.shape}")

    # ⚡ SAVE so you never recompute again
    np.save(tsne_path,  X_tsne)
    np.save(label_path, label_clean)
    np.save(mode_path,  mode_clean)
    print(f"✓ t-SNE results saved to disk")

# ============================================
# STEP 7: CREATE DRONE+MODE COMBINATIONS
# ============================================
print("\n[STEP 7] Creating combined labels...")

label_names = {0: "Background", 1: "Bebop", 2: "AR", 3: "Phantom"}
mode_names  = {0: "No Drone", 1: "Connected", 2: "Hovering", 3: "Flying NoVid", 4: "Flying WithVid"}

combined_labels = label_clean * 10 + mode_clean
print(f"✓ Combined labels created — {len(np.unique(combined_labels))} unique combinations")

for combo in sorted(np.unique(combined_labels)):
    label_id = combo // 10
    mode_id  = combo % 10
    count    = np.sum(combined_labels == combo)
    print(f"    {label_names.get(label_id, 'Unknown'):12} + {mode_names.get(mode_id, 'Unknown'):18} = {combo:2d} ({count:6d} samples)")

# ============================================
# STEP 8: PREPARE COLORS
# ============================================
print("\n[STEP 8] Preparing visualizations...")

label_colors = {
    0: '#808080',  # Gray  - Background
    1: '#FF4444',  # Red   - Bebop
    2: '#44FF44',  # Green - AR
    3: '#4444FF',  # Blue  - Phantom
}

mode_colors = {
    0: '#808080',  # Gray   - No Drone
    1: '#FF0000',  # Red    - Connected
    2: '#00FF00',  # Green  - Hovering
    3: '#0000FF',  # Blue   - Flying NoVid
    4: '#FFFF00',  # Yellow - Flying WithVid
}

base_colors = {
    0: (128, 128, 128),
    1: (255, 100, 100),
    2: (100, 255, 100),
    3: (100, 100, 255),
}

mode_brightness = {
    0: 0.4,
    1: 1.0,
    2: 0.7,
    3: 0.5,
    4: 0.3,
}

combined_colors_map = {}
for label_id in range(4):
    for mode_id in range(0, 5):
        combo = label_id * 10 + mode_id
        base_r, base_g, base_b = base_colors[label_id]
        brightness = mode_brightness[mode_id]
        combined_colors_map[combo] = (
            int(base_r * brightness) / 255,
            int(base_g * brightness) / 255,
            int(base_b * brightness) / 255,
        )

print("✓ Color maps created")

# ============================================
# STEP 9: PLOT 1 - BY DRONE TYPE
# ============================================
print("\n[STEP 9] Creating Plot 1: Colored by Label (Drone Type)...")

plt.figure(figsize=(14, 10))
for label_id in sorted(np.unique(label_clean)):
    indices = label_clean == label_id
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],
                c=[label_colors[label_id]], label=label_names[label_id],
                alpha=0.7, s=50, edgecolors='k', linewidth=0.3)

plt.xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
plt.ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
plt.title('Raw Features t-SNE: Colored by Drone Type', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
output_path_1 = os.path.join(OUTPUT_DIR, 'raw_tsne_by_label.png')
plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_1}")
plt.close()

# ============================================
# STEP 10: PLOT 2 - BY OPERATING MODE
# ============================================
print("\n[STEP 10] Creating Plot 2: Colored by Mode (Operating Mode)...")

plt.figure(figsize=(14, 10))
for mode_id in sorted(np.unique(mode_clean)):
    indices = mode_clean == mode_id
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],
                c=[mode_colors[mode_id]], label=mode_names[mode_id],
                alpha=0.7, s=50, edgecolors='k', linewidth=0.3)

plt.xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
plt.ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
plt.title('Raw Features t-SNE: Colored by Operating Mode', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
output_path_2 = os.path.join(OUTPUT_DIR, 'raw_tsne_by_mode.png')
plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_2}")
plt.close()

# ============================================
# STEP 11: PLOT 3 - COMBINED LABEL+MODE
# ============================================
print("\n[STEP 11] Creating Plot 3: Colored by Label+Mode (Combined)...")

plt.figure(figsize=(16, 12))
for combo in sorted(np.unique(combined_labels)):
    label_id = combo // 10
    mode_id  = combo % 10
    indices  = combined_labels == combo
    count    = np.sum(indices)

    if count > 0:
        label_name = label_names.get(label_id, "Unknown")
        mode_name  = mode_names.get(mode_id, "Unknown")
        color      = combined_colors_map.get(combo, (0.5, 0.5, 0.5))
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],
                    c=[color], label=f"{label_name}-{mode_name} ({count})",
                    alpha=0.7, s=60, edgecolors='k', linewidth=0.3)

plt.xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
plt.ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
plt.title('Raw Features t-SNE: Colored by Drone Type + Operating Mode', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best', ncol=2, framealpha=0.95)
plt.grid(True, alpha=0.3)
plt.tight_layout()
output_path_3 = os.path.join(OUTPUT_DIR, 'raw_tsne_by_label_mode_combined.png')
plt.savefig(output_path_3, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path_3}")
plt.close()

# ============================================
# STEP 12: SUMMARY
# ============================================
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print("\n[PLOT 1 - By Drone Type]")
for label_id in sorted(np.unique(label_clean)):
    count = np.sum(label_clean == label_id)
    print(f"    - {label_names[label_id]:12} : {count:6d} samples ({count/len(label_clean)*100:5.1f}%)")

print("\n[PLOT 2 - By Operating Mode]")
for mode_id in sorted(np.unique(mode_clean)):
    count = np.sum(mode_clean == mode_id)
    print(f"    - {mode_names[mode_id]:18} : {count:6d} samples ({count/len(mode_clean)*100:5.1f}%)")

print("\n[PLOT 3 - By Drone Type + Operating Mode]")
print("  This is what your models are trying to predict!")

print("\n" + "="*80)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print(f"\n✓ All plots saved to: {OUTPUT_DIR}")
print("  1. raw_tsne_by_label.png")
print("  2. raw_tsne_by_mode.png")
print("  3. raw_tsne_by_label_mode_combined.png")