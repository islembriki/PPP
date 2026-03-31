import pandas as pd
import os

# --- 1. CONFIGURATION ---
# Update these paths to where your files are located
path_me = r"C:\Users\HP\Desktop\RT3\PPP\processed data\ML\bebop_phantom_final_data.csv"
path_aziza = r"C:\Users\HP\Desktop\RT3\PPP\processed data\ML\final_features_dataset.csv"

# Destination for the final project file
output_dir = r"C:\Users\HP\Desktop\RT3\PPP\processed data\ML"
output_file = os.path.join(output_dir, "GLOBAL_DRONE_DATASET.csv")

print(">>> Starting the Final Data Merge <<<")

# --- 2. LOAD BOTH FILES ---
try:
    df_me = pd.read_csv(path_me)
    print(f"Your data loaded: {len(df_me)} segments (Bebop/Phantom)")
    
    df_aziza = pd.read_csv(path_aziza)
    print(f"Aziza's data loaded: {len(df_aziza)} segments (AR/Background)")

    # --- 3. CONCATENATE ---
    # ignore_index=True prevents row numbers from repeating
    df_final = pd.concat([df_me, df_aziza], ignore_index=True)
    
    # --- 4. DATA CLEANING (Safety First) ---
    # In case there are empty rows or weird values
    df_final = df_final.dropna()
    
    # --- 5. VERIFICATION ---
    print("\n--- FINAL DATASET SUMMARY ---")
    print(f"Total size: {len(df_final)} segments")
    print("\nSegments per Drone (Label):")
    # 0=BG, 1=Bebop, 2=AR, 3=Phantom
    counts = df_final['Label'].value_counts().sort_index()
    print(counts)

    # --- 6. SAVE ---
    df_final.to_csv(output_file, index=False)
    print(f"\nSUCCESS! Master dataset saved to: {output_file}")

except Exception as e:
    print(f"ERROR: {e}")