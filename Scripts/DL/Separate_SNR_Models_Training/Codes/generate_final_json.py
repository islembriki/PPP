import pandas as pd
import json
import os


summary_file = "synthese_performances_SNR_DL.csv"
detailed_files = {
    "30dB": "metrics_detailed_30dB.csv",
    "10dB": "metrics_detailed_10dB.csv",
    "0dB": "metrics_detailed_0dB.csv",
    "-10dB": "metrics_detailed_-10dB.csv"
}

def convert_to_json():
    # Vérification 
    if not os.path.exists(summary_file):
        print(f"Erreur : Je ne trouve pas {summary_file} dans ce dossier.")
        print(f"Dossier actuel : {os.getcwd()}")
        return

    data = {
        "model_info": "Deep Learning CNN - Multi-Experts SNR",
        "global_summary": [],
        "detailed_per_snr": {}
    }

    # 1. Traitement de la synthèse (Tableau global)
    print("Lecture de la synthèse...")
    df_summary = pd.read_csv(summary_file)
    data["global_summary"] = df_summary.to_dict(orient="records")

    # 2. Traitement des détails par SNR
    for snr, filename in detailed_files.items():
        if os.path.exists(filename):
            print(f"Lecture des détails pour {snr}...")
            # On lit le CSV détaillé
            df_det = pd.read_csv(filename, index_col=0)
            
            # On le transforme en dictionnaire (Propre pour le JSON)
            data["detailed_per_snr"][snr] = df_det.to_dict(orient="index")
        else:
            print(f"Warning : Fichier {filename} manquant.")

    # 3. Écriture du fichier JSON final
    output_name = "results_DL_experts.json"
    with open(output_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nTERMINÉ ! Le fichier '{output_name}' est prêt.")
    print("Tu peux maintenant le donner à la personne qui code l'interface.")

if __name__ == "__main__":
    convert_to_json()