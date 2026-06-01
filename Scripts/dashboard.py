import tkinter as tk
from tkinter import ttk
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os

# ============================================
# PATHS
# ============================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "rf_type":         os.path.join(BASE, "ml_trained_models_type_only", "rf_results.json"),
    "knn_type":        os.path.join(BASE, "ml_trained_models_type_only", "knn_results.json"),
    "xgb_type":        os.path.join(BASE, "ml_trained_models_type_only", "xgboost_type_results.json"),
    "rf_mode":         os.path.join(BASE, "ml_trained_models_mode_included", "rf_mode.json"),
    "knn_mode":        os.path.join(BASE, "ml_trained_models_mode_included", "knn_mode.json"),
    "xgb_mode":        os.path.join(BASE, "ml_trained_models_mode_included", "xgboost_mode.json"),
    "tsne_avant_type": os.path.join(BASE, "tsne", "tsne_avant_type_only.png"),
    "tsne_avant_mode": os.path.join(BASE, "tsne", "tsne_avant_type_mode.png"),
    "tsne_rf_type":    os.path.join(BASE, "tsne", "tsne_rf_post_type_only.png"),
    "tsne_knn_type":   os.path.join(BASE, "tsne", "tsne_knn_post_type_only.png"),
    "tsne_xgb_type":   os.path.join(BASE, "tsne", "tsne_xgboost_post_type_only.png"),
    "tsne_rf_mode":    os.path.join(BASE, "tsne", "tsne_post_rf_type_mode.png"),
    "tsne_knn_mode":   os.path.join(BASE, "tsne", "tsne_post_train_knn_type_mode.png"),
    "tsne_xgb_mode":   os.path.join(BASE, "tsne", "tsne_xgboost_post_type_mode.png"),


    "dl_results":    os.path.join(BASE, "Scripts", "DL", "Separate_SNR_Models_Training", "Metrics_Reports", "results_DL_experts.json"),
    "tsne_dl_avant": os.path.join(BASE, "Scripts", "DL", "final_version", "tsne_avant_entrainement.png"),
    "dl_dashboard":  os.path.join(BASE, "Scripts", "DL", "Separate_SNR_Models_Training", "Interfaces_graphiques", "dashboard_officiel_valeurs_csv.png"),
    "dl_confusion":  os.path.join(BASE, "Scripts", "DL", "Separate_SNR_Models_Training", "Interfaces_graphiques", "confusion_matrix_30dB.png"),
    "dl_tsne_post":  os.path.join(BASE, "Scripts", "DL", "Separate_SNR_Models_Training", "Interfaces_graphiques", "tsne_drone_final_fixed.png"),

    "belkis_results":  os.path.join(BASE, "results", "cnn_results.json"),
    "belkis_dist":     os.path.join(BASE, "results", "figure_distribution_white.png"),
    "belkis_weights":  os.path.join(BASE, "results", "figure_snr_weights_white.png"),
    "belkis_learning": os.path.join(BASE, "learning_curves_final.png"), # Vérifie le dossier exact
    "belkis_cm":       os.path.join(BASE, "results", "MATRICE_FINALE_85PC.png"),
    "belkis_tsne":     os.path.join(BASE, "results", "tsne_CLEAN_WHITE.png"),
}






# ============================================
# LOAD JSONs
# ============================================
def load_json(key):
    try:
        with open(PATHS[key], 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Impossible de charger {key}: {e}")
        return {}

rf_type  = load_json("rf_type")
knn_type = load_json("knn_type")
xgb_type = load_json("xgb_type")
rf_mode  = load_json("rf_mode")
knn_mode = load_json("knn_mode")
xgb_mode = load_json("xgb_mode")
dl_data = load_json("dl_results")
belkis_data = load_json("belkis_results")


# ============================================
# HELPERS
# ============================================
def get_acc(data):
    acc = data.get("accuracy", 0)
    if isinstance(acc, float) and acc <= 1:
        return acc * 100
    return float(acc)

def get_window_width():
    return root.winfo_width() - 40

# ============================================
# SCROLLABLE FRAME — fixes trackpad scrolling
# ============================================
def scrollable_frame(parent):
    container = tk.Frame(parent, bg="#f5f5f5")
    container.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(container, bg="#f5f5f5", highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg="#f5f5f5")

    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

    # Make inner frame fill full canvas width
    def on_canvas_configure(e):
        canvas.itemconfig(canvas_window, width=e.width)
    canvas.bind("<Configure>", on_canvas_configure)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Bind scroll to ALL widgets inside — fixes trackpad
    def _on_mousewheel(event):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def bind_scroll(widget):
        widget.bind("<MouseWheel>", _on_mousewheel)
        for child in widget.winfo_children():
            bind_scroll(child)

    # Rebind after children are added
    inner.bind("<Configure>", lambda e: (
        canvas.configure(scrollregion=canvas.bbox("all")),
        bind_scroll(inner)
    ))
    canvas.bind("<MouseWheel>", _on_mousewheel)
    inner.bind("<MouseWheel>", _on_mousewheel)

    return inner, canvas, _on_mousewheel

# ============================================
# UI COMPONENTS
# ============================================
def section_label(parent, text, color="#1a1a2e", size=12):
    tk.Label(parent, text=text, font=("Arial", size, "bold"),
             bg="#f5f5f5", fg=color, anchor="w").pack(
        fill=tk.X, padx=20, pady=(18, 4))

def divider(parent):
    ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=20, pady=8)

def stat_cards(parent, cards):
    row = tk.Frame(parent, bg="#f5f5f5")
    row.pack(fill=tk.X, padx=20, pady=10)
    for label, value, color in cards:
        card = tk.Frame(row, bg=color, padx=16, pady=14)
        card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=6)
        tk.Label(card, text=value, font=("Arial", 20, "bold"),
                 bg=color, fg="white").pack()
        tk.Label(card, text=label, font=("Arial", 9),
                 bg=color, fg="white").pack()

def embed_matplotlib(fig, parent):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    w = canvas.get_tk_widget()
    w.pack(fill=tk.BOTH, expand=True)
    plt.close(fig)
    return canvas

def bar_chart_figure(models, accuracies, colors, title):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("#f5f5f5")
    bars = ax.barh(models, accuracies, color=colors, height=0.45)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%", va='center', fontweight='bold', fontsize=10)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Précision (%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

def confusion_matrix_figure(cm_data, labels, title):
    import seaborn as sns
    cm = np.array(cm_data)
    n = len(labels)
    figsize = (max(6, n * 0.9), max(5, n * 0.8))
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f5f5f5")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.4, cbar=False,
                annot_kws={"size": 7})
    ax.set_xlabel('Prédit', fontsize=9)
    ax.set_ylabel('Réel', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    return fig

# ============================================
# FIX 2: tsne_image_block — responsive scaling
# clamp between 300 px (readable non-fullscreen)
# and 900 px (not oversized fullscreen)
# ============================================
def tsne_image_block(parent, key, caption, bind_fn=None):
    try:
        img_raw = Image.open(PATHS[key])
        W = min(900, max(300, root.winfo_width() - 80))
        ratio = img_raw.height / img_raw.width
        H = int(W * ratio)
        img_raw = img_raw.resize((W, H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img_raw)
        lbl = tk.Label(parent, image=photo, bg="#f5f5f5")
        lbl.image = photo
        lbl.pack(padx=20, pady=6, fill=tk.X)
        if bind_fn:
            lbl.bind("<MouseWheel>", bind_fn)
    except Exception as e:
        tk.Label(parent, text=f"[Image introuvable: {key}]\n{e}",
                 bg="#ffe0e0", fg="red", padx=10, pady=6).pack(padx=20)
    if caption:
        tk.Label(parent, text=caption, font=("Arial", 8, "italic"),
                 bg="#f5f5f5", fg="#666").pack(padx=20)

def report_table(parent, report_dict, target_names, bind_fn=None):
    frame = tk.Frame(parent, bg="#f5f5f5")
    frame.pack(fill=tk.X, padx=20, pady=6)
    headers = ["Classe", "Précision", "Rappel", "F1-Score", "Support"]
    widths  = [16, 11, 11, 11, 11]
    colors  = ["#2c3e50","#2980b9","#27ae60","#e67e22","#8e44ad"]
    for col, (h, w, c) in enumerate(zip(headers, widths, colors)):
        lbl = tk.Label(frame, text=h, font=("Courier New", 9, "bold"),
                 bg=c, fg="white", width=w, relief="flat", padx=4, pady=4)
        lbl.grid(row=0, column=col, padx=1, pady=1)
        if bind_fn: lbl.bind("<MouseWheel>", bind_fn)
    for row, cls in enumerate(target_names, 1):
        if cls in report_dict:
            r = report_dict[cls]
            vals = [cls,
                    f"{r['precision']:.3f}",
                    f"{r['recall']:.3f}",
                    f"{r['f1-score']:.3f}",
                    f"{int(r['support'])}"]
            bg_row = "#ffffff" if row % 2 == 0 else "#ecf0f1"
            for col, (v, w) in enumerate(zip(vals, widths)):
                lbl = tk.Label(frame, text=v, font=("Courier New", 9),
                         bg=bg_row, fg="#2c3e50", width=w, padx=4, pady=3)
                lbl.grid(row=row, column=col, padx=1, pady=1)
                if bind_fn: lbl.bind("<MouseWheel>", bind_fn)
    if "accuracy" in report_dict:
        acc_val = report_dict["accuracy"]
        if isinstance(acc_val, float) and acc_val <= 1:
            acc_val *= 100
        lbl = tk.Label(frame, text=f"Précision globale : {acc_val:.2f}%",
                 font=("Arial", 10, "bold"), bg="#f5f5f5", fg="#2c3e50")
        lbl.grid(row=len(target_names)+1, column=0, columnspan=5,
                 sticky="w", pady=(8,2))
        if bind_fn: lbl.bind("<MouseWheel>", bind_fn)

# ============================================
# TAB 1 — Expérience A : Type seulement
# ============================================
def build_tab_a(parent):
    inner, _, scroll_fn = scrollable_frame(parent)
    acc_rf  = get_acc(rf_type)
    acc_knn = get_acc(knn_type)
    acc_xgb = get_acc(xgb_type)
    best    = max(acc_rf, acc_knn, acc_xgb)

    stat_cards(inner, [
        ("Meilleure précision (RF)", f"{best:.1f}%", "#2980b9"),
        ("Nombre de classes", "4", "#27ae60"),
        ("Descripteurs utilisés", "8", "#8e44ad"),
    ])
    divider(inner)

    # FIX 1: single bar chart block — duplicate removed
    section_label(inner, "Précision par algorithme", "#2980b9")
    fr = tk.Frame(inner, bg="#f5f5f5")
    fr.pack(fill=tk.X, padx=20, pady=4)
    embed_matplotlib(bar_chart_figure(
        ["Forêt Aléatoire", "KNN (k=52)", "XGBoost"],
        [acc_rf, acc_knn, acc_xgb],
        ["#2980b9","#27ae60","#e67e22"],
        "Exp. A — Classification du type de drone (4 classes)"
    ), fr)
    divider(inner)

    section_label(inner, "t-SNE — Avant l'entraînement (caractéristiques brutes)", "#2980b9")
    tsne_image_block(inner, "tsne_avant_type",
        "t-SNE pré-entraînement : 4 types de drones, espace des caractéristiques brutes", scroll_fn)
    divider(inner)

    section_label(inner, "t-SNE — Post-entraînement (Forêt Aléatoire)", "#2980b9")
    tsne_image_block(inner, "tsne_rf_type",
        f"t-SNE RF post-entraînement · Précision : {acc_rf:.1f}%", scroll_fn)

    section_label(inner, "t-SNE — Post-entraînement (KNN)", "#27ae60")
    tsne_image_block(inner, "tsne_knn_type",
        f"t-SNE KNN post-entraînement · Précision : {acc_knn:.1f}%", scroll_fn)

    section_label(inner, "t-SNE — Post-entraînement (XGBoost)", "#e67e22")
    tsne_image_block(inner, "tsne_xgb_type",
        f"t-SNE XGBoost post-entraînement · Précision : {acc_xgb:.1f}%", scroll_fn)
    divider(inner)

    tnames_type = rf_type.get("target_names", ["Background","Bebop","AR_Drone","Phantom"])

    section_label(inner, "Rapport détaillé — Forêt Aléatoire", "#2980b9")
    report_table(inner, rf_type.get("classification_report", {}), tnames_type, scroll_fn)

    section_label(inner, "Matrice de confusion — Forêt Aléatoire", "#2980b9")
    fr1 = tk.Frame(inner, bg="#f5f5f5"); fr1.pack(fill=tk.X, padx=20)
    embed_matplotlib(confusion_matrix_figure(
        rf_type.get("confusion_matrix",[]), tnames_type,
        "Matrice de confusion — Forêt Aléatoire (Type seulement)"), fr1)

    section_label(inner, "Rapport détaillé — KNN", "#27ae60")
    report_table(inner, knn_type.get("classification_report",{}), tnames_type, scroll_fn)

    section_label(inner, "Matrice de confusion — KNN", "#27ae60")
    fr2 = tk.Frame(inner, bg="#f5f5f5"); fr2.pack(fill=tk.X, padx=20)
    embed_matplotlib(confusion_matrix_figure(
        knn_type.get("confusion_matrix",[]), tnames_type,
        "Matrice de confusion — KNN (Type seulement)"), fr2)

    section_label(inner, "Rapport détaillé — XGBoost", "#e67e22")
    report_table(inner, xgb_type.get("classification_report",{}), tnames_type, scroll_fn)

    section_label(inner, "Matrice de confusion — XGBoost", "#e67e22")
    fr3 = tk.Frame(inner, bg="#f5f5f5"); fr3.pack(fill=tk.X, padx=20)
    embed_matplotlib(confusion_matrix_figure(
        xgb_type.get("confusion_matrix",[]), tnames_type,
        "Matrice de confusion — XGBoost (Type seulement)"), fr3)

    tk.Label(inner, text="", bg="#f5f5f5").pack(pady=20)

# ============================================
# TAB 2 — Expérience B : Type + Mode
# ============================================
def build_tab_b(parent):
    inner, _, scroll_fn = scrollable_frame(parent)
    acc_rf  = get_acc(rf_mode)
    acc_knn = get_acc(knn_mode)
    acc_xgb = get_acc(xgb_mode)
    best    = max(acc_rf, acc_knn, acc_xgb)
    labels  = rf_mode.get("target_names", [])

    stat_cards(inner, [
        ("Meilleure précision (RF)", f"{best:.1f}%", "#d35400"),
        ("Nombre de classes", "10", "#c0392b"),
        ("Descripteurs utilisés", "12", "#8e44ad"),
    ])

    tk.Label(inner,
             text="⚠  Cause racine de la dégradation — La baisse n'est pas un échec algorithmique.\n"
                  "Les variations de mode de vol au sein d'un même drone produisent des différences\n"
                  "micro-spectrales perdues lors de la compression en descripteurs scalaires.\n"
                  "Capturer ces différences nécessite des représentations temps-fréquence (spectrogrammes).",
             font=("Arial", 9), bg="#fff3cd", fg="#856404",
             justify="left", padx=14, pady=10
             ).pack(fill=tk.X, padx=20, pady=8)
    divider(inner)

    section_label(inner, "Précision par algorithme", "#d35400")
    fr = tk.Frame(inner, bg="#f5f5f5"); fr.pack(fill=tk.X, padx=20, pady=4)
    embed_matplotlib(bar_chart_figure(
        ["Forêt Aléatoire", "KNN (k=5)", "XGBoost"],
        [acc_rf, acc_knn, acc_xgb],
        ["#d35400","#c0392b","#e67e22"],
        "Exp. B — Classification type + mode de vol (10 classes)"
    ), fr)
    divider(inner)

    section_label(inner, "t-SNE — Avant l'entraînement (Type + Mode)", "#d35400")
    tsne_image_block(inner, "tsne_avant_mode",
        "t-SNE pré-entraînement : 10 classes combinées (type × mode)", scroll_fn)
    divider(inner)

    section_label(inner, "t-SNE — Post-entraînement (Forêt Aléatoire)", "#d35400")
    tsne_image_block(inner, "tsne_rf_mode",
        f"t-SNE RF post-entraînement · Précision : {acc_rf:.1f}%", scroll_fn)

    section_label(inner, "t-SNE — Post-entraînement (KNN)", "#c0392b")
    tsne_image_block(inner, "tsne_knn_mode",
        f"t-SNE KNN post-entraînement · Précision : {acc_knn:.1f}%", scroll_fn)

    section_label(inner, "t-SNE — Post-entraînement (XGBoost)", "#e67e22")
    tsne_image_block(inner, "tsne_xgb_mode",
        f"t-SNE XGBoost post-entraînement · Précision : {acc_xgb:.1f}%", scroll_fn)
    divider(inner)

    section_label(inner, "Rapport détaillé — Forêt Aléatoire", "#d35400")
    report_table(inner, rf_mode.get("classification_report",{}), labels, scroll_fn)

    section_label(inner, "Matrice de confusion — Forêt Aléatoire", "#d35400")
    fr1 = tk.Frame(inner, bg="#f5f5f5"); fr1.pack(fill=tk.X, padx=20)
    embed_matplotlib(confusion_matrix_figure(
        rf_mode.get("confusion_matrix",[]), labels,
        "Matrice de confusion — Forêt Aléatoire (Type + Mode)"), fr1)

    section_label(inner, "Rapport détaillé — KNN", "#c0392b")
    report_table(inner, knn_mode.get("classification_report",{}), labels, scroll_fn)

    section_label(inner, "Matrice de confusion — KNN", "#c0392b")
    fr2 = tk.Frame(inner, bg="#f5f5f5"); fr2.pack(fill=tk.X, padx=20)
    embed_matplotlib(confusion_matrix_figure(
        knn_mode.get("confusion_matrix",[]), labels,
        "Matrice de confusion — KNN (Type + Mode)"), fr2)

    section_label(inner, "Rapport détaillé — XGBoost", "#e67e22")
    report_table(inner, xgb_mode.get("classification_report",{}), labels, scroll_fn)

    section_label(inner, "Matrice de confusion — XGBoost", "#e67e22")
    fr3 = tk.Frame(inner, bg="#f5f5f5"); fr3.pack(fill=tk.X, padx=20)
    embed_matplotlib(confusion_matrix_figure(
        xgb_mode.get("confusion_matrix",[]), labels,
        "Matrice de confusion — XGBoost (Type + Mode)"), fr3)

    tk.Label(inner, text="", bg="#f5f5f5").pack(pady=20)

# ============================================
# TAB 3 — Comparaison ML
# ============================================
def build_tab_comparison(parent):
    inner, _, scroll_fn = scrollable_frame(parent)
    acc_rf_a  = get_acc(rf_type)
    acc_knn_a = get_acc(knn_type)
    acc_xgb_a = get_acc(xgb_type)
    acc_rf_b  = get_acc(rf_mode)
    acc_knn_b = get_acc(knn_mode)
    acc_xgb_b = get_acc(xgb_mode)

    section_label(inner, "Exp. A vs Exp. B — Comparaison côte à côte", "#27ae60", size=13)

    row = tk.Frame(inner, bg="#f5f5f5")
    row.pack(fill=tk.X, padx=20, pady=10)

    def mini_card(parent, title, rows, color, note):
        c = tk.Frame(parent, bg=color, padx=2, pady=2)
        c.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=8)
        tk.Label(c, text=title, font=("Arial", 11, "bold"),
                 bg=color, fg="white", pady=6).pack()
        inner2 = tk.Frame(c, bg="white")
        inner2.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        for model, acc in rows:
            r2 = tk.Frame(inner2, bg="white")
            r2.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(r2, text=model, font=("Arial", 9),
                     bg="white", fg="#2c3e50", width=18, anchor="w").pack(side=tk.LEFT)
            tk.Label(r2, text=f"{acc:.1f}%", font=("Arial", 10, "bold"),
                     bg="white", fg=color).pack(side=tk.RIGHT)
        tk.Label(inner2, text=note, font=("Arial", 8, "italic"),
                 bg="white", fg="#888", pady=6).pack()

    mini_card(row, "Exp. A · 4 classes",
              [("Forêt Aléatoire", acc_rf_a),
               ("KNN (k=52)", acc_knn_a),
               ("XGBoost", acc_xgb_a)],
              "#2980b9", "✓ 8 descripteurs suffisants")

    mini_card(row, "Exp. B · 10 classes",
              [("Forêt Aléatoire", acc_rf_b),
               ("KNN (k=5)", acc_knn_b),
               ("XGBoost", acc_xgb_b)],
              "#d35400", "✗ Descripteurs insuffisants pour les modes intra-marque")

    divider(inner)

    section_label(inner, "Comparaison des précisions — tous les modèles", "#27ae60")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor("#f5f5f5")
    models = ["Forêt Aléatoire", "KNN", "XGBoost"]
    for ax in [ax1, ax2]:
        ax.set_facecolor("#f5f5f5")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("Précision (%)", fontsize=9)
        ax.set_xlim(0, 108)

    ax1.barh(models, [acc_rf_a, acc_knn_a, acc_xgb_a],
             color=["#2980b9","#27ae60","#e67e22"], height=0.45)
    ax1.set_title("Exp. A — 4 classes", fontsize=10, fontweight='bold')
    for i, v in enumerate([acc_rf_a, acc_knn_a, acc_xgb_a]):
        ax1.text(v+0.5, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')

    ax2.barh(models, [acc_rf_b, acc_knn_b, acc_xgb_b],
             color=["#d35400","#c0392b","#e67e22"], height=0.45)
    ax2.set_title("Exp. B — 10 classes", fontsize=10, fontweight='bold')
    for i, v in enumerate([acc_rf_b, acc_knn_b, acc_xgb_b]):
        ax2.text(v+0.5, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')

    fig.tight_layout()
    fr = tk.Frame(inner, bg="#f5f5f5"); fr.pack(fill=tk.X, padx=20)
    embed_matplotlib(fig, fr)
    divider(inner)

    section_label(inner, "t-SNE côte à côte — Avant entraînement", "#27ae60")
    img_row = tk.Frame(inner, bg="#f5f5f5")
    img_row.pack(fill=tk.X, padx=20, pady=6)
    for key, cap in [("tsne_avant_type", "Type seulement (4 classes)"),
                     ("tsne_avant_mode", "Type + Mode (10 classes)")]:
        col = tk.Frame(img_row, bg="#f5f5f5")
        col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=6)
        try:
            W = min(700, max(300, (root.winfo_width() - 100) // 2))
            img_raw = Image.open(PATHS[key])
            H = int(W * img_raw.height / img_raw.width)
            photo = ImageTk.PhotoImage(img_raw.resize((W, H), Image.LANCZOS))
            lbl = tk.Label(col, image=photo, bg="#f5f5f5")
            lbl.image = photo
            lbl.pack()
            lbl.bind("<MouseWheel>", scroll_fn)
        except:
            tk.Label(col, text=f"[{key}]", bg="#ffe0e0", fg="red").pack()
        tk.Label(col, text=cap, font=("Arial", 8, "italic"),
                 bg="#f5f5f5", fg="#666").pack()
    divider(inner)

    # NEW: RF post-training side by side
    section_label(inner, "t-SNE post-entraînement — Forêt Aléatoire (A vs B)", "#2980b9")
    img_row_rf = tk.Frame(inner, bg="#f5f5f5")
    img_row_rf.pack(fill=tk.X, padx=20, pady=6)
    for key, cap in [("tsne_rf_type", "RF — Type seulement (4 classes)"),
                     ("tsne_rf_mode", "RF — Type + Mode (10 classes)")]:
        col = tk.Frame(img_row_rf, bg="#f5f5f5")
        col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=6)
        try:
            W = min(700, max(300, (root.winfo_width() - 100) // 2))
            img_raw = Image.open(PATHS[key])
            H = int(W * img_raw.height / img_raw.width)
            photo = ImageTk.PhotoImage(img_raw.resize((W, H), Image.LANCZOS))
            lbl = tk.Label(col, image=photo, bg="#f5f5f5")
            lbl.image = photo
            lbl.pack()
            lbl.bind("<MouseWheel>", scroll_fn)
        except:
            tk.Label(col, text=f"[{key}]", bg="#ffe0e0", fg="red").pack()
        tk.Label(col, text=cap, font=("Arial", 8, "italic"),
                 bg="#f5f5f5", fg="#666").pack()
    divider(inner)

    # NEW: KNN post-training side by side
    section_label(inner, "t-SNE post-entraînement — KNN (A vs B)", "#27ae60")
    img_row_knn = tk.Frame(inner, bg="#f5f5f5")
    img_row_knn.pack(fill=tk.X, padx=20, pady=6)
    for key, cap in [("tsne_knn_type", "KNN — Type seulement (4 classes)"),
                     ("tsne_knn_mode", "KNN — Type + Mode (10 classes)")]:
        col = tk.Frame(img_row_knn, bg="#f5f5f5")
        col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=6)
        try:
            W = min(700, max(300, (root.winfo_width() - 100) // 2))
            img_raw = Image.open(PATHS[key])
            H = int(W * img_raw.height / img_raw.width)
            photo = ImageTk.PhotoImage(img_raw.resize((W, H), Image.LANCZOS))
            lbl = tk.Label(col, image=photo, bg="#f5f5f5")
            lbl.image = photo
            lbl.pack()
            lbl.bind("<MouseWheel>", scroll_fn)
        except:
            tk.Label(col, text=f"[{key}]", bg="#ffe0e0", fg="red").pack()
        tk.Label(col, text=cap, font=("Arial", 8, "italic"),
                 bg="#f5f5f5", fg="#666").pack()
    divider(inner)

    # NEW: XGBoost post-training side by side
    section_label(inner, "t-SNE post-entraînement — XGBoost (A vs B)", "#e67e22")
    img_row_xgb = tk.Frame(inner, bg="#f5f5f5")
    img_row_xgb.pack(fill=tk.X, padx=20, pady=6)
    for key, cap in [("tsne_xgb_type", "XGBoost — Type seulement (4 classes)"),
                     ("tsne_xgb_mode", "XGBoost — Type + Mode (10 classes)")]:
        col = tk.Frame(img_row_xgb, bg="#f5f5f5")
        col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=6)
        try:
            W = min(700, max(300, (root.winfo_width() - 100) // 2))
            img_raw = Image.open(PATHS[key])
            H = int(W * img_raw.height / img_raw.width)
            photo = ImageTk.PhotoImage(img_raw.resize((W, H), Image.LANCZOS))
            lbl = tk.Label(col, image=photo, bg="#f5f5f5")
            lbl.image = photo
            lbl.pack()
            lbl.bind("<MouseWheel>", scroll_fn)
        except:
            tk.Label(col, text=f"[{key}]", bg="#ffe0e0", fg="red").pack()
        tk.Label(col, text=cap, font=("Arial", 8, "italic"),
                 bg="#f5f5f5", fg="#666").pack()
    divider(inner)

    tk.Label(inner,
             text="Conclusion clé : les 8 descripteurs permettent une séparation nette au niveau de la marque,\n"
                  "mais ne peuvent résoudre les différences intra-marque entre modes de vol.\n"
                  "C'est la motivation structurelle pour passer au CNN + spectrogrammes.",
             font=("Arial", 9), bg="#d4edda", fg="#155724",
             padx=14, pady=12, justify="left"
             ).pack(fill=tk.X, padx=20, pady=10)

    tk.Label(inner, text="", bg="#f5f5f5").pack(pady=20)




# ============================================
# TAB 4 — Apprentissage Profond (placeholder) 
# ============================================
def build_tab_dl(parent):
    # Création du Notebook interne
    inner_notebook = ttk.Notebook(parent)
    inner_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    sub_tab_separate = tk.Frame(inner_notebook, bg="#f5f5f5")
    sub_tab_merged   = tk.Frame(inner_notebook, bg="#f5f5f5")

    inner_notebook.add(sub_tab_separate, text="  1. Separate SNR ")
    inner_notebook.add(sub_tab_merged,   text="  2. Merged SNR ")

    # --- CONSTRUCTION DE LA PARTIE AMANI (ORDRE LOGIQUE) ---
    inner_sep, _, scroll_fn = scrollable_frame(sub_tab_separate)

    # ==========================================================
    # ÉTAPE 1 : ÉTAT INITIAL (t-SNE AVANT)
    # ==========================================================
    section_label(inner_sep, "1. État Initial : Complexité des Données Brutes", "#022b55")
    tsne_image_block(inner_sep, "tsne_dl_avant", 
                     "Visualisation t-SNE pré-entraînement : Les signatures sont entrelacées et indiscriminables.", scroll_fn)

    # ==========================================================
    # ÉTAPE 2 : STRATÉGIE DE NETTOYAGE (DESIGN UI PRO)
    # ==========================================================
    divider(inner_sep)
    section_label(inner_sep, "2. Optimisation : Signal Activity Detection (SAD)", "#022b55")
    
    # Remplacement de l'image par un bloc de texte stylisé "Tech"
    clean_box = tk.Frame(inner_sep, bg="#ffffff", highlightbackground="#022b55", highlightthickness=1, padx=20, pady=15)
    clean_box.pack(fill=tk.X, padx=30, pady=10)
    
    sad_text = (
        "⚙️ MÉTHODOLOGIE DE PURIFICATION (GATING) :\n\n"
        "• Analyse des Bursts : Isolation des salves actives du drone par rapport au bruit thermique.\n"
        "• Seuil d'Activité : Application d'un filtre morphologique (Luminance Moyenne μ > 15).\n"
        "• Impact : Suppression de 90% des trames vides, éliminant le biais de confusion (Label Noise)."
    )
    lbl_sad = tk.Label(clean_box, text=sad_text, font=("Segoe UI", 10), bg="#ffffff", fg="#2c3e50", justify="left")
    lbl_sad.pack(anchor="w")
    lbl_sad.bind("<MouseWheel>", scroll_fn)

    # ==========================================================
    # ÉTAPE 3 : RÉSULTATS D'APPRENTISSAGE (ACCURACY)
    # ==========================================================
    divider(inner_sep)
    section_label(inner_sep, "3. Performances des Modèles Experts SNR", "#022b55")
    
    summary = dl_data.get("global_summary", [])
    acc_30 = next((item["Accuracy (%)"] for item in summary if item["SNR (dB)"] == 30), 0)
    acc_10 = next((item["Accuracy (%)"] for item in summary if item["SNR (dB)"] == 10), 0)
    acc_0  = next((item["Accuracy (%)"] for item in summary if item["SNR (dB)"] == 0), 0)
    acc_m10 = next((item["Accuracy (%)"] for item in summary if item["SNR (dB)"] == -10), 0)

    stat_cards(inner_sep, [
        ("Expert 30dB", f"{acc_30:.2f}%", "#1a936f"),
        ("Expert 10dB (Best)", f"{acc_10:.2f}%", "#18C15E"), # Vert plus vif car meilleur score
        ("Expert 0dB", f"{acc_0:.2f}%", "#f3a712"),
        ("Expert -10dB", f"{acc_m10:.2f}%", "#db2b39"),
    ])

    tsne_image_block(inner_sep, "dl_dashboard", 
                     "Dashboard de performance : Synthèse de la précision et courbe de résilience au bruit.", scroll_fn)

    # ==========================================================
    # ÉTAPE 4 : ANALYSE DES ERREURS (MATRICE DE CONFUSION)
    # ==========================================================
    divider(inner_sep)
    section_label(inner_sep, "4. Analyse Fine : Matrice de Confusion (Expert 30dB)", "#022b55")
    tsne_image_block(inner_sep, "dl_confusion", 
                     "Validation : Identification parfaite des types (98%) et défis sur les modes intra-marques.", scroll_fn)

    # ==========================================================
    # ÉTAPE 5 : VALIDATION FINALE (t-SNE POST-TRAINING)
    # ==========================================================
    divider(inner_sep)
    section_label(inner_sep, "5. Validation de l'Apprentissage : Espace Latent", "#022b55")
    tsne_image_block(inner_sep, "dl_tsne_post", 
                     "Post-Training : Séparation sémantique nette des signatures de drones après optimisation.", scroll_fn)

    # Footer pour finir proprement
    tk.Label(inner_sep, text="", bg="#f5f5f5").pack(pady=20)

    # --- PARTIE BELKIS (APPROCHE GÉNÉRALISTE) ---
    inner_merg, _, scroll_fn_merg = scrollable_frame(sub_tab_merged)

    # ==========================================================
    # ÉTAPE 1 : STRATÉGIE "ONE BRAIN"
    # ==========================================================
    section_label(inner_merg, "1. Concept : Modèle Unique Multi-SNR (Généraliste)", "#3f51b5")

    info_box = tk.Frame(inner_merg, bg="#e8eaf6", padx=20, pady=15)
    info_box.pack(fill=tk.X, padx=30, pady=10)

    concept_text = (
        "🎯 STRATÉGIE SCIENTIFIQUE :\n\n"
        "Contrairement à l'approche multi-modèles, nous avons entraîné un cerveau unique sur l'ensemble\n"
        "du spectre de bruit (-10dB à +30dB). Le modèle utilise les signaux clairs comme guide structurel\n"
        "pour identifier les drones quasi-invisibles dans le bruit extrême."
    )
    tk.Label(info_box, text=concept_text, font=("Segoe UI", 10, "italic"), bg="#e8eaf6", fg="#1a237e", justify="left").pack(anchor="w")

    # ==========================================================
    # ÉTAPE 2 : OPTIMISATION DES DONNÉES (SAD & WEIGHTING)
    # ==========================================================
    divider(inner_merg)
    section_label(inner_merg, "2. Prétraitement et Pondération SNR-Aware", "#3f51b5")

    # Bloc pour la Figure 5 (Distribution)
    tsne_image_block(inner_merg, "belkis_dist",
                     "Figure : Rééquilibrage par WeightedRandomSampler pour compenser le nettoyage SAD.", scroll_fn_merg)

    # Bloc pour la Figure 6 (Poids SNR)
    tsne_image_block(inner_merg, "belkis_weights",
                     "Figure : SNR-Aware Weighting. Les signaux à -10dB reçoivent une priorité 4x supérieure.", scroll_fn_merg)

    # ==========================================================
    # ÉTAPE 3 : PERFORMANCE GLOBALE
    # ==========================================================
    divider(inner_merg)
    section_label(inner_merg, "3. Résultats : Convergence et Précision Globale", "#3f51b5")

    acc_global = belkis_data.get("accuracy", 85.18)
    f1_global = belkis_data.get("f1_score", 0.85)

    stat_cards(inner_merg, [
        ("Accuracy Merged", f"{acc_global}%", "#3f51b5"),
        ("Score F1", f"{f1_global:.3f}", "#5c6bc0"),
        ("Échantillons", "2.27 Millions", "#7986cb"),
        ("Architecture", "CNN + Dropout", "#9fa8da"),
    ])

    tsne_image_block(inner_merg, "belkis_learning",
                     "Analyse de convergence : Stabilité du modèle grâce au Dropout (0.5) et Learning Rate Scheduler.", scroll_fn_merg)

    # ==========================================================
    # ÉTAPE 4 : VALIDATION FINALE (MATRICE & T-SNE)
    # ==========================================================
    divider(inner_merg)
    section_label(inner_merg, "4. Analyse Fine et Espace Latent", "#3f51b5")

    tsne_image_block(inner_merg, "belkis_cm",
                     "Matrice de Confusion : Identification robuste malgré le mélange des niveaux de bruit.", scroll_fn_merg)

    divider(inner_merg)
    section_label(inner_merg, "5. Cartographie de l'Intelligence (t-SNE)", "#3f51b5")

    tsne_image_block(inner_merg, "belkis_tsne",
                     "Projection t-SNE : Séparation sémantique parfaite des familles de drones en conditions réelles.", scroll_fn_merg)

    # Footer
    tk.Label(inner_merg, text="Fin du Rapport Merged SNR", font=("Arial", 9, "italic"), bg="#f5f5f5", fg="#999").pack(pady=20)













# ============================================
# TAB 5 — ML vs DL (placeholder)
# ============================================
def build_tab_mlvsdl(parent):
    inner, _, _ = scrollable_frame(parent)
    tk.Label(inner, text="ML vs Apprentissage Profond — Comparaison finale",
             font=("Arial", 16, "bold"), bg="#f5f5f5", fg="#8e44ad"
             ).pack(pady=30)
    tk.Label(inner,
             text="Cet onglet affichera l'analyse comparative finale\n"
                  "entre le pipeline ML et l'approche CNN\n"
                  "dès que tous les résultats DL seront disponibles.",
             font=("Arial", 11), bg="#e8daef", fg="#4a235a",
             padx=20, pady=20, justify="center"
             ).pack(padx=40)

# ============================================
# MAIN WINDOW
# ============================================
root = tk.Tk()
root.title("Drone RF Classification — Tableau de Bord")
root.geometry("1200x860")
root.configure(bg="#1a1a2e")
root.state('zoomed')  # start maximized

header = tk.Frame(root, bg="#1a1a2e", pady=12)
header.pack(fill=tk.X)
tk.Label(header, text="Drone RF Classification — Tableau de Bord des Résultats",
         font=("Arial", 15, "bold"), bg="#1a1a2e", fg="white").pack()
tk.Label(header, text="Dataset DroneRF  ·  Forêt Aléatoire · KNN · XGBoost · CNN",
         font=("Arial", 9), bg="#1a1a2e", fg="#aaa").pack()

style = ttk.Style()
style.theme_use("default")
style.configure("TNotebook", background="#1a1a2e", borderwidth=0)
style.configure("TNotebook.Tab", font=("Arial", 9, "bold"),
                padding=[14, 8], background="#2c3e50", foreground="white")
style.map("TNotebook.Tab",
          background=[("selected","#2980b9")],
          foreground=[("selected","white")])

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

tab_a    = tk.Frame(notebook, bg="#f5f5f5")
tab_b    = tk.Frame(notebook, bg="#f5f5f5")
tab_comp = tk.Frame(notebook, bg="#f5f5f5")
tab_dl   = tk.Frame(notebook, bg="#f5f5f5")
tab_vs   = tk.Frame(notebook, bg="#f5f5f5")

notebook.add(tab_a,    text="  Exp. A — Type seulement  ")
notebook.add(tab_b,    text="  Exp. B — Type + Mode  ")
notebook.add(tab_comp, text="  Comparaison ML  ")
notebook.add(tab_dl,   text="  Apprentissage Profond  ")
notebook.add(tab_vs,   text="  ML vs DL  ")

# Wait for window to render before building tabs (needed for responsive images)
root.update()

build_tab_a(tab_a)
build_tab_b(tab_b)
build_tab_comparison(tab_comp)
build_tab_dl(tab_dl)
build_tab_mlvsdl(tab_vs)

root.mainloop()