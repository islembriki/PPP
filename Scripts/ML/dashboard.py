import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DroneJSONDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone RF Identification - Results Dashboard")
        self.root.geometry("1400x850")
        
        # Path to your JSON folder
        self.results_dir = r"C:\Users\HP\Desktop\PPP\ml_trained_models_type_only"
        
        self.models_data = {}
        self.setup_ui()
        
        # Auto-load on startup
        self.load_all_jsons()

    def setup_ui(self):
        # --- TAB CONTROL ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # TAB 1: Summary & Charts
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="📊 Accuracy & Detailed Reports")

        # TAB 2: Confusion Matrices
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="🧩 Confusion Matrices")

        self.setup_tab1()
        self.setup_tab2()

        # --- BOTTOM BUTTONS ---
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill='x', side='bottom', padx=20, pady=10)
        
        ttk.Button(bottom_frame, text="📂 Reload JSONs", command=self.load_all_jsons).pack(side='left', padx=10)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom_frame, textvariable=self.status_var, font=("Arial", 10, "italic")).pack(side='right')

    def setup_tab1(self):
        # Horizontal Split: Left for Graph, Right for Text
        paned = ttk.PanedWindow(self.tab1, orient='horizontal')
        paned.pack(expand=True, fill='both')

        # Chart Zone
        self.fig_bar, self.ax_bar = plt.subplots(figsize=(6, 5))
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=paned)
        paned.add(self.canvas_bar.get_tk_widget(), weight=1)

        # Text Zone
        text_frame = ttk.Frame(paned)
        paned.add(text_frame, weight=1)
        
        ttk.Label(text_frame, text="Detailed Classification Reports", font=("Arial", 12, "bold")).pack(pady=5)
        self.report_area = tk.Text(text_frame, font=("Courier New", 10), bg="#f8f9fa", wrap='none')
        self.report_area.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Scrollbars
        yscroll = ttk.Scrollbar(self.report_area, orient='vertical', command=self.report_area.yview)
        yscroll.pack(side='right', fill='y')
        self.report_area['yscrollcommand'] = yscroll.set

    def setup_tab2(self):
        # Space for Confusion Matrix heatmaps
        self.fig_cm, self.axes_cm = plt.subplots(1, 2, figsize=(14, 6))
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=self.tab2)
        self.canvas_cm.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)

    def load_all_jsons(self):
        """Loads specific JSON files from your path"""
        self.models_data = {}
        files = {
            "Random Forest": "rf_results.json",
            "KNN": "knn_results.json"
        }

        try:
            for display_name, filename in files.items():
                path = os.path.join(self.results_dir, filename)
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        self.models_data[display_name] = json.load(f)
                else:
                    print(f"Warning: {filename} not found in {self.results_dir}")

            if self.models_data:
                self.update_visuals()
                self.status_var.set(f"Successfully loaded {len(self.models_data)} results.")
            else:
                self.status_var.set("Error: No JSON files found.")
                
        except Exception as e:
            messagebox.showerror("JSON Load Error", str(e))

    def format_report_dict(self, report_dict):
        """Converts the JSON report dictionary into a pretty table string"""
        lines = [f"{'Class':<15} | {'Prec.':<10} | {'Recall':<10} | {'F1':<10}"]
        lines.append("-" * 55)
        
        for key, metrics in report_dict.items():
            if isinstance(metrics, dict):
                lines.append(f"{key:<15} | {metrics['precision']:<10.2f} | {metrics['recall']:<10.2f} | {metrics['f1-score']:<10.2f}")
        
        return "\n".join(lines)

    def update_visuals(self):
        # 1. Update Accuracy Bar Chart
        self.ax_bar.clear()
        names = list(self.models_data.keys())
        # Note: RF JSON uses 'accuracy' value 82.707, KNN uses 72.73
        # We normalize to 100 scale
        accs = []
        for n in names:
            a = self.models_data[n]['accuracy']
            accs.append(a if a > 1 else a * 100) # handles both 0.82 and 82.7
            
        bars = self.ax_bar.bar(names, accs, color=['#3498db', '#2ecc71'])
        self.ax_bar.set_title("ML Accuracy Comparison", fontsize=14, fontweight='bold')
        self.ax_bar.set_ylim(0, 105)
        self.ax_bar.set_ylabel("Accuracy (%)")
        
        for bar in bars:
            self.ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, 
                             f"{bar.get_height():.2f}%", ha='center', va='bottom', fontweight='bold')
        self.canvas_bar.draw()

        # 2. Update Detailed Reports
        self.report_area.delete('1.0', tk.END)
        for name in names:
            self.report_area.insert(tk.END, f"========================================\n")
            self.report_area.insert(tk.END, f" MODEL: {name.upper()}\n")
            self.report_area.insert(tk.END, f"========================================\n")
            # RF uses 'classification_report', KNN uses 'classification_report' (dict format)
            report = self.models_data[name]['classification_report']
            self.report_area.insert(tk.END, self.format_report_dict(report) + "\n\n")
        
        # 3. Update Confusion Matrices
        for i, name in enumerate(names):
            ax = self.axes_cm[i]
            ax.clear()
            cm = np.array(self.models_data[name]['confusion_matrix'])
            # Normalizing CM for colors
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                        xticklabels=['BG', 'Bebop', 'AR', 'Phantom'],
                        yticklabels=['BG', 'Bebop', 'AR', 'Phantom'])
            ax.set_title(f"Confusion Matrix: {name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            
        self.fig_cm.tight_layout()
        self.canvas_cm.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneJSONDashboard(root)
    root.mainloop()