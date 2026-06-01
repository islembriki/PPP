# DroneRF Detection: Classification Dashboard

A comprehensive machine learning and deep learning project for **RF signal-based drone type and mode classification**.

---

## 📋 Project Overview

This project implements multiple classification approaches to identify drone types and flight modes based on RF (Radio Frequency) signal features. It compares:

1. **Machine Learning Models** (Random Forest, KNN, XGBoost)
2. **Deep Learning Models** (CNN with SNR-aware training)

**Best Overall Result:** CNN Expert at 10dB SNR achieving **79.67% accuracy** on 10-class classification.

---

## 🚀 Quick Start: Running the Dashboard

The main deliverable is an interactive Tkinter-based dashboard that visualizes all results.

### Prerequisites
```bash
pip install tkinter pillow numpy matplotlib seaborn xgboost scikit-learn joblib
```

### Launch the Dashboard
```bash
cd Scripts/
python dashboard.py
```

The dashboard will display:
- ✅ ML Model comparisons (Experiment A & B)
- ✅ Deep Learning CNN performance by SNR
- ✅ Confusion matrices and performance metrics
- ✅ t-SNE visualizations
- ✅ ML vs DL comparative analysis

---

## 📁 Repository Structure

```
PPP/
├── Scripts/
│   ├── dashboard.py                      ← MAIN FILE TO RUN
│   ├── ML/
│   │   ├── train_ml.py                   # ML training (SVM, RF, KNN)
│   │   ├── train_mode.py                 # Multi-class (type + mode) training
│   │   ├── approche_type/
│   │   │   ├── rf_training.py            # Random Forest (Type only)
│   │   │   └── xgboost_train_type_only.py # XGBoost (Type only)
│   │   └── ...
│   └── DL/
│       ├── Separate_SNR_Models_Training/  # Expert models per SNR level
│       ├── Architecture_CNN/              # Core CNN architecture
│       └── ...
│
├── ml_trained_models_type_only/          # ML results (4 classes: type only)
│   ├── rf_results.json
│   ├── knn_results.json
│   └── xgboost_type_results.json
│
├── ml_trained_models_mode_included/      # ML results (10 classes: type + mode)
│   ├── rf_mode.json
│   ├── knn_mode.json
│   └── xgboost_mode.json
│
├── tsne/                                 # t-SNE visualizations
│   ├── tsne_avant_type_only.png
│   ├── tsne_avant_type_mode.png
│   ├── tsne_rf_post_type_only.png
│   └── ...
│
├── results/                              # Deep Learning outputs
│   ├── cnn_results.json
│   ├── learning_curves_final.png
│   ├── MATRICE_FINALE_85PC.png
│   └── tsne_CLEAN_WHITE.png
│
└── processed data/                       # Preprocessed datasets
```

---

## 🎯 Key Results

### Experiment A: Drone Type Classification (4 Classes)
- **Background, Bebop, AR_Drone, Phantom**
- **Best ML Model:** Random Forest - **89.3%** accuracy
- **Features Used:** 8 spectral descriptors

| Model | Accuracy |
|-------|----------|
| Random Forest | 89.3% |
| XGBoost | 87.5% |
| KNN (k=52) | 85.2% |

### Experiment B: Type + Mode Classification (10 Classes)
- **4 drone types × 2-3 flight modes each**
- **Best ML Model:** Random Forest - **62.1%** accuracy
- **Challenge:** Scalar descriptors insufficient for intra-brand mode differentiation

| Model | Accuracy |
|-------|----------|
| Random Forest | 62.1% |
| XGBoost | 58.7% |
| KNN (k=5) | 54.3% |

### Deep Learning: CNN Expert Models by SNR
- **Architecture:** Convolutional Neural Network on spectrograms
- **Approach:** Separate experts trained per Signal-to-Noise Ratio

| SNR Level | Accuracy |
|-----------|----------|
| 30dB (Clean Signal) | 77.82% |
| **10dB (Best)** | **79.67%** ⭐ |
| 0dB (Noisy) | 73.69% |
| -10dB (Very Noisy) | 71.00% |

**Gain:** CNN at 10dB (+17.5pp) vs best ML model on 10-class task

---

## 📊 Dashboard Features

### Tab 1: ML Experiment A (Type Only)
- Bar chart of RF, KNN, XGBoost accuracies
- t-SNE pre and post-training visualizations
- Confusion matrices and detailed classification reports

### Tab 2: ML Experiment B (Type + Mode)
- Performance comparison on harder 10-class task
- Analysis of why mode differentiation is challenging
- Same visualization suite as Tab 1

### Tab 3: ML Comparison (Exp. A vs B)
- Side-by-side performance metrics
- Insights on descriptor limitations
- Combined t-SNE and confusion matrix analysis

### Tab 4: Deep Learning Approaches
- **Sub-tab 1 - Separate SNR Models:** Expert CNN trained per SNR level
  - Signal Activity Detection (SAD) methodology
  - Per-SNR confusion matrices
  - Overall performance dashboard
  
- **Sub-tab 2 - Merged SNR Model:** Single generalist CNN on all noise levels
  - SNR-aware weighting strategy
  - Learning curves and convergence
  - Robustness metrics

### Tab 5: ML vs DL Comparison
- Direct accuracy comparison across 10 classes
- Radar chart showing performance profiles
- Key insights on when to use ML vs DL

---

## 🔬 Data & Features

### Input Data
- **RF Signal Characteristics:** Frequency, power, modulation parameters
- **Preprocessing:** Normalization, feature scaling, signal activity detection
- **Dataset Split:** 80% train / 20% test

### ML Features (8 descriptors - Type Only)
1. Mean
2. Variance
3. Power Spectral Density
4. Peak-to-Average Power Ratio (PAPR)
5. Signal Power
6. Variance/PAPR Ratio
7. ...and more

### DL Input
- **Spectrograms:** 2D time-frequency representations
- **SAD Filtering:** 90% noise frame removal
- **SNR-Aware Weighting:** Priority given to harder (low-SNR) samples

---

## 💾 Model Files

Pre-trained models are saved as pickle files:

```python
# ML Models
ml_trained_models_type_only/
  - rf_model.pkl            # Random Forest (4 classes)
  - knn_model.pkl           # KNN (4 classes)
  - xgboost_type_model.pkl  # XGBoost (4 classes)

ml_trained_models_mode_included/
  - rf_model_mode.pkl       # Random Forest (10 classes)
  - knn_model_mode.pkl      # KNN (10 classes)
  - xgboost_mode.pkl        # XGBoost (10 classes)
```

Results are stored as JSON:
```json
{
  "accuracy": 0.8934,
  "target_names": ["Background", "Bebop", "AR_Drone", "Phantom"],
  "classification_report": {...},
  "confusion_matrix": [...],
  "feature_importances": {...}
}
```

---

## 🧠 Technical Highlights

### Machine Learning
- **Algorithms:** Random Forest, KNN, XGBoost
- **Feature Engineering:** 8-12 hand-crafted spectral descriptors
- **Scaling:** StandardScaler for normalization
- **Validation:** Stratified train-test split, confusion matrices

### Deep Learning
- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** 2D Spectrograms from RF signals
- **Noise Handling:** 
  - Signal Activity Detection (SAD) for preprocessing
  - SNR-aware weighted sampling
  - Dropout regularization (p=0.5)
- **Learning:** Adam optimizer, categorical cross-entropy loss
- **Training:** Separate models per SNR OR merged multi-SNR model

### Key Innovation
**CNN Expert Models:** Instead of training one model on mixed noise levels, we train specialized experts for each SNR level. This allows:
- Better signal understanding at each noise regime
- Clear visualization of robustness degradation
- Identification of the "best operating point" (10dB)

---

## 📈 Performance Analysis

### Why ML Struggles with Modes
Scalar descriptors (mean, variance, etc.) compress all time-frequency information into single numbers. Intra-brand flight mode variations (e.g., Phantom hovering vs. Phantom moving) produce subtle spectral shifts invisible to these features.

**Solution:** CNN on spectrograms preserves 2D structure → better mode differentiation.

### Why DL Excels at Noise Robustness
CNNs trained on spectrograms learn hierarchical features:
- **Low layers:** Micro-patterns (individual frequencies)
- **High layers:** Macro-patterns (energy distribution)

At -10dB (signal buried in noise), learned patterns still match drone signatures.

---

## 🎓 For Your Professor

### What to Highlight
1. **Methodological rigor:** Separate experiments (type-only vs. type+mode)
2. **Comparative analysis:** ML vs. DL with clear performance metrics
3. **Root cause analysis:** Understanding limitations of each approach
4. **Innovation:** SNR-expert models showing targeted optimization
5. **Reproducibility:** All results saved as JSON; dashboard loads them interactively

### How to Present
1. Run the dashboard
2. Navigate Tab 5 (ML vs DL Comparison) for executive summary
3. Dive into Tab 4 (Deep Learning) for technical depth
4. Show confusion matrices to explain class-specific challenges

### Key Takeaway
> *"While ML models efficiently classify drone types, deep learning with spectral analysis achieves superior performance and robustness, especially in noisy environments. The 17.5 percentage point improvement on 10-class classification demonstrates the value of learned representations over hand-crafted features."*

---

## 🛠️ Extending the Project

### To retrain models:
```bash
# ML (Type only)
python Scripts/ML/approche_type/rf_training.py

# ML (Type + Mode)
python Scripts/ML/train_mode.py

# DL (Separate SNR experts)
python Scripts/DL/Separate_SNR_Models_Training/train_experts.py

# DL (Merged)
python Scripts/DL/merged_training.py
```

### To generate new dashboards:
Edit `dashboard.py` PATHS dictionary to point to new result files.

---

## 📝 License & Attribution

**Author:** Islem Briki  
**Dataset:** DroneRF  
**Technologies:** Python, scikit-learn, TensorFlow/PyTorch, Tkinter

---

## 📞 Support

For questions about:
- **Model architecture:** See `Scripts/DL/Architecture_CNN/`
- **Feature engineering:** See `Scripts/ML/train_ml.py`
- **Dashboard UI:** See `Scripts/dashboard.py` (well-documented with helper functions)

---

**Last Updated:** June 2026  
**Status:** Complete & Ready for Evaluation ✅
