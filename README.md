# EEG-ML-Workflow

A lightweight EEG → Machine Learning workflow toolkit built on top of BrainFlow.

---

## 🔗 Quick Navigation

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Details](#details)
- [Artifacts](#artifacts)
- [Limitations](#limitations)

---

## Installation

git clone https://github.com/baris-talar/EEG-ML-Workflow.git  
cd EEG-ML-Workflow  

python3 -m venv .venv  
source .venv/bin/activate  

pip install -e .  

---

## Quick Start

eeg-calibrate  
eeg-train  
eeg-predict  

---

## Workflow

EEG Recording  
→ Preprocessing (optional)  
→ Windowing  
→ Feature Extraction  
→ Training  
→ Prediction  

---

## Details

<details>
<summary>Calibration (eeg-calibrate)</summary>

- Records EEG data  
- Splits into labeled blocks  
- Applies preprocessing (optional)  
- Extracts features  
- Saves dataset  

Output:  
artifacts/eeg_calibration_dataset.npz  

</details>

---

<details>
<summary>Training (eeg-train)</summary>

- Loads dataset  
- Trains Logistic Regression  
- Evaluates accuracy  
- Saves model  

Output:  
artifacts/eeg_logreg_model.joblib  

</details>

---

<details>
<summary>Prediction (eeg-predict)</summary>

- Records new EEG session  
- Applies same pipeline  
- Loads trained model  
- Predicts labels per window  

Optional synthetic validation mode available.

</details>

---

## Artifacts

All generated files are stored in:  
artifacts/  

Examples:  
- eeg_calibration_dataset.npz  
- eeg_logreg_model.joblib  

---

## Limitations

- Not a medical system  
- Synthetic validation only  
- Depends on calibration quality  
- Does NOT read thoughts  

---

## Dependencies

- numpy  
- scikit-learn  
- brainflow  
- matplotlib  

---

## License

MIT
