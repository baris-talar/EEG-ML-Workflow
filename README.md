# EEG-ML-Workflow

A lightweight EEG → Machine Learning workflow toolkit built on top of BrainFlow.

## Quick Navigation

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Commands](#commands)
- [Visualization](#visualization)
- [Artifacts](#artifacts)
- [Dependencies](#dependencies)
- [Limitations](#limitations)
- [License](#license)

## Installation

Clone the repository, create a virtual environment, and install the package in **editable** mode. Editable install (`pip install -e .`) puts `src/` on your path and registers the command-line entry points from `pyproject.toml` (`eeg-calibrate`, `eeg-train`, `eeg-predict`, `eeg-visualize`).

```bash
git clone https://github.com/baris-talar/EEG-ML-Workflow.git
cd EEG-ML-Workflow
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows, activate the venv with:

```bat
.venv\Scripts\activate
```

## Quick Start

After installation, run the tools from the same activated environment:

```bash
eeg-calibrate
eeg-train
eeg-predict
eeg-visualize
```

Each command is interactive unless you pipe defaults; follow the prompts.

## Workflow

High-level data flow:

```text
EEG Recording
  → Preprocessing (optional)
  → Windowing
  → Feature Extraction
  → Training
  → Prediction
  → Visualization / Inspection
```

`eeg-calibrate` / `eeg-train` / `eeg-predict` implement the supervised workflow. `eeg-visualize` is a separate inspection tool (see [Visualization](#visualization)).

## Commands

### eeg-calibrate

- Records EEG data (BrainFlow **synthetic board** in the default pipeline).
- Splits the session into **labeled time blocks** you define at the terminal.
- Optional **bandpass** preprocessing.
- Extracts **per-window mean and variance** features and aligns them with block labels.
- Saves a calibration dataset to **`artifacts/eeg_calibration_dataset.npz`**.

### eeg-train

- Loads **`artifacts/eeg_calibration_dataset.npz`** (path configurable when prompted).
- Trains **logistic regression** (scikit-learn).
- Prints a simple **accuracy** split on held-out data.
- Saves a model bundle to **`artifacts/eeg_logreg_model.joblib`** (path configurable when prompted).

### eeg-predict

- Records a **new** EEG session with BrainFlow.
- Applies preprocessing and feature extraction consistent with the saved model metadata.
- Loads **`artifacts/eeg_logreg_model.joblib`** and predicts **labels per window**.
- Supports an optional **synthetic validation** mode (injected reference states) for sanity checks.
- May open **matplotlib** figures when synthetic validation is enabled.

### eeg-visualize

- Streams EEG from the **synthetic board** for a fixed duration (~20 seconds).
- Writes **human-readable text reports** (no training or prediction).
- Outputs:
  - **`artifacts/raw_eeg_preview.txt`**
  - **`artifacts/preprocessed_eeg_preview.txt`**
- Useful for understanding raw traces, **3–45 Hz** bandpass preprocessing, **5 s** blocks, **1 s** windows, and **mean / variance** feature layout.

## Visualization

`eeg-visualize` is an **educational and debugging** helper. It does **not** train models or run classifiers. It saves two text files that summarize shapes, sample previews, per-channel statistics, block/window structure, and an example mean–variance feature vector for the first window of block 0—once for **raw** data and once for **bandpass-filtered** data.

## Artifacts

Generated files typically live under **`artifacts/`** (the directory is created by the tools when needed). Examples:

| File | Produced by |
|------|----------------|
| `artifacts/eeg_calibration_dataset.npz` | `eeg-calibrate` |
| `artifacts/eeg_logreg_model.joblib` | `eeg-train` |
| `artifacts/raw_eeg_preview.txt` | `eeg-visualize` |
| `artifacts/preprocessed_eeg_preview.txt` | `eeg-visualize` |

Your `.gitignore` may ignore some of these; keep local copies for your own runs.

## Dependencies

Declared in `pyproject.toml` and installed with `pip install -e .`:

- **numpy** — arrays and numerics  
- **scikit-learn** — logistic regression and metrics  
- **brainflow** — synthetic board streaming and `DataFilter` preprocessing  
- **matplotlib** — optional plots in `eeg-predict` (synthetic validation)  
- **joblib** — model artifact save/load used by the toolkit  

## Limitations

- **Not a medical or diagnostic system** — for experimentation and learning only.  
- Default workflows target BrainFlow’s **synthetic board**; real hardware would need different board configuration and protocols.  
- Model quality **depends on calibration** design, labels, and feature choices.  
- The toolkit does **not** “read thoughts” or infer mental content; it performs **supervised learning** on features you define during calibration.  

## License

MIT
