# Contributing to EEG-ML-Workflow

Thanks for your interest in contributing 🙌  
This project is a structured EEG → Machine Learning pipeline, and contributions are welcome.

---

## 🧠 Project Philosophy

This project focuses on:

- A clean, reproducible ML pipeline  
- Modular design (preprocessing → windowing → features → model)  
- Personalized EEG workflows (calibration-based learning)  

It is **not**:
- a neuroscience claim  
- a brain-reading system  
- a place for random experiments without structure  

Try to keep contributions aligned with this direction.

---

## ⚙️ Setup

```bash
git clone https://github.com/baris-talar/EEG-ML-Workflow.git
cd EEG-ML-Workflow

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

---

## 🚀 How to Contribute

Typical workflow:

```
Fork → Create branch → Make changes → Open Pull Request
```

- Please **do not push directly to `main`**
- Use descriptive branch names (e.g. `feature/fft-features`)

---

## 🧩 Project Structure Guidelines

Main code lives in:

```
src/eeg_toolkit/
```

Key modules:

- `pipelines/` → high-level workflows (calibrate / train / predict)
- `features.py` → feature extraction logic
- `preprocess.py` → signal preprocessing
- `windowing.py` → segmentation logic
- `modeling.py` → ML-related utilities
- `artifacts.py` → saving/loading data

### Guidelines

- Try to keep logic **modular and separated**
- Avoid mixing responsibilities across files
- Pipelines should orchestrate — not contain heavy logic

---

## 🧪 ML Pipeline Rules

Please try to follow the pipeline structure:

```
Preprocessing → Windowing → Feature Extraction → Model
```

- Avoid skipping steps unless there is a clear reason  
- Avoid hardcoded shortcuts  
- Keep the workflow reproducible  

If you're unsure, feel free to open an issue first.

---

## 🧼 Coding Guidelines

- Use Python (consistent with existing code)
- Keep functions small and readable
- Avoid duplicating logic — reuse existing modules
- Prefer clarity over cleverness

---

## 🔁 Pull Request Guidelines

When opening a PR:

- Clearly describe what you changed  
- Reference the related issue (if applicable)  
- Keep PRs focused (one feature or fix at a time)  

Small, clean contributions are preferred over large, messy ones.

---

## 📚 Documentation

If your change affects behavior or adds features:

- Please update the `README.md` accordingly  
- Add short explanations where needed  

Good documentation helps others understand and use the project.

---

## 🚫 What to Avoid

- Breaking existing pipeline functionality  
- Adding unrelated or experimental code without context  
- Large unstructured changes  

If you’re unsure about something, open an issue first — happy to discuss.

---

## 💡 Final Note

Contributions don’t have to be perfect.  
Clean ideas, improvements, and experiments are all welcome — just try to keep things consistent with the overall structure.

Thanks for contributing 🚀
