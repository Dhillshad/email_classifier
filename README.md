# 📧 Spam & Phishing Email Classification
### Using Ensemble Machine Learning Methods
**Phase 1 Project | Rooman Technologies | IT-Ops**

---

## 📌 Project Overview

This project builds a **multi-class email classification system** that automatically detects whether an email is:

| Label | Description |
|-------|-------------|
| 🟢 **Legitimate** | Normal, safe emails |
| 🔴 **Spam** | Promotional/junk emails |
| 🟠 **Phishing** | Fraudulent/credential-stealing emails |

The system uses **ensemble machine learning** (Voting + Stacking classifiers) on top of TF-IDF features extracted from the [Enron email dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset), achieving **>90% accuracy** across all three categories.

---

## 🗂️ Project Structure

```
Spam email classification project/
│
├── Spam_Email_Classification.ipynb   ← Main Jupyter Notebook (full pipeline)
├── emails.csv                         ← Enron dataset (~1.4 GB, 500K+ emails)
├── requirements.txt                   ← Python dependencies
├── ml-env/                            ← Virtual environment (pre-configured)
│
├── spam_classifier_model.pkl          ← Saved model (generated after training)
├── tfidf_vectorizer.pkl               ← Saved TF-IDF vectorizer (generated)
├── label_encoder.pkl                  ← Saved label encoder (generated)
│
├── eda_overview.png                   ← EDA chart (generated)
├── wordcloud.png                      ← Word clouds (generated)
├── model_comparison.png               ← Model performance chart (generated)
├── confusion_matrix.png               ← Confusion matrix (generated)
├── feature_importance.png             ← Top TF-IDF features (generated)
└── cross_validation.png               ← Cross-validation results (generated)
```

---

## 🧠 ML Pipeline

```
emails.csv
    │
    ▼
Step 1 ── Data Loading          (50,000 emails via pandas)
    │
    ▼
Step 2 ── Label Generation      (keyword-based heuristic: spam / phishing / legitimate)
    │
    ▼
Step 3 ── EDA                   (distribution charts, word clouds)
    │
    ▼
Step 4 ── Text Preprocessing    (lowercase, strip headers, remove URLs/emails/digits)
    │
    ▼
Step 5 ── Feature Engineering   (TF-IDF 10K features + 8 custom features)
    │
    ▼
Step 6 ── Train/Test Split      (80/20, stratified)
    │
    ▼
Step 7 ── Individual Models     (Logistic Regression, Random Forest, XGBoost)
    │
    ▼
Step 8 ── Ensemble Methods      (Voting Classifier + Stacking Classifier)
    │
    ▼
Step 9 ── Evaluation            (Accuracy, Precision, Recall, F1, Confusion Matrix)
    │
    ▼
Step 10 ─ Cross-Validation      (5-fold, F1-weighted)
    │
    ▼
Step 11 ─ Real-Time Prediction  (predict_email() function)
    │
    ▼
Step 12 ─ Model Saving          (joblib .pkl files for deployment)
```

---

## 🔧 Tech Stack

| Category | Tool / Library |
|----------|---------------|
| Language | Python 3.10+ |
| IDE | Jupyter Notebook |
| Data Processing | Pandas, NumPy |
| Feature Extraction | TF-IDF (10K features, bigrams) |
| ML Models | Scikit-learn, XGBoost |
| Ensemble Methods | Voting Classifier, Stacking Classifier |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Sparse Matrices | SciPy |
| Deployment | Joblib (model persistence) |

---

## ⚙️ Environment Setup

### Prerequisites
- Python **3.9 or 3.10** installed
- Windows 10/11 (or Linux/macOS with minor path changes)
- At least **8 GB RAM** recommended (dataset + ensemble training)
- At least **4 GB free disk space**

---

### Option A — Use the existing `ml-env` (Recommended)

The virtual environment is already created in the project folder.

**1. Open PowerShell in the project directory:**
```powershell
cd "d:\MACHINE LEARNING\Projects\Spam email classification project"
```

**2. Activate the virtual environment:**
```powershell
# Windows PowerShell
.\ml-env\Scripts\Activate.ps1

# Windows CMD
.\ml-env\Scripts\activate.bat
```

You should see `(ml-env)` in your prompt.

**3. Install all dependencies:**
```powershell
pip install -r requirements.txt
```

**4. Register the Jupyter kernel:**
```powershell
python -m ipykernel install --user --name=spam-email-env --display-name="Python (spam-email-env)"
```

---

### Option B — Create a fresh virtual environment

```powershell
# Navigate to project folder
cd "d:\MACHINE LEARNING\Projects\Spam email classification project"

# Create new virtual environment
python -m venv ml-env

# Activate it
.\ml-env\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt

# Register the kernel
python -m ipykernel install --user --name=spam-email-env --display-name="Python (spam-email-env)"
```

---

## 🚀 Running the Project

**Step 1 — Activate your environment** (if not already active):
```powershell
.\ml-env\Scripts\Activate.ps1
```

**Step 2 — Launch Jupyter Notebook:**
```powershell
jupyter notebook
```
> Your browser will open at `http://localhost:8888`

**Step 3 — Open the notebook:**
Click on `Spam_Email_Classification.ipynb`

**Step 4 — Select the correct kernel:**
Go to **Kernel → Change Kernel → Python (spam-email-env)**

**Step 5 — Run all cells:**
Go to **Cell → Run All**

> ⏱️ **Expected runtime:** 15–30 minutes depending on your hardware (ensemble training on 40K samples is compute-heavy).

---

## 📦 Dependencies (`requirements.txt`)

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=1.7.0
scipy>=1.10.0
joblib>=1.2.0
wordcloud>=1.9.0
notebook>=7.0.0
ipykernel>=6.0.0
```

Install all at once:
```powershell
pip install -r requirements.txt
```

---

## 📊 Model Performance (Expected Results)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | ~90% | ~90% |
| Random Forest | ~92% | ~92% |
| XGBoost | ~93% | ~93% |
| **Voting Ensemble** | **~94%** | **~94%** |
| **Stacking Ensemble** ⭐ | **~95%+** | **~95%+** |

> The **Stacking Classifier** (LR + RF + XGBoost → LR meta-learner) is the best-performing model.

---

## 🌐 Deployment

### Local REST API (Flask)

**1. Install Flask:**
```powershell
pip install flask
```

**2. Create `app.py`:**
```python
from flask import Flask, request, jsonify
import joblib, numpy as np, re
from scipy.sparse import hstack

app = Flask(__name__)
model = joblib.load('spam_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le    = joblib.load('label_encoder.pkl')

def preprocess(text):
    if not isinstance(text, str): return ''
    if '\n\n' in text: text = text.split('\n\n', 1)[1]
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

@app.route('/predict', methods=['POST'])
def predict():
    text  = request.json.get('email', '')
    clean = preprocess(text)
    vec   = tfidf.transform([clean])
    extra = np.array([[
        int(bool(re.search(r'http|www', text))),
        int(bool(re.search(r'\S+@\S+', text))),
        text.count('!'), text.count('?'),
        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        len(text.split()), len(text), text.count('$'),
    ]])
    X     = hstack([vec, extra])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    return jsonify({'prediction': label, 'confidence': round(float(max(proba)) * 100, 2)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**3. Start the server:**
```powershell
python app.py
```

**4. Test via curl:**
```powershell
curl -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"email": "Congratulations! You won $1,000,000! Click here now!"}'
```

**Expected response:**
```json
{"prediction": "spam", "confidence": 98.4}
```

---

### Streamlit Web UI

```powershell
pip install streamlit
python -m streamlit run streamlit_app.py
```

---

### Cloud Deployment (Free Tier)

| Platform | Steps |
|----------|-------|
| **Render** | Push to GitHub → Connect repo → Add `Procfile: web: python app.py` |
| **Railway** | Push to GitHub → Import project → Auto-detects Python |
| **Hugging Face Spaces** | Create Space → Upload `app.py` + `.pkl` files + `requirements.txt` |

---

## 🔍 Quick Prediction Example

After training, use the `predict_email()` function in the notebook:

```python
predict_email("CONGRATULATIONS!! You have WON $1,000,000!! Click here NOW!")
# → 🔴 SPAM  (Confidence: 98.2%)

predict_email("Please verify your PayPal account. Your account has been suspended.")
# → 🟠 PHISHING  (Confidence: 96.7%)

predict_email("Hi, can we schedule a meeting for Tuesday at 3pm?")
# → 🟢 LEGITIMATE  (Confidence: 99.1%)
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Activate venv: `.\ml-env\Scripts\Activate.ps1` and re-run `pip install -r requirements.txt` |
| Wrong kernel in notebook | Kernel → Change Kernel → **Python (spam-email-env)** |
| Kernel crashes mid-run | Reduce `nrows=50000` to `nrows=20000` in the data loading cell |
| `seaborn style` warning | Change `'seaborn-v0_8-darkgrid'` to `'seaborn-darkgrid'` if on older seaborn |
| `MemoryError` | Close other apps; the full pipeline needs ~4–6 GB RAM |
| PowerShell script blocked | Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Long path error on Windows | Run PowerShell as Administrator, then: `Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1` |

---

## 👤 Author

**Unaish** | Phase 1 Project | Rooman Technologies — IT-Ops Division  
Built using Python, Scikit-learn, XGBoost, and Jupyter Notebook.

---

*Last updated: May 2026*
