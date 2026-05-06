import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Email Classifier | Spam & Phishing Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hide Streamlit default header */
#MainMenu, footer, header { visibility: hidden; }

/* Hero section */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
}

/* Glassmorphism card */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(10px);
    margin: 1rem 0;
}

/* Result boxes */
.result-spam {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05));
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-phishing {
    background: linear-gradient(135deg, rgba(251,146,60,0.2), rgba(251,146,60,0.05));
    border: 2px solid #fb923c;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-legitimate {
    background: linear-gradient(135deg, rgba(52,211,153,0.2), rgba(52,211,153,0.05));
    border: 2px solid #34d399;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-label {
    font-size: 2rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.result-conf {
    color: #cbd5e1;
    font-size: 1rem;
    font-weight: 500;
}

/* Metric pill */
.metric-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 50px;
    padding: 0.4rem 1rem;
    font-size: 0.85rem;
    color: #e2e8f0;
    margin: 0.3rem;
}

/* Prob bar */
.prob-bar-wrap { margin: 0.5rem 0; }
.prob-label {
    display: flex;
    justify-content: space-between;
    color: #cbd5e1;
    font-size: 0.85rem;
    margin-bottom: 4px;
}
.prob-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* Textarea styling override */
textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #f1f5f9 !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.4);
}

/* Sample emails */
.sample-tag {
    display: inline-block;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 0.2rem;
    cursor: pointer;
}

/* Footer */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
}

/* Section label */
.section-label {
    color: #7c3aed;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}

/* Stats row */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.stat-item {
    text-align: center;
}
.stat-number {
    font-size: 1.6rem;
    font-weight: 800;
    color: #a78bfa;
}
.stat-text {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load("spam_classifier_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    le    = joblib.load("label_encoder.pkl")
    return model, tfidf, le

model, tfidf, le = load_models()


# ── Helpers ───────────────────────────────────────────────────
def preprocess(text):
    if not isinstance(text, str): return ''
    if '\n\n' in text: text = text.split('\n\n', 1)[1]
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(text):
    clean = preprocess(text)
    vec   = tfidf.transform([clean])
    extra = np.array([[
        int(bool(re.search(r'http|www', text))),
        int(bool(re.search(r'\S+@\S+', text))),
        text.count('!'),
        text.count('?'),
        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        len(text.split()),
        len(text),
        text.count('$'),
    ]])
    X     = hstack([vec, extra])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    proba_dict = dict(zip(le.classes_, proba))
    return label, proba_dict


# ── UI ────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <h1>📧 Email Classifier</h1>
    <p>AI-powered spam & phishing detection using Ensemble Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown("""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-number">95%+</div>
        <div class="stat-text">Accuracy</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">50K</div>
        <div class="stat-text">Emails Trained</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">3</div>
        <div class="stat-text">Categories</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">5</div>
        <div class="stat-text">ML Models</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Session State Init ──────────────────────────────────────
if "email_text" not in st.session_state:
    st.session_state.email_text = ""
if "result" not in st.session_state:
    st.session_state.result = None

# Sample emails for quick testing
SAMPLES = {
    "🔴 Spam Example"     : "CONGRATULATIONS!! You have WON $1,000,000!! Click here NOW to claim your FREE prize! Limited time offer! Act now! Buy now and save big!",
    "🟠 Phishing Example" : "Dear Customer, Your PayPal account has been suspended due to unusual activity. Please verify your account and confirm your credentials by clicking the link below. Urgent action required.",
    "🟢 Legit Example"    : "Hi John, hope you're doing well. Can we schedule a meeting for Tuesday at 3pm to discuss the quarterly budget report? Please let me know what works for you. Thanks.",
}

st.markdown('<div class="section-label">⚡ Quick Test Samples</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔴 Spam Sample"):
        st.session_state["email_textarea"] = SAMPLES["🔴 Spam Example"]
        st.session_state.result = None
        st.rerun()
with col2:
    if st.button("🟠 Phishing Sample"):
        st.session_state["email_textarea"] = SAMPLES["🟠 Phishing Example"]
        st.session_state.result = None
        st.rerun()
with col3:
    if st.button("🟢 Legit Sample"):
        st.session_state["email_textarea"] = SAMPLES["🟢 Legit Example"]
        st.session_state.result = None
        st.rerun()

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<div class="section-label">✏️ Or paste your own email</div>', unsafe_allow_html=True)

# text_area managed entirely by its key — do NOT pass value= when using key=
email_input = st.text_area(
    label="Email content",
    height=180,
    placeholder="Paste the email content here (subject + body)...",
    label_visibility="collapsed",
    key="email_textarea",
)

col_btn, col_clear = st.columns([4, 1])
with col_btn:
    classify_btn = st.button("🔍 Classify Email", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["email_textarea"] = ""
        st.session_state.result = None
        st.rerun()

# ── Results ───────────────────────────────────────────────────
if classify_btn:
    current_email = st.session_state.get("email_textarea", "")
    if not current_email.strip():
        st.warning("⚠️ Please enter some email text first.")
    else:
        with st.spinner("Analyzing email..."):
            label, proba_dict = predict(current_email)
        st.session_state.result = {
            "label":      label,
            "proba_dict": proba_dict,
            "text":       current_email,
        }

# Display stored result (persists across reruns)
if st.session_state.result:
    res        = st.session_state.result
    label      = res["label"]
    proba_dict = res["proba_dict"]
    email_used = res["text"]

    icons  = {"spam": "🔴", "phishing": "🟠", "legitimate": "🟢"}
    colors = {"spam": "#ef4444", "phishing": "#fb923c", "legitimate": "#34d399"}
    css_cl = {"spam": "result-spam", "phishing": "result-phishing", "legitimate": "result-legitimate"}
    desc   = {
        "spam":       "This email appears to be unsolicited commercial or junk mail.",
        "phishing":   "This email shows signs of a phishing or fraud attempt.",
        "legitimate": "This email appears to be safe and legitimate.",
    }

    confidence = round(proba_dict[label] * 100, 1)

    st.markdown("---")
    st.markdown('<div class="section-label">📊 Classification Result</div>', unsafe_allow_html=True)

    # Main result card
    st.markdown(f"""
    <div class="{css_cl[label]}">
        <div style="font-size:3rem; margin-bottom:0.2rem">{icons[label]}</div>
        <div class="result-label" style="color:{colors[label]}">{label.upper()}</div>
        <div class="result-conf">{desc[label]}</div>
        <br>
        <span class="metric-pill">Confidence: <strong>{confidence}%</strong></span>
        <span class="metric-pill">Words: {len(email_used.split())}</span>
        <span class="metric-pill">Characters: {len(email_used)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability bars
    st.markdown('<div class="section-label">📈 Probability Breakdown</div>', unsafe_allow_html=True)

    bar_colors_map = {
        "legitimate": "#34d399",
        "phishing"  : "#fb923c",
        "spam"      : "#ef4444",
    }

    for cls in ["legitimate", "spam", "phishing"]:
        pct       = round(proba_dict[cls] * 100, 1)
        bar_color = bar_colors_map[cls]
        emoji     = icons[cls]
        st.markdown(f"""
        <div class="prob-bar-wrap">
            <div class="prob-label">
                <span>{emoji} {cls.capitalize()}</span>
                <span><strong>{pct}%</strong></span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Key signals
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔎 Detected Signals</div>', unsafe_allow_html=True)

    signals = []
    if re.search(r'http|www', email_used):      signals.append("🔗 Contains URL")
    if re.search(r'\S+@\S+', email_used):       signals.append("📬 Contains Email Address")
    if email_used.count('!') > 2:               signals.append(f"❗ {email_used.count('!')} Exclamation Marks")
    if email_used.count('$') > 0:               signals.append(f"💰 {email_used.count('$')} Dollar Signs")
    if email_used.count('?') > 2:               signals.append(f"❓ {email_used.count('?')} Question Marks")
    upper_ratio = sum(1 for c in email_used if c.isupper()) / max(len(email_used), 1)
    if upper_ratio > 0.2:                       signals.append(f"🔠 {round(upper_ratio*100)}% Uppercase Text")

    if signals:
        pills = " ".join([f'<span class="metric-pill">{s}</span>' for s in signals])
        st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="metric-pill">✅ No suspicious signals detected</span>', unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("⚙️ How does this work?"):
    st.markdown("""
    This classifier uses a **Stacking Ensemble** — the best-performing model from the pipeline:

    | Step | Details |
    |------|---------|
    | **Dataset** | Enron Email Dataset (50,000 emails) |
    | **Features** | TF-IDF (10K bigrams) + 8 custom signals |
    | **Base Models** | Logistic Regression, Random Forest, XGBoost |
    | **Meta-Learner** | Logistic Regression (stacked on base predictions) |
    | **Accuracy** | 95%+ on held-out test set |

    Labels are assigned using keyword-based heuristics on the Enron dataset, which contains raw corporate emails.
    """)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    📧 Spam & Phishing Email Classifier &nbsp;|&nbsp; Phase 1 Project &nbsp;|&nbsp; Rooman Technologies — IT-Ops<br>
    Built with Python · Scikit-learn · XGBoost · Streamlit
</div>
""", unsafe_allow_html=True)
