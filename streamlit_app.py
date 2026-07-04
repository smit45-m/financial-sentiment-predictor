"""
Financial Sentiment Predictor — Streamlit App
Next Level UI with advanced CSS, animations, and glassmorphism.
"""

import streamlit as st
import numpy as np
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Sentiment AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Next Level CSS Injection
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Root Styling */
    :root {
        --bg-color: #050505;
        --card-bg: rgba(20, 20, 20, 0.6);
        --border-color: rgba(255, 255, 255, 0.08);
        --accent-glow: rgba(56, 189, 248, 0.15);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --positive: #10b981;
        --negative: #ef4444;
        --neutral: #f59e0b;
    }

    /* Force dark theme and base font */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.1) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.1) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-main);
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, span, p, div {
        font-family: 'Outfit', sans-serif !important;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out forwards;
    }

    .subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 3rem;
        animation: fadeInUp 1s ease-out forwards;
    }

    /* Premium Glass Cards */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease, border-color 0.3s ease;
        animation: zoomIn 0.5s ease-out forwards;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 20px var(--accent-glow);
        border-color: rgba(255, 255, 255, 0.15);
    }

    /* Sentiment Badges */
    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 0 15px currentColor;
    }

    .sentiment-positive {
        color: var(--positive);
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }

    .sentiment-negative {
        color: var(--negative);
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }

    .sentiment-neutral {
        color: var(--neutral);
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
    }

    /* Big Numbers */
    .confidence-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 1rem 0;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
    }

    .model-tag {
        position: absolute;
        top: -12px;
        left: 24px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        color: white;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 4px 10px rgba(56, 189, 248, 0.3);
    }

    /* Streamlit Overrides */
    
    /* Text Area */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: white !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    }

    /* Button Override */
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.5) !important;
        background: linear-gradient(135deg, #7dd3fc, #a5b4fc) !important;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: rgba(10, 10, 10, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Progress bars custom height and glow */
    .stProgress .st-bo {
        background-color: rgba(255, 255, 255, 0.1);
        height: 8px;
        border-radius: 4px;
    }
    
    .prob-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .prob-label-text {
        font-weight: 500;
        color: var(--text-muted);
        font-size: 0.9rem;
    }
    .prob-pct-text {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Animations */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes zoomIn {
        0% { opacity: 0; transform: scale(0.95); }
        100% { opacity: 1; transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Basic text preprocessing matching the training pipeline."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Model Loading (Cached)
# ---------------------------------------------------------------------------
LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "saved_models"


@st.cache_resource
def load_classical_models():
    """Load TF-IDF + classical ML models."""
    import joblib
    models = {}
    tfidf = None
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    if vec_path.exists():
        tfidf = joblib.load(vec_path)
    lr_path = MODELS_DIR / "logistic_regression.joblib"
    if lr_path.exists():
        models["logistic_regression"] = joblib.load(lr_path)
    xgb_path = MODELS_DIR / "xgboost_model.joblib"
    if xgb_path.exists():
        try:
            models["xgboost"] = joblib.load(xgb_path)
        except Exception:
            pass
    return tfidf, models


@st.cache_resource
def load_finbert():
    """Load FinBERT transformer pipeline."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
        )
        return pipe
    except Exception as e:
        logger.warning("Could not load FinBERT: %s", e)
        return None


# ---------------------------------------------------------------------------
# Prediction Functions
# ---------------------------------------------------------------------------
def predict_classical(text: str, model_name: str, tfidf, models):
    if tfidf is None or model_name not in models:
        return None
    cleaned = clean_text(text)
    features = tfidf.transform([cleaned])
    model = models[model_name]
    probas = model.predict_proba(features)[0]
    label = int(np.argmax(probas))
    return {
        "sentiment": LABEL_MAP[label],
        "confidence": float(probas[label]),
        "probabilities": {
            name: float(probas[i]) for i, name in enumerate(LABEL_NAMES)
        },
    }


def predict_finbert(text: str, pipe):
    if pipe is None:
        return None
    results = pipe(text[:512])
    if isinstance(results[0], list):
        results = results[0]

    prob_dict = {name: 0.0 for name in LABEL_NAMES}
    finbert_map = {"positive": 2, "neutral": 1, "negative": 0}

    for entry in results:
        lbl = entry["label"].lower()
        if lbl in prob_dict:
            prob_dict[lbl] = float(entry["score"])

    label = max(finbert_map, key=lambda k: prob_dict[k])
    return {
        "sentiment": label,
        "confidence": prob_dict[label],
        "probabilities": prob_dict,
    }


# ---------------------------------------------------------------------------
# UI - Main Structure
# ---------------------------------------------------------------------------
st.markdown('<h1 class="main-title">AI Financial Sentiment Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-gen institutional-grade market intelligence, powered by FinBERT & XGBoost.</p>', unsafe_allow_html=True)

# Load models
with st.spinner("Initializing AI Core..."):
    tfidf, classical_models = load_classical_models()
    finbert_pipe = load_finbert()

available_models = {}
if "logistic_regression" in classical_models:
    available_models["Logistic Regression"] = "logistic_regression"
if "xgboost" in classical_models:
    available_models["XGBoost"] = "xgboost"
if finbert_pipe is not None:
    available_models["FinBERT (Transformer)"] = "finbert"

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.markdown("---")

    model_choice = st.selectbox(
        "Select Active Engine",
        options=list(available_models.keys()) + ["🌌 Compare All Engines"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🔌 Core Systems Status")
    for name in available_models:
        st.markdown(f"🟢 {name} `ONLINE`")

    if finbert_pipe is None:
        st.markdown("🔴 FinBERT `OFFLINE`")

    st.markdown("---")
    st.markdown("### 🚀 Example Scenarios")
    examples = {
        "Bullish Earnings": "Revenue exceeded expectations with a 23% year-over-year increase, driven by strong enterprise cloud adoption.",
        "Bearish Headwinds": "Operating margins contracted significantly due to rising material costs and unfavorable foreign exchange headwinds.",
        "Neutral Guidance": "The company maintained its full-year EPS guidance of $4.20 to $4.40, assuming current macro conditions persist.",
    }

    selected_example = None
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            selected_example = text

# Input Layout
st.markdown("<br>", unsafe_allow_html=True)
col_in1, col_in2, col_in3 = st.columns([1, 6, 1])

with col_in2:
    default_text = selected_example if selected_example else ""
    text_input = st.text_area(
        "Financial Text Input",
        value=default_text,
        height=180,
        placeholder="Inject earnings reports, SEC filings, or financial news here...",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🚀 INITIATE ANALYSIS", use_container_width=True)

# ---------------------------------------------------------------------------
# Output / Results
# ---------------------------------------------------------------------------
if analyze_clicked and text_input.strip():
    models_to_run = []

    if model_choice == "🌌 Compare All Engines":
        models_to_run = list(available_models.values())
    else:
        models_to_run = [available_models[model_choice]]

    results = {}
    for model_key in models_to_run:
        with st.spinner(f"Computing forward pass on {model_key}..."):
            if model_key == "finbert":
                res = predict_finbert(text_input, finbert_pipe)
            else:
                res = predict_classical(text_input, model_key, tfidf, classical_models)

            if res:
                results[model_key] = res

    if not results:
        st.error("Engine failure. No models could process the input.")
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Center the results gracefully
        num_models = len(results)
        cols = st.columns(num_models)
        
        for idx, (model_key, res) in enumerate(results.items()):
            with cols[idx]:
                sentiment = res["sentiment"]
                confidence = res["confidence"]
                probs = res["probabilities"]

                display_name = model_key.replace("_", " ").title()
                if model_key == "finbert":
                    display_name = "FinBERT"

                st.markdown(f"""
                <div style="position: relative; margin-top: 20px;">
                    <div class="glass-card">
                        <div class="model-tag">{display_name}</div>
                        <div style="text-align: center; margin-top: 15px;">
                            <span class="sentiment-badge sentiment-{sentiment}">{sentiment}</span>
                            <div class="confidence-value">{confidence*100:.2f}%</div>
                        </div>
                        <hr style="border-color: var(--border-color); margin: 1.5rem 0;">
                        <div>
                            <div class="prob-row">
                                <span class="prob-label-text">Positive</span>
                                <span class="prob-pct-text" style="color: var(--positive)">{probs['positive']*100:.1f}%</span>
                            </div>
                            <div class="prob-row">
                                <span class="prob-label-text">Neutral</span>
                                <span class="prob-pct-text" style="color: var(--neutral)">{probs['neutral']*100:.1f}%</span>
                            </div>
                            <div class="prob-row">
                                <span class="prob-label-text">Negative</span>
                                <span class="prob-pct-text" style="color: var(--negative)">{probs['negative']*100:.1f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif analyze_clicked:
    st.warning("⚠️ Input sequence empty. Awaiting telemetry.")

