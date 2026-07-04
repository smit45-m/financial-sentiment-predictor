"""
Financial Sentiment Predictor — Streamlit App
Beautiful, interactive dashboard for financial text sentiment analysis.
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
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium glassmorphism look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .sentiment-positive {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }

    .sentiment-negative {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }

    .sentiment-neutral {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }

    .confidence-text {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .model-badge {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .prob-label {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.3s !important;
    }

    .stButton > button:hover {
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-2px) !important;
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
            pass  # xgboost might not be installed

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
    """Predict using a classical ML model."""
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
    """Predict using FinBERT."""
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
# UI
# ---------------------------------------------------------------------------
st.markdown('<h1 class="main-title">📈 Financial Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multi-model NLP system predicting market impact from earnings calls & financial news</p>', unsafe_allow_html=True)

# Load models
with st.spinner("Loading models..."):
    tfidf, classical_models = load_classical_models()
    finbert_pipe = load_finbert()

# Build available models list
available_models = {}
if "logistic_regression" in classical_models:
    available_models["Logistic Regression"] = "logistic_regression"
if "xgboost" in classical_models:
    available_models["XGBoost"] = "xgboost"
if finbert_pipe is not None:
    available_models["FinBERT (Transformer)"] = "finbert"

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    st.markdown("---")

    model_choice = st.selectbox(
        "Select Model",
        options=list(available_models.keys()) + ["🔄 Compare All Models"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📊 Loaded Models")
    for name in available_models:
        st.markdown(f"✅ {name}")

    if finbert_pipe is None:
        st.markdown("❌ FinBERT *(not loaded)*")

    st.markdown("---")
    st.markdown("### 💡 Quick Examples")
    examples = {
        "🟢 Strong Earnings": "Revenue exceeded expectations with a 23% year-over-year increase, driven by strong enterprise cloud adoption.",
        "🔴 Margin Pressure": "Operating margins contracted significantly due to rising material costs and unfavorable foreign exchange headwinds.",
        "🟡 Guidance Maintained": "The company maintained its full-year EPS guidance of $4.20 to $4.40, assuming current macro conditions persist.",
        "🔴 Supply Chain Issues": "Production delays at key manufacturing facilities will negatively impact Q3 deliveries.",
    }

    selected_example = None
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            selected_example = text

# Main Input Area
col1, col2 = st.columns([3, 1])

with col1:
    default_text = selected_example if selected_example else ""
    text_input = st.text_area(
        "Enter financial text to analyze",
        value=default_text,
        height=150,
        placeholder="Enter an earnings call excerpt, financial news headline, or analyst report...",
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True)

# ---------------------------------------------------------------------------
# Run Prediction
# ---------------------------------------------------------------------------
if analyze_clicked and text_input.strip():
    models_to_run = []

    if model_choice == "🔄 Compare All Models":
        models_to_run = list(available_models.values())
    else:
        models_to_run = [available_models[model_choice]]

    results = {}
    for model_key in models_to_run:
        with st.spinner(f"Running {model_key}..."):
            if model_key == "finbert":
                res = predict_finbert(text_input, finbert_pipe)
            else:
                res = predict_classical(text_input, model_key, tfidf, classical_models)

            if res:
                results[model_key] = res

    if not results:
        st.error("No models could process the input. Check model loading status in the sidebar.")
    else:
        st.markdown("---")
        st.markdown("### 🎯 Results")

        cols = st.columns(len(results))
        for idx, (model_key, res) in enumerate(results.items()):
            with cols[idx]:
                sentiment = res["sentiment"]
                confidence = res["confidence"]
                probs = res["probabilities"]

                # Model name formatting
                display_name = model_key.replace("_", " ").title()
                if model_key == "finbert":
                    display_name = "FinBERT"

                st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)

                st.markdown(f'<span class="model-badge">{display_name}</span>', unsafe_allow_html=True)

                st.markdown(f'<br><span class="sentiment-{sentiment}">{sentiment.upper()}</span>', unsafe_allow_html=True)

                st.markdown(f'<p class="confidence-text">{confidence*100:.1f}%</p>', unsafe_allow_html=True)

                # Probability bars
                st.markdown("**Probability Distribution**")

                neg_pct = probs["negative"]
                neu_pct = probs["neutral"]
                pos_pct = probs["positive"]

                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(neg_pct)
                with col_b:
                    st.markdown(f'<span class="prob-label">Neg: {neg_pct*100:.1f}%</span>', unsafe_allow_html=True)

                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(neu_pct)
                with col_b:
                    st.markdown(f'<span class="prob-label">Neu: {neu_pct*100:.1f}%</span>', unsafe_allow_html=True)

                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(pos_pct)
                with col_b:
                    st.markdown(f'<span class="prob-label">Pos: {pos_pct*100:.1f}%</span>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

elif analyze_clicked:
    st.warning("⚠️ Please enter some financial text to analyze.")
