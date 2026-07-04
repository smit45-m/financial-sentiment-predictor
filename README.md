# 📈 Financial Earnings Call Sentiment & Market Impact Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F7931E?logo=huggingface&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-sentiment-predictor-s6.streamlit.app/)

A production-quality machine learning system that predicts market sentiment from financial texts, earnings call transcripts, and news headlines. 

Built to demonstrate end-to-end ML engineering, this project compares **Classical ML**, **Deep Learning (BiLSTM)**, and **Transformer (FinBERT)** architectures, deployed via a high-performance FastAPI backend with a modern glassmorphism frontend UI.

---

## ✨ Key Features
- **Multi-Model Comparison**: Evaluates TF-IDF + Logistic Regression, XGBoost, PyTorch BiLSTMs, and HuggingFace FinBERT.
- **Domain-Specific NLP**: Uses models specifically tuned for financial lexicon where standard sentiment analyzers (like VADER) fail (e.g., "losses" = negative, "liabilities" = neutral).
- **Production-Ready API**: FastAPI backend with lazy-loading, CORS, batch prediction, and graceful model fallbacks.
- **Explainable AI**: Integrates SHAP and LIME for feature importance and prediction explainability.
- **Containerized**: Fully Dockerized for seamless deployment.
- **Interactive UI**: Stunning vanilla HTML/CSS/JS glassmorphism dashboard.

---

## 🏗️ Architecture

```text
Raw Text → Text Preprocessing 
             ├── TF-IDF Vectorizer → Logistic Regression / XGBoost
             ├── Custom Tokenizer  → PyTorch BiLSTM
             └── FinBERT Tokenizer → HuggingFace FinBERT 
                   ↓
             Model Ensemble & Evaluation 
                   ↓
             FastAPI Serving Layer 
                   ↓
             Frontend Dashboard
```

---

## 📊 Dataset
Trained and evaluated on the [Financial PhraseBank dataset](https://huggingface.co/datasets/financial_phrasebank) (Malo et al., 2014), which contains ~4,840 sentences from English-language financial news categorized by finance experts.
- **Split**: `sentences_allagree` (highest quality annotations)
- **Labels**: Positive, Negative, Neutral

---

## 🚀 Quick Start

### 1. Local Development (Backend)
```bash
# Clone the repository
git clone https://github.com/yourusername/financial-sentiment-predictor.git
cd financial-sentiment-predictor

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --reload
```
The FastAPI backend will run at `http://localhost:8000`.

### 2. Local Development (Frontend)
The frontend is a standalone, decoupled HTML/JS application located in the `frontend/` directory.
1. Open `frontend/index.html` in your browser.
2. By default, it will detect `localhost` and point its requests to `http://localhost:8000`.

---

## 🌍 Production Deployment

This project uses a modern decoupled architecture.

### 1. Deploy the Backend (FastAPI) to Render / AWS App Runner
The backend is fully Dockerized.
- **Render**: Create a new "Web Service", connect your GitHub repo, and Render will automatically detect the `Dockerfile` and build it.
- **Environment**: You must set `PORT=8000`.

### 2. Deploy the Frontend (UI) to Vercel / Netlify
1. Log in to [Vercel](https://vercel.com/) and click **Add New Project**.
2. Import your GitHub repository.
3. In the Vercel configuration, set the **Root Directory** to `frontend`.
4. Click **Deploy**.
5. *Important*: Once your backend is deployed and gives you a live URL, open `frontend/index.html` in your codebase, change the `API_BASE_URL` constant on line ~275 to your live backend URL, commit, and push. Vercel will instantly update!

---

## 📡 API Reference

- `GET /health`: Check system status and loaded models.
- `POST /predict`: Predict sentiment using a specific model.
  ```json
  {
    "text": "Revenue increased by 20% year-over-year.",
    "model": "xgboost"
  }
  ```
- `POST /predict/all`: Get predictions from all loaded models simultaneously.
- `POST /predict/batch`: Process multiple texts at once.

---

## 💼 Resume Bullet Points

- **Engineered an end-to-end financial NLP pipeline** analyzing earnings transcripts, comparing TF-IDF + XGBoost, PyTorch BiLSTM, and HuggingFace FinBERT architectures.
- **Built a production REST API** with FastAPI and Docker, implementing lazy model loading, graceful degradation, and batch inference capabilities.
- **Implemented Explainable AI (XAI)** using SHAP/LIME to interpret financial market sentiment signals, increasing model transparency for stakeholder review.
- **Designed a responsive frontend dashboard** demonstrating real-time model comparisons with vanilla JS/CSS glassmorphism aesthetics.

---

## 📄 License
MIT License.
