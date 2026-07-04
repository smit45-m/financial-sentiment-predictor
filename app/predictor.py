"""
Sentiment Predictor — Unified inference interface for all model types.

Supports:
  - Classical ML: TF-IDF + Logistic Regression, TF-IDF + XGBoost
  - Deep Learning: BiLSTM (PyTorch)
  - Transformer: FinBERT (HuggingFace)

Each predict method returns a standardised dict:
    {"sentiment": str, "label": int, "confidence": float,
     "probabilities": {"negative": float, "neutral": float, "positive": float}}
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]
LABEL_MAP: dict[int, str] = {i: name for i, name in enumerate(LABEL_NAMES)}


class SentimentPredictor:
    """Unified sentiment prediction across Classical ML, BiLSTM, and FinBERT models."""

    def __init__(self) -> None:
        """Initialise with empty model registry."""
        self.models: dict[str, Any] = {}
        self.tfidf_vectorizer: Any | None = None
        self.label_encoder: Any | None = None
        self.bilstm_vocab: dict[str, int] = {}
        self.bilstm_config: dict[str, Any] = {}
        self._finbert_pipeline: Any | None = None
        self._loaded_flags: dict[str, bool] = {
            "classical": False,
            "bilstm": False,
            "finbert": False,
        }

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_classical_models(self, path: str = "saved_models") -> list[str]:
        """Load TF-IDF vectorizer, label encoder, Logistic Regression, and XGBoost.

        Args:
            path: Directory containing the joblib artefacts.

        Returns:
            List of successfully loaded model names.
        """
        import joblib

        base = Path(path)
        loaded: list[str] = []

        # Vectorizer (required for all classical models)
        vectorizer_path = base / "tfidf_vectorizer.joblib"
        if not vectorizer_path.exists():
            logger.warning("TF-IDF vectorizer not found at %s — skipping classical models.", vectorizer_path)
            return loaded
        self.tfidf_vectorizer = joblib.load(vectorizer_path)
        logger.info("Loaded TF-IDF vectorizer from %s", vectorizer_path)

        # Label encoder (optional — fall back to LABEL_MAP)
        le_path = base / "label_encoder.joblib"
        if le_path.exists():
            self.label_encoder = joblib.load(le_path)
            logger.info("Loaded label encoder from %s", le_path)

        # Logistic Regression
        lr_path = base / "logistic_regression.joblib"
        if lr_path.exists():
            self.models["logistic_regression"] = joblib.load(lr_path)
            loaded.append("logistic_regression")
            logger.info("Loaded Logistic Regression from %s", lr_path)

        # XGBoost
        xgb_path = base / "xgboost_model.joblib"
        if xgb_path.exists():
            self.models["xgboost"] = joblib.load(xgb_path)
            loaded.append("xgboost")
            logger.info("Loaded XGBoost from %s", xgb_path)

        self._loaded_flags["classical"] = len(loaded) > 0
        return loaded

    def load_bilstm(self, path: str = "saved_models") -> bool:
        """Load BiLSTM model, vocabulary, and config.

        Args:
            path: Directory containing bilstm_model.pt, vocab.json, bilstm_config.json.

        Returns:
            True if successfully loaded, False otherwise.
        """
        import torch

        base = Path(path)

        # Config
        config_path = base / "bilstm_config.json"
        if not config_path.exists():
            logger.warning("BiLSTM config not found at %s — skipping.", config_path)
            return False
        with open(config_path, "r", encoding="utf-8") as f:
            self.bilstm_config = json.load(f)

        # Vocabulary
        vocab_path = base / "vocab.json"
        if not vocab_path.exists():
            logger.warning("BiLSTM vocab not found at %s — skipping.", vocab_path)
            return False
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.bilstm_vocab = json.load(f)

        # Model weights
        model_path = base / "bilstm_model.pt"
        if not model_path.exists():
            logger.warning("BiLSTM weights not found at %s — skipping.", model_path)
            return False

        try:
            from src.deep_learning import BiLSTMClassifier
        except ImportError:
            logger.warning(
                "Could not import BiLSTMClassifier from src.deep_learning. "
                "Defining a fallback BiLSTM architecture."
            )
            BiLSTMClassifier = self._get_fallback_bilstm_class()

        model = BiLSTMClassifier(
            vocab_size=self.bilstm_config.get("vocab_size", len(self.bilstm_vocab)),
            embedding_dim=self.bilstm_config.get("embedding_dim", 128),
            hidden_dim=self.bilstm_config.get("hidden_dim", 256),
            output_dim=self.bilstm_config.get("output_dim", 3),
            n_layers=self.bilstm_config.get("n_layers", 2),
            dropout=self.bilstm_config.get("dropout", 0.3),
            bidirectional=self.bilstm_config.get("bidirectional", True),
            pad_idx=self.bilstm_config.get("pad_idx", 0),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self.models["bilstm"] = model
        self._loaded_flags["bilstm"] = True
        logger.info("Loaded BiLSTM model from %s (device=%s)", model_path, device)
        return True

    def load_finbert(self) -> bool:
        """Load FinBERT sentiment pipeline from HuggingFace.

        Returns:
            True if successfully loaded, False otherwise.
        """
        try:
            from transformers import pipeline

            self._finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,  # return all class probabilities
            )
            self.models["finbert"] = self._finbert_pipeline
            self._loaded_flags["finbert"] = True
            logger.info("Loaded FinBERT pipeline from ProsusAI/finbert.")
            return True
        except Exception as exc:
            logger.warning("Failed to load FinBERT: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_available_models(self) -> list[str]:
        """Return names of all currently loaded models."""
        return list(self.models.keys())

    def predict(self, text: str, model_name: str) -> dict[str, Any]:
        """Route prediction to the appropriate model.

        Args:
            text: Input text to classify.
            model_name: One of 'logistic_regression', 'xgboost', 'bilstm', 'finbert'.

        Returns:
            Prediction dict with sentiment, label, confidence, and probabilities.

        Raises:
            ValueError: If model_name is not loaded.
        """
        if model_name not in self.models:
            available = self.get_available_models()
            raise ValueError(
                f"Model '{model_name}' is not available. "
                f"Loaded models: {available}"
            )

        cleaned = self.clean_text(text)

        if model_name in ("logistic_regression", "xgboost"):
            return self._predict_classical(cleaned, model_name)
        elif model_name == "bilstm":
            return self._predict_bilstm(cleaned)
        elif model_name == "finbert":
            return self._predict_finbert(text)  # FinBERT uses raw text
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def predict_all(self, text: str) -> dict[str, dict[str, Any]]:
        """Predict with every available model and return combined results.

        Args:
            text: Input text to classify.

        Returns:
            Dict mapping model_name → prediction dict.
        """
        results: dict[str, dict[str, Any]] = {}
        for model_name in self.get_available_models():
            try:
                results[model_name] = self.predict(text, model_name)
            except Exception as exc:
                logger.error("Prediction failed for %s: %s", model_name, exc)
                results[model_name] = {"error": str(exc)}
        return results

    # ------------------------------------------------------------------
    # Text Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text preprocessing mirroring src/preprocessing.py logic.

        Steps:
            1. Lowercase
            2. Remove URLs
            3. Remove HTML tags
            4. Remove special characters (keep letters, digits, whitespace)
            5. Collapse whitespace
            6. Strip leading/trailing whitespace
        """
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Private Prediction Methods
    # ------------------------------------------------------------------

    def _predict_classical(self, text: str, model_name: str) -> dict[str, Any]:
        """Predict using a classical ML model (LR or XGBoost).

        Args:
            text: Pre-cleaned input text.
            model_name: 'logistic_regression' or 'xgboost'.

        Returns:
            Standardised prediction dict.
        """
        model = self.models[model_name]
        features = self.tfidf_vectorizer.transform([text])

        # Probability estimates
        probas = model.predict_proba(features)[0]
        label = int(np.argmax(probas))
        confidence = float(probas[label])
        sentiment = LABEL_MAP.get(label, "unknown")

        return {
            "sentiment": sentiment,
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(probas[i]), 4) for i, name in enumerate(LABEL_NAMES)
            },
        }

    def _predict_bilstm(self, text: str) -> dict[str, Any]:
        """Predict using the BiLSTM model.

        Args:
            text: Pre-cleaned input text.

        Returns:
            Standardised prediction dict.
        """
        import torch

        model = self.models["bilstm"]
        device = next(model.parameters()).device

        # Tokenise
        tokens = text.split()
        max_len = self.bilstm_config.get("max_length", 128)
        unk_idx = self.bilstm_vocab.get("<unk>", 1)
        indices = [self.bilstm_vocab.get(t, unk_idx) for t in tokens[:max_len]]

        # Pad
        pad_idx = self.bilstm_config.get("pad_idx", 0)
        if len(indices) < max_len:
            indices += [pad_idx] * (max_len - len(indices))

        tensor = torch.tensor([indices], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(tensor)
            probas = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        label = int(np.argmax(probas))
        confidence = float(probas[label])
        sentiment = LABEL_MAP.get(label, "unknown")

        return {
            "sentiment": sentiment,
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(probas[i]), 4) for i, name in enumerate(LABEL_NAMES)
            },
        }

    def _predict_finbert(self, text: str) -> dict[str, Any]:
        """Predict using the FinBERT transformer pipeline.

        Args:
            text: Raw input text (FinBERT handles its own tokenisation).

        Returns:
            Standardised prediction dict.
        """
        pipe = self._finbert_pipeline
        results = pipe(text[:512])  # Truncate to model max length

        # HuggingFace returns list of dicts with 'label' and 'score'
        # When top_k=None, results is [[{label, score}, ...]]
        if isinstance(results[0], list):
            results = results[0]

        # Map FinBERT labels to our label scheme
        finbert_map = {"positive": 2, "neutral": 1, "negative": 0}
        prob_dict = {name: 0.0 for name in LABEL_NAMES}

        for entry in results:
            lbl = entry["label"].lower()
            if lbl in prob_dict:
                prob_dict[lbl] = round(float(entry["score"]), 4)

        label = max(finbert_map, key=lambda k: prob_dict[k])
        label_idx = finbert_map[label]
        confidence = prob_dict[label]

        return {
            "sentiment": label,
            "label": label_idx,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    # ------------------------------------------------------------------
    # Fallback BiLSTM Definition
    # ------------------------------------------------------------------

    @staticmethod
    def _get_fallback_bilstm_class() -> type:
        """Return a minimal BiLSTM classifier when src.deep_learning is unavailable."""
        import torch
        import torch.nn as nn

        class BiLSTMClassifier(nn.Module):
            """Bidirectional LSTM text classifier (fallback definition)."""

            def __init__(
                self,
                vocab_size: int,
                embedding_dim: int = 128,
                hidden_dim: int = 256,
                output_dim: int = 3,
                n_layers: int = 2,
                dropout: float = 0.3,
                bidirectional: bool = True,
                pad_idx: int = 0,
            ) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    num_layers=n_layers,
                    bidirectional=bidirectional,
                    dropout=dropout if n_layers > 1 else 0.0,
                    batch_first=True,
                )
                lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(lstm_output_dim, output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                embedded = self.dropout(self.embedding(x))
                output, (hidden, _) = self.lstm(embedded)
                if self.lstm.bidirectional:
                    hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
                else:
                    hidden = hidden[-1]
                hidden = self.dropout(hidden)
                return self.fc(hidden)

        return BiLSTMClassifier
