import logging
import numpy as np
from transformers import pipeline
from typing import List, Dict, Any
from .evaluation import compute_metrics

logger = logging.getLogger(__name__)

class FinBERTSentimentAnalyzer:
    def __init__(self):
        logger.info("Initializing FinBERT Pipeline...")
        # ProsusAI/finbert predicts: positive, negative, neutral
        try:
            self.pipe = pipeline('sentiment-analysis', model='ProsusAI/finbert')
            self.is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.is_loaded = False
            
        # Target mapping: 0=negative, 1=neutral, 2=positive
        self.label_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        self.idx_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            raise ValueError("FinBERT model is not loaded.")
            
        result = self.pipe(text, top_k=None) # Get all scores
        # Returns list of dicts like: [{'label': 'positive', 'score': 0.9}, ...]
        
        # Sort by score to get top prediction
        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]
            
        top_pred = max(result, key=lambda x: x['score'])
        top_label_str = top_pred['label']
        label_idx = self.label_map[top_label_str]
        
        probs = {0: 0.0, 1: 0.0, 2: 0.0}
        for item in result:
            idx = self.label_map[item['label']]
            probs[idx] = float(item['score'])
            
        return {
            "label": label_idx,
            "sentiment": self.idx_to_label[label_idx],
            "confidence": float(top_pred['score']),
            "probabilities": {
                "negative": probs[0],
                "neutral": probs[1],
                "positive": probs[2]
            },
            "raw_label": top_label_str
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        if not self.is_loaded:
            raise ValueError("FinBERT model is not loaded.")
            
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self.pipe(batch_texts, top_k=None)
            
            for res in batch_results:
                if isinstance(res, list): # For older transformers versions
                    top_pred = max(res, key=lambda x: x['score'])
                    probs = {self.label_map[item['label']]: float(item['score']) for item in res}
                else:
                    top_pred = max(res, key=lambda x: x['score'])
                    probs = {self.label_map[item['label']]: float(item['score']) for item in res}
                    
                label_idx = self.label_map[top_pred['label']]
                results.append({
                    "label": label_idx,
                    "sentiment": self.idx_to_label[label_idx],
                    "confidence": float(top_pred['score']),
                    "probabilities": {
                        "negative": probs.get(0, 0.0),
                        "neutral": probs.get(1, 0.0),
                        "positive": probs.get(2, 0.0)
                    },
                    "raw_label": top_pred['label']
                })
        return results

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        logger.info("Evaluating FinBERT...")
        if not self.is_loaded:
            raise ValueError("FinBERT model is not loaded.")
            
        predictions = self.predict_batch(texts)
        y_pred = [p['label'] for p in predictions]
        
        y_prob = []
        for p in predictions:
            probs = p['probabilities']
            y_prob.append([probs['negative'], probs['neutral'], probs['positive']])
            
        y_prob = np.array(y_prob)
        metrics = compute_metrics(labels, y_pred, y_prob)
        return metrics
