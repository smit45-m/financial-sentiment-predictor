import os
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .evaluation import compute_metrics

logger = logging.getLogger(__name__)

class ClassicalMLPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        self.xgb_model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            use_label_encoder=False
        )
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.is_fitted = False

    def build_tfidf_features(self, train_texts, val_texts, test_texts):
        logger.info("Building TF-IDF features...")
        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)
        X_test = self.vectorizer.transform(test_texts)
        self.is_fitted = True
        return X_train, X_val, X_test

    def train_logistic_regression(self, X_train, y_train):
        logger.info("Training Logistic Regression...")
        self.lr_model.fit(X_train, y_train)

    def train_xgboost(self, X_train, y_train):
        logger.info("Training XGBoost...")
        self.xgb_model.fit(X_train, y_train)

    def evaluate(self, model, X_test, y_test, model_name: str):
        logger.info(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        return metrics

    def save_models(self, path: str = 'saved_models'):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(path, 'tfidf_vectorizer.joblib'))
        joblib.dump(self.lr_model, os.path.join(path, 'logistic_regression.joblib'))
        joblib.dump(self.xgb_model, os.path.join(path, 'xgboost_model.joblib'))
        # Save a mock label encoder for completeness if downstream tasks expect it
        joblib.dump(self.label_mapping, os.path.join(path, 'label_encoder.joblib'))
        logger.info(f"Models saved to {path}")

    def load_models(self, path: str = 'saved_models'):
        self.vectorizer = joblib.load(os.path.join(path, 'tfidf_vectorizer.joblib'))
        self.lr_model = joblib.load(os.path.join(path, 'logistic_regression.joblib'))
        self.xgb_model = joblib.load(os.path.join(path, 'xgboost_model.joblib'))
        self.label_mapping = joblib.load(os.path.join(path, 'label_encoder.joblib'))
        self.is_fitted = True
        logger.info(f"Models loaded from {path}")

    def predict(self, text: str, model_type: str = 'xgboost'):
        """Predicts sentiment for a single string.
        Args:
            text: Cleaned text string
            model_type: 'lr' or 'xgboost'
        """
        if not self.is_fitted:
            raise ValueError("Models are not loaded or fitted yet.")
        
        X = self.vectorizer.transform([text])
        
        if model_type == 'lr':
            model = self.lr_model
        else:
            model = self.xgb_model
            
        pred_idx = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        
        return {
            "label": pred_idx,
            "sentiment": self.label_mapping[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2])
            }
        }
