import os
import json
import logging
import argparse
from src.data_loader import load_financial_phrasebank, split_data
from src.preprocessing import preprocess_dataframe
from src.classical_ml import ClassicalMLPipeline
from src.deep_learning import BiLSTMPipeline
from src.transformer_model import FinBERTSentimentAnalyzer
from src.evaluation import compare_models, plot_confusion_matrix, plot_model_comparison
from src.explainability import get_feature_importance, plot_feature_importance

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    logger.info("--- Phase 1: Data Loading & Preprocessing ---")
    df = load_financial_phrasebank()
    df = preprocess_dataframe(df)
    
    train_df, val_df, test_df = split_data(df)
    
    train_texts = train_df['cleaned_sentence'].tolist()
    train_labels = train_df['label'].tolist()
    val_texts = val_df['cleaned_sentence'].tolist()
    val_labels = val_df['label'].tolist()
    test_texts = test_df['cleaned_sentence'].tolist()
    test_labels = test_df['label'].tolist()
    
    labels_names = ['negative', 'neutral', 'positive']
    all_results = {}
    
    logger.info("--- Phase 2: Classical ML ---")
    classical_pipeline = ClassicalMLPipeline()
    X_train, X_val, X_test = classical_pipeline.build_tfidf_features(train_texts, val_texts, test_texts)
    
    # Logistic Regression
    classical_pipeline.train_logistic_regression(X_train, train_labels)
    lr_metrics = classical_pipeline.evaluate(classical_pipeline.lr_model, X_test, test_labels, "Logistic Regression")
    all_results["Logistic Regression"] = lr_metrics
    plot_confusion_matrix(test_labels, classical_pipeline.lr_model.predict(X_test), labels_names, "Logistic Regression CM", "outputs/figures/cm_lr.png")
    
    # XGBoost
    classical_pipeline.train_xgboost(X_train, train_labels)
    xgb_metrics = classical_pipeline.evaluate(classical_pipeline.xgb_model, X_test, test_labels, "XGBoost")
    all_results["XGBoost"] = xgb_metrics
    plot_confusion_matrix(test_labels, classical_pipeline.xgb_model.predict(X_test), labels_names, "XGBoost CM", "outputs/figures/cm_xgb.png")
    
    # Feature Importance for LR
    lr_importance = get_feature_importance(classical_pipeline.lr_model, classical_pipeline.vectorizer)
    plot_feature_importance(lr_importance, "Top Features (Logistic Regression)", "outputs/figures/feature_importance_lr.png")
    
    classical_pipeline.save_models()
    
    if not args.skip_dl:
        logger.info("--- Phase 3: Deep Learning (BiLSTM) ---")
        bilstm_pipeline = BiLSTMPipeline()
        history = bilstm_pipeline.train(train_texts, train_labels, val_texts, val_labels, epochs=args.epochs)
        
        from src.evaluation import plot_training_history
        plot_training_history(history, "outputs/figures/bilstm_training.png")
        
        bilstm_metrics = bilstm_pipeline.evaluate(test_texts, test_labels)
        all_results["BiLSTM"] = bilstm_metrics
        bilstm_pipeline.save_model()
        
        # We don't have easy CM for BiLSTM without re-running predict, but we can extract it from metrics
        # plot_confusion_matrix(test_labels, bilstm_preds, labels_names, "BiLSTM CM", "outputs/figures/cm_bilstm.png")
        
    if not args.skip_transformer:
        logger.info("--- Phase 4: Transformer (FinBERT) ---")
        finbert_analyzer = FinBERTSentimentAnalyzer()
        if finbert_analyzer.is_loaded:
            finbert_metrics = finbert_analyzer.evaluate(test_texts, test_labels)
            all_results["FinBERT"] = finbert_metrics
            
            # Predict for CM
            preds = finbert_analyzer.predict_batch(test_texts)
            y_pred = [p['label'] for p in preds]
            plot_confusion_matrix(test_labels, y_pred, labels_names, "FinBERT CM", "outputs/figures/cm_finbert.png")
    
    logger.info("--- Phase 5: Evaluation & Comparison ---")
    comparison_df = compare_models(all_results)
    print("\n--- Model Comparison ---")
    print(comparison_df.to_string(index=False))
    
    plot_model_comparison(comparison_df, "outputs/figures/model_comparison.png")
    
    # Save results dictionary
    with open('saved_models/model_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    logger.info("Training complete. All models and artifacts saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models for Financial Sentiment Analysis")
    parser.add_argument('--skip-dl', action='store_true', help="Skip BiLSTM training")
    parser.add_argument('--skip-transformer', action='store_true', help="Skip FinBERT inference")
    parser.add_argument('--epochs', type=int, default=15, help="Epochs for BiLSTM")
    args = parser.parse_args()
    
    main(args)
