import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    """Computes standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm
    }

def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Creates a comparison DataFrame from a dictionary of model results."""
    records = []
    for model_name, metrics in results_dict.items():
        records.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "F1 (Macro)": metrics["f1_macro"],
            "F1 (Weighted)": metrics["f1_weighted"]
        })
    return pd.DataFrame(records).sort_values(by="F1 (Weighted)", ascending=False)

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    """Plots a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_model_comparison(comparison_df: pd.DataFrame, save_path=None):
    """Plots grouped bar chart for model metrics comparison."""
    plt.figure(figsize=(10, 6))
    melted_df = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette='viridis')
    plt.title("Model Comparison")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_training_history(history: Dict[str, list], save_path=None):
    """Plots training and validation loss/accuracy."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
