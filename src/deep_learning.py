import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from tqdm import tqdm
from .evaluation import compute_metrics
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {'sequence': torch.tensor(self.sequences[idx], dtype=torch.long)}
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 3)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Concat the final forward and backward hidden state
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.dropout(hidden_cat)
        out = self.fc(out)
        return out

class BiLSTMPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.config = {
            "vocab_size": 2,
            "embed_dim": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "max_len": 128
        }
        self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        logger.info(f"Using device: {self.device}")

    def build_vocab(self, texts: List[str], max_vocab: int = 20000):
        logger.info("Building vocabulary...")
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
                
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words[:max_vocab-2]):
            self.vocab[word] = i + 2
            
        self.config["vocab_size"] = len(self.vocab)
        logger.info(f"Vocabulary built with {self.config['vocab_size']} tokens.")

    def texts_to_sequences(self, texts: List[str], max_len: int = 128):
        self.config["max_len"] = max_len
        sequences = []
        for text in texts:
            seq = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [self.vocab["<PAD>"]] * (max_len - len(seq))
            sequences.append(seq)
        return sequences

    def train(self, train_texts, train_labels, val_texts, val_labels, epochs=15, batch_size=64, lr=1e-3):
        self.build_vocab(train_texts)
        
        train_seqs = self.texts_to_sequences(train_texts)
        val_seqs = self.texts_to_sequences(val_texts)
        
        train_dataset = TextDataset(train_seqs, train_labels)
        val_dataset = TextDataset(val_seqs, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        self.model = BiLSTMClassifier(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"]
        ).to(self.device)
        
        # Calculate class weights for imbalanced dataset
        class_counts = np.bincount(train_labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        logger.info("Starting BiLSTM training...")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                seqs = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(seqs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    seqs = batch['sequence'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(seqs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Could save best model state here, but we will save at the end or use best state
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    self.model.load_state_dict(best_model_state)
                    break
                    
        return history

    def evaluate(self, test_texts, test_labels):
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        test_seqs = self.texts_to_sequences(test_texts)
        test_dataset = TextDataset(test_seqs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                seqs = batch['sequence'].to(self.device)
                labels = batch['label'].numpy()
                outputs = self.model(seqs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels)
                
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        return metrics

    def save_model(self, path: str = 'saved_models'):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'bilstm_model.pt'))
        with open(os.path.join(path, 'vocab.json'), 'w') as f:
            json.dump(self.vocab, f)
        with open(os.path.join(path, 'bilstm_config.json'), 'w') as f:
            json.dump(self.config, f)
        logger.info(f"BiLSTM model saved to {path}")

    def load_model(self, path: str = 'saved_models'):
        with open(os.path.join(path, 'vocab.json'), 'r') as f:
            self.vocab = json.load(f)
        with open(os.path.join(path, 'bilstm_config.json'), 'r') as f:
            self.config = json.load(f)
            
        self.model = BiLSTMClassifier(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"]
        ).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(path, 'bilstm_model.pt'), map_location=self.device))
        self.model.eval()
        logger.info(f"BiLSTM model loaded from {path}")

    def predict(self, text: str) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
            
        seq = self.texts_to_sequences([text])[0]
        tensor_seq = torch.tensor([seq], dtype=torch.long).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor_seq)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        pred_idx = int(np.argmax(probs))
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
