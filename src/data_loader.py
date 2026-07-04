import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

import requests
import zipfile
import io

def load_financial_phrasebank() -> pd.DataFrame:
    """
    Loads the Financial PhraseBank dataset directly from its zip file.
    Uses the 'Sentences_AllAgree' split.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['sentence', 'label', 'label_name']
    """
    logger.info("Downloading Financial PhraseBank dataset (sentences_allagree)...")
    url = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
    r = requests.get(url)
    r.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open('FinancialPhraseBank-v1.0/Sentences_AllAgree.txt') as f:
            lines = f.read().decode('iso-8859-1').splitlines()
            
    sentences = []
    labels = []
    for line in lines:
        if '@' in line:
            parts = line.rsplit('@', 1)
            if len(parts) == 2:
                sentences.append(parts[0].strip())
                labels.append(parts[1].strip())
                
    df = pd.DataFrame({'sentence': sentences, 'label_name': labels})
    
    # Map labels: 0=negative, 1=neutral, 2=positive
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['label_name'].map(label_map)
    
    logger.info(f"Loaded {len(df)} sentences.")
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """
    Splits the dataframe into train, validation, and test sets.
    
    Args:
        df: Input dataframe
        test_size: Proportion of dataset to include in test split
        val_size: Proportion of train set to include in validation split
        random_state: Random seed
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info("Splitting data into train/val/test sets (stratified)...")
    
    # First split: Train+Val and Test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # Second split: Train and Val
    # Adjust val_size to be relative to the original dataset size
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, random_state=random_state, stratify=train_val_df['label']
    )
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    return train_df, val_df, test_df
