import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Cleans financial text for modeling.
    - Lowercases text
    - Keeps alphanumeric characters, basic punctuation, $, and %
    - Removes extra whitespace
    
    Args:
        text (str): Raw text string
        
    Returns:
        str: Cleaned text string
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Keep only alphanumeric and specific punctuation relevant for finance (., %, $)
    # Replace anything else with space
    text = re.sub(r'[^a-z0-9\s.,%$]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text cleaning to the 'sentence' column of the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with a new 'cleaned_sentence' column
    """
    logger.info("Preprocessing text data...")
    df_clean = df.copy()
    df_clean['cleaned_sentence'] = df_clean['sentence'].apply(clean_text)
    return df_clean
