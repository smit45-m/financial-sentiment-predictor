import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import logging
from src.data_loader import load_financial_phrasebank
from src.preprocessing import preprocess_dataframe

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directories exist
os.makedirs('outputs/figures', exist_ok=True)

def plot_class_distribution(df):
    """Plots the distribution of sentiment classes."""
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='label_name', order=['negative', 'neutral', 'positive'], palette='Set2')
    
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%\n({p.get_height()})'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom')
        
    plt.title('Sentiment Class Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('outputs/figures/class_distribution.png')
    plt.close()
    logger.info("Saved class_distribution.png")

def plot_text_length(df):
    """Plots distribution of text lengths."""
    df['text_length'] = df['sentence'].apply(len)
    df['word_count'] = df['sentence'].apply(lambda x: len(x.split()))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Text Length Histogram
    sns.histplot(data=df, x='text_length', bins=50, ax=axes[0,0], kde=True)
    axes[0,0].set_title('Character Length Distribution')
    
    # Word Count Histogram
    sns.histplot(data=df, x='word_count', bins=30, ax=axes[0,1], kde=True)
    axes[0,1].set_title('Word Count Distribution')
    
    # Text Length by Class Boxplot
    sns.boxplot(data=df, x='label_name', y='text_length', ax=axes[1,0], order=['negative', 'neutral', 'positive'])
    axes[1,0].set_title('Character Length by Sentiment')
    
    # Word Count by Class Boxplot
    sns.boxplot(data=df, x='label_name', y='word_count', ax=axes[1,1], order=['negative', 'neutral', 'positive'])
    axes[1,1].set_title('Word Count by Sentiment')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/text_length_analysis.png')
    plt.close()
    logger.info("Saved text_length_analysis.png")

def plot_word_clouds(df):
    """Generates word clouds for each sentiment class."""
    for label in ['negative', 'neutral', 'positive']:
        text = " ".join(df[df['label_name'] == label]['cleaned_sentence'])
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud: {label.capitalize()} Sentiment')
        plt.tight_layout()
        plt.savefig(f'outputs/figures/wordcloud_{label}.png')
        plt.close()
        logger.info(f"Saved wordcloud_{label}.png")

def plot_top_words(df, n=20):
    """Plots the top N most frequent words for each class."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, label in enumerate(['negative', 'neutral', 'positive']):
        text = " ".join(df[df['label_name'] == label]['cleaned_sentence'])
        words = [word for word in text.split() if len(word) > 2] # simple filter
        top_words = Counter(words).most_common(n)
        
        words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
        sns.barplot(data=words_df, y='Word', x='Count', ax=axes[i], palette='viridis')
        axes[i].set_title(f'Top {n} Words: {label.capitalize()}')
        
    plt.tight_layout()
    plt.savefig('outputs/figures/top_words.png')
    plt.close()
    logger.info("Saved top_words.png")

if __name__ == "__main__":
    logger.info("Starting Exploratory Data Analysis...")
    
    # Load data
    df = load_financial_phrasebank()
    
    # Preprocess
    df = preprocess_dataframe(df)
    
    # Print basic stats
    print("\n--- Dataset Statistics ---")
    print(f"Total samples: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nClass Distribution:\n{df['label_name'].value_counts(normalize=True)*100}")
    
    # Generate Plots
    logger.info("Generating plots...")
    plot_class_distribution(df)
    plot_text_length(df)
    plot_word_clouds(df)
    plot_top_words(df)
    
    logger.info("EDA complete. Check outputs/figures/ directory for visualizations.")
